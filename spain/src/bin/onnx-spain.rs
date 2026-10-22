//! Execute compiled ONNX and drive the existing Spain protocol actors.
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use model::AFloat;
use protocol::machine::ProtocolState;
use serde_json::{Value, json};
use spain::{
    actor::{SpainMessage, SpainProver, SpainVerifier},
    onnx_v2::{Artifact, OnnxV2Executor, denominator, execute_forward, file_sha256},
    prover::ProverState,
    traits::R1CSInstance,
    verifier::VerifierState,
};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(about = "High-precision ONNX execution and application-bound Spain proofs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Forward(Options),
    Prove(Options),
}
#[derive(clap::Args)]
struct Options {
    #[arg(long)]
    artifact_dir: PathBuf,
    #[arg(long)]
    input_ids: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 75)]
    scale_bits: usize,
    #[arg(long, default_value_t = 9.094947017729282e-13)]
    epsilon: f64,
    #[arg(long, default_value_t = 256)]
    dark_chunks: usize,
    #[arg(long, default_value_t = 200)]
    memory_budget_gib: usize,
    #[arg(long, default_value_t = 30)]
    disk_budget_gib: usize,
}
fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let temp = path.with_extension("json.tmp");
    let mut writer = BufWriter::new(File::create(&temp)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    fs::rename(temp, path)?;
    Ok(())
}
fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")
                .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
        })
}
fn executable_sha256() -> Result<String> {
    // Opening proc's executable link hashes the running inode even if another
    // build replaces its pathname while a long proof is executing.
    #[cfg(target_os = "linux")]
    let path = PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe()?;
    file_sha256(&path)
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Forward(opts) => run(opts, false),
        Command::Prove(opts) => run(opts, true),
    }
}
fn run(opts: Options, prove: bool) -> Result<()> {
    let all_started = Instant::now();
    ensure!(
        opts.threads > 0 && opts.scale_bits <= 100,
        "invalid numerical/thread parameters"
    );
    ensure!(
        opts.epsilon.is_finite() && opts.epsilon > 0.,
        "invalid epsilon"
    );
    ensure!(
        opts.dark_chunks.is_power_of_two(),
        "DARK chunks must be a positive power of two"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build_global()?;
    fs::create_dir_all(&opts.output_dir)?;
    for name in [
        "statement.json",
        "forward.json",
        "primary_witness.i128le",
        "transcript.jsonl",
        "proof_report.json",
    ] {
        ensure!(
            !opts.output_dir.join(name).exists(),
            "output directory contains previous {name}; use a fresh run directory"
        );
    }
    eprintln!("Authenticating compiled ONNX artifacts");
    let artifact = Artifact::load(&opts.artifact_dir)?;
    let m = &artifact.meta;
    let nnz = m.matrix_entries.values().sum::<usize>() as u64;
    let witness = m.protocol_metadata().get_witness_size() as u64;
    let asset_bytes = m.artifacts.iter().map(|a| a.bytes).sum::<u64>();
    // COO records occupy 32 bytes in memory (two usize indices and i128).
    // Allow four simultaneous copies across the two actors, conversion and
    // replacement. Row products use three I512 arrays; the graph has at least
    // as many witness values as rows in the supported compiled artifacts.
    // The second term covers the larger of those arrays and sum-check buffers.
    // Eight GiB additionally covers mmap weights, DARK bases and activations.
    let row_or_witness = witness.max(m.num_constraints as u64);
    let estimated_memory = nnz.saturating_mul(if prove { 128 } else { 0 })
        + row_or_witness.saturating_mul(if prove { 192 } else { 128 })
        + (8u64 << 30);
    let estimated_disk = asset_bytes
        + (m.num_witness_values as u64) * 16
        + (m.num_public_values as u64 + m.num_secondary_witness_values as u64) * 160
        + (2u64 << 30);
    atomic_json(
        &opts.output_dir.join("resource_estimate.json"),
        &json!({
        "constraints":m.num_constraints,"matrix_entries":m.matrix_entries,
        "witness_values":witness,"asset_bytes":asset_bytes,
        "estimated_peak_bytes":estimated_memory,"estimated_artifact_bytes":estimated_disk,
        "memory_budget_gib":opts.memory_budget_gib,"disk_budget_gib":opts.disk_budget_gib,
        "estimate_method":"128*COO_entries + 192*max(rows,witness) + 8GiB for proof; includes both actors"}),
    )?;
    ensure!(
        estimated_memory <= (opts.memory_budget_gib as u64) << 30,
        "estimated memory exceeds budget"
    );
    ensure!(
        estimated_disk <= (opts.disk_budget_gib as u64) << 30,
        "estimated disk exceeds budget"
    );
    let raw: Value = serde_json::from_reader(File::open(&opts.input_ids)?)?;
    let ids: Vec<i64> = serde_json::from_value(if raw.is_array() {
        raw
    } else {
        raw.get("input_ids")
            .context("input JSON requires input_ids")?
            .clone()
    })?;
    let inputs = artifact.input_tensors(&ids)?;
    eprintln!("Executing complete ONNX with MPFR-128");
    let timer = Instant::now();
    let prepared = execute_forward(&artifact, &inputs, opts.scale_bits)?;
    let forward_seconds = timer.elapsed().as_secs_f64();
    atomic_json(
        &opts.output_dir.join("forward.json"),
        &json!({
        "schema_version":2,"input_ids":ids,"fingerprint_sha256":m.fingerprint_sha256,
        "arithmetic":"MPFR-128 ONNX","precision_bits":128,
        "outputs":prepared.outputs,"nodes_executed":prepared.nodes_executed,
        "peak_live_elements":prepared.peak_live_elements,"forward_seconds":forward_seconds,
        "peak_rss_kib":peak_rss_kib()}),
    )?;
    if !prove {
        return Ok(());
    }
    let claims: HashMap<String, Vec<i128>> = prepared.claims.clone();
    let scaled: BTreeMap<_, _> = claims
        .iter()
        .map(|(name, values)| {
            (
                name.clone(),
                values.iter().map(ToString::to_string).collect::<Vec<_>>(),
            )
        })
        .collect();
    atomic_json(
        &opts.output_dir.join("statement.json"),
        &json!({
        "schema":"spain-onnx-statement-v2","fingerprint_sha256":m.fingerprint_sha256,
        "input_ids":ids,"positions":(0..ids.len()).collect::<Vec<_>>(),
        "scale_bits":opts.scale_bits,"epsilon":opts.epsilon,"claims_scaled":scaled,
        "status":"claim_fixed_before_challenge","arithmetic":"MPFR-128 ONNX"}),
    )?;
    let timer = Instant::now();
    let verifier_exec =
        OnnxV2Executor::verifier(artifact.clone(), &inputs, &claims, opts.scale_bits)?;
    let verifier_frontend_seconds = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let prover_exec = OnnxV2Executor::prover(artifact.clone(), &inputs, prepared, opts.scale_bits)?;
    let prover_frontend_seconds = timer.elapsed().as_secs_f64();
    let metadata = prover_exec.get_meta();
    ensure!(
        opts.dark_chunks < metadata.num_witness_values.next_power_of_two(),
        "DARK chunks must be less than padded witness length"
    );
    let mut primary_file = BufWriter::new(File::create(
        opts.output_dir.join("primary_witness.i128le"),
    )?);
    prover_exec.write_primary(&mut primary_file)?;
    primary_file.flush()?;
    let mut sources = BTreeMap::new();
    for name in [
        "tools/qwen.py",
        "tools/qwen_onnx.py",
        "circuit/onnx_v2.py",
        "model/src/onnx_v2.rs",
        "spain/src/onnx_v2.rs",
        "spain/src/bin/onnx-spain.rs",
        "spain/src/actor.rs",
        "spain/src/prover.rs",
        "spain/src/verifier.rs",
        "spain/src/traits.rs",
        "spain/src/inputs.rs",
        "Cargo.lock",
    ] {
        if Path::new(name).is_file() {
            sources.insert(name, file_sha256(Path::new(name))?);
        }
    }
    atomic_json(
        &opts.output_dir.join("implementation_manifest.json"),
        &json!({
        "executable_sha256":executable_sha256()?,
        "cargo_lock_sha256":file_sha256(Path::new("Cargo.lock"))?,
        "artifact_fingerprint":m.fingerprint_sha256,
        "compiled_artifact_dir":artifact.directory,
        "threads":opts.threads,"source_sha256":sources}),
    )?;
    eprintln!(
        "Initializing Spain: {} constraints, {} primary values",
        m.num_constraints, m.num_witness_values
    );
    let timer = Instant::now();
    let p = ProverState::new(
        prover_exec,
        denominator(opts.scale_bits),
        metadata.clone(),
        1,
        false,
    );
    let prover_setup_seconds = timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let v = VerifierState::<i128, AFloat, _>::new(
        opts.epsilon,
        1,
        opts.scale_bits,
        30000,
        128,
        opts.dark_chunks,
        false,
        verifier_exec,
        metadata.clone(),
    );
    let verifier_setup_seconds = timer.elapsed().as_secs_f64();
    let mut prover = SpainProver::new(p);
    let mut verifier = SpainVerifier::new(v);
    let mut transcript = BufWriter::new(File::create(opts.output_dir.join("transcript.jsonl"))?);
    let protocol_started = Instant::now();
    let mut prover_seconds = 0.;
    let mut verifier_seconds = 0.;
    let mut prover_serialization_seconds = 0.;
    let mut verifier_serialization_seconds = 0.;
    let mut prover_bytes = 0usize;
    let mut verifier_bytes = 0usize;
    let mut proof_discounted = 0usize;
    let mut messages = 0;
    let mut squared_error = None;
    let timer = Instant::now();
    let mut message = verifier.init_message()?;
    verifier_seconds += timer.elapsed().as_secs_f64();
    let mut to_prover = true;
    loop {
        let timer = Instant::now();
        let encoded = serde_json::to_vec(&message)?;
        if to_prover {
            verifier_serialization_seconds += timer.elapsed().as_secs_f64();
            verifier_bytes += encoded.len();
        } else {
            prover_serialization_seconds += timer.elapsed().as_secs_f64();
            prover_bytes += encoded.len();
            proof_discounted += match &message {
                SpainMessage::WitnessOpenings(openings) => {
                    serde_json::to_vec(&SpainMessage::WitnessOpenings(
                        openings
                            .iter()
                            .filter(|o| o.range_index != 0)
                            .cloned()
                            .collect(),
                    ))?
                    .len()
                }
                _ => encoded.len(),
            };
        }
        transcript.write_all(&encoded)?;
        transcript.write_all(b"\n")?;
        messages += 1;
        match &message {
            SpainMessage::ErrorClaim(error) => squared_error = Some(error.clone()),
            SpainMessage::BoundErrorClaim {
                squared_error: error,
                ..
            } => squared_error = Some(error.clone()),
            _ => {}
        }
        eprintln!(
            "Protocol message {messages}, {} bytes, to {}",
            encoded.len(),
            if to_prover { "prover" } else { "verifier" }
        );
        let timer = Instant::now();
        let next = if to_prover {
            prover.handle_message(&message)
        } else {
            verifier.handle_message(&message)
        };
        if to_prover {
            prover_seconds += timer.elapsed().as_secs_f64();
        } else {
            verifier_seconds += timer.elapsed().as_secs_f64();
        }
        match next {
            Some(next) => {
                message = next;
                to_prover = !to_prover;
            }
            None => break,
        }
    }
    transcript.flush()?;
    ensure!(
        prover.is_done() && verifier.is_done(),
        "Spain protocol did not finish both actors"
    );
    ensure!(squared_error.is_some(), "missing checked residual");
    atomic_json(
        &opts.output_dir.join("proof_report.json"),
        &json!({
        "measurement_schema_version":2,"verified":true,"protocol":"Spain interactive commitment/sum-check/DARK",
        "input_tokens":ids.len(),"fingerprint_sha256":m.fingerprint_sha256,
        "constraints":m.num_constraints,"primary_values":metadata.num_witness_values,
        "secondary_values":metadata.num_secondary_witness_values,"public_values":metadata.num_public_values,
        "random_values":metadata.num_random_values,"messages":messages,
        "communication_bytes_including_public_statement":prover_bytes+verifier_bytes,
        "proof_size_bytes":prover_bytes,"proof_size_excluding_public_bytes":proof_discounted,
        "prover_message_bytes":prover_bytes,"verifier_message_bytes":verifier_bytes,
        "constraint_squared_error_integer":squared_error,"scale_bits":opts.scale_bits,"epsilon":opts.epsilon,
        "dark_chunks":opts.dark_chunks,"forward_seconds":forward_seconds,
        "prover_protocol_seconds":prover_seconds,"verifier_protocol_seconds":verifier_seconds,
        "prover_setup_seconds":prover_setup_seconds,"verifier_setup_seconds":verifier_setup_seconds,
        "prover_frontend_seconds":prover_frontend_seconds,"verifier_frontend_seconds":verifier_frontend_seconds,
        "prover_seconds":forward_seconds+prover_frontend_seconds+prover_setup_seconds+prover_seconds+prover_serialization_seconds,
        "verifier_seconds":verifier_frontend_seconds+verifier_setup_seconds+verifier_seconds+verifier_serialization_seconds,
        "proof_wall_seconds":protocol_started.elapsed().as_secs_f64(),
        "end_to_end_seconds":all_started.elapsed().as_secs_f64(),"peak_rss_kib":peak_rss_kib(),
        "memory_scope":"whole process including prover, verifier and forward",
        "prover_statistics":format!("{:#?}",prover.get_eval_result()),
        "verifier_statistics":format!("{:#?}",verifier.get_eval_result())}),
    )?;
    eprintln!("Complete ONNX Spain proof VERIFIED");
    Ok(())
}
