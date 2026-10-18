use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use llama_spain::{
    circuit::{LlamaExecutor, model_fingerprint, quantize, scale},
    model::{LlamaModel, write_tiny_fixture},
    trace::Trace,
};
use model::ToPrimitive;
use parse::mat::MatrixData;
use protocol::machine::ProtocolState;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use spain::{
    actor::{SpainMessage, SpainProver, SpainVerifier},
    prover::ProverState,
    traits::R1CSInstance,
    verifier::VerifierState,
};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Complete native Llama execution and Spain interactive proof")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[arg(long, global = true, default_value_t = 8)]
    threads: usize,
}
#[derive(Subcommand)]
enum Command {
    Forward(Options),
    Prove(Options),
    /// Create a deterministic small complete model and exercise the actual proof protocol.
    TinyTest {
        #[arg(long)]
        output_dir: PathBuf,
    },
}
#[derive(clap::Args)]
struct Options {
    #[arg(long, default_value = "/home/yliangh/models/meta-llama/Llama-3.2-1B")]
    model_dir: PathBuf,
    /// JSON array of token IDs or object containing input_ids; no implicit BOS insertion.
    #[arg(long)]
    input_ids: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 75)]
    scale_bits: usize,
    #[arg(long, default_value_t = 9.094947017729282e-13)]
    epsilon: f64,
    #[arg(long, default_value_t = 200)]
    memory_budget_gib: usize,
    #[arg(long, default_value_t = 60)]
    disk_budget_gib: usize,
    /// Power-of-two DARK commitment chunks; larger values reduce setup bases.
    #[arg(long, default_value_t = 256)]
    dark_chunks: usize,
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.threads > 0, "threads must be positive");
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()?;
    match cli.command {
        Command::Forward(options) => run(options, false),
        Command::Prove(options) => run(options, true),
        Command::TinyTest { output_dir } => {
            let dir = output_dir.join("model");
            write_tiny_fixture(&dir)?;
            fs::create_dir_all(&output_dir)?;
            let ids = output_dir.join("input_ids.json");
            write_json(&ids, &json!([1, 4, 7]))?;
            run(
                Options {
                    model_dir: dir,
                    input_ids: ids,
                    output_dir,
                    scale_bits: 75,
                    epsilon: 2f64.powi(-40),
                    memory_budget_gib: 200,
                    disk_budget_gib: 60,
                    dark_chunks: 256,
                },
                true,
            )
        }
    }
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}
fn tokens(path: &Path) -> Result<Vec<u32>> {
    let value: Value = serde_json::from_reader(File::open(path)?)?;
    Ok(serde_json::from_value(if value.is_array() {
        value
    } else {
        value.get("input_ids").context("input_ids missing")?.clone()
    })?)
}
fn digest(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
fn model_identity(directory: &Path) -> Result<Value> {
    let mut names = fs::read_dir(directory)?
        .map(|e| Ok(e?.path()))
        .collect::<Result<Vec<_>>>()?;
    names.sort();
    let mut result = serde_json::Map::new();
    for path in names {
        let name = path.file_name().unwrap().to_string_lossy();
        if name.ends_with(".safetensors")
            || matches!(
                name.as_ref(),
                "config.json"
                    | "tokenizer.json"
                    | "tokenizer_config.json"
                    | "generation_config.json"
                    | "model.safetensors.index.json"
            )
        {
            result.insert(
                name.into_owned(),
                json!({"sha256":digest(&path)?,"bytes":path.metadata()?.len()}),
            );
        }
    }
    Ok(Value::Object(result))
}
fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
fn run(options: Options, prove: bool) -> Result<()> {
    for name in [
        "statement.json",
        "primary_witness.i128le",
        "transcript.jsonl",
        "proof_report.json",
    ] {
        ensure!(
            !options.output_dir.join(name).exists(),
            "output directory already contains {name}; choose a fresh directory to keep each proof run's artifacts consistent"
        );
    }
    ensure!(
        options.epsilon.is_finite() && options.epsilon > 0.,
        "invalid epsilon"
    );
    ensure!(
        options.scale_bits <= 100,
        "fixed point scale would leave insufficient integer headroom"
    );
    ensure!(
        options.dark_chunks.is_power_of_two(),
        "dark-chunks must be a positive power of two"
    );
    fs::create_dir_all(&options.output_dir)?;
    let input = tokens(&options.input_ids)?;
    let model = Arc::new(LlamaModel::load(&options.model_dir)?);
    let layout = model.layout(&input)?;
    let activation_count: usize = layout
        .nodes
        .iter()
        .map(|n| n.shape.iter().product::<usize>())
        .sum();
    let nonlinear_count: usize = layout
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                &n.op,
                llama_spain::trace::Op::Silu { .. } | llama_spain::trace::Op::SoftmaxCausal { .. }
            )
        })
        .map(|n| n.shape.iter().product::<usize>())
        .sum();
    let weight_bytes: u64 = fs::read_dir(&options.model_dir)?
        .map(|entry| -> Result<u64> {
            let path = entry?.path();
            Ok(
                if path
                    .extension()
                    .is_some_and(|extension| extension == "safetensors")
                {
                    path.metadata()?.len()
                } else {
                    0
                },
            )
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum();
    let estimated_bytes = weight_bytes
        + activation_count as u64 * 256
        + if prove {
            nonlinear_count as u64 * 16_384
        } else {
            0
        };
    let estimated_disk = activation_count as u64 * 32
        + if prove {
            nonlinear_count as u64 * 1024
        } else {
            0
        };
    ensure!(
        estimated_bytes <= options.memory_budget_gib as u64 * (1 << 30),
        "estimated working memory exceeds budget: {estimated_bytes} bytes"
    );
    ensure!(
        estimated_disk <= options.disk_budget_gib as u64 * (1 << 30),
        "estimated artifacts exceed disk budget"
    );
    write_json(
        &options.output_dir.join("resource_estimate.json"),
        &json!({"activation_values":activation_count,"nonlinear_values":nonlinear_count,"estimated_peak_bytes":estimated_bytes,"estimated_artifact_bytes":estimated_disk,"memory_budget_gib":options.memory_budget_gib,"disk_budget_gib":options.disk_budget_gib}),
    )?;
    let identity = model_identity(&options.model_dir)?;
    let fingerprint = model_fingerprint(&model)?
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let start = Instant::now();
    let trace = model.forward(&input)?;
    let seconds = start.elapsed().as_secs_f64();
    save_forward(&options.output_dir, &trace, seconds, &identity)?;
    if !prove {
        return Ok(());
    }
    let denominator = scale(options.scale_bits);
    let claim = trace
        .logits()
        .iter()
        .map(|x| quantize(x, &denominator))
        .collect::<Vec<_>>();
    write_json(
        &options.output_dir.join("statement.json"),
        &json!({"schema":"spain-llama-statement-v1","model":identity,"model_fingerprint_sha256":fingerprint,"input_ids":input,"positions":(0..input.len()).collect::<Vec<_>>(),"scale_bits":options.scale_bits,"epsilon":options.epsilon,"logits_scaled":claim.iter().map(ToString::to_string).collect::<Vec<_>>(),"layers":model.config.num_hidden_layers,"vocabulary":model.config.vocab_size,"arithmetic":"MPFR-128 / negative-exp Pade[3/3] / 10 squares","status":"claim_fixed_before_challenge"}),
    )?;
    eprintln!("Building primary circuit and high precision witness");
    let prover_exec = LlamaExecutor::new(model.clone(), trace, &claim, options.scale_bits)?;
    // Derive the verifier's program from its model/config, without executing a forward pass.
    let verifier_exec = LlamaExecutor::new(model.clone(), layout, &claim, options.scale_bits)?;
    let metadata = prover_exec.get_meta();
    ensure!(
        options.dark_chunks < metadata.num_witness_values.next_power_of_two(),
        "dark-chunks must be smaller than the padded primary witness"
    );
    ensure!(
        metadata.get_witness_size() == verifier_exec.get_meta().get_witness_size(),
        "party layouts differ"
    );
    let mut primary = BufWriter::new(File::create(
        options.output_dir.join("primary_witness.i128le"),
    )?);
    if let MatrixData::Dense(values) = prover_exec.primary_witness().data() {
        for value in values.iter() {
            primary.write_all(&value.to_le_bytes())?;
        }
    }
    primary.flush()?;
    write_json(
        &options.output_dir.join("circuit.json"),
        &json!({"num_public_values":metadata.num_public_values,"num_random_values":metadata.num_random_values,"num_primary_witness_values":metadata.num_witness_values,"num_secondary_witness_values":metadata.num_secondary_witness_values,"witness_file":"primary_witness.i128le","encoding":"signed little-endian i128, implicit denominator 2^scale_bits"}),
    )?;
    eprintln!(
        "Running actual Spain protocol (primary witness: {} elements)",
        metadata.num_witness_values
    );
    let p = ProverState::new(prover_exec, denominator, metadata.clone(), 1, false);
    let v = VerifierState::new(
        options.epsilon,
        1,
        options.scale_bits,
        30000,
        128,
        options.dark_chunks,
        false,
        verifier_exec,
        metadata,
    );
    let mut prover = SpainProver::new(p);
    let mut verifier = SpainVerifier::new(v);
    let constraints = prover.num_constraints();
    let mut transcript = BufWriter::new(File::create(options.output_dir.join("transcript.jsonl"))?);
    let (mut message, mut to_prover) = (verifier.init_message()?, true);
    let mut messages = 0usize;
    let mut bytes = 0usize;
    let mut squared_error = None;
    let started = Instant::now();
    loop {
        let encoded = serde_json::to_vec(&message)?;
        bytes += encoded.len();
        messages += 1;
        transcript.write_all(&encoded)?;
        transcript.write_all(b"\n")?;
        match &message {
            SpainMessage::ErrorClaim(j) => squared_error = Some(j.clone()),
            SpainMessage::BoundErrorClaim {
                squared_error: j, ..
            } => squared_error = Some(j.clone()),
            _ => {}
        }
        eprintln!("Protocol message {messages}: {}", message_name(&message));
        let response = if to_prover {
            prover.handle_message(&message)
        } else {
            verifier.handle_message(&message)
        };
        match response {
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
        "protocol ended before both parties completed"
    );
    let p = prover.get_eval_result();
    let v = verifier.get_eval_result();
    write_json(
        &options.output_dir.join("proof_report.json"),
        &json!({"verified":true,"protocol":"Spain interactive commitment/sum-check/DARK","scope":"all decoder layers, final RMSNorm, full-vocabulary last-position logits","layers":model.config.num_hidden_layers,"vocabulary":model.config.vocab_size,"input_tokens":input.len(),"constraints":constraints,"messages":messages,"communication_bytes_including_public_statement":bytes,"constraint_squared_error_integer":squared_error,"scale_bits":options.scale_bits,"epsilon":options.epsilon,"dark_chunks":options.dark_chunks,"proof_wall_seconds":started.elapsed().as_secs_f64(),"forward_seconds":seconds,"peak_rss_kib":peak_rss_kib(),"prover_statistics":format!("{p:#?}"),"verifier_statistics":format!("{v:#?}")}),
    )?;
    eprintln!("Complete model Spain proof VERIFIED");
    Ok(())
}
fn save_forward(directory: &Path, trace: &Trace, seconds: f64, identity: &Value) -> Result<()> {
    let logits = trace
        .logits()
        .iter()
        .map(|x| x.to_f64().unwrap())
        .collect::<Vec<_>>();
    let top = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let mut layers = serde_json::Map::new();
    for node in &trace.nodes {
        if node.name.ends_with("mlp_residual") {
            let index = node.name.split('.').nth(2).unwrap();
            let width = node.shape[1];
            let rows = node
                .values
                .chunks(width)
                .map(|row| row.iter().map(|x| x.to_f64().unwrap()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            layers.insert(format!("layer_{index}"), json!(rows));
        }
    }
    write_json(
        &directory.join("forward.json"),
        &json!({"input_ids":trace.tokens,"logits":logits,"top_token":top,"layer_hidden_states":layers,"model":identity,"precision_bits":128,"forward_seconds":seconds,"peak_rss_kib":peak_rss_kib(),"nodes":trace.nodes.iter().map(|n|json!({"name":n.name,"shape":n.shape,"values":n.values.len()})).collect::<Vec<_>>() }),
    )
}
fn message_name(message: &SpainMessage<i128>) -> &'static str {
    match message {
        SpainMessage::Setup(_) => "setup",
        SpainMessage::Commitment(_) => "commitment",
        SpainMessage::Randomness(_) => "randomness",
        SpainMessage::ErrorClaim(_) => "error",
        SpainMessage::BoundErrorClaim { .. } => "bound error and secondary witness",
        SpainMessage::SampleMont { .. } => "field",
        SpainMessage::WitnessOpenings(_) => "openings",
        SpainMessage::DarkRoundResponse(_) => "DARK response",
        SpainMessage::DarkChallenge(_) => "DARK challenge",
        _ => "sum-check / evaluation",
    }
}
