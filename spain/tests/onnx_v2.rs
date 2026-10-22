//! End-to-end statement and witness binding for the versioned ONNX frontend.
use model::AFloat;
use protocol::machine::ProtocolState;
use spain::{
    actor::{SpainMessage, SpainProver, SpainVerifier},
    onnx_v2::{Artifact, OnnxV2Executor, denominator, execute_forward},
    prover::ProverState,
    traits::R1CSInstance,
    verifier::VerifierState,
};
use std::{
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/onnx_v2")
}

#[derive(Clone, Copy)]
enum Mutation {
    None,
    Primary,
    Tokens,
    Claim,
    Embedding,
    Secondary,
}

fn protocol(mutation: Mutation) {
    let artifact = Artifact::load(fixture()).unwrap();
    let inputs = artifact.input_tensors(&[1, 3]).unwrap();
    let mut prepared = execute_forward(&artifact, &inputs, 75).unwrap();
    let mut claims = prepared.claims.clone();
    if matches!(mutation, Mutation::Primary) {
        prepared.primary[0] += 1_i128 << 75;
    }
    if matches!(mutation, Mutation::Claim) {
        claims.values_mut().next().unwrap()[0] += 1_i128 << 75;
    }
    let verifier_inputs = if matches!(mutation, Mutation::Tokens) {
        artifact.input_tensors(&[2, 3]).unwrap()
    } else {
        inputs.clone()
    };
    let v = OnnxV2Executor::verifier(artifact.clone(), &verifier_inputs, &claims, 75).unwrap();
    let mut p = OnnxV2Executor::prover(artifact.clone(), &inputs, prepared, 75).unwrap();
    if matches!(mutation, Mutation::Embedding) {
        p.public[artifact.meta.public.gathers[0].offset] += 1_i128 << 75;
    }
    let meta = p.get_meta();
    let mut prover = SpainProver::new(ProverState::new(p, denominator(75), meta.clone(), 1, false));
    let mut verifier = SpainVerifier::new(VerifierState::<i128, AFloat, _>::new(
        2f64.powi(-40),
        1,
        75,
        30000,
        128,
        16,
        false,
        v,
        meta,
    ));
    let mut message = verifier.init_message().unwrap();
    let mut to_prover = true;
    let mut altered = false;
    for _ in 0..300 {
        if !to_prover && matches!(mutation, Mutation::Secondary) && !altered {
            if let SpainMessage::BoundErrorClaim {
                secondary_openings, ..
            } = &mut message
            {
                secondary_openings[0].values[0] += 1_i128 << 75;
                altered = true;
            }
        }
        let next = if to_prover {
            prover.handle_message(&message)
        } else {
            verifier.handle_message(&message)
        };
        match next {
            Some(next) => {
                message = next;
                to_prover = !to_prover;
            }
            None => break,
        }
    }
    if matches!(mutation, Mutation::Secondary) {
        assert!(altered, "test did not reach secondary binding");
    }
    assert!(prover.is_done() && verifier.is_done());
}

#[test]
fn complete_onnx_protocol_accepts_real_computation() {
    protocol(Mutation::None);
}

#[test]
fn protocol_rejects_changed_witness_statement_and_secondary() {
    for mutation in [
        Mutation::Primary,
        Mutation::Tokens,
        Mutation::Claim,
        Mutation::Embedding,
        Mutation::Secondary,
    ] {
        let error = catch_unwind(AssertUnwindSafe(|| protocol(mutation)))
            .expect_err("a changed computation or statement was accepted");
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            !message.contains("test did not reach"),
            "mutation was not exercised"
        );
    }
}

#[test]
fn malformed_tokens_and_nonfinite_quantization_fail_closed() {
    let artifact = Artifact::load(fixture()).unwrap();
    assert!(artifact.input_tensors(&[1]).is_err());
    assert!(artifact.input_tensors(&[-1, 3]).is_err());
    let out_of_range = artifact.input_tensors(&[i64::MAX, 3]).unwrap();
    assert!(execute_forward(&artifact, &out_of_range, 75).is_err());
    let nan = AFloat(rug::Float::with_val(128, f64::NAN));
    assert!(spain::onnx_v2::quantize(&nan, &denominator(75)).is_err());
}

#[test]
fn artifact_hashes_bind_weights_projections_layout_and_constraints() {
    let root = std::env::temp_dir().join(format!("spain-onnx-v2-tamper-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    for file in fs::read_dir(fixture()).unwrap() {
        let file = file.unwrap();
        if file.file_type().unwrap().is_file() {
            fs::copy(file.path(), root.join(file.file_name())).unwrap();
        }
    }
    Artifact::load(&root).unwrap();
    for name in [
        "original_model.onnx",
        "primary_model.onnx",
        "projection_model.onnx",
        "secondary_model.onnx",
        "layout.json",
        "A.bin",
        "B.bin",
        "C.bin",
    ] {
        let path = root.join(name);
        let original = fs::read(&path).unwrap();
        let mut changed = original.clone();
        let last = changed.len() - 1;
        changed[last] ^= 1;
        fs::write(&path, &changed).unwrap();
        assert!(Artifact::load(&root).is_err(), "unbound artifact: {name}");
        fs::write(path, original).unwrap();
    }
    fs::remove_dir_all(root).unwrap();
}
