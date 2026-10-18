//! Cross-component regression tests for the actual full-model arithmetization.
//! These inspect residuals using unbounded integers, so rejection cannot be an
//! artifact of the protocol's deliberately checked I512 overflow handling.
use std::{fs, io::{Read, Seek, SeekFrom, Write}, path::PathBuf, sync::{Arc, OnceLock}};

use llama_spain::{
    circuit::{LlamaExecutor, quantize, scale},
    model::{LlamaModel, write_tiny_fixture},
    numeric,
    trace::{Op, Trace},
};
use model::{AFloat, FromPrimitive};
use rug::Integer;
use spain::{
    inputs::Metadata,
    traits::R1CSInstance,
    verifier::{VerifierState, ZRangeOpening},
};

const BITS: usize = 75;
const EPSILON: f64 = 9.094947017729282e-13; // 2^-40, identical to the real proof.

struct Fixture { model: Arc<LlamaModel>, trace: Trace }
static FIXTURE: OnceLock<Fixture> = OnceLock::new();

fn fixture() -> &'static Fixture {
    FIXTURE.get_or_init(|| {
        let path = temporary("base");
        write_tiny_fixture(&path).unwrap();
        let model = Arc::new(LlamaModel::load(path).unwrap());
        let trace = model.forward(&[1, 4, 7]).unwrap();
        Fixture { model, trace }
    })
}

fn temporary(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("spain-arithmetization-{}-{label}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()))
}

fn claim(trace: &Trace) -> Vec<i128> {
    trace.logits().iter().map(|value| quantize(value, &scale(BITS))).collect()
}

fn challenges(meta: &Metadata) -> (Vec<AFloat>, Vec<i128>) {
    // All coefficients are nonzero dyadics. This avoids a deterministic zero
    // coefficient accidentally hiding the specifically altered test element.
    let values = (0..meta.num_random_values).map(|i| {
        let magnitude = ((i * 37 % 97) + 1) as f64 / 64.0;
        numeric::scalar(if i % 2 == 0 { magnitude } else { -magnitude })
    }).collect::<Vec<_>>();
    let fixed = values.iter().map(|value| quantize(value, &scale(BITS))).collect();
    (values, fixed)
}

fn residual(trace: Trace, public_claim: &[i128]) -> Integer {
    let mut executor = LlamaExecutor::new(fixture().model.clone(), trace, public_claim, BITS).unwrap();
    let meta = executor.get_meta();
    let (random, fixed) = challenges(&meta);
    let (matrices, _) = executor.get_matrices(scale(BITS), Some(&fixed));
    let witness = executor.compute_full_witness(&meta, random, scale(BITS));
    spain::prover::compute_squared_error_hp(&matrices, &witness, &(i256::I512::from(1) << BITS), false)
}

fn accepted(error: &Integer) -> bool {
    std::panic::catch_unwind(|| spain::verifier::epsilon_check_rug(error, BITS, EPSILON, false)).is_ok()
}

#[test]
fn honest_full_decoder_meets_real_proof_epsilon() {
    let trace = fixture().trace.clone();
    assert!(accepted(&residual(trace.clone(), &claim(&trace))));
}

#[test]
fn tampered_outputs_of_every_gadget_fail_the_residual_bound() {
    type Predicate = fn(&Op) -> bool;
    let gadgets: &[(&str, Predicate)] = &[
        ("RMSNorm", |op| matches!(op, Op::RmsNorm { .. })),
        ("RoPE", |op| matches!(op, Op::Rope { .. })),
        ("attention scores", |op| matches!(op, Op::AttentionScores { .. })),
        ("attention values", |op| matches!(op, Op::AttentionValues { .. })),
        ("causal softmax", |op| matches!(op, Op::SoftmaxCausal { .. })),
        ("SiLU", |op| matches!(op, Op::Silu { .. })),
        ("SwiGLU multiplication", |op| matches!(op, Op::Mul { .. })),
        ("residual addition", |op| matches!(op, Op::Add { .. })),
        ("fixed-weight linear", |op| matches!(op, Op::Linear { .. })),
    ];
    for (name, predicate) in gadgets {
        let mut trace = fixture().trace.clone();
        let public_claim = claim(&trace);
        let node = trace.nodes.iter_mut().find(|node| predicate(&node.op)).unwrap();
        node.values[0] = numeric::add(&node.values[0], &numeric::scalar(0.01));
        assert!(!accepted(&residual(trace, &public_claim)), "unconstrained {name}");
    }
}

#[test]
fn future_attention_probability_is_explicitly_constrained_to_zero() {
    let mut trace = fixture().trace.clone();
    let public_claim = claim(&trace);
    let probabilities = trace.nodes.iter_mut().find(|node| matches!(node.op, Op::SoftmaxCausal { .. })).unwrap();
    assert_eq!(probabilities.values[1].0, 0);
    probabilities.values[1] = numeric::scalar(0.01);
    assert!(!accepted(&residual(trace, &public_claim)));
}

#[test]
fn both_final_logits_and_public_logit_claim_are_bound() {
    let original = fixture().trace.clone();
    let public_claim = claim(&original);
    let mut modified = original.clone();
    let logits = modified.logits;
    modified.nodes[logits].values[17] = numeric::add(&modified.nodes[logits].values[17], &numeric::scalar(0.01));
    assert!(!accepted(&residual(modified, &public_claim)));
    let mut wrong_claim = public_claim;
    wrong_claim[17] += quantize(&numeric::scalar(0.01), &scale(BITS));
    assert!(!accepted(&residual(original, &wrong_claim)));
}

#[test]
fn changed_input_token_cannot_reuse_previous_activations() {
    let mut trace = fixture().trace.clone();
    let public_claim = claim(&trace);
    trace.tokens[0] = 2;
    assert!(!accepted(&residual(trace, &public_claim)));
}

#[test]
fn changed_model_weight_cannot_reuse_previous_witness() {
    let path = temporary("changed-weight");
    write_tiny_fixture(&path).unwrap();
    let mut file = fs::OpenOptions::new().read(true).write(true).open(path.join("model.safetensors")).unwrap();
    let mut length = [0u8; 8]; file.read_exact(&mut length).unwrap();
    let header_len = u64::from_le_bytes(length);
    let mut header = vec![0; header_len as usize]; file.read_exact(&mut header).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    let offset = header["model.layers.0.self_attn.q_proj.weight"]["data_offsets"][0].as_u64().unwrap();
    file.seek(SeekFrom::Start(8 + header_len + offset)).unwrap();
    let mut old = [0u8; 4]; file.read_exact(&mut old).unwrap();
    file.seek(SeekFrom::Start(8 + header_len + offset)).unwrap();
    file.write_all(&(f32::from_le_bytes(old) + 0.125).to_le_bytes()).unwrap();
    file.flush().unwrap();
    let model = Arc::new(LlamaModel::load(&path).unwrap());
    let trace = fixture().trace.clone();
    let public_claim = claim(&trace);
    let mut prover = LlamaExecutor::new(fixture().model.clone(), trace, &public_claim, BITS).unwrap();
    let verifier = LlamaExecutor::new(model.clone(), model.layout(&[1, 4, 7]).unwrap(), &public_claim, BITS).unwrap();
    let meta = prover.get_meta(); let (random, fixed) = challenges(&meta);
    prover.get_matrices(scale(BITS), Some(&fixed));
    let witness = prover.compute_full_witness(&meta, random, scale(BITS));
    let (matrices, _) = verifier.get_matrices(scale(BITS), Some(&fixed));
    let error = spain::prover::compute_squared_error_hp(&matrices, &witness, &(i256::I512::from(1) << BITS), false);
    assert!(!accepted(&error));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn different_token_ids_are_bound_even_if_embedding_rows_are_equal() {
    let model = fixture().model.clone();
    // Deterministic fixture rows repeat every 29 tokens, exposing token-ID
    // binding separately from the already checked embedding-value binding.
    let embedding = model.weights.tensor("model.embed_tokens.weight").unwrap();
    for d in 0..8 { assert_eq!(embedding.get_f64(8 + d), embedding.get_f64(30 * 8 + d)); }
    let public_claim = claim(&fixture().trace);
    let correct = LlamaExecutor::new(model.clone(), model.layout(&[1, 4, 7]).unwrap(), &public_claim, BITS).unwrap();
    let wrong = LlamaExecutor::new(model.clone(), model.layout(&[30, 4, 7]).unwrap(), &public_claim, BITS).unwrap();
    assert_ne!(correct.public_values(), wrong.public_values(), "token IDs missing from the public statement");
    let meta = correct.get_meta();
    let mut verifier = VerifierState::new(EPSILON, 1, BITS, 30000, 128, 16, false, correct, meta.clone());
    let random = verifier.sample_normal_randomness();
    let mut openings = meta.get_ranges().iter().enumerate().filter(|(i, _)| *i != 1).map(|(range_index, range)| ZRangeOpening { range_index, width: 1, values: vec![0i128; range.len()] }).collect::<Vec<_>>();
    let public = openings.iter_mut().find(|opening| opening.range_index == 0).unwrap();
    public.values[..meta.num_public_values].copy_from_slice(wrong.public_values());
    for (i, value) in random.into_iter().enumerate() {
        public.values[meta.num_public_values + i] = quantize(&AFloat::from_f64(f64::from_bits(value)).unwrap(), &scale(BITS));
    }
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| verifier.validate_witness_openings(&openings))).is_err());
}

#[test]
fn exponential_domain_is_checked_in_executor_and_circuit() {
    assert!(numeric::exp_negative(&numeric::scalar(-128.0001)).is_err());
    assert!(numeric::exp_negative(&numeric::scalar(0.0001)).is_err());
    assert!(numeric::exp_negative(&numeric::scalar(-128.0)).is_ok());
    let mut trace = fixture().trace.clone();
    let public_claim = claim(&trace);
    let gate = trace.nodes.iter().find_map(|node| if let Op::Silu { input } = node.op { Some(input) } else { None }).unwrap();
    trace.nodes[gate].values[0] = numeric::scalar(-129.0);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| LlamaExecutor::new(fixture().model.clone(), trace, &public_claim, BITS))).is_err());
}
