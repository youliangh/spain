//! Architecture-derived approximate R1CS and high-precision witnesses.
//!
//! Fixed weights are projected by the verifier after the primary commitment.
//! This avoids a coefficient for every parameter in the sparse R1CS matrices.
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use model::{AFloat, FromPrimitive, HighPrecision, ToPrimitiveExt};
use parse::{generalized::InjectionInfo, mat::Matrix};
use rug::Float;
use sha2::{Digest, Sha256};
use spain::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};
use std::{
    fs::{self, File},
    io::{BufReader, Read},
};

use crate::{
    model::LlamaModel,
    trace::{Op, Trace},
};

const SECONDARY: usize = 1usize << (usize::BITS - 1);
const IDENTITY_BYTES: usize = 32;
type Var = usize;
type Lc = Vec<(Var, AFloat)>;
fn f(x: f64) -> AFloat {
    AFloat::from_f64(x).unwrap()
}
fn v(x: Var) -> Lc {
    vec![(x, f(1.))]
}
fn c(x: f64) -> Lc {
    vec![(0, f(x))]
}
fn scaled(x: Var, coefficient: f64) -> Lc {
    vec![(x, f(coefficient))]
}

/// Bind the local model and tokenizer bytes to the verifier's public statement.
pub fn model_fingerprint(model: &LlamaModel) -> Result<[u8; 32]> {
    let mut paths = fs::read_dir(&model.weights.directory)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy();
        if name.ends_with(".safetensors")
            || matches!(
                name.as_ref(),
                "config.json"
                    | "model.safetensors.index.json"
                    | "tokenizer.json"
                    | "tokenizer_config.json"
                    | "generation_config.json"
            )
        {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(path.metadata()?.len().to_le_bytes());
            let mut reader = BufReader::new(File::open(path)?);
            loop {
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
        }
    }
    Ok(hash.finalize().into())
}

pub fn scale(bits: usize) -> AFloat {
    AFloat(Float::with_val(128, 1) << bits as i32)
}
pub fn quantize(x: &AFloat, denominator: &AFloat) -> i128 {
    let scaled = x.clone() * denominator.clone();
    assert!(scaled.0.is_finite(), "non-finite witness/coefficient");
    scaled
        .to_rug_integer()
        .to_i128()
        .expect("fixed point value exceeds i128")
}

struct Builder {
    denominator: AFloat,
    public: Vec<AFloat>,
    random: Vec<AFloat>,
    primary: Vec<AFloat>,
    secondary: Vec<AFloat>,
    a: Vec<(usize, usize, i128)>,
    b: Vec<(usize, usize, i128)>,
    c: Vec<(usize, usize, i128)>,
    rows: usize,
    witness: bool,
}

impl Builder {
    fn primary(&mut self, x: AFloat) -> Var {
        let id = self.public.len() + self.random.len() + self.primary.len();
        self.primary.push(x);
        id
    }
    fn secondary(&mut self, x: AFloat) -> Var {
        let id = SECONDARY | self.secondary.len();
        self.secondary.push(x);
        id
    }
    fn value(&self, id: Var) -> AFloat {
        if id & SECONDARY != 0 {
            return self.secondary[id ^ SECONDARY].clone();
        }
        if id < self.public.len() {
            return self.public[id].clone();
        }
        let id = id - self.public.len();
        if id < self.random.len() {
            return self.random[id].clone();
        }
        self.primary[id - self.random.len()].clone()
    }
    fn eval(&self, terms: &Lc) -> AFloat {
        if !self.witness {
            return f(0.);
        }
        terms
            .iter()
            .fold(f(0.), |s, (id, coeff)| s + self.value(*id) * coeff.clone())
    }
    fn constrain(&mut self, a: Lc, b: Lc, c: Lc) {
        for (terms, dst) in [(a, &mut self.a), (b, &mut self.b), (c, &mut self.c)] {
            for (id, coefficient) in terms {
                // Keep zero challenge coefficients: the structure must be fixed before commitment.
                dst.push((self.rows, id, quantize(&coefficient, &self.denominator)));
            }
        }
        self.rows += 1;
    }
    fn equal(&mut self, x: Lc, y: Lc) {
        self.constrain(x, c(1.), y);
    }
    fn linear(&mut self, x: Lc) -> Var {
        let out = self.primary(self.eval(&x));
        self.equal(x, v(out));
        out
    }
    fn product(&mut self, a: Lc, b: Lc) -> Var {
        let out = self.primary(self.eval(&a) * self.eval(&b));
        self.constrain(a, b, v(out));
        out
    }
    fn positive(&mut self, x: Var) {
        let value = if self.witness {
            let value = self.value(x);
            assert!(
                value >= f(0.),
                "negative value in nonnegative gadget: {value}"
            );
            value.sqrt()
        } else {
            f(0.)
        };
        let root = self.primary(value);
        self.constrain(v(root), v(root), v(x));
    }
    fn quotient(&mut self, num: Lc, den: Lc) -> Var {
        let out_value = if self.witness {
            self.eval(&num) / self.eval(&den)
        } else {
            f(0.)
        };
        let out = self.primary(out_value);
        self.constrain(den, v(out), num);
        out
    }
    fn exp_neg(&mut self, x: Var) -> Var {
        let negative = self.linear(scaled(x, -1.));
        self.positive(negative);
        let lower = self.linear(vec![(x, f(1.)), (0, f(128.))]);
        self.positive(lower);
        let z = self.linear(scaled(x, 1. / 1024.));
        let z2 = self.product(v(z), v(z));
        let z3 = self.product(v(z2), v(z));
        let p = vec![(0, f(120.)), (z, f(60.)), (z2, f(12.)), (z3, f(1.))];
        let q = vec![(0, f(120.)), (z, f(-60.)), (z2, f(12.)), (z3, f(-1.))];
        let mut e = self.quotient(p, q);
        for _ in 0..10 {
            e = self.product(v(e), v(e));
        }
        e
    }
    fn finish(mut self) -> Compiled {
        let metadata = Metadata {
            num_public_values: self.public.len(),
            num_random_values: self.random.len(),
            num_witness_values: self.primary.len(),
            num_secondary_witness_values: self.secondary.len(),
            num_secondary_constraint_variables: 0,
            primary_output_labels: vec![],
            secondary_output_labels: vec![],
        };
        let secondary_offset = metadata.secondary_witness_shift();
        for entries in [&mut self.a, &mut self.b, &mut self.c] {
            for (_, col, _) in entries {
                if *col & SECONDARY != 0 {
                    *col = secondary_offset + (*col ^ SECONDARY);
                }
            }
        }
        let width = metadata.get_witness_size();
        let matrices = R1CSMatrices {
            a: Matrix::from_coo(self.a, width, self.rows),
            b: Matrix::from_coo(self.b, width, self.rows),
            c: Matrix::from_coo(self.c, width, self.rows),
        };
        let primary = self
            .primary
            .iter()
            .map(|x| quantize(x, &self.denominator))
            .collect::<Vec<_>>();
        let mut values = self
            .public
            .iter()
            .chain(&self.random)
            .map(|x| quantize(x, &self.denominator))
            .collect::<Vec<_>>();
        values.extend_from_slice(&primary);
        values.extend(
            self.secondary
                .iter()
                .map(|x| quantize(x, &self.denominator)),
        );
        let mut witness = Matrix::from_vec(values);
        witness.set_ranges(&metadata.get_ranges());
        Compiled {
            matrices,
            primary: Matrix::from_vec(primary),
            witness,
            metadata,
        }
    }
}

struct Compiled {
    matrices: R1CSMatrices<i128>,
    primary: Matrix<i128>,
    witness: Matrix<i128>,
    metadata: Metadata,
}

fn randomness_count(trace: &Trace) -> usize {
    trace
        .nodes
        .iter()
        .map(|n| match &n.op {
            Op::Linear { .. } => n.shape[1],
            Op::AttentionScores { num_heads, .. } => num_heads * n.shape[2],
            Op::AttentionValues {
                num_heads,
                head_dim,
                ..
            } => num_heads * head_dim,
            _ => 0,
        })
        .sum()
}

fn compile(
    model: &LlamaModel,
    trace: &Trace,
    public: &[i128],
    denominator: AFloat,
    randomness: Option<&[AFloat]>,
) -> Compiled {
    let count = randomness_count(trace);
    let random = randomness
        .map(|r| {
            assert_eq!(r.len(), count);
            r.to_vec()
        })
        .unwrap_or_else(|| vec![f(0.); count]);
    let witness = !trace.nodes[0].values.is_empty();
    let mut b = Builder {
        denominator: denominator.clone(),
        public: public
            .iter()
            .map(|x| AFloat::from_i128(*x).unwrap() / denominator.clone())
            .collect(),
        random,
        primary: vec![],
        secondary: vec![],
        a: vec![],
        b: vec![],
        c: vec![],
        rows: 0,
        witness,
    };
    let mut variables: Vec<Vec<Var>> = Vec::new();
    for node in &trace.nodes {
        let len = node.shape.iter().product::<usize>();
        let vars = match &node.op {
            Op::Input => (1 + IDENTITY_BYTES + trace.tokens.len()
                ..1 + IDENTITY_BYTES + trace.tokens.len() + len)
                .collect(),
            Op::LastToken { input } => {
                let source = &variables[*input];
                source[source.len() - len..].to_vec()
            }
            _ => (0..len)
                .map(|i| {
                    b.primary(if witness {
                        node.values[i].clone()
                    } else {
                        f(0.)
                    })
                })
                .collect(),
        };
        variables.push(vars);
    }
    let mut random_offset = 0;
    for (node_id, node) in trace.nodes.iter().enumerate() {
        let out = &variables[node_id];
        match &node.op {
            Op::Input | Op::LastToken { .. } => {}
            Op::Add { left, right } => {
                for i in 0..out.len() {
                    b.equal(
                        vec![(variables[*left][i], f(1.)), (variables[*right][i], f(1.))],
                        v(out[i]),
                    );
                }
            }
            Op::Mul { left, right } => {
                for i in 0..out.len() {
                    b.constrain(v(variables[*left][i]), v(variables[*right][i]), v(out[i]));
                }
            }
            Op::Linear { input, weight } => {
                let x = &variables[*input];
                let k = trace.nodes[*input].shape[1];
                let n = node.shape[1];
                let rows = node.shape[0];
                let r = &b.random[random_offset..random_offset + n];
                let w = model.weights.tensor(weight).unwrap();
                assert_eq!(w.shape, &[n, k]);
                let projection = if randomness.is_some() {
                    eprintln!("Projecting weights for {}", node.name);
                    model.weights.project_transpose(weight, r).unwrap()
                } else {
                    vec![f(0.); k]
                };
                for row in 0..rows {
                    let lhs: Lc = (0..k)
                        .map(|col| (x[row * k + col], projection[col].clone()))
                        .collect();
                    let rhs: Lc = (0..n)
                        .map(|col| (out[row * n + col], b.random[random_offset + col].clone()))
                        .collect();
                    let s = b.secondary(b.eval(&lhs));
                    b.equal(lhs, v(s));
                    b.equal(v(s), rhs);
                }
                random_offset += n;
            }
            Op::RmsNorm {
                input,
                weight,
                epsilon,
            } => {
                let x = &variables[*input];
                let width = node.shape[1];
                let w = model.weights.tensor(weight).unwrap();
                for row in 0..node.shape[0] {
                    let squares = (0..width)
                        .map(|j| b.product(v(x[row * width + j]), v(x[row * width + j])))
                        .collect::<Vec<_>>();
                    let mut mean: Lc = squares.iter().map(|i| (*i, f(1. / width as f64))).collect();
                    mean.push((0, f(*epsilon)));
                    let mean = b.linear(mean);
                    let root_value = if witness { b.value(mean).sqrt() } else { f(0.) };
                    let root = b.primary(root_value);
                    b.constrain(v(root), v(root), v(mean));
                    b.positive(root);
                    for j in 0..width {
                        let normalized = b.quotient(v(x[row * width + j]), v(root));
                        b.equal(scaled(normalized, w.get_f64(j)), v(out[row * width + j]));
                    }
                }
            }
            Op::Silu { input } => {
                for i in 0..out.len() {
                    let x = variables[*input][i];
                    let bit = b.primary(if witness && b.value(x) >= f(0.) {
                        f(1.)
                    } else {
                        f(0.)
                    });
                    b.constrain(v(bit), vec![(bit, f(1.)), (0, f(-1.))], c(0.));
                    let absolute = b.product(vec![(bit, f(2.)), (0, f(-1.))], v(x));
                    b.positive(absolute);
                    let negative = b.linear(scaled(absolute, -1.));
                    let e = b.exp_neg(negative);
                    let h = b.product(vec![(0, f(1.)), (bit, f(-1.))], v(e));
                    let numerator = b.product(v(x), vec![(bit, f(1.)), (h, f(1.))]);
                    b.constrain(v(out[i]), vec![(0, f(1.)), (e, f(1.))], v(numerator));
                }
            }
            Op::Rope {
                input,
                positions,
                heads,
                head_dim,
                cos,
                sin,
            } => {
                let x = &variables[*input];
                let half = head_dim / 2;
                for pos in 0..*positions {
                    for head in 0..*heads {
                        for d in 0..half {
                            let a = pos * heads * head_dim + head * head_dim + d;
                            let z = a + half;
                            let co = cos[pos * half + d];
                            let si = sin[pos * half + d];
                            b.equal(vec![(x[a], f(co)), (x[z], f(-si))], v(out[a]));
                            b.equal(vec![(x[z], f(co)), (x[a], f(si))], v(out[z]));
                        }
                    }
                }
            }
            Op::AttentionScores {
                query,
                key,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            } => {
                let seq = node.shape[1];
                let q = &variables[*query];
                let k = &variables[*key];
                for head in 0..*num_heads {
                    let kv = head / (num_heads / num_kv_heads);
                    let mut projected = Vec::with_capacity(*head_dim);
                    for d in 0..*head_dim {
                        let lc: Lc = (0..seq)
                            .map(|j| {
                                (
                                    k[j * num_kv_heads * head_dim + kv * head_dim + d],
                                    b.random[random_offset + j].clone(),
                                )
                            })
                            .collect();
                        let u = b.secondary(b.eval(&lc));
                        b.equal(lc, v(u));
                        projected.push(u);
                    }
                    for row in 0..seq {
                        let mut lhs = vec![];
                        for d in 0..*head_dim {
                            let qv = q[row * num_heads * head_dim + head * head_dim + d];
                            let u = projected[d];
                            let p = b.secondary(b.value(qv) * b.value(u));
                            b.constrain(v(qv), v(u), v(p));
                            lhs.push((p, f(*scale)));
                        }
                        let rhs = (0..seq)
                            .map(|j| {
                                (
                                    out[(head * seq + row) * seq + j],
                                    b.random[random_offset + j].clone(),
                                )
                            })
                            .collect();
                        b.equal(lhs, rhs);
                    }
                    random_offset += seq;
                }
            }
            Op::AttentionValues {
                probabilities,
                value,
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let seq = node.shape[0];
                let p = &variables[*probabilities];
                let values = &variables[*value];
                for head in 0..*num_heads {
                    let kv = head / (num_heads / num_kv_heads);
                    let mut projected = Vec::with_capacity(seq);
                    for row in 0..seq {
                        let lc = (0..*head_dim)
                            .map(|d| {
                                (
                                    values[row * num_kv_heads * head_dim + kv * head_dim + d],
                                    b.random[random_offset + d].clone(),
                                )
                            })
                            .collect();
                        let u = b.secondary(b.eval(&lc));
                        b.equal(lc, v(u));
                        projected.push(u);
                    }
                    for row in 0..seq {
                        let mut lhs = vec![];
                        for j in 0..seq {
                            let pv = p[(head * seq + row) * seq + j];
                            let u = projected[j];
                            let product = b.secondary(b.value(pv) * b.value(u));
                            b.constrain(v(pv), v(u), v(product));
                            lhs.push((product, f(1.)));
                        }
                        let rhs = (0..*head_dim)
                            .map(|d| {
                                (
                                    out[row * num_heads * head_dim + head * head_dim + d],
                                    b.random[random_offset + d].clone(),
                                )
                            })
                            .collect();
                        b.equal(lhs, rhs);
                    }
                    random_offset += head_dim;
                }
            }
            Op::SoftmaxCausal {
                input,
                heads,
                seq_len,
            } => {
                let x = &variables[*input];
                let seq = *seq_len;
                for head in 0..*heads {
                    for row in 0..seq {
                        let base = (head * seq + row) * seq;
                        let best = if witness {
                            (0..=row)
                                .max_by(|a, z| {
                                    b.value(x[base + *a])
                                        .partial_cmp(&b.value(x[base + *z]))
                                        .unwrap()
                                        .then_with(|| z.cmp(a))
                                })
                                .unwrap()
                        } else {
                            0
                        };
                        let maximum = b.primary(if witness {
                            b.value(x[base + best])
                        } else {
                            f(0.)
                        });
                        let mut flags = vec![];
                        let mut exps = vec![];
                        for j in 0..=row {
                            let bit = b.primary(if witness && j == best { f(1.) } else { f(0.) });
                            flags.push((bit, f(1.)));
                            b.constrain(v(bit), vec![(bit, f(1.)), (0, f(-1.))], c(0.));
                            let gap = b.linear(vec![(maximum, f(1.)), (x[base + j], f(-1.))]);
                            b.positive(gap);
                            b.constrain(v(bit), v(gap), c(0.));
                            let diff = b.linear(vec![(x[base + j], f(1.)), (maximum, f(-1.))]);
                            exps.push(b.exp_neg(diff));
                        }
                        b.equal(flags, c(1.));
                        let denominator = b.linear(exps.iter().map(|i| (*i, f(1.))).collect());
                        for j in 0..=row {
                            b.constrain(v(out[base + j]), v(denominator), v(exps[j]));
                        }
                        for j in row + 1..seq {
                            b.equal(v(out[base + j]), c(0.));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(random_offset, count);
    let logits = &variables[trace.logits];
    let public_logits =
        1 + IDENTITY_BYTES + trace.tokens.len() + trace.nodes[0].shape.iter().product::<usize>();
    for (i, id) in logits.iter().enumerate() {
        b.equal(v(*id), v(public_logits + i));
    }
    b.finish()
}

/// A fixed, architecture-derived statement. The verifier gets a layout without activations.
pub struct LlamaExecutor {
    model: Arc<LlamaModel>,
    trace: Arc<Trace>,
    public: Vec<i128>,
    denominator: AFloat,
    metadata: Metadata,
    primary: Matrix<i128>,
    initial: Mutex<Option<R1CSMatrices<i128>>>,
    full: Mutex<Option<Matrix<i128>>>,
}

impl LlamaExecutor {
    pub fn new(model: Arc<LlamaModel>, trace: Trace, logits: &[i128], bits: usize) -> Result<Self> {
        let denominator = scale(bits);
        let embedding = model.weights.tensor("model.embed_tokens.weight")?;
        let hidden = embedding.shape[1];
        ensure!(
            logits.len() == embedding.shape[0],
            "logits must cover entire vocabulary"
        );
        let mut public = vec![quantize(&f(1.), &denominator)];
        public.extend(
            model_fingerprint(&model)?
                .iter()
                .map(|byte| quantize(&f(*byte as f64), &denominator)),
        );
        public.extend(
            trace
                .tokens
                .iter()
                .map(|id| quantize(&f(*id as f64), &denominator)),
        );
        for token in &trace.tokens {
            for d in 0..hidden {
                public.push(quantize(
                    &f(embedding.get_f64(*token as usize * hidden + d)),
                    &denominator,
                ));
            }
        }
        public.extend_from_slice(logits);
        let compiled = compile(&model, &trace, &public, denominator.clone(), None);
        Ok(Self {
            model,
            trace: Arc::new(trace),
            public,
            denominator,
            metadata: compiled.metadata,
            primary: compiled.primary,
            initial: Mutex::new(Some(compiled.matrices)),
            full: Mutex::new(None),
        })
    }
    pub fn primary_witness(&self) -> &Matrix<i128> {
        &self.primary
    }
    pub fn public_values(&self) -> &[i128] {
        &self.public
    }
}

impl R1CSInstance<AFloat, i128> for LlamaExecutor {
    fn get_meta(&self) -> Metadata {
        self.metadata.clone()
    }
    fn requires_challenge_matrices(&self) -> bool {
        true
    }
    fn expected_public_values(&self, denominator: AFloat, batch: usize) -> Option<Vec<i128>> {
        assert_eq!(batch, 1);
        assert_eq!(denominator, self.denominator);
        Some(self.public.clone())
    }
    fn get_matrices(
        &self,
        denominator: AFloat,
        randomness: Option<&Vec<i128>>,
    ) -> (R1CSMatrices<i128>, Option<InjectionInfo>) {
        assert_eq!(denominator, self.denominator);
        if randomness.is_none() {
            if let Some(m) = self.initial.lock().unwrap().take() {
                return (m, None);
            }
        }
        let r = randomness.map(|r| {
            r.iter()
                .map(|x| AFloat::from_i128(*x).unwrap() / denominator.clone())
                .collect::<Vec<_>>()
        });
        let compiled = compile(
            &self.model,
            &self.trace,
            &self.public,
            denominator,
            r.as_deref(),
        );
        assert_eq!(
            compiled.metadata.get_witness_size(),
            self.metadata.get_witness_size()
        );
        if randomness.is_some() {
            if !self.trace.nodes[0].values.is_empty() {
                assert_eq!(
                    compiled.primary.data(),
                    self.primary.data(),
                    "primary witness changed after verifier challenge"
                );
            }
            *self.full.lock().unwrap() = Some(compiled.witness);
        }
        (compiled.matrices, None)
    }
    fn compute_commit_witness(&mut self, denominator: AFloat, batch: usize) -> Matrix<i128> {
        assert_eq!(batch, 1);
        assert_eq!(denominator, self.denominator);
        assert!(
            !self.trace.nodes[0].values.is_empty(),
            "verifier cannot act as witness generator"
        );
        self.primary.clone()
    }
    fn compute_full_witness(
        &mut self,
        _metadata: &Metadata,
        _random: Vec<AFloat>,
        denominator: AFloat,
    ) -> Matrix<i128> {
        assert_eq!(denominator, self.denominator);
        self.full
            .lock()
            .unwrap()
            .take()
            .expect("challenge-dependent matrices must be constructed first")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::write_tiny_fixture;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn fixture() -> Arc<LlamaModel> {
        let path = std::env::temp_dir().join(format!(
            "spain-circuit-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        write_tiny_fixture(&path).unwrap();
        Arc::new(LlamaModel::load(&path).unwrap())
    }
    fn residual(tamper: bool) {
        let model = fixture();
        let mut trace = model.forward(&[1, 4, 7]).unwrap();
        let denominator = scale(75);
        let claim = trace
            .logits()
            .iter()
            .map(|x| quantize(x, &denominator))
            .collect::<Vec<_>>();
        if tamper {
            let node = trace
                .nodes
                .iter_mut()
                .find(|n| matches!(n.op, Op::Silu { .. }))
                .unwrap();
            node.values[0] = node.values[0].clone() + f(0.01);
        }
        let mut executor = LlamaExecutor::new(model, trace, &claim, 75).unwrap();
        let metadata = executor.get_meta();
        let random = (0..metadata.num_random_values)
            .map(|i| f(((i * 37 % 101) as f64 - 50.) / 32.))
            .collect::<Vec<_>>();
        let random_int = random
            .iter()
            .map(|x| quantize(x, &denominator))
            .collect::<Vec<_>>();
        let (matrices, _) = executor.get_matrices(denominator.clone(), Some(&random_int));
        let witness = executor.compute_full_witness(&metadata, random, denominator);
        let error = spain::prover::compute_squared_error(
            &matrices,
            &witness,
            &(i256::I512::from(1) << 75),
            false,
        );
        spain::verifier::epsilon_check(&error, 75, 2f64.powi(-40), false);
    }
    #[test]
    fn complete_tiny_model_constraints() {
        residual(false);
    }
    #[test]
    #[should_panic]
    fn changed_silu_activation_is_rejected() {
        residual(true);
    }
    #[test]
    fn verifier_layout_has_identical_coefficients() {
        let model = fixture();
        let trace = model.forward(&[1, 4, 7]).unwrap();
        let denominator = scale(75);
        let claim = trace
            .logits()
            .iter()
            .map(|x| quantize(x, &denominator))
            .collect::<Vec<_>>();
        let prover = LlamaExecutor::new(model.clone(), trace, &claim, 75).unwrap();
        let verifier =
            LlamaExecutor::new(model.clone(), model.layout(&[1, 4, 7]).unwrap(), &claim, 75)
                .unwrap();
        let random = (0..prover.get_meta().num_random_values)
            .map(|i| quantize(&f((i % 17) as f64 / 16.), &denominator))
            .collect::<Vec<_>>();
        let (p, _) = prover.get_matrices(denominator.clone(), Some(&random));
        let (v, _) = verifier.get_matrices(denominator, Some(&random));
        assert_eq!(p.a.data(), v.a.data());
        assert_eq!(p.b.data(), v.b.data());
        assert_eq!(p.c.data(), v.c.data());
    }
}
