//! Complete Llama decoder execution with MPFR 128-bit accumulation.
use crate::{
    numeric as n,
    trace::{NodeId, Op, Trace},
    weights::WeightStore,
};
use ::model::AFloat;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use rug::{Assign, Float};
use serde::Deserialize;
use std::{fs::File, path::Path, time::Instant};

#[derive(Clone, Debug, Deserialize)]
pub struct RopeScaling {
    pub factor: f64,
    pub high_freq_factor: f64,
    pub low_freq_factor: f64,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
}

#[derive(Clone, Debug)]
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub weights: WeightStore,
}

impl LlamaModel {
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let config: LlamaConfig =
            serde_json::from_reader(File::open(directory.join("config.json"))?)?;
        ensure!(
            config.hidden_size == config.num_attention_heads * config.head_dim,
            "hidden_size must equal query heads × head_dim"
        );
        ensure!(
            config.num_key_value_heads > 0
                && config.num_attention_heads % config.num_key_value_heads == 0,
            "invalid grouped-query head counts"
        );
        ensure!(
            config.head_dim > 0 && config.head_dim % 2 == 0,
            "RoPE head dimension must be positive and even"
        );
        ensure!(
            config.hidden_act == "silu",
            "only the Llama SiLU activation is supported"
        );
        ensure!(
            !config.attention_bias && !config.mlp_bias,
            "bias-bearing model is not supported"
        );
        ensure!(
            config.rms_norm_eps > 0.0 && config.rms_norm_eps.is_finite(),
            "invalid RMS epsilon"
        );
        ensure!(
            config.rope_theta > 0.0 && config.rope_theta.is_finite(),
            "invalid RoPE theta"
        );
        if let Some(scaling) = &config.rope_scaling {
            ensure!(
                scaling.rope_type == "llama3",
                "unsupported RoPE scaling {}",
                scaling.rope_type
            );
            ensure!(
                scaling.factor > 0.0
                    && scaling.high_freq_factor > scaling.low_freq_factor
                    && scaling.low_freq_factor > 0.0,
                "invalid Llama 3 RoPE scaling"
            );
        }
        Ok(Self {
            config,
            weights: WeightStore::load(directory)?,
        })
    }

    /// Build the trusted graph without evaluating any model operation.
    pub fn layout(&self, tokens: &[u32]) -> Result<Trace> {
        let c = &self.config;
        ensure!(!tokens.is_empty(), "at least one input token is required");
        ensure!(
            tokens.len() <= c.max_position_embeddings,
            "input exceeds maximum position count"
        );
        ensure!(
            tokens.iter().all(|&token| (token as usize) < c.vocab_size),
            "token id outside vocabulary"
        );
        ensure!(
            self.weights.tensor("model.embed_tokens.weight")?.shape
                == [c.vocab_size, c.hidden_size],
            "invalid embedding matrix shape"
        );
        let seq = tokens.len();
        let mut trace = Trace::new(tokens);
        let mut hidden = trace.push_layout("embedding", vec![seq, c.hidden_size], Op::Input);
        let (cos, sin) = rope_coefficients(c, seq);
        for layer in 0..c.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let input_norm = self.norm_layout(
                &mut trace,
                hidden,
                &format!("{prefix}.input_layernorm.weight"),
            )?;
            let query = self.linear_layout(
                &mut trace,
                input_norm,
                &format!("{prefix}.self_attn.q_proj.weight"),
            )?;
            let key = self.linear_layout(
                &mut trace,
                input_norm,
                &format!("{prefix}.self_attn.k_proj.weight"),
            )?;
            let value = self.linear_layout(
                &mut trace,
                input_norm,
                &format!("{prefix}.self_attn.v_proj.weight"),
            )?;
            ensure!(
                trace.nodes[query].shape[1] == c.num_attention_heads * c.head_dim,
                "invalid q projection shape"
            );
            ensure!(
                trace.nodes[key].shape[1] == c.num_key_value_heads * c.head_dim
                    && trace.nodes[value].shape == trace.nodes[key].shape,
                "invalid KV projection shape"
            );
            let query = trace.push_layout(
                format!("{prefix}.query_rope"),
                vec![seq, c.num_attention_heads * c.head_dim],
                Op::Rope {
                    input: query,
                    positions: seq,
                    heads: c.num_attention_heads,
                    head_dim: c.head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                },
            );
            let key = trace.push_layout(
                format!("{prefix}.key_rope"),
                vec![seq, c.num_key_value_heads * c.head_dim],
                Op::Rope {
                    input: key,
                    positions: seq,
                    heads: c.num_key_value_heads,
                    head_dim: c.head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                },
            );
            let scores = trace.push_layout(
                format!("{prefix}.attention_scores"),
                vec![c.num_attention_heads, seq, seq],
                Op::AttentionScores {
                    query,
                    key,
                    num_heads: c.num_attention_heads,
                    num_kv_heads: c.num_key_value_heads,
                    head_dim: c.head_dim,
                    scale: 1.0 / (c.head_dim as f64).sqrt(),
                },
            );
            let probabilities = trace.push_layout(
                format!("{prefix}.attention_probabilities"),
                vec![c.num_attention_heads, seq, seq],
                Op::SoftmaxCausal {
                    input: scores,
                    heads: c.num_attention_heads,
                    seq_len: seq,
                },
            );
            let attended = trace.push_layout(
                format!("{prefix}.attention_values"),
                vec![seq, c.hidden_size],
                Op::AttentionValues {
                    probabilities,
                    value,
                    num_heads: c.num_attention_heads,
                    num_kv_heads: c.num_key_value_heads,
                    head_dim: c.head_dim,
                },
            );
            let attention_output = self.linear_layout(
                &mut trace,
                attended,
                &format!("{prefix}.self_attn.o_proj.weight"),
            )?;
            hidden = trace.push_layout(
                format!("{prefix}.attention_residual"),
                vec![seq, c.hidden_size],
                Op::Add {
                    left: hidden,
                    right: attention_output,
                },
            );
            let post_norm = self.norm_layout(
                &mut trace,
                hidden,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?;
            let gate = self.linear_layout(
                &mut trace,
                post_norm,
                &format!("{prefix}.mlp.gate_proj.weight"),
            )?;
            let up = self.linear_layout(
                &mut trace,
                post_norm,
                &format!("{prefix}.mlp.up_proj.weight"),
            )?;
            ensure!(
                trace.nodes[gate].shape == [seq, c.intermediate_size]
                    && trace.nodes[up].shape == trace.nodes[gate].shape,
                "invalid MLP projection shape"
            );
            let gate = trace.push_layout(
                format!("{prefix}.silu"),
                vec![seq, c.intermediate_size],
                Op::Silu { input: gate },
            );
            let product = trace.push_layout(
                format!("{prefix}.swiglu"),
                vec![seq, c.intermediate_size],
                Op::Mul {
                    left: gate,
                    right: up,
                },
            );
            let down = self.linear_layout(
                &mut trace,
                product,
                &format!("{prefix}.mlp.down_proj.weight"),
            )?;
            ensure!(
                trace.nodes[down].shape == [seq, c.hidden_size],
                "invalid down projection shape"
            );
            hidden = trace.push_layout(
                format!("{prefix}.mlp_residual"),
                vec![seq, c.hidden_size],
                Op::Add {
                    left: hidden,
                    right: down,
                },
            );
        }
        hidden = self.norm_layout(&mut trace, hidden, "model.norm.weight")?;
        let last = trace.push_layout(
            "last_hidden_state",
            vec![1, c.hidden_size],
            Op::LastToken { input: hidden },
        );
        let head = if c.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        trace.logits = self.linear_layout(&mut trace, last, head)?;
        trace.nodes[trace.logits].name = "logits".into();
        ensure!(
            trace.nodes[trace.logits].shape == [1, c.vocab_size],
            "invalid LM head shape"
        );
        Ok(trace)
    }

    fn linear_layout(&self, trace: &mut Trace, input: NodeId, weight: &str) -> Result<NodeId> {
        let tensor = self.weights.tensor(weight)?;
        let input_shape = &trace.nodes[input].shape;
        ensure!(
            tensor.shape.len() == 2 && input_shape.len() == 2 && tensor.shape[1] == input_shape[1],
            "linear shape mismatch: {weight}"
        );
        Ok(trace.push_layout(
            weight.trim_end_matches(".weight"),
            vec![input_shape[0], tensor.shape[0]],
            Op::Linear {
                input,
                weight: weight.into(),
            },
        ))
    }

    fn norm_layout(&self, trace: &mut Trace, input: NodeId, weight: &str) -> Result<NodeId> {
        let shape = trace.nodes[input].shape.clone();
        ensure!(
            self.weights.tensor(weight)?.shape == [shape[1]],
            "RMS norm shape mismatch: {weight}"
        );
        Ok(trace.push_layout(
            weight.trim_end_matches(".weight"),
            shape,
            Op::RmsNorm {
                input,
                weight: weight.into(),
                epsilon: self.config.rms_norm_eps,
            },
        ))
    }

    pub fn forward(&self, tokens: &[u32]) -> Result<Trace> {
        let mut trace = self.layout(tokens)?;
        let start = Instant::now();
        for id in 0..trace.nodes.len() {
            let node_start = Instant::now();
            let values = self
                .evaluate(&trace, id)
                .with_context(|| format!("evaluating {}", trace.nodes[id].name))?;
            ensure!(
                values.len() == trace.nodes[id].shape.iter().product::<usize>(),
                "incorrect result shape for {}",
                trace.nodes[id].name
            );
            ensure!(
                values.iter().all(|x| x.0.is_finite()),
                "nonfinite value in {}",
                trace.nodes[id].name
            );
            trace.nodes[id].values = values;
            eprintln!(
                "HP128 [{}/{}] {} {:.3}s (total {:.1}s)",
                id + 1,
                trace.nodes.len(),
                trace.nodes[id].name,
                node_start.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64()
            );
        }
        Ok(trace)
    }

    fn evaluate(&self, trace: &Trace, id: NodeId) -> Result<Vec<AFloat>> {
        let node = &trace.nodes[id];
        let get = |input: NodeId| -> &[AFloat] { &trace.nodes[input].values };
        match &node.op {
            Op::Input => {
                let weight = self.weights.tensor("model.embed_tokens.weight")?;
                let dim = self.config.hidden_size;
                Ok(trace
                    .tokens
                    .iter()
                    .flat_map(|&token| {
                        (0..dim).map(move |j| n::scalar(weight.get_f64(token as usize * dim + j)))
                    })
                    .collect())
            }
            Op::Linear { input, weight } => {
                let weight = self.weights.tensor(weight)?;
                let (rows, out_dim, in_dim) = (node.shape[0], weight.shape[0], weight.shape[1]);
                let input = get(*input);
                // Each tile shares its decoded BF16 rows across every sequence position.
                let tile_size = 16;
                let tiles: Vec<Vec<AFloat>> = (0..out_dim.div_ceil(tile_size))
                    .into_par_iter()
                    .map(|tile| {
                        let start = tile * tile_size;
                        let count = tile_size.min(out_dim - start);
                        let mut output = Vec::with_capacity(count * rows);
                        let mut product = Float::with_val(n::PRECISION, 0);
                        let mut sum = Float::with_val(n::PRECISION, 0);
                        let mut decoded = vec![0.0f64; in_dim];
                        for out in start..start + count {
                            for (k, item) in decoded.iter_mut().enumerate() {
                                *item = weight.get_f64(out * in_dim + k);
                            }
                            for row in 0..rows {
                                sum.assign(0);
                                for k in 0..in_dim {
                                    product.assign(&input[row * in_dim + k].0 * decoded[k]);
                                    sum += &product;
                                }
                                output.push(AFloat(sum.clone()));
                            }
                        }
                        output
                    })
                    .collect();
                let mut result = vec![n::zero(); rows * out_dim];
                for (tile, values) in tiles.into_iter().enumerate() {
                    for (index, value) in values.into_iter().enumerate() {
                        let out = tile * tile_size + index / rows;
                        result[(index % rows) * out_dim + out] = value;
                    }
                }
                Ok(result)
            }
            Op::RmsNorm {
                input,
                weight,
                epsilon,
            } => {
                let input = get(*input);
                let weights = self.weights.get_f64(weight)?;
                let dim = weights.len();
                Ok(input
                    .par_chunks(dim)
                    .flat_map_iter(|row| {
                        let mean = n::scale(&n::dot(row, row), 1.0 / dim as f64);
                        let inverse =
                            n::div(&n::one(), &n::sqrt(&n::add(&mean, &n::scalar(*epsilon))));
                        row.iter()
                            .zip(&weights)
                            .map(|(x, &weight)| n::scale(&n::mul(x, &inverse), weight))
                            .collect::<Vec<_>>()
                    })
                    .collect())
            }
            Op::Rope {
                input,
                positions,
                heads,
                head_dim,
                cos,
                sin,
            } => {
                let input = get(*input);
                let mut output = vec![n::zero(); input.len()];
                let half = head_dim / 2;
                for position in 0..*positions {
                    for head in 0..*heads {
                        let base = (position * heads + head) * head_dim;
                        for d in 0..half {
                            let (c, s) = (cos[position * half + d], sin[position * half + d]);
                            output[base + d] = n::sub(
                                &n::scale(&input[base + d], c),
                                &n::scale(&input[base + half + d], s),
                            );
                            output[base + half + d] = n::add(
                                &n::scale(&input[base + half + d], c),
                                &n::scale(&input[base + d], s),
                            );
                        }
                    }
                }
                Ok(output)
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
                let (query, key) = (get(*query), get(*key));
                let groups = num_heads / num_kv_heads;
                Ok((0..num_heads * seq * seq)
                    .into_par_iter()
                    .map(|index| {
                        let head = index / (seq * seq);
                        let row = (index / seq) % seq;
                        let col = index % seq;
                        let q = (row * num_heads + head) * head_dim;
                        let k = (col * num_kv_heads + head / groups) * head_dim;
                        n::scale(
                            &n::dot(&query[q..q + head_dim], &key[k..k + head_dim]),
                            *scale,
                        )
                    })
                    .collect())
            }
            Op::SoftmaxCausal {
                input,
                heads: _,
                seq_len,
            } => {
                let input = get(*input);
                let rows: Result<Vec<Vec<AFloat>>> = input
                    .par_chunks(*seq_len)
                    .enumerate()
                    .map(|(index, row)| {
                        let allowed = index % seq_len + 1;
                        let max = row[..allowed]
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap();
                        let exps: Vec<AFloat> = row[..allowed]
                            .iter()
                            .map(|x| n::exp_negative(&n::sub(x, max)))
                            .collect::<Result<_>>()?;
                        let sum = exps.iter().fold(n::zero(), |sum, e| n::add(&sum, e));
                        let mut output: Vec<AFloat> =
                            exps.iter().map(|e| n::div(e, &sum)).collect();
                        output.resize_with(*seq_len, n::zero);
                        Ok(output)
                    })
                    .collect();
                Ok(rows?.into_iter().flatten().collect())
            }
            Op::AttentionValues {
                probabilities,
                value,
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let seq = node.shape[0];
                let (probabilities, value) = (get(*probabilities), get(*value));
                let groups = num_heads / num_kv_heads;
                Ok((0..seq * num_heads * head_dim)
                    .into_par_iter()
                    .map(|index| {
                        let row = index / (num_heads * head_dim);
                        let head = index / head_dim % num_heads;
                        let dim = index % head_dim;
                        let mut sum = Float::with_val(n::PRECISION, 0);
                        let mut product = Float::with_val(n::PRECISION, 0);
                        for token in 0..=row {
                            let p = &probabilities[(head * seq + row) * seq + token].0;
                            let v =
                                &value[(token * num_kv_heads + head / groups) * head_dim + dim].0;
                            product.assign(p * v);
                            sum += &product;
                        }
                        AFloat(sum)
                    })
                    .collect())
            }
            Op::Silu { input } => get(*input).par_iter().map(n::silu).collect(),
            Op::Mul { left, right } => {
                ensure!(
                    get(*left).len() == get(*right).len(),
                    "elementwise multiplication shape mismatch"
                );
                Ok(get(*left)
                    .par_iter()
                    .zip(get(*right))
                    .map(|(a, b)| n::mul(a, b))
                    .collect())
            }
            Op::Add { left, right } => {
                ensure!(
                    get(*left).len() == get(*right).len(),
                    "elementwise addition shape mismatch"
                );
                Ok(get(*left)
                    .par_iter()
                    .zip(get(*right))
                    .map(|(a, b)| n::add(a, b))
                    .collect())
            }
            Op::LastToken { input } => {
                let input = get(*input);
                let dim = node.shape[1];
                Ok(input[input.len() - dim..].to_vec())
            }
        }
    }
}

/// Public deterministic constants; these same exact f64 values enter the circuit.
pub fn rope_coefficients(config: &LlamaConfig, positions: usize) -> (Vec<f64>, Vec<f64>) {
    let half = config.head_dim / 2;
    let frequencies: Vec<f64> = (0..half)
        .map(|index| {
            let mut inverse = 1.0
                / config
                    .rope_theta
                    .powf((2 * index) as f64 / config.head_dim as f64);
            if let Some(scaling) = &config.rope_scaling {
                let wavelength = std::f64::consts::TAU / inverse;
                let low = scaling.original_max_position_embeddings as f64 / scaling.low_freq_factor;
                let high =
                    scaling.original_max_position_embeddings as f64 / scaling.high_freq_factor;
                if wavelength > low {
                    inverse /= scaling.factor;
                } else if wavelength >= high {
                    let smooth = (scaling.original_max_position_embeddings as f64 / wavelength
                        - scaling.low_freq_factor)
                        / (scaling.high_freq_factor - scaling.low_freq_factor);
                    inverse = (1.0 - smooth) * inverse / scaling.factor + smooth * inverse;
                }
            }
            inverse
        })
        .collect();
    let mut cos = Vec::with_capacity(positions * half);
    let mut sin = Vec::with_capacity(positions * half);
    for position in 0..positions {
        for inverse in &frequencies {
            let angle = position as f64 * inverse;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

pub fn forward(model: &LlamaModel, tokens: &[u32]) -> Result<Trace> {
    model.forward(tokens)
}

/// Write a deterministic two-layer test model; never used in place of user weights.
/// The shape includes GQA and a nontrivial half-rotation: H=8, Q=2, KV=1, D=4.
pub fn write_tiny_fixture(directory: impl AsRef<Path>) -> Result<()> {
    use std::io::Write;
    let directory = directory.as_ref();
    ensure!(
        !directory.join("config.json").exists() && !directory.join("model.safetensors").exists(),
        "fixture destination already contains model files"
    );
    std::fs::create_dir_all(directory)?;
    let config = serde_json::json!({
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "vocab_size": 32, "rms_norm_eps": 0.00001, "rope_theta": 500000.0,
        "max_position_embeddings": 32, "tie_word_embeddings": true,
        "bos_token_id": 0, "eos_token_id": 31, "hidden_act": "silu",
        "rope_scaling": { "factor": 32.0, "high_freq_factor": 4.0,
            "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3" }
    });
    serde_json::to_writer(File::create(directory.join("config.json"))?, &config)?;
    let mut header = serde_json::Map::new();
    let mut data = Vec::<u8>::new();
    let mut write = |name: String, shape: Vec<usize>, norm: bool| {
        let begin = data.len();
        for index in 0..shape.iter().product::<usize>() {
            let value = if norm {
                1.0f32
            } else {
                (((index * 17 + begin / 4) % 29) as f32 - 14.0) / 64.0
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name,
            serde_json::json!({ "dtype":"F32", "shape":shape, "data_offsets":[begin,data.len()] }),
        );
    };
    write("model.embed_tokens.weight".into(), vec![32, 8], false);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        write(format!("{p}.input_layernorm.weight"), vec![8], true);
        write(
            format!("{p}.post_attention_layernorm.weight"),
            vec![8],
            true,
        );
        for (name, shape) in [
            ("self_attn.q_proj", vec![8, 8]),
            ("self_attn.k_proj", vec![4, 8]),
            ("self_attn.v_proj", vec![4, 8]),
            ("self_attn.o_proj", vec![8, 8]),
            ("mlp.gate_proj", vec![16, 8]),
            ("mlp.up_proj", vec![16, 8]),
            ("mlp.down_proj", vec![8, 16]),
        ] {
            write(format!("{p}.{name}.weight"), shape, false);
        }
    }
    write("model.norm.weight".into(), vec![8], true);
    let mut header = serde_json::to_vec(&header)?;
    // Safetensors requires the header to be aligned to eight bytes.
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut file = File::create(directory.join("model.safetensors"))?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small real safetensors fixture exercising the same loader and all graph operations.
    fn tiny_model() -> (LlamaModel, std::path::PathBuf) {
        let directory = std::env::temp_dir().join(format!(
            "spain-llama-unit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_fixture(&directory).unwrap();
        (LlamaModel::load(&directory).unwrap(), directory)
    }

    #[test]
    fn complete_tiny_forward_is_causal_and_uses_full_vocabulary() {
        let (model, directory) = tiny_model();
        let one = model.forward(&[1]).unwrap();
        let three = model.forward(&[1, 2, 3]).unwrap();
        assert_eq!(one.logits().len(), 32);
        assert_eq!(three.logits().len(), 32);
        assert!(three.logits().iter().any(|x| x.0 != 0));
        // Independent scalar f64 implementation, including GQA, scaled RoPE,
        // both layers, the same rational activation, and tied output weights.
        let expected = [
            0.2065938317124112,
            -0.283757936990073,
            0.0739492363201047,
            0.43690258116745956,
            -0.4010259176013811,
            -0.5575540557654882,
            0.31963460055615145,
            -0.36313439254543656,
            0.21002483390702625,
            0.20236661994484328,
            0.021157175151198825,
            0.09275685329571806,
            0.24985120723706694,
            -0.09611080546010933,
            -0.2251239466844189,
            0.13258322662575883,
            0.14795984140675714,
            -0.3423919272957271,
            -0.4989200654598342,
            0.37826859086180553,
            -0.45965990790703526,
            0.2686588242126803,
            0.26100061025049737,
            -0.4217683828510906,
            0.15139084360137217,
            0.308485197542721,
            -0.03747681515445523,
            0.034122862990063986,
            0.1912172169314129,
            0.2065938317124112,
            -0.283757936990073,
            0.0739492363201047,
        ];
        for (actual, expected) in three.logits().iter().zip(expected) {
            assert!((actual.0.to_f64() - expected).abs() < 5e-12);
        }
        let challenge: Vec<AFloat> = (0..32)
            .map(|i| n::scalar((i as f64 - 16.0) / 32.0))
            .collect();
        let projection = model
            .weights
            .project_transpose("model.embed_tokens.weight", &challenge)
            .unwrap();
        let weights = model.weights.tensor("model.embed_tokens.weight").unwrap();
        for (column, actual) in projection.iter().enumerate() {
            let expected = challenge
                .iter()
                .enumerate()
                .fold(n::zero(), |sum, (row, r)| {
                    n::add(&sum, &n::scale(r, weights.get_f64(row * 8 + column)))
                });
            assert_eq!(*actual, expected);
        }
        for (short, longer) in one.nodes.iter().zip(&three.nodes) {
            match short.op {
                Op::AttentionScores { .. } | Op::SoftmaxCausal { .. } | Op::LastToken { .. } => {
                    continue;
                }
                _ if short.name == "logits" => continue,
                _ => {}
            }
            let width = short.shape[1];
            assert_eq!(
                short.values[..width],
                longer.values[..width],
                "future tokens changed {}",
                short.name
            );
        }
        let probs = three
            .nodes
            .iter()
            .find(|node| matches!(node.op, Op::SoftmaxCausal { .. }))
            .unwrap();
        for head in 0..2 {
            for row in 0..3 {
                let sum = probs.values[(head * 3 + row) * 3..(head * 3 + row + 1) * 3]
                    .iter()
                    .fold(n::zero(), |sum, x| n::add(&sum, x));
                assert!((sum.0.to_f64() - 1.0).abs() < 1e-15);
                for col in row + 1..3 {
                    assert_eq!(probs.values[(head * 3 + row) * 3 + col].0, 0);
                }
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn installed_llama_layout_includes_every_layer_and_full_vocabulary() {
        let path = Path::new("/home/yliangh/models/meta-llama/Llama-3.2-1B");
        if !path.is_dir() {
            return;
        }
        let model = LlamaModel::load(path).unwrap();
        let trace = model.layout(&[128000, 791, 25]).unwrap();
        assert!(trace.nodes.iter().all(|node| node.values.is_empty()));
        assert_eq!(trace.nodes[trace.logits].shape, [1, 128256]);
        assert_eq!(
            trace
                .nodes
                .iter()
                .filter(|node| matches!(node.op, Op::Silu { .. }))
                .count(),
            16
        );
        assert_eq!(
            trace
                .nodes
                .iter()
                .filter(|node| matches!(node.op, Op::Linear { .. }))
                .count(),
            113
        );
        assert!(model.layout(&[]).is_err());
        assert!(model.layout(&[128256]).is_err());
    }
}
