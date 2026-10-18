//! Typed, complete execution trace consumed by the Spain circuit compiler.
use ::model::AFloat;

pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum Op {
    Input,
    Linear {
        input: NodeId,
        weight: String,
    },
    RmsNorm {
        input: NodeId,
        weight: String,
        epsilon: f64,
    },
    Rope {
        input: NodeId,
        positions: usize,
        heads: usize,
        head_dim: usize,
        /// Row-major [positions, head_dim / 2]; exact f64 public coefficients.
        cos: Vec<f64>,
        sin: Vec<f64>,
    },
    AttentionScores {
        query: NodeId,
        key: NodeId,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f64,
    },
    SoftmaxCausal {
        input: NodeId,
        heads: usize,
        seq_len: usize,
    },
    AttentionValues {
        probabilities: NodeId,
        value: NodeId,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    Silu {
        input: NodeId,
    },
    Mul {
        left: NodeId,
        right: NodeId,
    },
    Add {
        left: NodeId,
        right: NodeId,
    },
    LastToken {
        input: NodeId,
    },
}

#[derive(Clone, Debug)]
pub struct TensorNode {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<AFloat>,
    pub op: Op,
}

#[derive(Clone, Debug)]
pub struct Trace {
    pub tokens: Vec<u32>,
    pub nodes: Vec<TensorNode>,
    pub logits: NodeId,
}

impl Trace {
    pub fn new(tokens: &[u32]) -> Self {
        Self {
            tokens: tokens.to_vec(),
            nodes: Vec::new(),
            logits: 0,
        }
    }

    pub fn push(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<AFloat>,
        op: Op,
    ) -> NodeId {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "trace shape mismatch"
        );
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            name: name.into(),
            shape,
            values,
            op,
        });
        id
    }

    pub fn node(&self, id: NodeId) -> &TensorNode {
        &self.nodes[id]
    }
    pub fn logits(&self) -> &[AFloat] {
        &self.nodes[self.logits].values
    }
    pub fn push_layout(&mut self, name: impl Into<String>, shape: Vec<usize>, op: Op) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            name: name.into(),
            shape,
            values: Vec::new(),
            op,
        });
        id
    }
}
