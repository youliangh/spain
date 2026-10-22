//! ONNX execution with MPFR activations and immutable, memory-mapped weights.
//! The graph, including all rational-function auxiliary nodes, is the executable
//! specification. Selected outputs execute only their dependency closure.
use crate::AFloat;
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use prost::Message;
use rayon::prelude::*;
use rug::{Assign, Float, float::Round};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

const PRECISION: u32 = 128;

/// Minimal standard ONNX protobuf schema. Unknown fields are skipped by prost.
pub mod proto {
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Model {
        #[prost(message, optional, tag = "7")]
        pub graph: Option<Graph>,
        #[prost(message, repeated, tag = "14")]
        pub metadata_props: Vec<KeyValue>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Graph {
        #[prost(message, repeated, tag = "1")]
        pub node: Vec<Node>,
        #[prost(message, repeated, tag = "5")]
        pub initializer: Vec<TensorProto>,
        #[prost(message, repeated, tag = "11")]
        pub input: Vec<ValueInfo>,
        #[prost(message, repeated, tag = "12")]
        pub output: Vec<ValueInfo>,
        #[prost(message, repeated, tag = "13")]
        pub value_info: Vec<ValueInfo>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Node {
        #[prost(string, repeated, tag = "1")]
        pub input: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        pub output: Vec<String>,
        #[prost(string, tag = "3")]
        pub name: String,
        #[prost(string, tag = "4")]
        pub op_type: String,
        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<Attribute>,
        #[prost(string, tag = "7")]
        pub domain: String,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Attribute {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(float, optional, tag = "2")]
        pub f: Option<f32>,
        #[prost(int64, optional, tag = "3")]
        pub i: Option<i64>,
        #[prost(bytes = "vec", tag = "4")]
        pub s: Vec<u8>,
        #[prost(message, optional, tag = "5")]
        pub t: Option<TensorProto>,
        #[prost(float, repeated, tag = "7")]
        pub floats: Vec<f32>,
        #[prost(int64, repeated, tag = "8")]
        pub ints: Vec<i64>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct TensorProto {
        #[prost(int64, repeated, tag = "1")]
        pub dims: Vec<i64>,
        #[prost(int32, tag = "2")]
        pub data_type: i32,
        #[prost(float, repeated, tag = "4")]
        pub float_data: Vec<f32>,
        #[prost(int32, repeated, tag = "5")]
        pub int32_data: Vec<i32>,
        #[prost(int64, repeated, tag = "7")]
        pub int64_data: Vec<i64>,
        #[prost(string, tag = "8")]
        pub name: String,
        #[prost(bytes = "vec", tag = "9")]
        pub raw_data: Vec<u8>,
        #[prost(double, repeated, tag = "10")]
        pub double_data: Vec<f64>,
        #[prost(message, repeated, tag = "13")]
        pub external_data: Vec<KeyValue>,
        #[prost(int32, tag = "14")]
        pub data_location: i32,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct KeyValue {
        #[prost(string, tag = "1")]
        pub key: String,
        #[prost(string, tag = "2")]
        pub value: String,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ValueInfo {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(message, optional, tag = "2")]
        pub r#type: Option<Type>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Type {
        #[prost(message, optional, tag = "1")]
        pub tensor_type: Option<TensorType>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct TensorType {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<Shape>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Shape {
        #[prost(message, repeated, tag = "1")]
        pub dim: Vec<Dimension>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Dimension {
        #[prost(int64, optional, tag = "1")]
        pub dim_value: Option<i64>,
        #[prost(string, optional, tag = "2")]
        pub dim_param: Option<String>,
    }
}

#[derive(Clone, Debug)]
pub enum TensorData {
    Mpfr(Arc<Vec<AFloat>>),
    I64(Arc<Vec<i64>>),
    F64(Arc<Vec<f64>>),
    Mapped {
        bytes: Arc<Mmap>,
        byte_offset: usize,
        dtype: i32,
    },
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
    strides: Vec<usize>,
    offset: usize,
}

fn elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b).context("tensor size overflow")
    })
}
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    let mut out = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        out[i] = stride;
        stride *= shape[i];
    }
    out
}
impl Tensor {
    fn new(shape: Vec<usize>, data: TensorData) -> Self {
        Self {
            strides: strides(&shape),
            shape,
            data,
            offset: 0,
        }
    }
    pub fn from_i64(shape: Vec<usize>, values: Vec<i64>) -> Result<Self> {
        ensure!(
            elements(&shape)? == values.len(),
            "integer tensor shape mismatch"
        );
        Ok(Self::new(shape, TensorData::I64(Arc::new(values))))
    }
    pub fn from_f64(shape: Vec<usize>, values: Vec<f64>) -> Result<Self> {
        ensure!(
            elements(&shape)? == values.len(),
            "float tensor shape mismatch"
        );
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite tensor constant"
        );
        Ok(Self::new(shape, TensorData::F64(Arc::new(values))))
    }
    pub fn from_mpfr(shape: Vec<usize>, values: Vec<AFloat>) -> Result<Self> {
        ensure!(
            elements(&shape)? == values.len(),
            "MPFR tensor shape mismatch"
        );
        ensure!(
            values.iter().all(|v| v.0.is_finite()),
            "nonfinite MPFR output"
        );
        Ok(Self::new(shape, TensorData::Mpfr(Arc::new(values))))
    }
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn storage_index(&self, mut i: usize) -> usize {
        assert!(i < self.len());
        let mut p = self.offset;
        for d in (0..self.shape.len()).rev() {
            p += (i % self.shape[d]) * self.strides[d];
            i /= self.shape[d];
        }
        p
    }
    fn integer(&self, i: usize) -> Option<i64> {
        let p = self.storage_index(i);
        match &self.data {
            TensorData::I64(v) => Some(v[p]),
            TensorData::Mapped {
                bytes,
                byte_offset,
                dtype: 7,
            } => {
                let b = byte_offset + p * 8;
                Some(i64::from_le_bytes(bytes[b..b + 8].try_into().unwrap()))
            }
            _ => None,
        }
    }
    fn fixed(&self, i: usize) -> Option<f64> {
        let p = self.storage_index(i);
        match &self.data {
            TensorData::Mpfr(_) => None,
            TensorData::I64(v) => Some(v[p] as f64),
            TensorData::F64(v) => Some(v[p]),
            TensorData::Mapped {
                bytes,
                byte_offset,
                dtype,
            } => Some(match dtype {
                11 => {
                    let b = byte_offset + p * 8;
                    f64::from_le_bytes(bytes[b..b + 8].try_into().unwrap())
                }
                1 => {
                    let b = byte_offset + p * 4;
                    f32::from_le_bytes(bytes[b..b + 4].try_into().unwrap()) as f64
                }
                7 => {
                    let b = byte_offset + p * 8;
                    i64::from_le_bytes(bytes[b..b + 8].try_into().unwrap()) as f64
                }
                16 => {
                    let b = byte_offset + p * 2;
                    f32::from_bits(
                        (u16::from_le_bytes(bytes[b..b + 2].try_into().unwrap()) as u32) << 16,
                    ) as f64
                }
                _ => unreachable!(),
            }),
        }
    }
    pub fn get(&self, i: usize) -> AFloat {
        match &self.data {
            TensorData::Mpfr(v) => v[self.storage_index(i)].clone(),
            TensorData::I64(v) => AFloat(Float::with_val(PRECISION, v[self.storage_index(i)])),
            TensorData::Mapped { dtype: 7, .. } => {
                AFloat(Float::with_val(PRECISION, self.integer(i).unwrap()))
            }
            _ => AFloat(Float::with_val(PRECISION, self.fixed(i).unwrap())),
        }
    }
    pub fn get_f64(&self, i: usize) -> f64 {
        self.fixed(i).unwrap_or_else(|| self.get(i).0.to_f64())
    }
    pub fn to_f64(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.get_f64(i)).collect()
    }
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        (0..self.len())
            .map(|i| {
                if let Some(v) = self.integer(i) {
                    return Ok(v);
                }
                let v = self.get(i).0;
                ensure!(v.is_integer(), "noninteger index/shape value");
                v.to_integer()
                    .and_then(|v| v.to_i64())
                    .context("integer value outside i64")
            })
            .collect()
    }
    pub fn mpfr_values(&self) -> Vec<AFloat> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }
    pub fn write_scaled_i128(&self, writer: &mut impl Write, scale_bits: u32) -> Result<()> {
        ensure!(scale_bits < 127, "invalid fixed-point scale");
        for i in 0..self.len() {
            let mut v = self.get(i).0;
            ensure!(v.is_finite(), "nonfinite witness");
            v <<= scale_bits;
            let q = v
                .to_integer_round(Round::Nearest)
                .context("nonfinite witness")?
                .0
                .to_i128()
                .context("quantized witness exceeds i128")?;
            writer.write_all(&q.to_le_bytes())?;
        }
        Ok(())
    }
    fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        ensure!(
            elements(&shape)? == self.len(),
            "reshape element count mismatch"
        );
        if self.strides == strides(&self.shape) {
            let mut t = self.clone();
            t.shape = shape;
            t.strides = strides(&t.shape);
            Ok(t)
        } else if matches!(self.data, TensorData::I64(_)) {
            Self::from_i64(shape, self.to_i64()?)
        } else {
            Self::from_mpfr(shape, self.mpfr_values())
        }
    }
    fn transpose(&self, perm: &[usize]) -> Result<Self> {
        ensure!(
            perm.len() == self.shape.len(),
            "transpose permutation rank mismatch"
        );
        ensure!(
            perm.iter().copied().collect::<HashSet<_>>().len() == perm.len()
                && perm.iter().all(|&i| i < perm.len()),
            "invalid transpose permutation"
        );
        let mut t = self.clone();
        t.shape = perm.iter().map(|&i| self.shape[i]).collect();
        t.strides = perm.iter().map(|&i| self.strides[i]).collect();
        Ok(t)
    }
}

pub struct RunResult {
    pub outputs: Vec<(String, Tensor)>,
    pub nodes_executed: usize,
    pub peak_live_elements: usize,
}
pub struct OnnxV2 {
    pub graph: proto::Graph,
    initializers: HashMap<String, Tensor>,
    shapes: HashMap<String, Vec<usize>>,
    base: PathBuf,
    domains: HashMap<String, Vec<(f64, f64)>>,
}

impl OnnxV2 {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let model =
            proto::Model::decode(std::fs::read(path)?.as_slice()).context("decoding ONNX model")?;
        let mut domains: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        for property in &model.metadata_props {
            if property.key == "spain.domains" {
                let data: serde_json::Value = serde_json::from_str(&property.value)
                    .context("invalid spain.domains metadata")?;
                for rule in data.as_array().context("spain.domains must be an array")? {
                    let tensor = rule["tensor"].as_str().context("domain tensor absent")?;
                    let lower = rule["lower"]
                        .as_f64()
                        .context("domain lower bound absent")?;
                    let upper = rule["upper"]
                        .as_f64()
                        .context("domain upper bound absent")?;
                    ensure!(
                        lower.is_finite() && upper.is_finite() && lower <= upper,
                        "invalid domain bounds for {tensor}"
                    );
                    domains
                        .entry(tensor.to_string())
                        .or_default()
                        .push((lower, upper));
                }
            }
        }
        let graph = model.graph.context("ONNX graph absent")?;
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut files = HashMap::new();
        let mut initializers = HashMap::new();
        let mut shapes = HashMap::new();
        for t in &graph.initializer {
            ensure!(
                !initializers.contains_key(&t.name),
                "duplicate initializer {}",
                t.name
            );
            let tensor = load_tensor(t, &base, &mut files)?;
            shapes.insert(t.name.clone(), tensor.shape.clone());
            initializers.insert(t.name.clone(), tensor);
        }
        for v in graph
            .input
            .iter()
            .chain(&graph.output)
            .chain(&graph.value_info)
        {
            if let Some(shape) = v
                .r#type
                .as_ref()
                .and_then(|t| t.tensor_type.as_ref())
                .and_then(|t| t.shape.as_ref())
            {
                if let Some(dims) = shape
                    .dim
                    .iter()
                    .map(|d| d.dim_value.and_then(|d| usize::try_from(d).ok()))
                    .collect::<Option<Vec<_>>>()
                {
                    shapes.insert(v.name.clone(), dims);
                }
            }
        }
        let mut produced: HashSet<_> = initializers
            .keys()
            .cloned()
            .chain(graph.input.iter().map(|v| v.name.clone()))
            .collect();
        for n in &graph.node {
            ensure!(
                n.domain.is_empty() || n.domain == "ai.onnx",
                "unsupported ONNX domain {}",
                n.domain
            );
            for i in n.input.iter().filter(|s| !s.is_empty()) {
                ensure!(
                    produced.contains(i),
                    "ONNX graph is not topological: {} input {}",
                    n.name,
                    i
                );
            }
            for o in &n.output {
                ensure!(produced.insert(o.clone()), "duplicate graph output {}", o);
            }
        }
        Ok(Self {
            graph,
            initializers,
            shapes,
            base,
            domains,
        })
    }
    fn check_domain(&self, name: &str, tensor: &Tensor) -> Result<()> {
        if let Some(rules) = self.domains.get(name) {
            for &(lower, upper) in rules {
                for i in 0..tensor.len() {
                    let value = tensor.get(i).0;
                    ensure!(
                        value.is_finite() && value >= lower && value <= upper,
                        "approximation domain violation: {name}[{i}]={} outside [{lower}, {upper}]",
                        value.to_f64()
                    );
                }
            }
        }
        Ok(())
    }
    pub fn input_names(&self) -> Vec<String> {
        self.graph
            .input
            .iter()
            .filter(|v| !self.initializers.contains_key(&v.name))
            .map(|v| v.name.clone())
            .collect()
    }
    pub fn output_names(&self) -> Vec<String> {
        self.graph.output.iter().map(|v| v.name.clone()).collect()
    }
    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(Vec::as_slice)
    }
    pub fn initializer(&self, name: &str) -> Option<&Tensor> {
        self.initializers.get(name)
    }
    pub fn run(
        &self,
        inputs: HashMap<String, Tensor>,
        selected_outputs: Option<&[String]>,
    ) -> Result<RunResult> {
        self.run_observed(inputs, selected_outputs, true, |_, _| Ok(()))
    }
    pub fn run_observed(
        &self,
        inputs: HashMap<String, Tensor>,
        selected_outputs: Option<&[String]>,
        retain_outputs: bool,
        mut observer: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<RunResult> {
        let outputs = selected_outputs
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.output_names());
        let declared: HashSet<_> = self
            .graph
            .node
            .iter()
            .flat_map(|n| n.output.iter())
            .chain(self.graph.input.iter().map(|v| &v.name))
            .chain(self.initializers.keys())
            .collect();
        for output in &outputs {
            ensure!(
                declared.contains(output),
                "selected tensor {output} does not exist"
            );
        }
        let mut needed: HashSet<_> = outputs.iter().cloned().collect();
        let mut active = vec![false; self.graph.node.len()];
        for (i, n) in self.graph.node.iter().enumerate().rev() {
            if n.output.iter().any(|o| needed.contains(o)) {
                active[i] = true;
                needed.extend(n.input.iter().filter(|s| !s.is_empty()).cloned());
            }
        }
        let mut values = self.initializers.clone();
        for (name, tensor) in inputs {
            ensure!(
                self.graph.input.iter().any(|v| v.name == name),
                "unexpected input {name}"
            );
            if let Some(s) = self.tensor_shape(&name) {
                ensure!(s == tensor.shape, "input shape mismatch for {name}");
            }
            values.insert(name, tensor);
        }
        for name in self.input_names() {
            if needed.contains(&name) {
                ensure!(values.contains_key(&name), "missing input {name}");
            }
        }
        for (name, tensor) in &values {
            if needed.contains(name) {
                self.check_domain(name, tensor)?;
            }
        }
        let mut uses: HashMap<String, usize> = HashMap::new();
        for (i, n) in self.graph.node.iter().enumerate() {
            if active[i] {
                for input in &n.input {
                    *uses.entry(input.clone()).or_default() += 1;
                }
            }
        }
        let retained: HashSet<_> = if retain_outputs {
            outputs.iter().cloned().collect()
        } else {
            HashSet::new()
        };
        let mut count = 0;
        let mut peak = 0;
        for (i, n) in self.graph.node.iter().enumerate() {
            if !active[i] {
                continue;
            }
            let args = n
                .input
                .iter()
                .map(|s| {
                    if s.is_empty() {
                        Ok(None)
                    } else {
                        values
                            .get(s)
                            .map(Some)
                            .with_context(|| format!("missing tensor {s}"))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let result = evaluate(n, &args, &self.base)
                .with_context(|| format!("executing ONNX {} ({})", n.name, n.op_type))?;
            ensure!(
                result.len() == n.output.len(),
                "output count mismatch for {}",
                n.name
            );
            for (name, t) in n.output.iter().zip(result) {
                if let Some(shape) = self.tensor_shape(name) {
                    ensure!(
                        shape == t.shape,
                        "declared ONNX shape mismatch for {name}: expected {shape:?}, computed {:?}",
                        t.shape
                    );
                }
                self.check_domain(name, &t)?;
                observer(name, &t)?;
                values.insert(name.clone(), t);
            }
            count += 1;
            peak = peak.max(
                values
                    .values()
                    .filter(|t| matches!(t.data, TensorData::Mpfr(_)))
                    .map(Tensor::len)
                    .sum(),
            );
            for input in &n.input {
                if let Some(u) = uses.get_mut(input) {
                    *u -= 1;
                    if *u == 0
                        && !retained.contains(input)
                        && !self.initializers.contains_key(input)
                    {
                        values.remove(input);
                    }
                }
            }
            for output in &n.output {
                if uses.get(output).copied().unwrap_or(0) == 0 && !retained.contains(output) {
                    values.remove(output);
                }
            }
        }
        let result = if retain_outputs {
            outputs
                .into_iter()
                .map(|n| {
                    Ok((
                        n.clone(),
                        values
                            .remove(&n)
                            .with_context(|| format!("selected output {n} not found"))?,
                    ))
                })
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        Ok(RunResult {
            outputs: result,
            nodes_executed: count,
            peak_live_elements: peak,
        })
    }
}

fn load_tensor(
    t: &proto::TensorProto,
    base: &Path,
    files: &mut HashMap<PathBuf, Arc<Mmap>>,
) -> Result<Tensor> {
    let shape = t
        .dims
        .iter()
        .map(|&v| usize::try_from(v).context("negative tensor dimension"))
        .collect::<Result<Vec<_>>>()?;
    let count = elements(&shape)?;
    if t.data_location == 1 || !t.external_data.is_empty() {
        let ext: HashMap<_, _> = t
            .external_data
            .iter()
            .map(|v| (v.key.as_str(), v.value.as_str()))
            .collect();
        let loc = ext
            .get("location")
            .context("external tensor location absent")?;
        let path = base.join(loc);
        let byte_offset = ext.get("offset").unwrap_or(&"0").parse::<usize>()?;
        let width = match t.data_type {
            11 | 7 => 8,
            1 => 4,
            16 => 2,
            _ => bail!("unsupported external dtype {}", t.data_type),
        };
        let length = count
            .checked_mul(width)
            .context("external tensor length overflow")?;
        if let Some(n) = ext.get("length") {
            ensure!(
                n.parse::<usize>()? == length,
                "external tensor length mismatch"
            );
        }
        let bytes = if let Some(m) = files.get(&path) {
            m.clone()
        } else {
            let file = File::open(&path)
                .with_context(|| format!("opening external ONNX data {}", path.display()))?;
            let m = Arc::new(unsafe { Mmap::map(&file)? });
            files.insert(path, m.clone());
            m
        };
        ensure!(
            byte_offset
                .checked_add(length)
                .is_some_and(|end| end <= bytes.len()),
            "external tensor out of file bounds"
        );
        // Validate raw floating bits before any comparison or view can hide NaN.
        // This is a sequential mmap scan, with no allocation or MPFR expansion.
        let raw = &bytes[byte_offset..byte_offset + length];
        let finite = match t.data_type {
            11 => raw.chunks_exact(8).all(|b| {
                u64::from_le_bytes(b.try_into().unwrap()) & 0x7ff0_0000_0000_0000
                    != 0x7ff0_0000_0000_0000
            }),
            1 => raw
                .chunks_exact(4)
                .all(|b| u32::from_le_bytes(b.try_into().unwrap()) & 0x7f80_0000 != 0x7f80_0000),
            16 => raw
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80),
            _ => true,
        };
        ensure!(finite, "nonfinite external tensor {}", t.name);
        return Ok(Tensor::new(
            shape,
            TensorData::Mapped {
                bytes,
                byte_offset,
                dtype: t.data_type,
            },
        ));
    }
    let raw = &t.raw_data;
    if !raw.is_empty() {
        let width = match t.data_type {
            11 | 7 => 8,
            1 | 6 => 4,
            16 => 2,
            9 => 1,
            _ => bail!("unsupported inline dtype {}", t.data_type),
        };
        ensure!(
            count.checked_mul(width) == Some(raw.len()),
            "inline raw tensor length mismatch"
        );
    }
    match t.data_type {
        11 => Tensor::from_f64(
            shape,
            if raw.is_empty() {
                t.double_data.clone()
            } else {
                raw.chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            },
        ),
        1 => Tensor::from_f64(
            shape,
            if raw.is_empty() {
                t.float_data.iter().map(|&v| v as f64).collect()
            } else {
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
                    .collect()
            },
        ),
        7 => Tensor::from_i64(
            shape,
            if raw.is_empty() {
                t.int64_data.clone()
            } else {
                raw.chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            },
        ),
        6 | 9 => Tensor::from_i64(
            shape,
            if raw.is_empty() {
                t.int32_data.iter().map(|&v| v as i64).collect()
            } else if t.data_type == 9 {
                raw.iter().map(|&v| v as i64).collect()
            } else {
                raw.chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
                    .collect()
            },
        ),
        16 => Tensor::from_f64(
            shape,
            raw.chunks_exact(2)
                .map(|b| {
                    f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16) as f64
                })
                .collect(),
        ),
        _ => bail!("unsupported ONNX dtype {}", t.data_type),
    }
}

fn attr_i(n: &proto::Node, name: &str, default: i64) -> i64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| a.i)
        .unwrap_or(default)
}
fn attr_f(n: &proto::Node, name: &str, default: f64) -> f64 {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| a.f)
        .map(|v| v as f64)
        .unwrap_or(default)
}
fn attr_ints(n: &proto::Node, name: &str) -> Option<Vec<i64>> {
    n.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}
fn axis(a: i64, rank: usize) -> Result<usize> {
    let v = if a < 0 { a + rank as i64 } else { a };
    ensure!(v >= 0 && (v as usize) < rank, "axis {a} out of rank {rank}");
    Ok(v as usize)
}
fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1; rank];
    for i in 0..rank {
        let x = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let y = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        ensure!(
            x == y || x == 1 || y == 1,
            "incompatible broadcast {:?} {:?}",
            a,
            b
        );
        out[i] = if x == 1 { y } else { x };
    }
    Ok(out)
}
fn broadcast_index(mut i: usize, out: &[usize], input: &[usize]) -> usize {
    let mut j = 0;
    let mut stride = 1;
    for d in (0..out.len()).rev() {
        let c = i % out[d];
        i /= out[d];
        if d >= out.len() - input.len() {
            let dim = input[d - (out.len() - input.len())];
            if dim != 1 {
                j += c * stride;
            }
            stride *= dim;
        }
    }
    j
}
fn coordinate(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        out[d] = i % shape[d];
        i /= shape[d];
    }
    out
}
fn flat(coords: &[usize], shape: &[usize]) -> usize {
    coords.iter().zip(shape).fold(0, |a, (&c, &s)| a * s + c)
}

fn evaluate(n: &proto::Node, args: &[Option<&Tensor>], base: &Path) -> Result<Vec<Tensor>> {
    let arg = |i: usize| -> Result<&Tensor> {
        args.get(i)
            .copied()
            .flatten()
            .with_context(|| format!("missing operand {i}"))
    };
    let axes = |input: usize, rank: usize| -> Result<Vec<usize>> {
        let raw = if let Some(Some(t)) = args.get(input) {
            t.to_i64()?
        } else {
            attr_ints(n, "axes").unwrap_or_default()
        };
        raw.into_iter().map(|a| axis(a, rank)).collect()
    };
    let one = match n.op_type.as_str() {
        "Constant" => {
            if let Some(t) = n
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .and_then(|a| a.t.as_ref())
            {
                load_tensor(t, base, &mut HashMap::new())?
            } else if let Some(a) = n.attribute.iter().find(|a| a.name == "value_int") {
                Tensor::from_i64(vec![], vec![a.i.context("constant missing int")?])?
            } else if let Some(a) = n.attribute.iter().find(|a| a.name == "value_float") {
                Tensor::from_f64(vec![], vec![a.f.context("constant missing float")? as f64])?
            } else {
                bail!("unsupported Constant attribute")
            }
        }
        "Identity" => arg(0)?.clone(),
        "MatMul" => matmul(arg(0)?, arg(1)?)?,
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Max" | "Min" | "Equal" | "Greater"
        | "GreaterOrEqual" | "Less" | "LessOrEqual" | "And" | "Or" => {
            let a = arg(0)?;
            let b = arg(1)?;
            let shape = broadcast(&a.shape, &b.shape)?;
            let mut out = Vec::with_capacity(elements(&shape)?);
            for i in 0..elements(&shape)? {
                let x = a.get(broadcast_index(i, &shape, &a.shape)).0;
                let y = b.get(broadcast_index(i, &shape, &b.shape)).0;
                let z = match n.op_type.as_str() {
                    "Add" => Float::with_val(PRECISION, &x + &y),
                    "Sub" => Float::with_val(PRECISION, &x - &y),
                    "Mul" => Float::with_val(PRECISION, &x * &y),
                    "Div" => {
                        ensure!(!y.is_zero(), "division by zero");
                        Float::with_val(PRECISION, &x / &y)
                    }
                    "Pow" => {
                        use rug::ops::Pow;
                        ensure!(y.is_integer(), "only integer Pow is supported");
                        x.pow(
                            y.to_integer()
                                .and_then(|v| v.to_i32())
                                .context("Pow exponent outside i32")?,
                        )
                    }
                    "Max" => {
                        if x >= y {
                            x
                        } else {
                            y
                        }
                    }
                    "Min" => {
                        if x <= y {
                            x
                        } else {
                            y
                        }
                    }
                    op => Float::with_val(
                        PRECISION,
                        match op {
                            "Equal" => x == y,
                            "Greater" => x > y,
                            "GreaterOrEqual" => x >= y,
                            "Less" => x < y,
                            "LessOrEqual" => x <= y,
                            "And" => !x.is_zero() && !y.is_zero(),
                            "Or" => !x.is_zero() || !y.is_zero(),
                            _ => unreachable!(),
                        } as i32,
                    ),
                };
                out.push(AFloat(z));
            }
            Tensor::from_mpfr(shape, out)?
        }
        "Sqrt" | "Reciprocal" | "Abs" | "Neg" | "Exp" | "Log" | "Sigmoid" | "Softplus" | "Tanh"
        | "Erf" | "Floor" | "Not" => {
            let a = arg(0)?;
            let values = (0..a.len())
                .map(|i| {
                    let x = a.get(i).0;
                    let v = match n.op_type.as_str() {
                        "Sqrt" => {
                            ensure!(x >= 0, "sqrt of negative value");
                            x.sqrt()
                        }
                        "Reciprocal" => {
                            ensure!(!x.is_zero(), "reciprocal of zero");
                            x.recip()
                        }
                        "Abs" => x.abs(),
                        "Neg" => -x,
                        "Exp" => x.exp(),
                        "Log" => {
                            ensure!(x > 0, "log of nonpositive value");
                            x.ln()
                        }
                        "Tanh" => x.tanh(),
                        "Erf" => x.erf(),
                        "Floor" => x.floor(),
                        "Not" => Float::with_val(PRECISION, x.is_zero() as i32),
                        "Sigmoid" => {
                            let e = (-x.clone().abs()).exp();
                            if x >= 0 {
                                Float::with_val(PRECISION, 1) / (1 + e)
                            } else {
                                e.clone() / (1 + e)
                            }
                        }
                        "Softplus" => {
                            let positive = if x > 0 {
                                x.clone()
                            } else {
                                Float::with_val(PRECISION, 0)
                            };
                            positive + (-x.abs()).exp().ln_1p()
                        }
                        _ => unreachable!(),
                    };
                    Ok(AFloat(v))
                })
                .collect::<Result<_>>()?;
            Tensor::from_mpfr(a.shape.clone(), values)?
        }
        "Where" => {
            let c = arg(0)?;
            let a = arg(1)?;
            let b = arg(2)?;
            let shape = broadcast(&broadcast(&a.shape, &b.shape)?, &c.shape)?;
            let values = (0..elements(&shape)?)
                .map(|i| {
                    if c.get(broadcast_index(i, &shape, &c.shape)).0.is_zero() {
                        b.get(broadcast_index(i, &shape, &b.shape))
                    } else {
                        a.get(broadcast_index(i, &shape, &a.shape))
                    }
                })
                .collect();
            Tensor::from_mpfr(shape, values)?
        }
        "Reshape" => {
            let a = arg(0)?;
            let raw = arg(1)?.to_i64()?;
            let mut shape = Vec::new();
            let mut inferred = None;
            for (i, d) in raw.into_iter().enumerate() {
                if d == -1 {
                    ensure!(inferred.is_none(), "multiple inferred dimensions");
                    inferred = Some(i);
                    shape.push(1);
                } else if d == 0 && attr_i(n, "allowzero", 0) == 0 {
                    shape.push(*a.shape.get(i).context("reshape zero axis out of rank")?);
                } else {
                    shape.push(usize::try_from(d)?);
                }
            }
            if let Some(i) = inferred {
                let p = elements(&shape)?;
                ensure!(p != 0 && a.len() % p == 0, "invalid inferred reshape");
                shape[i] = a.len() / p;
            }
            a.reshape(shape)?
        }
        "Flatten" => {
            let a = arg(0)?;
            let raw = attr_i(n, "axis", 1);
            let ax = if raw < 0 {
                raw + a.shape.len() as i64
            } else {
                raw
            };
            ensure!(
                ax >= 0 && ax as usize <= a.shape.len(),
                "invalid flatten axis"
            );
            a.reshape(vec![
                elements(&a.shape[..ax as usize])?,
                elements(&a.shape[ax as usize..])?,
            ])?
        }
        "Transpose" => {
            let a = arg(0)?;
            let perm = attr_ints(n, "perm")
                .map(|v| {
                    v.into_iter()
                        .map(usize::try_from)
                        .collect::<std::result::Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_else(|| (0..a.shape.len()).rev().collect());
            a.transpose(&perm)?
        }
        "Unsqueeze" => {
            let a = arg(0)?;
            let raw = if let Some(Some(t)) = args.get(1) {
                t.to_i64()?
            } else {
                attr_ints(n, "axes").context("Unsqueeze axes absent")?
            };
            let rank = a.shape.len() + raw.len();
            let ax = raw
                .into_iter()
                .map(|i| axis(i, rank))
                .collect::<Result<HashSet<_>>>()?;
            ensure!(ax.len() == rank - a.shape.len(), "duplicate unsqueeze axis");
            let mut old = a.shape.iter();
            a.reshape(
                (0..rank)
                    .map(|i| {
                        if ax.contains(&i) {
                            1
                        } else {
                            *old.next().unwrap()
                        }
                    })
                    .collect(),
            )?
        }
        "Squeeze" => {
            let a = arg(0)?;
            let mut ax = axes(1, a.shape.len())?;
            if ax.is_empty() {
                ax = a
                    .shape
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| (d == 1).then_some(i))
                    .collect();
            }
            for &i in &ax {
                ensure!(a.shape[i] == 1, "cannot squeeze nonunit axis");
            }
            a.reshape(
                a.shape
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| (!ax.contains(&i)).then_some(d))
                    .collect(),
            )?
        }
        "Gather" => gather(
            arg(0)?,
            arg(1)?,
            axis(attr_i(n, "axis", 0), arg(0)?.shape.len())?,
        )?,
        "Concat" => {
            let tensors = args
                .iter()
                .map(|v| v.context("Concat input absent"))
                .collect::<Result<Vec<_>>>()?;
            concat(
                &tensors,
                axis(attr_i(n, "axis", 0), tensors[0].shape.len())?,
            )?
        }
        "Split" => {
            let a = arg(0)?;
            let ax = axis(attr_i(n, "axis", 0), a.shape.len())?;
            let sizes = if let Some(Some(t)) = args.get(1) {
                t.to_i64()?
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else if let Some(v) = attr_ints(n, "split") {
                v.into_iter()
                    .map(usize::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                ensure!(
                    a.shape[ax] % n.output.len() == 0,
                    "unequal implicit Split unsupported"
                );
                vec![a.shape[ax] / n.output.len(); n.output.len()]
            };
            ensure!(
                sizes.len() == n.output.len() && sizes.iter().sum::<usize>() == a.shape[ax],
                "Split sizes mismatch"
            );
            let mut start = 0;
            return sizes
                .into_iter()
                .map(|size| {
                    let mut t = a.clone();
                    t.offset += start * a.strides[ax];
                    t.shape[ax] = size;
                    start += size;
                    Ok(t)
                })
                .collect();
        }
        "Slice" => slice(n, args)?,
        "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" => {
            let a = arg(0)?;
            let mut ax = axes(1, a.shape.len())?;
            if ax.is_empty() {
                if attr_i(n, "noop_with_empty_axes", 0) != 0 {
                    return Ok(vec![a.clone()]);
                }
                ax = (0..a.shape.len()).collect();
            }
            reduce(a, &ax, attr_i(n, "keepdims", 1) != 0, &n.op_type)?
        }
        "Shape" => {
            let a = arg(0)?;
            let rank = a.shape.len() as i64;
            let norm = |v: i64| {
                if v < 0 {
                    (v + rank).max(0)
                } else {
                    v.min(rank)
                }
            };
            let start = norm(attr_i(n, "start", 0)) as usize;
            let end = norm(attr_i(n, "end", rank)) as usize;
            let v = if end >= start {
                a.shape[start..end].iter().map(|&v| v as i64).collect()
            } else {
                vec![]
            };
            Tensor::from_i64(vec![end.saturating_sub(start)], v)?
        }
        "Cast" => {
            let a = arg(0)?;
            match attr_i(n, "to", 0) {
                7 | 6 => {
                    let values = (0..a.len())
                        .map(|i| {
                            a.get(i)
                                .0
                                .to_integer_round(Round::Zero)
                                .and_then(|v| v.0.to_i64())
                                .context("Cast integer overflow")
                        })
                        .collect::<Result<_>>()?;
                    Tensor::from_i64(a.shape.clone(), values)?
                }
                1 | 11 | 16 => Tensor::from_mpfr(a.shape.clone(), a.mpfr_values())?,
                9 => Tensor::from_i64(
                    a.shape.clone(),
                    (0..a.len())
                        .map(|i| (!a.get(i).0.is_zero()) as i64)
                        .collect(),
                )?,
                dtype => bail!("unsupported cast dtype {dtype}"),
            }
        }
        "Expand" => {
            let a = arg(0)?;
            let target = arg(1)?
                .to_i64()?
                .into_iter()
                .map(usize::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let shape = broadcast(&a.shape, &target)?;
            Tensor::from_mpfr(
                shape.clone(),
                (0..elements(&shape)?)
                    .map(|i| a.get(broadcast_index(i, &shape, &a.shape)))
                    .collect(),
            )?
        }
        "ConstantOfShape" => {
            let shape = arg(0)?
                .to_i64()?
                .into_iter()
                .map(usize::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let value = if let Some(t) = n
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .and_then(|a| a.t.as_ref())
            {
                load_tensor(t, base, &mut HashMap::new())?.get(0)
            } else {
                AFloat(Float::with_val(PRECISION, 0))
            };
            Tensor::from_mpfr(shape.clone(), vec![value; elements(&shape)?])?
        }
        "Softmax" => softmax(arg(0)?, axis(attr_i(n, "axis", -1), arg(0)?.shape.len())?)?,
        "LayerNormalization" => layer_norm(n, &args)?,
        op => bail!("unsupported standard ONNX operator {op}"),
    };
    Ok(vec![one])
}

fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure!(
        a.shape.len() >= 2 && b.shape.len() >= 2,
        "MatMul requires rank >=2"
    );
    let ar = a.shape.len();
    let br = b.shape.len();
    let (m, k, n) = (a.shape[ar - 2], a.shape[ar - 1], b.shape[br - 1]);
    ensure!(k == b.shape[br - 2], "MatMul inner dimension mismatch");
    let batch = broadcast(&a.shape[..ar - 2], &b.shape[..br - 2])?;
    let mut shape = batch.clone();
    shape.extend([m, n]);
    let total = elements(&shape)?;
    let data = (0..total)
        .into_par_iter()
        .map(|i| {
            let ba = i / (m * n);
            let row = (i / n) % m;
            let col = i % n;
            let ab = broadcast_index(ba, &batch, &a.shape[..ar - 2]) * m * k;
            let bb = broadcast_index(ba, &batch, &b.shape[..br - 2]) * k * n;
            let mut sum = Float::with_val(PRECISION, 0);
            let mut product = Float::with_val(PRECISION, 0);
            for d in 0..k {
                let ai = ab + row * k + d;
                let bi = bb + d * n + col;
                match (&a.data, &b.data) {
                    (TensorData::Mpfr(av), TensorData::Mpfr(bv)) => {
                        product.assign(&av[a.storage_index(ai)].0 * &bv[b.storage_index(bi)].0)
                    }
                    (TensorData::Mpfr(av), _) => {
                        if let Some(v) = b.integer(bi) {
                            product.assign(&av[a.storage_index(ai)].0 * v)
                        } else {
                            product.assign(&av[a.storage_index(ai)].0 * b.fixed(bi).unwrap())
                        }
                    }
                    (_, TensorData::Mpfr(bv)) => {
                        if let Some(v) = a.integer(ai) {
                            product.assign(&bv[b.storage_index(bi)].0 * v)
                        } else {
                            product.assign(&bv[b.storage_index(bi)].0 * a.fixed(ai).unwrap())
                        }
                    }
                    _ => {
                        product.assign(a.get(ai).0);
                        product *= b.get(bi).0;
                    }
                }
                sum += &product;
            }
            AFloat(sum)
        })
        .collect();
    Tensor::from_mpfr(shape, data)
}
fn gather(a: &Tensor, indices: &Tensor, ax: usize) -> Result<Tensor> {
    let idx = indices.to_i64()?;
    let mut shape = a.shape[..ax].to_vec();
    shape.extend(&indices.shape);
    shape.extend(&a.shape[ax + 1..]);
    let inner = elements(&a.shape[ax + 1..])?;
    let outer = elements(&a.shape[..ax])?;
    let mut values = Vec::with_capacity(elements(&shape)?);
    for o in 0..outer {
        for &i in &idx {
            let i = if i < 0 { i + a.shape[ax] as i64 } else { i };
            ensure!(
                i >= 0 && (i as usize) < a.shape[ax],
                "Gather index out of range"
            );
            for j in 0..inner {
                values.push(a.get((o * a.shape[ax] + i as usize) * inner + j));
            }
        }
    }
    Tensor::from_mpfr(shape, values)
}
fn concat(ts: &[&Tensor], ax: usize) -> Result<Tensor> {
    ensure!(!ts.is_empty(), "empty Concat");
    let mut shape = ts[0].shape.clone();
    shape[ax] = 0;
    for t in ts {
        ensure!(t.shape.len() == shape.len(), "Concat rank mismatch");
        for d in 0..shape.len() {
            if d != ax {
                ensure!(t.shape[d] == shape[d], "Concat shape mismatch");
            }
        }
        shape[ax] += t.shape[ax];
    }
    let outer = elements(&shape[..ax])?;
    let inner = elements(&shape[ax + 1..])?;
    let mut v = Vec::with_capacity(elements(&shape)?);
    for o in 0..outer {
        for t in ts {
            let block = t.shape[ax] * inner;
            for j in 0..block {
                v.push(t.get(o * block + j));
            }
        }
    }
    Tensor::from_mpfr(shape, v)
}
fn slice(n: &proto::Node, args: &[Option<&Tensor>]) -> Result<Tensor> {
    let a = args[0].context("Slice data absent")?;
    let read = |i: usize, name: &str| -> Result<Vec<i64>> {
        if let Some(Some(t)) = args.get(i) {
            t.to_i64()
        } else {
            attr_ints(n, name).with_context(|| format!("Slice {name} absent"))
        }
    };
    let starts = read(1, "starts")?;
    let ends = read(2, "ends")?;
    let axes = if args.get(3).is_some_and(Option::is_some) {
        read(3, "axes")?
    } else {
        (0..starts.len() as i64).collect()
    };
    let steps = if args.get(4).is_some_and(Option::is_some) {
        read(4, "steps")?
    } else {
        vec![1; starts.len()]
    };
    ensure!(
        starts.len() == ends.len() && starts.len() == axes.len() && starts.len() == steps.len(),
        "Slice parameter sizes mismatch"
    );
    let mut lists: Vec<Vec<usize>> = a.shape.iter().map(|&d| (0..d).collect()).collect();
    for i in 0..starts.len() {
        let ax = axis(axes[i], a.shape.len())?;
        let d = a.shape[ax] as i64;
        let step = steps[i];
        ensure!(step != 0, "Slice step zero");
        let norm = |x: i64| {
            let x = if x < 0 { x.saturating_add(d) } else { x };
            if step > 0 {
                x.clamp(0, d)
            } else {
                x.clamp(-1, d - 1)
            }
        };
        let mut cur = norm(starts[i]);
        let end = norm(ends[i]);
        let mut list = Vec::new();
        while if step > 0 { cur < end } else { cur > end } {
            list.push(cur as usize);
            cur = cur.saturating_add(step);
        }
        lists[ax] = list;
    }
    let shape = lists.iter().map(Vec::len).collect::<Vec<_>>();
    let values = (0..elements(&shape)?)
        .map(|i| {
            let c = coordinate(i, &shape);
            a.get(flat(
                &c.iter()
                    .enumerate()
                    .map(|(d, &x)| lists[d][x])
                    .collect::<Vec<_>>(),
                &a.shape,
            ))
        })
        .collect();
    Tensor::from_mpfr(shape, values)
}
fn reduce(a: &Tensor, axes: &[usize], keep: bool, op: &str) -> Result<Tensor> {
    ensure!(
        axes.iter().copied().collect::<HashSet<_>>().len() == axes.len(),
        "duplicate reduce axes"
    );
    let mut shape = Vec::new();
    for (i, &d) in a.shape.iter().enumerate() {
        if axes.contains(&i) {
            if keep {
                shape.push(1)
            }
        } else {
            shape.push(d)
        }
    }
    let count = elements(&shape)?;
    let mut values = vec![Float::with_val(PRECISION, 0); count];
    let mut seen = vec![false; count];
    for i in 0..a.len() {
        let c = coordinate(i, &a.shape);
        let mut out = Vec::new();
        for (d, &v) in c.iter().enumerate() {
            if axes.contains(&d) {
                if keep {
                    out.push(0)
                }
            } else {
                out.push(v)
            }
        }
        let j = flat(&out, &shape);
        let x = a.get(i).0;
        match op {
            "ReduceSum" | "ReduceMean" => values[j] += &x,
            "ReduceMax" => {
                if !seen[j] || x > values[j] {
                    values[j] = x;
                }
            }
            "ReduceMin" => {
                if !seen[j] || x < values[j] {
                    values[j] = x;
                }
            }
            _ => unreachable!(),
        }
        seen[j] = true;
    }
    if op == "ReduceMean" {
        let n = axes.iter().map(|&i| a.shape[i]).product::<usize>();
        ensure!(n > 0, "empty ReduceMean");
        for v in &mut values {
            *v /= n;
        }
    }
    if op == "ReduceMax" || op == "ReduceMin" {
        ensure!(seen.iter().all(|&v| v), "empty min/max reduction");
    }
    Tensor::from_mpfr(shape, values.into_iter().map(AFloat).collect())
}
fn softmax(a: &Tensor, ax: usize) -> Result<Tensor> {
    let outer = elements(&a.shape[..ax])?;
    let inner = elements(&a.shape[ax + 1..])?;
    let width = a.shape[ax];
    ensure!(width > 0, "empty softmax");
    let mut values = vec![AFloat(Float::with_val(PRECISION, 0)); a.len()];
    for o in 0..outer {
        for j in 0..inner {
            let mut max = a.get(o * width * inner + j).0;
            for d in 1..width {
                let x = a.get((o * width + d) * inner + j).0;
                if x > max {
                    max = x
                }
            }
            let mut sum = Float::with_val(PRECISION, 0);
            for d in 0..width {
                let p = (o * width + d) * inner + j;
                let e = Float::with_val(PRECISION, a.get(p).0 - &max).exp();
                sum += &e;
                values[p] = AFloat(e);
            }
            for d in 0..width {
                values[(o * width + d) * inner + j].0 /= &sum;
            }
        }
    }
    Tensor::from_mpfr(a.shape.clone(), values)
}
fn layer_norm(n: &proto::Node, args: &[Option<&Tensor>]) -> Result<Tensor> {
    ensure!(
        n.output.len() == 1,
        "LayerNormalization auxiliary outputs unsupported"
    );
    let a = args[0].context("LayerNorm input absent")?;
    let scale = args[1].context("LayerNorm scale absent")?;
    let bias = args.get(2).copied().flatten();
    let ax = axis(attr_i(n, "axis", -1), a.shape.len())?;
    let width = elements(&a.shape[ax..])?;
    ensure!(
        width > 0 && scale.len() == width,
        "LayerNorm scale shape mismatch"
    );
    let eps = Float::with_val(PRECISION, attr_f(n, "epsilon", 1e-5));
    let mut values = Vec::with_capacity(a.len());
    for row in 0..a.len() / width {
        let mut mean = Float::with_val(PRECISION, 0);
        for j in 0..width {
            mean += a.get(row * width + j).0;
        }
        mean /= width;
        let mut variance = Float::with_val(PRECISION, 0);
        for j in 0..width {
            let d = Float::with_val(PRECISION, a.get(row * width + j).0 - &mean);
            variance += Float::with_val(PRECISION, &d * &d);
        }
        variance /= width;
        variance += &eps;
        let denom = variance.sqrt();
        for j in 0..width {
            let mut x = Float::with_val(PRECISION, a.get(row * width + j).0 - &mean);
            x /= &denom;
            x *= scale.get(j).0;
            if let Some(b) = bias {
                x += b.get(j).0;
            }
            values.push(AFloat(x));
        }
    }
    Tensor::from_mpfr(a.shape.clone(), values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "spain-onnx-v2-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
        fn write(&self, graph: proto::Graph) -> PathBuf {
            let p = self.0.join("model.onnx");
            std::fs::write(
                &p,
                proto::Model {
                    graph: Some(graph),
                    ..Default::default()
                }
                .encode_to_vec(),
            )
            .unwrap();
            p
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn vi(name: &str) -> proto::ValueInfo {
        proto::ValueInfo {
            name: name.into(),
            ..Default::default()
        }
    }
    fn node(op: &str, ins: &[&str], out: &str) -> proto::Node {
        proto::Node {
            op_type: op.into(),
            name: out.into(),
            input: ins.iter().map(|v| v.to_string()).collect(),
            output: vec![out.into()],
            ..Default::default()
        }
    }
    fn ai(name: &str, i: i64) -> proto::Attribute {
        proto::Attribute {
            name: name.into(),
            i: Some(i),
            ..Default::default()
        }
    }
    fn ints(name: &str, v: Vec<i64>) -> proto::Attribute {
        proto::Attribute {
            name: name.into(),
            ints: v,
            ..Default::default()
        }
    }
    fn i64_const(name: &str, shape: Vec<i64>, values: Vec<i64>) -> proto::TensorProto {
        proto::TensorProto {
            name: name.into(),
            data_type: 7,
            dims: shape,
            int64_data: values,
            ..Default::default()
        }
    }
    fn f64_const(name: &str, shape: Vec<i64>, values: Vec<f64>) -> proto::TensorProto {
        proto::TensorProto {
            name: name.into(),
            data_type: 11,
            dims: shape,
            double_data: values,
            ..Default::default()
        }
    }
    #[test]
    fn mapped_weight_transpose_and_last_token_gather() {
        let f = Fixture::new();
        let weights = [1f64, 2., 3., 4., 5., 6.];
        std::fs::write(
            f.0.join("weights.bin"),
            weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let w = proto::TensorProto {
            name: "W".into(),
            dims: vec![2, 3],
            data_type: 11,
            data_location: 1,
            external_data: vec![
                proto::KeyValue {
                    key: "location".into(),
                    value: "weights.bin".into(),
                },
                proto::KeyValue {
                    key: "length".into(),
                    value: "48".into(),
                },
            ],
            ..Default::default()
        };
        let mut tr = node("Transpose", &["W"], "Wt");
        tr.attribute.push(ints("perm", vec![1, 0]));
        let mut take = node("Gather", &["product", "last"], "last_logits");
        take.attribute.push(ai("axis", 1));
        let path = f.write(proto::Graph {
            initializer: vec![w, i64_const("last", vec![1], vec![-1])],
            input: vec![vi("X")],
            node: vec![tr, node("MatMul", &["X", "Wt"], "product"), take],
            output: vec![vi("last_logits")],
            ..Default::default()
        });
        let model = OnnxV2::load(path).unwrap();
        assert!(matches!(
            model.initializer("W").unwrap().data,
            TensorData::Mapped { .. }
        ));
        let x = Tensor::from_f64(vec![1, 2, 3], vec![1., 2., 3., -1., 0., 1.]).unwrap();
        let r = model.run(HashMap::from([("X".into(), x)]), None).unwrap();
        assert_eq!(r.outputs[0].1.shape, vec![1, 1, 2]);
        assert_eq!(r.outputs[0].1.to_f64(), vec![2., 2.]);
        assert_eq!(r.nodes_executed, 3);
    }
    #[test]
    fn matmul_accumulates_beyond_binary64_precision() {
        let a = Tensor::from_f64(vec![1, 3], vec![1e20, 1., -1e20]).unwrap();
        let b = Tensor::from_f64(vec![3, 1], vec![1., 1., 1.]).unwrap();
        assert_eq!(matmul(&a, &b).unwrap().get_f64(0), 1.);
    }
    #[test]
    fn selected_output_prunes_unrelated_graph_and_missing_inputs() {
        let f = Fixture::new();
        let path = f.write(proto::Graph {
            input: vec![vi("tokens"), vi("unprovided")],
            initializer: vec![f64_const(
                "embedding",
                vec![3, 2],
                vec![1., 2., 3., 4., 5., 6.],
            )],
            node: vec![
                node("Gather", &["embedding", "tokens"], "embedded"),
                node("Unsupported", &["unprovided"], "other"),
            ],
            output: vec![vi("embedded"), vi("other")],
            ..Default::default()
        });
        let model = OnnxV2::load(path).unwrap();
        let inputs =
            HashMap::from([("tokens".into(), Tensor::from_i64(vec![1], vec![2]).unwrap())]);
        let result = model
            .run(inputs.clone(), Some(&["embedded".into()]))
            .unwrap();
        assert_eq!(result.nodes_executed, 1);
        assert_eq!(result.outputs[0].1.to_f64(), vec![5., 6.]);
        assert!(model.run(inputs, None).is_err());
    }
    #[test]
    fn observed_outputs_stream_in_graph_order_and_are_released() {
        let f = Fixture::new();
        let mut ns = vec![];
        let mut prev = "X".to_string();
        for i in 0..20 {
            let next = format!("v{i}");
            ns.push(node("Add", &[&prev, "one"], &next));
            prev = next;
        }
        let path = f.write(proto::Graph {
            input: vec![vi("X")],
            initializer: vec![f64_const("one", vec![], vec![1.])],
            output: ns.iter().map(|n| vi(&n.output[0])).collect(),
            node: ns,
            ..Default::default()
        });
        let model = OnnxV2::load(path).unwrap();
        let mut saved = vec![];
        let r = model
            .run_observed(
                HashMap::from([("X".into(), Tensor::from_f64(vec![2], vec![0., 1.]).unwrap())]),
                None,
                false,
                |name, t| {
                    saved.push((name.to_string(), t.to_f64()));
                    Ok(())
                },
            )
            .unwrap();
        assert!(r.outputs.is_empty());
        assert_eq!(saved.len(), 20);
        assert_eq!(saved.last().unwrap().1, vec![20., 21.]);
        assert!(r.peak_live_elements <= 4);
    }
    #[test]
    fn integer_indices_are_not_rounded_through_binary64() {
        let x = Tensor::from_i64(vec![2], vec![i64::MAX, -1]).unwrap();
        assert_eq!(x.to_i64().unwrap(), vec![i64::MAX, -1]);
        let data = Tensor::from_f64(vec![2], vec![3., 4.]).unwrap();
        assert!(gather(&data, &x, 0).is_err());
    }
    #[test]
    fn slice_concat_reduction_and_broadcast_follow_onnx_semantics() {
        let a = Tensor::from_f64(vec![2, 3], vec![0., 1., 2., 3., 4., 5.]).unwrap();
        let begin = Tensor::from_i64(vec![1], vec![2]).unwrap();
        let end = Tensor::from_i64(vec![1], vec![i64::MIN]).unwrap();
        let axis = Tensor::from_i64(vec![1], vec![1]).unwrap();
        let step = Tensor::from_i64(vec![1], vec![-1]).unwrap();
        let s = slice(
            &node("Slice", &[], "s"),
            &[Some(&a), Some(&begin), Some(&end), Some(&axis), Some(&step)],
        )
        .unwrap();
        assert_eq!(s.to_f64(), vec![2., 1., 0., 5., 4., 3.]);
        let c = concat(&[&a, &s], 0).unwrap();
        assert_eq!(c.shape, vec![4, 3]);
        let r = reduce(&c, &[1], true, "ReduceMean").unwrap();
        assert_eq!(r.to_f64(), vec![1., 4., 1., 4.]);
        let v = Tensor::from_f64(vec![3], vec![10., 20., 30.]).unwrap();
        let out = evaluate(
            &node("Add", &[], "out"),
            &[Some(&a), Some(&v)],
            Path::new("."),
        )
        .unwrap();
        assert_eq!(out[0].to_f64(), vec![10., 21., 32., 13., 24., 35.]);
    }
    #[test]
    fn quantization_rounds_ties_to_even_and_rejects_overflow() {
        let t = Tensor::from_f64(vec![4], vec![0.125, 0.375, -0.125, -0.375]).unwrap();
        let mut bytes = vec![];
        t.write_scaled_i128(&mut bytes, 2).unwrap();
        let values = bytes
            .chunks_exact(16)
            .map(|v| i128::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0, 2, 0, -2]);
        let large = Tensor::from_f64(vec![1], vec![1e30]).unwrap();
        assert!(large.write_scaled_i128(&mut vec![], 75).is_err());
    }
    #[test]
    fn external_truncation_and_wrong_lengths_fail_closed() {
        let f = Fixture::new();
        std::fs::write(f.0.join("weights.bin"), [0; 8]).unwrap();
        let t = proto::TensorProto {
            name: "W".into(),
            dims: vec![2],
            data_type: 11,
            data_location: 1,
            external_data: vec![proto::KeyValue {
                key: "location".into(),
                value: "weights.bin".into(),
            }],
            ..Default::default()
        };
        let path = f.write(proto::Graph {
            initializer: vec![t],
            ..Default::default()
        });
        assert!(OnnxV2::load(path).is_err());
    }
    #[test]
    fn external_nonfinite_cannot_hide_behind_comparison_or_view() {
        let f = Fixture::new();
        for (dtype, bits) in [
            (11, f64::NAN.to_le_bytes().to_vec()),
            (11, f64::NEG_INFINITY.to_le_bytes().to_vec()),
            (1, f32::INFINITY.to_le_bytes().to_vec()),
            (16, 0x7fc0u16.to_le_bytes().to_vec()),
        ] {
            std::fs::write(f.0.join("weights.bin"), bits).unwrap();
            let t = proto::TensorProto {
                name: "W".into(),
                dims: vec![1],
                data_type: dtype,
                data_location: 1,
                external_data: vec![proto::KeyValue {
                    key: "location".into(),
                    value: "weights.bin".into(),
                }],
                ..Default::default()
            };
            let path = f.write(proto::Graph {
                initializer: vec![t],
                ..Default::default()
            });
            let error = OnnxV2::load(path).err().unwrap().to_string();
            assert!(error.contains("nonfinite external tensor W"), "{error}");
        }
    }
    #[test]
    fn square_root_sign_and_invalid_arithmetic_fail_closed() {
        let positive = Tensor::from_f64(vec![1], vec![4.]).unwrap();
        let negative = Tensor::from_f64(vec![1], vec![-1.]).unwrap();
        let zero = Tensor::from_f64(vec![1], vec![0.]).unwrap();
        let n = node("Sqrt", &[], "Y");
        assert_eq!(
            evaluate(&n, &[Some(&positive)], Path::new(".")).unwrap()[0].to_f64(),
            vec![2.]
        );
        assert!(evaluate(&n, &[Some(&negative)], Path::new(".")).is_err());
        assert!(
            evaluate(
                &node("Div", &[], "Y"),
                &[Some(&positive), Some(&zero)],
                Path::new(".")
            )
            .is_err()
        );
    }
    #[test]
    fn approximation_metadata_rejects_domain_violations_before_observer() {
        let f = Fixture::new();
        let model = proto::Model {
            graph: Some(proto::Graph {
                input: vec![vi("X")],
                output: vec![vi("Y")],
                node: vec![node("Identity", &["X"], "Y")],
                ..Default::default()
            }),
            metadata_props: vec![proto::KeyValue {
                key: "spain.domains".into(),
                value: r#"[{"tensor":"Y","lower":-1,"upper":1}]"#.into(),
            }],
        };
        let path = f.0.join("model.onnx");
        std::fs::write(&path, model.encode_to_vec()).unwrap();
        let runner = OnnxV2::load(path).unwrap();
        let mut observed = false;
        let result = runner.run_observed(
            HashMap::from([("X".into(), Tensor::from_f64(vec![1], vec![1.1]).unwrap())]),
            None,
            false,
            |_, _| {
                observed = true;
                Ok(())
            },
        );
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("domain violation")
        );
        assert!(!observed);
    }
}
