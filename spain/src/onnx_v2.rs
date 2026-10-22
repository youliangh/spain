//! Versioned ONNX artifact executor for application-bound Spain proofs.
//!
//! All graph evaluation is performed by the generic high-precision ONNX runner.
//! The verifier evaluates only public lookups and fixed-weight projections.
use crate::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};
use anyhow::{Context, Result, bail, ensure};
use model::onnx_v2::{OnnxV2, Tensor};
use model::{AFloat, FromPrimitive, ToPrimitiveExt};
use parse::{
    generalized::InjectionInfo,
    mat::{Matrix, MatrixData},
};
use rug::Float;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use stream::bigvec::BigVec;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorRange {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PublicGather {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
    pub table: String,
    pub indices: String,
    pub axis: i64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PublicLayout {
    pub constant_offset: usize,
    pub fingerprint_offset: usize,
    pub fingerprint_length: usize,
    pub inputs: Vec<TensorRange>,
    pub gathers: Vec<PublicGather>,
    pub claims: Vec<TensorRange>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChallengeSlot {
    pub matrix: String,
    pub entry_offset: usize,
    pub length: usize,
    pub source: String,
    #[serde(default)]
    pub tensor: Option<String>,
    pub tensor_offset: usize,
    #[serde(default = "one")]
    pub multiplier: f64,
}
fn one() -> f64 {
    1.
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Asset {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArtifactMeta {
    pub schema_version: u32,
    pub num_constraints: usize,
    pub num_public_values: usize,
    pub num_random_values: usize,
    pub num_witness_values: usize,
    pub num_secondary_witness_values: usize,
    pub num_secondary_constraint_variables: usize,
    pub matrix_entries: BTreeMap<String, usize>,
    pub primary_outputs: Vec<TensorRange>,
    pub secondary_outputs: Vec<TensorRange>,
    pub secondary_inputs: Vec<NamedTensor>,
    pub public: PublicLayout,
    pub challenge_slots: Vec<ChallengeSlot>,
    pub projection_outputs: Vec<NamedTensor>,
    pub artifacts: Vec<Asset>,
    pub fingerprint_sha256: String,
}
impl ArtifactMeta {
    pub fn protocol_metadata(&self) -> Metadata {
        Metadata {
            num_witness_values: self.num_witness_values,
            num_public_values: self.num_public_values,
            num_random_values: self.num_random_values,
            num_secondary_witness_values: self.num_secondary_witness_values,
            num_secondary_constraint_variables: 0,
            primary_output_labels: vec![],
            secondary_output_labels: vec![],
        }
    }
}

pub fn denominator(bits: usize) -> AFloat {
    AFloat(Float::with_val(128, 1) << bits as i32)
}
pub fn quantize(value: &AFloat, scale: &AFloat) -> Result<i128> {
    ensure!(value.0.is_finite(), "nonfinite ONNX witness/coefficient");
    let scaled = value.clone() * scale.clone();
    scaled
        .to_rug_integer()
        .to_i128()
        .context("fixed-point i128 overflow")
}
pub fn file_sha256(path: &Path) -> Result<String> {
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
fn fingerprint_bytes(value: &str) -> Result<Vec<u8>> {
    ensure!(
        value.len() == 64 && value.is_ascii(),
        "invalid model fingerprint"
    );
    (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&value[i..i + 2], 16).map_err(Into::into))
        .collect()
}

#[derive(Debug)]
pub struct Artifact {
    pub directory: PathBuf,
    pub meta: ArtifactMeta,
}
impl Artifact {
    pub fn load(directory: impl AsRef<Path>) -> Result<Arc<Self>> {
        let directory = directory.as_ref().canonicalize()?;
        let raw: serde_json::Value =
            serde_json::from_reader(File::open(directory.join("meta.json"))?)?;
        let meta: ArtifactMeta = serde_json::from_value(raw.clone())?;
        ensure!(meta.schema_version == 2, "expected ONNX artifact schema 2");
        ensure!(
            meta.num_secondary_constraint_variables == 0,
            "deprecated secondary coefficient oracle"
        );
        ensure!(
            meta.num_constraints > 0 && meta.num_witness_values > 0,
            "empty circuit"
        );
        ensure!(
            meta.public.constant_offset == 0 && meta.public.fingerprint_length == 32,
            "invalid public constant/fingerprint layout"
        );
        fingerprint_bytes(&meta.fingerprint_sha256)?;
        let canonical_assets =
            serde_json::to_vec(raw.get("artifacts").context("missing artifact manifest")?)?;
        ensure!(
            format!("{:x}", Sha256::digest(&canonical_assets)) == meta.fingerprint_sha256,
            "artifact fingerprint does not match canonical manifest"
        );
        let layout: serde_json::Value =
            serde_json::from_reader(File::open(directory.join("layout.json"))?)?;
        let mut core = raw.clone();
        core.as_object_mut()
            .context("metadata must be an object")?
            .remove("artifacts");
        core.as_object_mut().unwrap().remove("fingerprint_sha256");
        ensure!(
            core == layout,
            "metadata does not match authenticated layout"
        );
        let mut names = HashSet::new();
        for asset in &meta.artifacts {
            ensure!(names.insert(asset.path.clone()), "duplicate artifact path");
            let path = directory.join(&asset.path);
            ensure!(
                path.metadata()?.len() == asset.bytes,
                "artifact size changed: {}",
                asset.path
            );
            ensure!(
                file_sha256(&path)? == asset.sha256,
                "artifact hash changed: {}",
                asset.path
            );
        }
        for required in [
            "original_model.onnx",
            "primary_model.onnx",
            "secondary_model.onnx",
            "projection_model.onnx",
            "layout.json",
            "A.bin",
            "B.bin",
            "C.bin",
        ] {
            ensure!(
                names.contains(required),
                "missing authenticated artifact: {required}"
            );
        }
        for (ranges, count) in [
            (&meta.primary_outputs, meta.num_witness_values),
            (&meta.secondary_outputs, meta.num_secondary_witness_values),
        ] {
            let mut ordered: Vec<_> = ranges.iter().collect();
            ordered.sort_by_key(|r| r.offset);
            let mut end = 0usize;
            for range in ordered {
                ensure!(range.offset == end, "noncontiguous witness layout");
                ensure!(
                    range.length == range.shape.iter().product::<usize>(),
                    "tensor layout shape mismatch"
                );
                end = end
                    .checked_add(range.length)
                    .context("witness layout overflow")?;
            }
            ensure!(end == count, "witness layout count mismatch");
        }
        Ok(Arc::new(Self { directory, meta }))
    }

    pub fn input_tensors(&self, ids: &[i64]) -> Result<HashMap<String, Tensor>> {
        ensure!(
            self.meta.public.inputs.len() == 1,
            "token CLI expects exactly one model input"
        );
        let input = &self.meta.public.inputs[0];
        ensure!(
            ids.len() == input.length,
            "token count does not match compiled sequence length"
        );
        ensure!(ids.iter().all(|x| *x >= 0), "negative token ID");
        Ok(HashMap::from([(
            input.name.clone(),
            Tensor::from_i64(input.shape.clone(), ids.to_vec())?,
        )]))
    }

    pub fn public_values(
        &self,
        inputs: &HashMap<String, Tensor>,
        claims: &HashMap<String, Vec<i128>>,
        scale: &AFloat,
    ) -> Result<Vec<i128>> {
        let m = &self.meta;
        let mut values = vec![0; m.num_public_values];
        let mut set = vec![false; m.num_public_values];
        let mut put = |offset: usize, incoming: &[i128]| -> Result<()> {
            ensure!(
                offset
                    .checked_add(incoming.len())
                    .is_some_and(|end| end <= values.len()),
                "public range outside layout"
            );
            for (i, value) in incoming.iter().enumerate() {
                ensure!(!set[offset + i], "overlapping public ranges");
                values[offset + i] = *value;
                set[offset + i] = true;
            }
            Ok(())
        };
        put(0, &[quantize(&AFloat::from_i64(1).unwrap(), scale)?])?;
        let fingerprint = fingerprint_bytes(&m.fingerprint_sha256)?
            .into_iter()
            .map(|b| quantize(&AFloat::from_i64(b as i64).unwrap(), scale))
            .collect::<Result<Vec<_>>>()?;
        put(m.public.fingerprint_offset, &fingerprint)?;
        for range in &m.public.inputs {
            let tensor = inputs.get(&range.name).context("missing public input")?;
            ensure!(tensor.shape == range.shape, "public input shape mismatch");
            put(range.offset, &quantized_tensor(tensor, scale)?)?;
        }
        if !m.public.gathers.is_empty() {
            let selected = m
                .public
                .gathers
                .iter()
                .map(|g| g.name.clone())
                .collect::<Vec<_>>();
            let runner = OnnxV2::load(self.directory.join("primary_model.onnx"))?;
            let result = runner.run(inputs.clone(), Some(&selected))?;
            let lookups: HashMap<_, _> = result.outputs.into_iter().collect();
            for gather in &m.public.gathers {
                let tensor = lookups.get(&gather.name).context("missing public lookup")?;
                ensure!(
                    tensor.shape == gather.shape && tensor.len() == gather.length,
                    "public lookup shape mismatch"
                );
                put(gather.offset, &quantized_tensor(tensor, scale)?)?;
            }
        }
        ensure!(
            claims.len() == m.public.claims.len(),
            "claim tensor coverage mismatch"
        );
        for range in &m.public.claims {
            let claim = claims.get(&range.name).context("missing claimed output")?;
            ensure!(claim.len() == range.length, "claim shape mismatch");
            put(range.offset, claim)?;
        }
        ensure!(set.iter().all(|v| *v), "unbound public values");
        Ok(values)
    }
}

fn quantized_tensor(tensor: &Tensor, scale: &AFloat) -> Result<Vec<i128>> {
    (0..tensor.len())
        .map(|i| quantize(&tensor.get(i), scale))
        .collect()
}

#[derive(Debug, Serialize)]
pub struct ForwardTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}
pub struct PreparedForward {
    pub primary: Vec<i128>,
    pub secondary_inputs: HashMap<String, Tensor>,
    pub claims: HashMap<String, Vec<i128>>,
    pub outputs: BTreeMap<String, ForwardTensor>,
    pub nodes_executed: usize,
    pub peak_live_elements: usize,
}

pub fn execute_forward(
    artifact: &Artifact,
    inputs: &HashMap<String, Tensor>,
    bits: usize,
) -> Result<PreparedForward> {
    let meta = &artifact.meta;
    let scale = denominator(bits);
    let mut primary = vec![0; meta.num_witness_values];
    let mut seen = HashSet::new();
    let ranges: HashMap<_, _> = meta
        .primary_outputs
        .iter()
        .map(|r| (r.name.as_str(), r))
        .collect();
    let retain: HashSet<_> = meta
        .secondary_inputs
        .iter()
        .map(|r| r.name.as_str())
        .collect();
    let claim_ranges: HashMap<_, _> = meta
        .public
        .claims
        .iter()
        .map(|r| (r.name.as_str(), r))
        .collect();
    let mut secondary_inputs = HashMap::new();
    for name in &retain {
        if let Some(value) = inputs.get(*name) {
            secondary_inputs.insert(name.to_string(), value.clone());
        }
    }
    let mut claims = HashMap::new();
    let mut outputs = BTreeMap::new();
    let runner = OnnxV2::load(artifact.directory.join("primary_model.onnx"))?;
    let result = runner.run_observed(inputs.clone(), None, false, |name, tensor| {
        if let Some(range) = ranges.get(name) {
            ensure!(
                tensor.shape == range.shape,
                "primary witness shape mismatch for {name}"
            );
            ensure!(
                seen.insert(name.to_owned()),
                "duplicate primary output {name}"
            );
            for i in 0..range.length {
                primary[range.offset + i] = quantize(&tensor.get(i), &scale)?;
            }
        }
        if retain.contains(name) {
            secondary_inputs.insert(name.to_owned(), tensor.clone());
        }
        if claim_ranges.contains_key(name) {
            claims.insert(name.to_owned(), quantized_tensor(tensor, &scale)?);
        }
        if claim_ranges.contains_key(name)
            || name == "embedding"
            || name == "final_norm"
            || (name.starts_with("layer_") && name.len() == 8)
        {
            outputs.insert(
                name.to_owned(),
                ForwardTensor {
                    shape: tensor.shape.clone(),
                    values: (0..tensor.len()).map(|i| tensor.get_f64(i)).collect(),
                },
            );
        }
        Ok(())
    })?;
    ensure!(
        seen.len() == ranges.len(),
        "primary outputs missing from ONNX execution"
    );
    ensure!(
        claims.len() == claim_ranges.len(),
        "claims missing from ONNX execution"
    );
    ensure!(
        secondary_inputs.len() == retain.len(),
        "secondary inputs missing from primary execution"
    );
    Ok(PreparedForward {
        primary,
        secondary_inputs,
        claims,
        outputs,
        nodes_executed: result.nodes_executed,
        peak_live_elements: result.peak_live_elements,
    })
}

pub struct OnnxV2Executor {
    pub artifact: Arc<Artifact>,
    pub scale: AFloat,
    pub public: Vec<i128>,
    primary: Option<Vec<i128>>,
    secondary_inputs: Option<HashMap<String, Tensor>>,
}
impl OnnxV2Executor {
    pub fn prover(
        artifact: Arc<Artifact>,
        inputs: &HashMap<String, Tensor>,
        prepared: PreparedForward,
        bits: usize,
    ) -> Result<Self> {
        let scale = denominator(bits);
        let public = artifact.public_values(inputs, &prepared.claims, &scale)?;
        Ok(Self {
            artifact,
            scale,
            public,
            primary: Some(prepared.primary),
            secondary_inputs: Some(prepared.secondary_inputs),
        })
    }
    pub fn verifier(
        artifact: Arc<Artifact>,
        inputs: &HashMap<String, Tensor>,
        claims: &HashMap<String, Vec<i128>>,
        bits: usize,
    ) -> Result<Self> {
        let scale = denominator(bits);
        let public = artifact.public_values(inputs, claims, &scale)?;
        Ok(Self {
            artifact,
            scale,
            public,
            primary: None,
            secondary_inputs: None,
        })
    }
    pub fn write_primary(&self, writer: &mut impl Write) -> Result<()> {
        for value in self.primary.as_ref().context("verifier has no witness")? {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    fn matrices(&self, randomness: Option<&[i128]>) -> Result<R1CSMatrices<i128>> {
        let meta = &self.artifact.meta;
        let mut projected = HashMap::<String, Tensor>::new();
        if let Some(random) = randomness {
            ensure!(
                random.len() == meta.num_random_values,
                "challenge count mismatch"
            );
            if !meta.projection_outputs.is_empty() {
                let r = Tensor::from_mpfr(
                    vec![random.len()],
                    random
                        .iter()
                        .map(|x| AFloat::from_i128(*x).unwrap() / self.scale.clone())
                        .collect(),
                )?;
                let runner = OnnxV2::load(self.artifact.directory.join("projection_model.onnx"))?;
                let result = runner.run(HashMap::from([("R".into(), r)]), None)?;
                projected.extend(result.outputs);
            }
        }
        let width = meta.protocol_metadata().get_witness_size();
        let mut matrices = Vec::new();
        let mut coefficient_cache = HashMap::new();
        for name in ["A", "B", "C"] {
            let count = *meta.matrix_entries.get(name).context("missing COO count")?;
            let path = self.artifact.directory.join(format!("{name}.bin"));
            ensure!(
                path.metadata()?.len() == count as u64 * 16,
                "COO byte count mismatch"
            );
            let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
            let mut entries = if count == 0 {
                BigVec::from_vec(Vec::new())
            } else {
                BigVec::new(count).map_err(anyhow::Error::msg)?
            };
            let mut record = [0u8; 16];
            for index in 0..count {
                reader.read_exact(&mut record)?;
                let row = u32::from_le_bytes(record[..4].try_into().unwrap()) as usize;
                let col = u32::from_le_bytes(record[4..8].try_into().unwrap()) as usize;
                let value = f64::from_le_bytes(record[8..].try_into().unwrap());
                ensure!(
                    row < meta.num_constraints && col < width,
                    "COO index outside circuit"
                );
                ensure!(value.is_finite(), "nonfinite COO coefficient");
                let key = value.to_bits();
                let coefficient = if let Some(cached) = coefficient_cache.get(&key) {
                    *cached
                } else {
                    let coefficient = quantize(&AFloat::from_f64(value).unwrap(), &self.scale)?;
                    if coefficient_cache.len() < 4096 {
                        coefficient_cache.insert(key, coefficient);
                    }
                    coefficient
                };
                entries[index] = (row, col, coefficient);
            }
            let mut slots: Vec<_> = meta
                .challenge_slots
                .iter()
                .filter(|s| s.matrix == name)
                .collect();
            slots.sort_by_key(|s| s.entry_offset);
            let mut previous_end = 0;
            for slot in slots {
                ensure!(
                    slot.multiplier.is_finite(),
                    "nonfinite challenge multiplier"
                );
                ensure!(
                    slot.entry_offset
                        .checked_add(slot.length)
                        .is_some_and(|end| end <= count),
                    "challenge slot outside COO"
                );
                ensure!(
                    slot.entry_offset >= previous_end,
                    "overlapping challenge coefficient slots"
                );
                previous_end = slot.entry_offset + slot.length;
                for j in 0..slot.length {
                    let index = slot.entry_offset + j;
                    ensure!(
                        entries[index].2 == 0,
                        "challenge slot must contain a zero placeholder"
                    );
                    if let Some(random) = randomness {
                        let value = match slot.source.as_str() {
                            "random" => {
                                AFloat::from_i128(
                                    *random
                                        .get(slot.tensor_offset + j)
                                        .context("random slot outside challenge")?,
                                )
                                .unwrap()
                                    / self.scale.clone()
                            }
                            "projection" => {
                                let tensor = projected
                                    .get(
                                        slot.tensor
                                            .as_deref()
                                            .context("missing projection name")?,
                                    )
                                    .context("missing projection output")?;
                                ensure!(
                                    slot.tensor_offset + j < tensor.len(),
                                    "projection coefficient outside tensor"
                                );
                                tensor.get(slot.tensor_offset + j)
                            }
                            other => bail!("unknown challenge slot source: {other}"),
                        };
                        entries[index].2 = quantize(
                            &(value * AFloat::from_f64(slot.multiplier).unwrap()),
                            &self.scale,
                        )?;
                    }
                }
            }
            matrices.push(Matrix::new(
                MatrixData::COO(entries),
                width,
                meta.num_constraints,
                None,
                format!("Authenticated ONNX v2 {name}"),
            ));
        }
        let c = matrices.pop().unwrap();
        let b = matrices.pop().unwrap();
        let a = matrices.pop().unwrap();
        Ok(R1CSMatrices { a, b, c })
    }
}

impl R1CSInstance<AFloat, i128> for OnnxV2Executor {
    fn requires_challenge_matrices(&self) -> bool {
        true
    }
    fn expected_public_values(&self, scale: AFloat, batch: usize) -> Option<Vec<i128>> {
        assert_eq!(batch, 1);
        assert_eq!(scale, self.scale);
        Some(self.public.clone())
    }
    fn get_meta(&self) -> Metadata {
        self.artifact.meta.protocol_metadata()
    }
    fn get_matrices(
        &self,
        scale: AFloat,
        randomness: Option<&Vec<i128>>,
    ) -> (R1CSMatrices<i128>, Option<InjectionInfo>) {
        assert_eq!(scale, self.scale);
        (
            self.matrices(randomness.map(Vec::as_slice))
                .expect("ONNX challenge matrix construction failed"),
            None,
        )
    }
    fn compute_commit_witness(&mut self, scale: AFloat, batch: usize) -> Matrix<i128> {
        assert_eq!(batch, 1);
        assert_eq!(scale, self.scale);
        Matrix::from_vec(
            self.primary
                .as_ref()
                .expect("verifier cannot generate witness")
                .clone(),
        )
    }
    fn compute_full_witness(
        &mut self,
        metadata: &Metadata,
        random: Vec<AFloat>,
        scale: AFloat,
    ) -> Matrix<i128> {
        assert_eq!(scale, self.scale);
        assert_eq!(random.len(), metadata.num_random_values);
        let result = (|| -> Result<Matrix<i128>> {
            let mut inputs = self
                .secondary_inputs
                .take()
                .context("secondary witness already consumed or not available")?;
            inputs.insert(
                "R".into(),
                Tensor::from_mpfr(vec![random.len()], random.clone())?,
            );
            let mut secondary = vec![0i128; metadata.num_secondary_witness_values];
            let ranges: HashMap<_, _> = self
                .artifact
                .meta
                .secondary_outputs
                .iter()
                .map(|r| (r.name.as_str(), r))
                .collect();
            let mut seen = HashSet::new();
            if !ranges.is_empty() {
                let runner = OnnxV2::load(self.artifact.directory.join("secondary_model.onnx"))?;
                runner.run_observed(inputs, None, false, |name, tensor| {
                    if let Some(range) = ranges.get(name) {
                        ensure!(
                            tensor.shape == range.shape,
                            "secondary shape mismatch for {name}"
                        );
                        ensure!(seen.insert(name.to_owned()), "duplicate secondary output");
                        for i in 0..range.length {
                            secondary[range.offset + i] = quantize(&tensor.get(i), &scale)?;
                        }
                    }
                    Ok(())
                })?;
            }
            ensure!(
                seen.len() == ranges.len(),
                "missing secondary witness tensor"
            );
            let mut full = Vec::with_capacity(metadata.get_witness_size());
            full.extend_from_slice(&self.public);
            for r in &random {
                full.push(quantize(r, &scale)?);
            }
            full.extend(self.primary.take().context("primary witness consumed")?);
            full.extend(secondary);
            ensure!(
                full.len() == metadata.get_witness_size(),
                "full witness size mismatch"
            );
            let mut matrix = Matrix::from_vec(full);
            matrix.set_ranges(&metadata.get_ranges());
            Ok(matrix)
        })();
        result.expect("ONNX secondary witness construction failed")
    }
}
