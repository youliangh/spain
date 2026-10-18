//! Read-only Hugging Face safetensors loader. BF16 values are decoded exactly.
use ::model::AFloat;
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use rayon::prelude::*;
use rug::{Assign, Float};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Clone, Debug)]
struct TensorLocation {
    file: usize,
    data_start: usize,
    info: TensorInfo,
}

#[derive(Clone, Debug)]
pub struct WeightStore {
    pub directory: PathBuf,
    files: Vec<Arc<Mmap>>,
    tensors: HashMap<String, TensorLocation>,
}

#[derive(Clone, Copy, Debug)]
pub struct WeightTensor<'a> {
    pub shape: &'a [usize],
    pub dtype: &'a str,
    bytes: &'a [u8],
}

impl<'a> WeightTensor<'a> {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get_f64(&self, index: usize) -> f64 {
        assert!(index < self.len(), "weight tensor index out of bounds");
        match self.dtype {
            "BF16" => {
                let offset = 2 * index;
                let bits = u16::from_le_bytes([self.bytes[offset], self.bytes[offset + 1]]);
                f32::from_bits((bits as u32) << 16) as f64
            }
            "F32" => {
                let offset = 4 * index;
                f32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap()) as f64
            }
            "F64" => {
                let offset = 8 * index;
                f64::from_le_bytes(self.bytes[offset..offset + 8].try_into().unwrap())
            }
            _ => unreachable!("dtype validated by WeightStore::load"),
        }
    }
    pub fn to_f64(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.get_f64(i)).collect()
    }
    pub fn get(&self, index: usize) -> f64 {
        self.get_f64(index)
    }
}

impl WeightStore {
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let single = directory.join("model.safetensors");
        let files = if single.is_file() {
            vec![single]
        } else {
            let index: serde_json::Value = serde_json::from_reader(File::open(
                directory.join("model.safetensors.index.json"),
            )?)?;
            let map = index
                .get("weight_map")
                .and_then(|v| v.as_object())
                .context("safetensors index missing weight_map")?;
            let mut names = map
                .values()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .context("invalid shard filename")
                })
                .collect::<Result<Vec<_>>>()?;
            names.sort();
            names.dedup();
            names.into_iter().map(|name| directory.join(name)).collect()
        };
        let mut store = Self {
            directory: directory.to_path_buf(),
            files: Vec::new(),
            tensors: HashMap::new(),
        };
        for path in files {
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // The model files are immutable inputs for the lifetime of the execution.
            let mmap = unsafe { Mmap::map(&file)? };
            ensure!(mmap.len() >= 8, "truncated safetensors header");
            let header_len = usize::try_from(u64::from_le_bytes(mmap[..8].try_into().unwrap()))?;
            let data_start = 8usize
                .checked_add(header_len)
                .context("header size overflow")?;
            ensure!(data_start <= mmap.len(), "truncated safetensors header");
            let header: HashMap<String, serde_json::Value> =
                serde_json::from_slice(&mmap[8..data_start])?;
            let file_index = store.files.len();
            for (name, value) in header {
                if name == "__metadata__" {
                    continue;
                }
                let info: TensorInfo = serde_json::from_value(value)?;
                let item_size = match info.dtype.as_str() {
                    "BF16" => 2,
                    "F32" => 4,
                    "F64" => 8,
                    other => bail!("unsupported weight dtype {other} for {name}"),
                };
                let expected = info
                    .shape
                    .iter()
                    .try_fold(item_size, |n: usize, &d| n.checked_mul(d))
                    .context("tensor shape overflow")?;
                let [start, end] = info.data_offsets;
                ensure!(
                    end >= start && end - start == expected,
                    "invalid tensor byte length for {name}"
                );
                ensure!(
                    data_start
                        .checked_add(end)
                        .is_some_and(|end| end <= mmap.len()),
                    "tensor outside safetensors data: {name}"
                );
                ensure!(
                    store
                        .tensors
                        .insert(
                            name.clone(),
                            TensorLocation {
                                file: file_index,
                                data_start,
                                info
                            }
                        )
                        .is_none(),
                    "duplicate weight {name}"
                );
            }
            store.files.push(Arc::new(mmap));
        }
        Ok(store)
    }

    pub fn tensor(&self, name: &str) -> Result<WeightTensor<'_>> {
        let location = self
            .tensors
            .get(name)
            .with_context(|| format!("missing weight {name}"))?;
        let [start, end] = location.info.data_offsets;
        Ok(WeightTensor {
            shape: &location.info.shape,
            dtype: &location.info.dtype,
            bytes: &self.files[location.file]
                [location.data_start + start..location.data_start + end],
        })
    }
    pub fn get_f64(&self, name: &str) -> Result<Vec<f64>> {
        Ok(self.tensor(name)?.to_f64())
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// For HF [output,input] weights, compute W^T r at 128-bit precision.
    /// Column tiles preserve sequential reads within a cache line and avoid
    /// allocating an MPFR temporary for each of the billion multiplications.
    pub fn project_transpose(&self, name: &str, r: &[AFloat]) -> Result<Vec<AFloat>> {
        let weight = self.tensor(name)?;
        ensure!(
            weight.shape.len() == 2 && weight.shape[0] == r.len(),
            "projection shape mismatch for {name}"
        );
        let columns = weight.shape[1];
        let tile_size = 32;
        let tiles: Vec<Vec<AFloat>> = (0..columns.div_ceil(tile_size))
            .into_par_iter()
            .map(|tile| {
                let first = tile * tile_size;
                let count = tile_size.min(columns - first);
                let mut sums: Vec<Float> = (0..count).map(|_| Float::with_val(128, 0)).collect();
                let mut product = Float::with_val(128, 0);
                for (row, random) in r.iter().enumerate() {
                    for (column, sum) in sums.iter_mut().enumerate() {
                        product.assign(&random.0 * weight.get_f64(row * columns + first + column));
                        *sum += &product;
                    }
                }
                sums.into_iter().map(AFloat).collect()
            })
            .collect();
        Ok(tiles.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bf16_decoding_is_exact() {
        let bytes = [0x80, 0x3f, 0x80, 0xbf, 0x01, 0x00];
        let tensor = WeightTensor {
            shape: &[3],
            dtype: "BF16",
            bytes: &bytes,
        };
        assert_eq!(tensor.get_f64(0), 1.0);
        assert_eq!(tensor.get_f64(1), -1.0);
        assert_eq!(tensor.get_f64(2), f32::from_bits(1 << 16) as f64);
    }
}
