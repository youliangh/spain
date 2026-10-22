//! Diagnostic entry point for the generic ONNX MPFR executor.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use model::onnx_v2::{OnnxV2, Tensor};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Options {
    #[arg(long)]
    model: PathBuf,
    /// JSON object: input name -> {shape: [...], dtype: "int64"|"float64", values: [...]}.
    #[arg(long)]
    inputs: PathBuf,
    #[arg(long, required_unless_present = "output_dir", conflicts_with = "output_dir")]
    output: Option<PathBuf>,
    /// Stream selected tensors to f64le files and a small manifest; directory must be empty.
    #[arg(long, required_unless_present = "output", conflicts_with = "output")]
    output_dir: Option<PathBuf>,
    /// Repeat to evaluate only the dependency closure of named tensors.
    #[arg(long)]
    selected_output: Vec<String>,
}

fn main() -> Result<()> {
    let options = Options::parse();
    let graph = OnnxV2::load(&options.model)?;
    let data: Value = serde_json::from_reader(File::open(&options.inputs)?)?;
    let mut inputs = HashMap::new();
    for (name, input) in data.as_object().context("inputs must be an object")? {
        let shape: Vec<usize> = serde_json::from_value(input["shape"].clone())?;
        let tensor = match input["dtype"].as_str().unwrap_or("float64") {
            "int64" => Tensor::from_i64(shape, serde_json::from_value(input["values"].clone())?)?,
            "float64" => Tensor::from_f64(shape, serde_json::from_value(input["values"].clone())?)?,
            other => anyhow::bail!("unsupported input dtype {other}"),
        };
        inputs.insert(name.clone(), tensor);
    }
    let selected =
        (!options.selected_output.is_empty()).then_some(options.selected_output.as_slice());
    let started = Instant::now();
    if let Some(directory) = &options.output_dir {
        fs::create_dir_all(directory)?;
        ensure!(fs::read_dir(directory)?.next().is_none(), "stream output directory must be empty");
        let names = selected.map(ToOwned::to_owned).unwrap_or_else(|| graph.output_names());
        let requested: HashSet<_> = names.iter().cloned().collect();
        let mut outputs = serde_json::Map::new();
        let mut observer = |name: &str, tensor: &Tensor| -> Result<()> {
            if !requested.contains(name) || outputs.contains_key(name) {
                return Ok(());
            }
            let filename = format!("{:04}.f64le", outputs.len());
            let path = directory.join(&filename);
            let temporary = path.with_extension("tmp");
            let mut writer = BufWriter::with_capacity(1 << 20, File::create(&temporary)?);
            for i in 0..tensor.len() {
                let value = tensor.get_f64(i);
                ensure!(value.is_finite(), "output exceeds binary64 report range: {name}");
                writer.write_all(&value.to_le_bytes())?;
            }
            writer.flush()?;
            drop(writer);
            fs::rename(temporary, path)?;
            outputs.insert(name.to_string(), json!({
                "shape": tensor.shape, "file": filename, "elements": tensor.len(),
            }));
            println!("Streamed {name}: {} binary64 values", tensor.len());
            Ok(())
        };
        for name in &names {
            if let Some(tensor) = inputs.get(name).or_else(|| graph.initializer(name)) {
                observer(name, tensor)?;
            }
        }
        let result = graph.run_observed(inputs, selected, false, &mut observer)?;
        let elapsed = started.elapsed().as_secs_f64();
        ensure!(outputs.len() == requested.len(), "not all requested outputs were streamed");
        let report = json!({
            "arithmetic": "MPFR-128", "storage": "f64le", "model": options.model,
            "forward_seconds": elapsed, "nodes_executed": result.nodes_executed,
            "peak_live_activation_elements": result.peak_live_elements, "outputs": outputs,
        });
        let temporary = directory.join("manifest.json.tmp");
        serde_json::to_writer(File::create(&temporary)?, &report)?;
        fs::rename(temporary, directory.join("manifest.json"))?;
        println!("MPFR ONNX forward completed: {} nodes, {:.3} seconds, {}",
                 result.nodes_executed, elapsed, directory.display());
        return Ok(());
    }
    let result = graph.run(inputs, selected)?;
    let elapsed = started.elapsed().as_secs_f64();
    let mut outputs = serde_json::Map::new();
    for (name, tensor) in result.outputs {
        let data = tensor.to_f64();
        ensure!(
            data.iter().all(|v| v.is_finite()),
            "output exceeds binary64 report range"
        );
        outputs.insert(name, json!({"shape": tensor.shape, "data": data}));
    }
    let report = json!({"arithmetic": "MPFR-128", "model": options.model,
        "forward_seconds": elapsed, "nodes_executed": result.nodes_executed,
        "peak_live_activation_elements": result.peak_live_elements, "outputs": outputs});
    let output = options.output.context("missing --output")?;
    serde_json::to_writer(File::create(&output)?, &report)?;
    println!(
        "MPFR ONNX forward completed: {} nodes, {:.3} seconds, {}",
        result.nodes_executed,
        elapsed,
        output.display()
    );
    Ok(())
}
