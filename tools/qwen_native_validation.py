#!/usr/bin/env python3
"""Compare every declared Qwen ONNX output against actual MPFR-128 execution.

Native tensors are streamed to temporary f64le files.  The report retains all
per-tensor metrics and execution metadata; successful raw tensor dumps are
removed after comparison.  Existing successful, identically hashed runs resume.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qwen import LIMITS, atomic_json, tensor_error
from tools.qwen_onnx import sha256_file


def verify_shared_weights(root: Path):
    """Hash actual shared bytes before accepting new or resumed measurements."""
    manifest = json.loads((root / "weights" / "weights_manifest.json").read_text())
    filename = manifest["external_file"]
    if Path(filename).name != filename:
        raise ValueError("shared weights manifest must use a local filename")
    path = root / "weights" / filename
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError("shared weights changed during hash verification")
    if after.st_size != manifest["external_bytes"] or digest != manifest["external_sha256"]:
        raise ValueError("shared weights bytes differ from the exported manifest")
    return {"path": str(path), "sha256": digest, "bytes": after.st_size}, identity(after)


def verify_graph_weights(graph: Path, weights: dict):
    model = onnx.load_model(graph, load_external_data=False)
    metadata = {p.key: p.value for p in model.metadata_props}
    if metadata.get("spain.weights_sha256") != weights["sha256"]:
        raise ValueError(f"graph does not bind verified shared weights: {graph}")
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            data = {p.key: p.value for p in tensor.external_data}
            if not (graph.parent / data["location"]).samefile(weights["path"]):
                raise ValueError(f"graph external initializer does not reference verified shared bytes: {tensor.name}")
    return {value.name for value in model.graph.output}


def run(args):
    root = Path(args.artifact_dir).resolve()
    directory = root / "native-validation"
    directory.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else directory / "validation.json"
    acceptance = json.loads(Path(args.validation).read_text())
    inputs = {row["length"]: row["input_ids"] for row in acceptance["lengths"]}
    missing = set(args.lengths) - set(inputs)
    if missing:
        raise ValueError(f"HF/ORT validation has no recorded inputs for lengths {sorted(missing)}")
    weights, weights_identity = verify_shared_weights(root)
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise ValueError(f"missing native executable: {binary}")
    binary_hash = sha256_file(binary)
    snapshot = directory / f"onnx-forward-{binary_hash[:16]}"
    if not snapshot.exists():
        shutil.copyfile(binary, snapshot)
        snapshot.chmod(0o755)
    if sha256_file(snapshot) != binary_hash:
        raise ValueError("native executable snapshot hash differs")
    binary = snapshot
    old = json.loads(output.read_text()) if output.exists() else {}
    completed = {row["length"]: row for row in old.get("lengths", []) if row.get("passed")}
    report = {"schema_version": 1, "status": "running", "passed": False,
              "backend": "MPFR-128 ONNX", "reference": "ORT FP64 approximate ONNX",
              "max_abs_limit": LIMITS["native_max_abs"], "threads": args.threads,
              "binary": str(binary), "binary_sha256": binary_hash,
              "weights": weights, "validator_source_sha256": sha256_file(__file__),
              "lengths": [], "failures": []}
    atomic_json(output, report)
    environment = dict(os.environ, RAYON_NUM_THREADS=str(args.threads),
                       OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    for length in args.lengths:
        graph = root / f"seq-{length}-approx-debug" / "approximate_model.onnx"
        if not graph.is_file():
            raise ValueError(f"missing previously validated debug ONNX: {graph}")
        graph_hash = sha256_file(graph)
        output_names = verify_graph_weights(graph, weights)
        ids = inputs[length]
        previous = completed.get(length)
        if (previous and previous.get("graph_sha256") == graph_hash
                and previous.get("input_ids") == ids and previous.get("binary_sha256") == binary_hash
                and previous.get("arithmetic") == "MPFR-128"
                and set(previous.get("tensors", {})) == output_names
                and all(np.isfinite(m.get("max_abs", np.inf))
                        and 0 <= m["max_abs"] <= LIMITS["native_max_abs"]
                        and ("logits" not in name or m.get("margin_violations") == 0)
                        for name, m in previous["tensors"].items())):
            report["lengths"].append(previous)
            atomic_json(output, report)
            print(f"Native length {length}: reuse matching successful metrics", flush=True)
            continue
        raw = directory / f"seq-{length}-raw"
        execution = directory / f"seq-{length}-execution.json"
        identity = {"graph_sha256": graph_hash, "input_ids": ids, "binary_sha256": binary_hash}
        input_file = directory / f"seq-{length}-inputs.json"
        atomic_json(input_file, {"input_ids": {"shape": [1, length], "dtype": "int64", "values": ids}})
        print(f"Native length {length}: executing all debug outputs", flush=True)
        started = time.monotonic()
        raw_reused = (raw / "manifest.json").exists()
        if raw_reused:
            if not execution.exists() or json.loads(execution.read_text()) != identity:
                raise ValueError(f"existing raw dump has different or missing execution identity: {raw}")
        if not (raw / "manifest.json").exists():
            if raw.exists() and any(raw.iterdir()):
                raise ValueError(f"incomplete native dump exists; inspect/remove before rerun: {raw}")
            atomic_json(execution, identity)
            with (directory / f"seq-{length}.log").open("w") as log:
                subprocess.run([str(binary), "--model", str(graph), "--inputs", str(input_file),
                                "--output-dir", str(raw)], env=environment, cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        native = json.loads((raw / "manifest.json").read_text())
        if native["arithmetic"] != "MPFR-128" or native["storage"] != "f64le":
            raise ValueError("native report does not use required MPFR-128 arithmetic")
        comparison_started = time.monotonic()
        execution_threads = None if raw_reused else args.threads
        precompute = directory / f"seq-{length}-precompute.json"
        if raw_reused and precompute.is_file():
            metadata = json.loads(precompute.read_text())
            if metadata.get("status") == "complete" and metadata.get("identity") == identity:
                execution_threads = metadata.get("threads")
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        names = [o.name for o in session.get_outputs()]
        if set(names) != set(native["outputs"]):
            raise ValueError("native tensor coverage differs from declared ONNX outputs")
        values = session.run(None, {"input_ids": np.array([ids], dtype=np.int64)})
        del session
        row = {"length": length, "input_ids": ids, "graph_sha256": graph_hash,
               "binary_sha256": binary_hash,
               "arithmetic": native["arithmetic"], "forward_seconds": native["forward_seconds"],
               "nodes_executed": native["nodes_executed"],
               "peak_live_activation_elements": native["peak_live_activation_elements"],
               "tensors": {}, "failures": [], "raw_bytes": 0,
               "raw_reused": raw_reused, "execution_threads": execution_threads}
        for name, reference in zip(names, values):
            item = native["outputs"][name]
            filename = item["file"]
            if Path(filename).name != filename or not filename.endswith(".f64le"):
                raise ValueError(f"unsafe native output filename: {filename}")
            path = raw / item["file"]
            if path.is_symlink():
                raise ValueError(f"native output must be an ordinary generated file: {path}")
            shape = tuple(item["shape"])
            expected_bytes = int(np.prod(shape)) * 8
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"truncated binary output: {name}")
            actual = np.memmap(path, dtype="<f8", mode="r", shape=shape)
            row["raw_bytes"] += expected_bytes
            metrics = tensor_error(actual, reference, logits="logits" in name)
            row["tensors"][name] = metrics
            if metrics["max_abs"] > LIMITS["native_max_abs"]:
                row["failures"].append(f"{name}: native maximum absolute error exceeded")
            if "logits" in name and metrics["margin_violations"]:
                row["failures"].append(f"{name}: top-1 margin bound violated")
            del actual
        del values
        gc.collect()
        row["passed"] = not row["failures"]
        row["wall_seconds"] = time.monotonic() - started
        row["comparison_wall_seconds"] = time.monotonic() - comparison_started
        report["lengths"].append(row)
        if not row["passed"]:
            report["failures"].append(f"native length {length}")
        atomic_json(output, report)
        print(f"Native length {length}: {'PASS' if row['passed'] else 'FAIL'}; "
              f"max_abs={max(m['max_abs'] for m in row['tensors'].values()):.3g}; "
              f"MPFR seconds={row['forward_seconds']:.3f}", flush=True)
        if row["passed"]:
            # Delete only files explicitly generated by this native invocation.
            for item in native["outputs"].values():
                (raw / item["file"]).unlink()
            (raw / "manifest.json").unlink()
            raw.rmdir()
    current = Path(weights["path"]).stat()
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != weights_identity:
        raise ValueError("shared weights changed during native validation")
    report["status"] = "complete"
    report["passed"] = not report["failures"]
    atomic_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", default=str(ROOT / "artifacts/qwen"))
    p.add_argument("--validation", default=str(ROOT / "artifacts/qwen/acceptance-validation.json"))
    p.add_argument("--output")
    p.add_argument("--binary", default=str(ROOT / "target/release/onnx-forward"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 8, 32, 63, 64, 65, 128])
    args = p.parse_args()
    if args.threads < 1 or any(n < 1 for n in args.lengths):
        p.error("threads and lengths must be positive")
    return 0 if run(args)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
