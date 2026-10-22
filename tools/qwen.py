#!/usr/bin/env python3
"""Local Qwen3.5 text inference, ONNX validation, and Spain proof orchestration.

Transformers is used only by the independent reference and comparison commands.
The implemented generator executes the same approximate ONNX graph as the proof.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

DEFAULT_MODEL_DIR = "/home/yliangh/models/Qwen/Qwen3.5-0.8B"
ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (1, 2, 8, 32, 63, 64, 65, 128)
# Freeze the acceptance suite before model iteration; do not select successful outputs.
PROMPTS = (
    ("请用一句话介绍你自己。", ()),
    ("中国的首都是哪里？请简短回答。", (("北京",),)),
    ("What is the capital of France? Answer briefly.", (("paris",),)),
    ("计算 17 + 25，只给出结果。", (("42",),)),
    ("Translate 'Good morning' into Chinese.", (("早上好", "早安"),)),
    ("Write two sentences about a cat sitting by a window.", (("cat",),)),
)
LIMITS = {"max_abs": .01, "max_rmse": .001, "max_relative_l2": .001,
          "native_max_abs": 1e-7, "greedy_total": .99, "greedy_per_prompt": .95}


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def tokenizer_for(model_dir: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)


def encode_prompt(tokenizer, prompt: str, raw: bool = False) -> list[int]:
    if raw:
        return tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )


def validate_ids(ids, model_dir: str) -> list[int]:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    if not isinstance(ids, list) or not ids:
        raise ValueError("input_ids must be a nonempty JSON array")
    if any(type(x) is not int or not 0 <= x < cfg["vocab_size"] for x in ids):
        raise ValueError("input_ids must contain integers in the model vocabulary")
    if len(ids) > cfg["max_position_embeddings"]:
        raise ValueError("input prefix exceeds model position limit")
    return ids


def parse_ids(args, tokenizer) -> list[int]:
    if args.input_ids:
        value = json.loads(Path(args.input_ids).read_text())
        ids = value.get("input_ids") if isinstance(value, dict) else value
    else:
        ids = encode_prompt(tokenizer, args.prompt, args.raw)
    return validate_ids(ids, args.model_dir)


def eos_ids(model_dir: str, tokenizer) -> set[int]:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    eos = cfg.get("text_config", cfg).get("eos_token_id")
    result = set(eos if isinstance(eos, list) else [eos])
    result.add(tokenizer.eos_token_id)
    result.discard(None)
    return result


def versions() -> dict:
    return {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "onnx", "onnxruntime", "safetensors", "numpy")}


class OnnxModelRunner:
    """One active CPU session, with length-specific graphs and shared weights."""
    def __init__(self, model_dir: str, artifact_dir: str | Path, threads: int = 4):
        self.model_dir = str(Path(model_dir).resolve())
        self.artifact_dir = Path(artifact_dir).resolve()
        self.threads = threads
        self._key = None
        self._session = None

    def graph(self, length: int, approximate: bool = True, debug: bool = False) -> Path:
        from tools.qwen_onnx import export_model
        suffix = ("approx" if approximate else "original") + ("-debug" if debug else "")
        directory = self.artifact_dir / f"seq-{length}-{suffix}"
        return Path(export_model(self.model_dir, directory, length,
                                 approximate=approximate, debug=debug,
                                 weights_dir=self.artifact_dir / "weights"))

    def close(self):
        self._session = None
        self._key = None
        gc.collect()

    def forward(self, ids: list[int], *, approximate: bool = True,
                capture_layers: bool = False) -> dict[str, np.ndarray]:
        import onnxruntime as ort
        ids = validate_ids(ids, self.model_dir)
        key = (len(ids), approximate, capture_layers)
        if self._key != key:
            self.close()
            path = self.graph(len(ids), approximate, capture_layers)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            self._session = ort.InferenceSession(str(path), sess_options=opts,
                                                 providers=["CPUExecutionProvider"])
            self._key = key
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(int64)":
            raise ValueError("Qwen graph must accept exactly one INT64 token tensor")
        outputs = self._session.run(None, {inputs[0].name: np.asarray([ids], dtype=np.int64)})
        result = dict(zip((o.name for o in self._session.get_outputs()), outputs))
        for name, value in result.items():
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite ONNX output: {name}")
        return result


def load_reference(model_dir: str, threads: int):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.set_num_threads(threads)
    cfg = Qwen3_5TextConfig.from_pretrained(model_dir, local_files_only=True)
    return Qwen3_5ForCausalLM.from_pretrained(
        model_dir, config=cfg, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
    ).eval()


def reference_forward(model, ids: list[int], capture_layers: bool = False) -> dict[str, np.ndarray]:
    import torch
    result, hooks = {}, []
    def capture(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            result[name] = tensor.detach().cpu().numpy().copy()
        return hook
    if capture_layers:
        hooks.append(model.model.embed_tokens.register_forward_hook(capture("embedding")))
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(capture(f"layer_{i:02d}")))
        hooks.append(model.model.norm.register_forward_hook(capture("final_norm")))
    try:
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False,
                           logits_to_keep=0 if capture_layers else 1)
            values = output.logits.detach().cpu().numpy()
            result["last_logits"] = values[:, -1:, :].copy()
            if capture_layers:
                result["logits"] = values.copy()
    finally:
        for hook in hooks:
            hook.remove()
    return result


def tensor_error(actual, expected, logits: bool = False) -> dict:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError("nonfinite tensor in comparison")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    diff = a - b
    metrics = {"max_abs": float(np.abs(diff).max()), "rmse": float(np.sqrt(np.mean(diff * diff))),
               "relative_l2": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))}
    if logits:
        rows = diff.reshape(-1, diff.shape[-1])
        ar, br = a.reshape(rows.shape), b.reshape(rows.shape)
        row_max = np.abs(rows).max(axis=-1)
        order = np.argsort(br, axis=-1)[:, -2:]
        top = np.argmax(ar, axis=-1)
        margin = np.take_along_axis(br, order[:, -1:], axis=-1)[:, 0] - np.take_along_axis(br, order[:, :1], axis=-1)[:, 0]
        metrics.update(max_row_rmse=float(np.sqrt(np.mean(rows * rows, axis=-1)).max()),
                       top1_agreement=float(np.mean(top == order[:, -1])),
                       margin_violations=int(np.sum((margin > 2 * row_max) & (top != order[:, -1]))))
    return metrics


def compare_outputs(actual: dict, expected: dict, *, native: bool = False) -> dict:
    if set(actual) != set(expected):
        raise ValueError(f"tensor coverage differs: missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}")
    metrics, failures = {}, []
    for name, expected_value in expected.items():
        is_logits = "logits" in name
        value = metrics[name] = tensor_error(actual[name], expected_value, is_logits)
        if value["max_abs"] > (LIMITS["native_max_abs"] if native else LIMITS["max_abs"]):
            failures.append(f"{name}: maximum absolute error exceeded")
        if not native:
            if is_logits and value["max_row_rmse"] > LIMITS["max_rmse"]:
                failures.append(f"{name}: row RMSE exceeded")
            if not is_logits and value["relative_l2"] > LIMITS["max_relative_l2"]:
                failures.append(f"{name}: relative L2 error exceeded")
        if is_logits and value["margin_violations"]:
            failures.append(f"{name}: top-1 margin bound violated")
    return {"tensors": metrics, "passed": not failures, "failures": failures}


def generate(model_dir: str, tokenizer, ids: list[int], max_new_tokens: int, forward,
             reference=None) -> dict:
    prefix, generated, steps = list(ids), [], []
    stops = eos_ids(model_dir, tokenizer)
    started = time.monotonic()
    stop_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        logits = forward(prefix)["last_logits"]
        next_id = int(np.argmax(logits.reshape(-1)))
        record = {"step": step, "prefix_length": len(prefix), "token_id": next_id}
        if reference is not None:
            ref = reference_forward(reference, prefix)["last_logits"]
            record["reference_token_id"] = int(np.argmax(ref.reshape(-1)))
            record["metrics"] = tensor_error(logits, ref, logits=True)
        if next_id >= len(tokenizer):
            raise ValueError(f"model selected padded, undecodable vocabulary ID {next_id}")
        generated.append(next_id)
        prefix.append(next_id)
        steps.append(record)
        if next_id in stops:
            stop_reason = "eos"
            break
    completion = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"input_ids": ids, "generated_ids": generated, "token_ids": prefix,
            "completion": completion, "steps": steps, "stop_reason": stop_reason,
            "seconds": time.monotonic() - started}


def validate(args) -> dict:
    tokenizer = tokenizer_for(args.model_dir)
    ref = load_reference(args.model_dir, args.threads)
    runner = OnnxModelRunner(args.model_dir, args.artifact_dir, args.threads)
    report = {"schema_version": 1, "status": "running", "passed": False,
              "model_dir": str(Path(args.model_dir).resolve()), "versions": versions(),
              "limits": LIMITS, "lengths": [], "generation": [], "failures": []}
    output = args.output or str(Path(args.artifact_dir) / "validation.json")
    atomic_json(output, report)
    seed = tokenizer.encode("The quiet library has many books. 北京是中国的首都。", add_special_tokens=False)
    for length in args.lengths:
        ids = (seed * math.ceil(length / len(seed)))[:length]
        expected = reference_forward(ref, ids, True)
        original = runner.forward(ids, approximate=False, capture_layers=True)
        approximate = runner.forward(ids, approximate=True, capture_layers=True)
        checks = {"original_vs_hf": compare_outputs(original, expected),
                  "approximate_vs_hf": compare_outputs(approximate, expected),
                  "approximate_vs_original": compare_outputs(approximate, original)}
        row = {"length": length, "input_ids": ids, "checks": checks,
               "passed": all(c["passed"] for c in checks.values())}
        report["lengths"].append(row)
        if not row["passed"]:
            report["failures"].append(f"forward length {length}")
        atomic_json(output, report)
        print(f"Forward length {length}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
        del original, approximate, expected
    total_matches = total_steps = 0
    for prompt, groups in PROMPTS:
        ids = encode_prompt(tokenizer, prompt)
        item = generate(args.model_dir, tokenizer, ids, args.max_new_tokens, runner.forward, ref)
        item["prompt"] = prompt
        matches = sum(s["token_id"] == s["reference_token_id"] for s in item["steps"])
        count = len(item["steps"])
        total_matches += matches
        total_steps += count
        item["greedy_agreement"] = matches / max(count, 1)
        content = item["completion"].strip().lower()
        item["content_checks"] = {"nonempty_readable_text": len(content) >= 2 and any(c.isalpha() or c.isdigit() for c in content),
                                  "expected_answer": all(any(x in content for x in group) for group in groups)}
        metrics_ok = all(s["metrics"]["max_abs"] <= LIMITS["max_abs"] and
                         s["metrics"]["max_row_rmse"] <= LIMITS["max_rmse"] and
                         s["metrics"]["margin_violations"] == 0 for s in item["steps"])
        item["passed"] = (count > 0 and metrics_ok and all(item["content_checks"].values()) and
                          item["greedy_agreement"] >= LIMITS["greedy_per_prompt"])
        if not item["passed"]:
            report["failures"].append(f"generation: {prompt}")
        report["generation"].append(item)
        atomic_json(output, report)
        print(f"Generation {'PASS' if item['passed'] else 'FAIL'}: {prompt}\n{item['completion']}", flush=True)
    report["greedy_total_agreement"] = total_matches / max(total_steps, 1)
    if not total_steps or report["greedy_total_agreement"] < LIMITS["greedy_total"]:
        report["failures"].append("total greedy agreement")
    report["status"] = "complete"
    report["passed"] = not report["failures"]
    atomic_json(output, report)
    runner.close()
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("export", "generate", "reference", "validate", "prove", "forward"))
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--artifact-dir", default=str(ROOT / "artifacts/qwen"))
    p.add_argument("--compiled-dir", help="Reuse an authenticated v2 circuit for native forward/prove")
    p.add_argument("--output")
    p.add_argument("--output-dir")
    p.add_argument("--input-ids")
    p.add_argument("--prompt", default="中国的首都是哪里？请简短回答。")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=2)
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    p.add_argument("--capture-layers", action="store_true")
    p.add_argument("--original", action="store_true")
    p.add_argument("--binary", default=str(ROOT / "target/release/onnx-spain"))
    p.add_argument("--memory-budget-gib", type=int, default=200)
    p.add_argument("--disk-budget-gib", type=int, default=30)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.threads < 1 or args.max_new_tokens < 1 or args.seq_len < 1 or any(x < 1 for x in args.lengths):
        raise ValueError("threads, lengths, and generation limit must be positive")
    if args.command == "prove" and args.capture_layers:
        raise ValueError("Layer capture is diagnostic only; proof claims exactly the final position's logits")
    if args.command == "validate":
        report = validate(args)
        return 0 if report["passed"] else 1
    runner = OnnxModelRunner(args.model_dir, args.artifact_dir, args.threads)
    if args.command == "export":
        path = runner.graph(args.seq_len, not args.original, args.capture_layers)
        print(path)
        return 0
    tokenizer = tokenizer_for(args.model_dir)
    ids = parse_ids(args, tokenizer)
    if args.command in ("prove", "forward"):
        from circuit.v2 import compile_model
        if not args.output_dir:
            raise ValueError("--output-dir is required for native forward/proof artifacts")
        if args.compiled_dir:
            compiled = Path(args.compiled_dir).resolve()
            if not (compiled / "meta.json").is_file():
                raise ValueError("--compiled-dir does not contain a compiled v2 artifact")
        else:
            graph = runner.graph(len(ids), True, args.capture_layers)
            suffix = "-debug" if args.capture_layers else ""
            compiled = Path(args.artifact_dir).resolve() / f"compiled-seq-{len(ids)}{suffix}"
            if (compiled / "meta.json").exists():
                previous = json.loads((compiled / "meta.json").read_text())
                graph_hash = hashlib.sha256(graph.read_bytes()).hexdigest()
                compiler_hash = hashlib.sha256((ROOT / "circuit/onnx_v2.py").read_bytes()).hexdigest()
                if (graph_hash != previous.get("source_graph_sha256") or
                        compiler_hash != previous.get("compiler_source_sha256")):
                    raise ValueError("Existing circuit uses a different graph/compiler; select a fresh --artifact-dir")
            else:
                compile_model(graph, compiled)
        meta = json.loads((compiled / "meta.json").read_text())
        if meta["public"]["inputs"][0]["length"] != len(ids):
            raise ValueError("Token count does not match the compiled circuit")
        import onnx
        compiled_graph = onnx.load(compiled / "original_model.onnx", load_external_data=False)
        properties = {p.key: p.value for p in compiled_graph.metadata_props}
        if properties.get("spain.model") != "qwen3_5_text" or properties.get("spain.layers") != "24":
            raise ValueError("Qwen command requires the complete 24-layer Qwen3.5 text graph")
        if args.command == "prove" and (len(meta["public"]["claims"]) != 1 or
                meta["public"]["claims"][0]["name"] != "last_logits" or
                meta["public"]["claims"][0]["shape"] != [1, 1, 248320]):
            raise ValueError("Qwen proof must claim exactly the last position's complete vocabulary logits")
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if any((output_dir / name).exists() for name in
               ("statement.json", "forward.json", "primary_witness.i128le", "transcript.jsonl", "proof_report.json")):
            raise ValueError("Proof/forward output directory contains an earlier run; use a fresh directory")
        input_path = output_dir / "input_ids.json"
        atomic_json(input_path, ids)
        atomic_json(output_dir / "python_environment.json", {"versions": versions(),
                    "python": sys.version, "model_dir": str(Path(args.model_dir).resolve()),
                    "command": sys.argv})
        cmd = [args.binary, args.command, "--artifact-dir", str(compiled), "--input-ids", str(input_path),
               "--output-dir", str(output_dir), "--threads", str(args.threads),
               "--memory-budget-gib", str(args.memory_budget_gib), "--disk-budget-gib", str(args.disk_budget_gib)]
        return subprocess.run(cmd, cwd=ROOT, check=False).returncode
    if args.command == "reference":
        ref = load_reference(args.model_dir, args.threads)
        result = generate(args.model_dir, tokenizer, ids, args.max_new_tokens,
                          lambda prefix: reference_forward(ref, prefix))
        result["backend"] = "huggingface-fp32-eager"
    else:
        result = generate(args.model_dir, tokenizer, ids, args.max_new_tokens, runner.forward)
        result["backend"] = "onnx-approximate-fp64-cpu"
    result["model_dir"] = str(Path(args.model_dir).resolve())
    if args.output:
        atomic_json(args.output, result)
    print(result["completion"])
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
