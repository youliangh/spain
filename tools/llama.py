#!/usr/bin/env python3
"""Run and validate the complete Llama 3.2 model used by the Spain extension."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from llama_model import DEFAULT_MODEL_DIR, PROFILE, LlamaConfig, LlamaModel


PROMPTS = (
    ("The capital of France is", (("paris",),)),
    ("Once upon a time, there was a little", (("girl", "boy", "rabbit", "mouse", "town", "village"),)),
    ("The opposite of hot is", (("cold",),)),
    ("The colors of the French flag are", (("blue",), ("white",), ("red",))),
    ("Water freezes at a temperature of", (("0", "zero"), ("celsius", "°c", "degrees c"))),
)
LENGTHS = (1, 2, 8, 32, 128)


def repetition_summary(text: str) -> dict:
    """An informational repetition count, independent of numerical acceptance."""
    words = text.lower().split()
    counts = Counter(" ".join(words[index:index + 4]) for index in range(max(0, len(words) - 3)))
    phrase, count = counts.most_common(1)[0] if counts else ("", 0)
    return {"ngram_words": 4, "most_repeated_phrase": phrase, "occurrences": count,
            "repetition_observed": count >= 3}


def atomic_json(path: str | Path, value: dict | list) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(destination)


def load_reference(model_dir: str):
    # The Transformers model is an independent validation oracle, never the
    # implementation behind `generate`.
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
    ).eval()


@torch.inference_mode()
def reference_forward(model, input_ids: list[int], capture_layers: bool = False):
    states, hooks = {}, []

    def capture(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            states[name] = tensor[0].detach().clone()
        return hook

    if capture_layers:
        hooks.append(model.model.embed_tokens.register_forward_hook(capture("embedding")))
        for index, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(capture(f"layer_{index}")))
        hooks.append(model.model.norm.register_forward_hook(capture("final_norm")))
    try:
        output = model(input_ids=torch.tensor([input_ids], dtype=torch.long), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return output.logits[0].detach(), states


def tensor_error(actual: torch.Tensor, expected: torch.Tensor, *, logits: bool = False) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"comparison shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError("comparison received non-finite values")
    actual, expected = actual.to(torch.float64), expected.to(torch.float64)
    difference = actual - expected
    result = {"max_abs": float(difference.abs().max()),
              "rmse": float(torch.sqrt(torch.mean(difference * difference))),
              "relative_l2": float(torch.linalg.vector_norm(difference) /
                                   torch.linalg.vector_norm(expected).clamp_min(1e-30))}
    if logits:
        if actual.ndim != 2:
            raise ValueError("logits must have [sequence, vocabulary] shape")
        row_error = difference.abs().amax(dim=-1)
        expected_logp = torch.log_softmax(expected, dim=-1)
        actual_logp = torch.log_softmax(actual, dim=-1)
        kl = torch.sum(expected_logp.exp() * (expected_logp - actual_logp), dim=-1)
        k = min(5, actual.shape[-1])
        expected_values, expected_ids = expected.topk(k, dim=-1)
        actual_ids = actual.topk(k, dim=-1).indices
        top1_matches = actual_ids[:, 0] == expected_ids[:, 0]
        overlap = (actual_ids[:, :, None] == expected_ids[:, None, :]).any(dim=-1).double().mean()
        margin = expected_values[:, 0] - expected_values[:, 1] if k > 1 else torch.full_like(row_error, math.inf)
        result.update(
            max_row_rmse=float(torch.sqrt(torch.mean(difference * difference, dim=-1)).max()),
            kl_max=max(0.0, float(kl.max())),
            top1_agreement=float(top1_matches.double().mean()), top5_overlap=float(overlap),
            margin_violations=int(((margin > 2 * row_error) & ~top1_matches).sum()),
            positions=actual.shape[0],
        )
    return result


def metric_failures(metrics: dict, limits: dict, *, logits: bool, name: str) -> list[str]:
    failures = []
    if metrics["max_abs"] > limits["max_abs"]:
        failures.append(f"{name}: max_abs {metrics['max_abs']:.6g} > {limits['max_abs']}")
    if logits:
        for metric, threshold in (("max_row_rmse", "max_rmse"), ("kl_max", "max_kl")):
            if metrics[metric] > limits[threshold]:
                failures.append(f"{name}: {metric} {metrics[metric]:.6g} > {limits[threshold]}")
        if metrics["margin_violations"]:
            failures.append(f"{name}: top-1 margin bound violated")
    elif metrics["relative_l2"] > limits["max_relative_l2"]:
        failures.append(f"{name}: relative_l2 {metrics['relative_l2']:.6g} > {limits['max_relative_l2']}")
    return failures


def compare_forward(actual, expected_logits, expected_layers, limits: dict) -> dict:
    if set(actual.layer_hidden_states) != set(expected_layers):
        raise ValueError("reference and implementation layer coverage differ")
    layers = {key: tensor_error(actual.layer_hidden_states[key], value)
              for key, value in expected_layers.items()}
    metrics = tensor_error(actual.logits, expected_logits, logits=True)
    failures = metric_failures(metrics, limits, logits=True, name="logits")
    for name, values in layers.items():
        failures.extend(metric_failures(values, limits, logits=False, name=name))
    return {"logits": metrics, "layers": layers, "arithmetic": actual.arithmetic,
            "passed": not failures, "failures": failures}


def parse_input(args, model: LlamaModel) -> list[int]:
    if args.input_ids:
        value = json.loads(Path(args.input_ids).read_text())
        ids = value.get("input_ids") if isinstance(value, dict) else value
        if not isinstance(ids, list):
            raise ValueError("--input-ids must contain a JSON array or {input_ids: [...]} object")
        return ids
    return model.encode(args.prompt)


def model_metadata(model: LlamaModel, model_dir: str) -> dict:
    return {"model_dir": str(Path(model_dir).resolve()), "profile": PROFILE,
            "config": asdict(model.config), "parameters": sum(value.numel() for name, value in model.weights.items()
                if name != "lm_head.weight" or not model.config.tie_word_embeddings),
            "dtype": "float64", "device": "cpu", "kv_cache": False,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "safetensors", "tokenizers")}}


def compare_native(path: str, model: LlamaModel, max_abs: float) -> dict:
    source = json.loads(Path(path).read_text())
    ids = source["input_ids"]
    actual = model.forward(ids, capture_layers=bool(source.get("layer_hidden_states")), last_logits_only=True)
    native = torch.tensor(source["logits"], dtype=torch.float64)
    if native.ndim == 1:
        native = native[None, :]
    if native.shape[0] != 1:
        actual = model.forward(ids, capture_layers=bool(source.get("layer_hidden_states")))
    metrics = tensor_error(actual.logits, native, logits=True)
    layers = {}
    for name, value in source.get("layer_hidden_states", {}).items():
        canonical = "final_norm" if name == "model.norm" else name
        if name.startswith("model.layers.") and name.endswith(".mlp_residual"):
            canonical = "layer_" + name.split(".")[2]
        if canonical not in actual.layer_hidden_states:
            raise ValueError(f"unknown native layer: {name}")
        layers[name] = tensor_error(actual.layer_hidden_states[canonical], torch.tensor(value, dtype=torch.float64))
    passed = metrics["max_abs"] <= max_abs and all(value["max_abs"] <= max_abs for value in layers.values())
    return {"path": str(Path(path).resolve()), "input_ids": ids, "logits": metrics,
            "layers": layers, "max_abs_limit": max_abs, "passed": passed}


def run_generate(args) -> dict:
    start = time.monotonic()
    model = LlamaModel.from_pretrained(args.model_dir)
    load_seconds = time.monotonic() - start
    ids = parse_input(args, model)
    start = time.monotonic()
    result = model.generate(ids, args.max_new_tokens)
    result.update(model=model_metadata(model, args.model_dir), model_load_seconds=load_seconds,
                  generation_seconds=time.monotonic() - start)
    if args.logits_json:
        result["native_comparison"] = compare_native(args.logits_json, model, args.native_max_abs)
        if not result["native_comparison"]["passed"]:
            result["passed"] = False
    if args.output:
        atomic_json(args.output, result)
    print(result["text"])
    return result


def run_reference(args) -> dict:
    from transformers import AutoTokenizer
    model = load_reference(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    if args.input_ids:
        source = json.loads(Path(args.input_ids).read_text())
        ids = source["input_ids"] if isinstance(source, dict) else source
    else:
        ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    logits, layers = reference_forward(model, ids, args.capture_layers)
    with torch.inference_mode():
        output = model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                                max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=False,
                                temperature=None, top_p=None, pad_token_id=model.config.eos_token_id)
    generated = output[0, len(ids):].tolist()
    result = {"backend": "huggingface-fp32-eager", "input_ids": ids,
              "generated_ids": generated, "token_ids": output[0].tolist(),
              "text": tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False),
              "completion": tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
              "logits": logits[-1].tolist(),
              "layer_hidden_states": {name: value.tolist() for name, value in layers.items()}}
    if args.output:
        atomic_json(args.output, result)
    print(result["text"])
    return result


def run_prove(args) -> dict:
    """Tokenize locally, then delegate the actual proof to the native Spain CLI."""
    from tokenizers import Tokenizer
    binary = Path(args.binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"native prover is unavailable: {binary}; build with cargo build --release -p llama-spain")
    directory = Path(args.model_dir).resolve()
    config = LlamaConfig.from_dict(json.loads((directory / "config.json").read_text()))
    if args.input_ids:
        source = json.loads(Path(args.input_ids).read_text())
        ids = source.get("input_ids") if isinstance(source, dict) else source
    else:
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        ids = tokenizer.encode(args.prompt, add_special_tokens=True).ids
    if not isinstance(ids, list) or not ids or len(ids) > config.max_position_embeddings:
        raise ValueError("proof input must be a non-empty token list within the model context limit")
    if any(type(token) is not int or not 0 <= token < config.vocab_size for token in ids):
        raise ValueError("proof input contains an invalid token ID")
    destination = Path(args.output_dir).resolve()
    previous_artifacts = [name for name in ("statement.json", "primary_witness.i128le", "transcript.jsonl", "proof_report.json")
                          if (destination / name).exists()]
    if previous_artifacts:
        raise ValueError(f"proof output already contains {', '.join(previous_artifacts)}; choose a fresh --output-dir")
    destination.mkdir(parents=True, exist_ok=True)
    input_path = destination / "input_ids.json"
    atomic_json(input_path, ids)
    command = [str(binary), "prove", "--model-dir", str(directory), "--input-ids", str(input_path),
               "--output-dir", str(destination), "--threads", str(args.threads),
               "--memory-budget-gib", str(args.memory_budget_gib), "--disk-budget-gib", str(args.disk_budget_gib),
               "--dark-chunks", str(getattr(args, "dark_chunks", 256))]
    process = subprocess.run(command, check=False)
    if process.returncode:
        return {"passed": False, "native_exit_code": process.returncode, "output_dir": str(destination)}
    proof_report_path = destination / "proof_report.json"
    if not proof_report_path.exists():
        raise ValueError("native prover returned without a verification report")
    report = json.loads(proof_report_path.read_text())
    if report.get("verified") is not True:
        raise ValueError("native proof was not verified")
    result = {"passed": True, "input_ids": ids, "output_dir": str(destination), "proof": report}
    print(json.dumps({"verified": True, "output_dir": str(destination)}, indent=2))
    return result


def run_validate(args) -> dict:
    limits = {"max_abs": args.max_abs_error, "max_rmse": args.max_rmse,
              "max_kl": args.max_kl, "max_relative_l2": args.max_relative_l2,
              "min_greedy_agreement": args.min_greedy_agreement}
    report = {"complete": False, "passed": False, "limits": limits,
              "reference": "huggingface-fp32-eager", "lengths": [], "prompts": [], "failures": []}
    start = time.monotonic()
    if args.output:
        atomic_json(args.output, report)
    try:
        model = LlamaModel.from_pretrained(args.model_dir)
        reference = load_reference(args.model_dir)
        report["model"] = model_metadata(model, args.model_dir)
        report["reference_version"] = importlib.metadata.version("transformers")
        expected_architecture = {"num_hidden_layers": 16, "hidden_size": 2048, "intermediate_size": 8192,
                                 "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64,
                                 "vocab_size": 128256, "tie_word_embeddings": True}
        if any(getattr(model.config, key) != value for key, value in expected_architecture.items()) or report["model"]["parameters"] != 1235814400:
            raise ValueError("full validation requires the complete 1,235,814,400-parameter Llama-3.2-1B architecture")
        source = model.encode(("A small library stood near the river. Every morning, the librarian opened the doors "
                               "and welcomed readers who came to learn about history, science, and distant places. ") * 16)
        for length in LENGTHS:
            ids = source[:length]
            if len(ids) != length:
                raise ValueError("validation text did not cover the requested sequence length")
            began = time.monotonic()
            actual = model.forward(ids, capture_layers=True)
            expected_logits, expected_layers = reference_forward(reference, ids, True)
            comparison = compare_forward(actual, expected_logits, expected_layers, limits)
            comparison.update(length=length, input_ids=ids, seconds=time.monotonic() - began)
            report["lengths"].append(comparison)
            report["failures"].extend(f"length {length}: {failure}" for failure in comparison["failures"])
            print(f"length={length} max_abs={comparison['logits']['max_abs']:.6g} passed={comparison['passed']}", flush=True)
            del actual, expected_logits, expected_layers
            if args.output:
                atomic_json(args.output, report)
        for prompt, required_terms in PROMPTS:
            began = time.monotonic()
            ids = model.encode(prompt)
            step_checks = []

            def check_step(prefix, result, chosen):
                expected, _ = reference_forward(reference, prefix)
                metrics = tensor_error(result.logits, expected[-1:], logits=True)
                failures = metric_failures(metrics, limits, logits=True, name=f"step {len(step_checks)}")
                step_checks.append({"step": len(step_checks), "reference_token_id": int(expected[-1].argmax()),
                                    "token_id": chosen, "metrics": metrics, "passed": not failures,
                                    "failures": failures})
                if len(step_checks) % 16 == 0:
                    print(f"prompt={prompt!r} tokens={len(step_checks)}/{args.max_new_tokens} "
                          f"last_max_abs={metrics['max_abs']:.6g}", flush=True)
                    report["in_progress_prompt"] = {"prompt": prompt, "completed_tokens": len(step_checks),
                                                     "text": model.decode(prefix + [chosen]),
                                                     "last_logits_metrics": metrics}
                    if args.output:
                        atomic_json(args.output, report)

            result = model.generate(ids, args.max_new_tokens, on_step=check_step)
            completion = result["completion"].lower()
            semantic_pass = all(any(term in completion for term in alternatives) for alternatives in required_terms)
            # This is deliberately a small, explicit smoke test, not a general
            # claim of instruction following or factual benchmark accuracy.
            printable_pass = bool(completion.strip()) and "\ufffd" not in completion and all(
                character.isprintable() or character in "\n\r\t" for character in completion)
            sequence_pass = len(result["generated_ids"]) == args.max_new_tokens or result["stop_reason"] == "eos"
            greedy_matches = sum(item["token_id"] == item["reference_token_id"] for item in step_checks)
            agreement = greedy_matches / len(step_checks)
            failures = [failure for check in step_checks for failure in check["failures"]]
            if not semantic_pass:
                failures.append("continuation failed the explicit meaningful-content smoke check")
            if not printable_pass or not sequence_pass:
                failures.append("continuation is empty, malformed, or stopped unexpectedly")
            if agreement < args.min_greedy_agreement:
                failures.append(f"greedy agreement {agreement:.6g} < {args.min_greedy_agreement}")
            result.update(prompt=prompt, validation_steps=step_checks, semantic_pass=semantic_pass,
                          repetition=repetition_summary(result["completion"]),
                          greedy_agreement=agreement, seconds=time.monotonic() - began,
                          passed=not failures, failures=failures)
            report["prompts"].append(result)
            report.pop("in_progress_prompt", None)
            report["failures"].extend(f"{prompt!r}: {failure}" for failure in failures)
            print(f"prompt={prompt!r} passed={result['passed']} greedy_agreement={agreement:.3f}\n{result['text']}", flush=True)
            if args.output:
                atomic_json(args.output, report)
        if args.logits_json:
            report["native_comparison"] = compare_native(args.logits_json, model, args.native_max_abs)
            if not report["native_comparison"]["passed"]:
                report["failures"].append("native logits differ beyond the declared tolerance")
        report["complete"] = True
        report["passed"] = not report["failures"]
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        if args.output:
            atomic_json(args.output, report)
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        if args.output:
            atomic_json(args.output, report)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("generate", "reference", "validate", "prove"):
        command = commands.add_parser(name)
        command.add_argument("--model-dir", "--model", default=DEFAULT_MODEL_DIR)
        command.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
        if name != "prove":
            command.add_argument("--max-new-tokens", type=int, default=64)
            command.add_argument("--output", help="JSON output; validation updates it after every completed case")
        if name != "validate":
            inputs = command.add_mutually_exclusive_group()
            inputs.add_argument("--prompt", default=PROMPTS[0][0])
            inputs.add_argument("--input-ids", help="JSON file containing token IDs")
        if name in ("generate", "validate"):
            command.add_argument("--logits-json", help="compare native {input_ids, logits, layer_hidden_states?} JSON")
            command.add_argument("--native-max-abs", type=float, default=1e-7)
        if name == "reference":
            command.add_argument("--capture-layers", action="store_true")
        if name == "prove":
            command.add_argument("--binary", default=str(Path(__file__).resolve().parents[1] / "target/release/llama-spain"))
            command.add_argument("--output-dir", default="artifacts/llama/proof")
            command.add_argument("--memory-budget-gib", type=int, default=200)
            command.add_argument("--disk-budget-gib", type=int, default=60)
            command.add_argument("--dark-chunks", type=int, default=256)
        if name == "validate":
            command.add_argument("--max-abs-error", type=float, default=0.01)
            command.add_argument("--max-rmse", type=float, default=0.001)
            command.add_argument("--max-kl", type=float, default=1e-5)
            command.add_argument("--max-relative-l2", type=float, default=0.001)
            command.add_argument("--min-greedy-agreement", type=float, default=0.99)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.threads <= 0 or getattr(args, "max_new_tokens", 1) <= 0:
        raise ValueError("threads and max-new-tokens must be positive")
    if any(getattr(args, key, 1) <= 0 for key in ("memory_budget_gib", "disk_budget_gib")):
        raise ValueError("proof memory and disk budgets must be positive")
    chunks = getattr(args, "dark_chunks", 1)
    if chunks <= 0 or chunks & (chunks - 1):
        raise ValueError("dark-chunks must be a positive power of two")
    for key in ("native_max_abs", "max_abs_error", "max_rmse", "max_kl", "max_relative_l2", "min_greedy_agreement"):
        value = getattr(args, key, None)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be non-negative and finite")
    if getattr(args, "min_greedy_agreement", 0) > 1:
        raise ValueError("min-greedy-agreement must be at most one")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    result = {"generate": run_generate, "reference": run_reference, "validate": run_validate,
              "prove": run_prove}[args.command](args)
    if args.command == "validate":
        print(json.dumps({"complete": result["complete"], "passed": result["passed"],
                          "failures": result["failures"], "seconds": result["seconds"]}, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
