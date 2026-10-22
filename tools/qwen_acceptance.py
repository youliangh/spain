#!/usr/bin/env python3
"""Fail-closed audit of completed Qwen numerical, generation and proof reports.

This aggregates evidence; it does not replace executing the validation or the
interactive proof. Paths to the actual runs are required explicitly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import onnx
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qwen import LENGTHS, LIMITS, PROMPTS, atomic_json, tensor_error


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(args):
    checks, evidence = {}, {}
    numerical = read(args.validation)
    native = read(args.native_validation)
    evidence["numerical_report"] = {"path": str(Path(args.validation).resolve()), "sha256": digest(args.validation)}
    evidence["native_report"] = {"path": str(Path(args.native_validation).resolve()), "sha256": digest(args.native_validation)}
    arithmetic_prompt = PROMPTS[3][0]
    def accepted_reference_error(row):
        # Explicitly approved exception for this checkpoint and this frozen
        # sample. The raw validation remains failed and is never rewritten.
        return (args.accept_reference_arithmetic_error and row.get("prompt") == arithmetic_prompt
            and row.get("completion", "").strip() == "52"
            and row.get("content_checks") == {"nonempty_readable_text": True, "expected_answer": False}
            and bool(row.get("steps")) and row.get("greedy_agreement") == 1.0
            and all(s.get("token_id") == s.get("reference_token_id")
                    and s["metrics"]["max_abs"] <= LIMITS["max_abs"]
                    and s["metrics"]["max_row_rmse"] <= LIMITS["max_rmse"]
                    and s["metrics"]["margin_violations"] == 0 for s in row["steps"]))
    raw_failures = numerical.get("failures", [])
    exception_only = (args.accept_reference_arithmetic_error
                      and raw_failures == [f"generation: {arithmetic_prompt}"])
    checks["numerical_complete"] = (numerical.get("status") == "complete"
                                     and (numerical.get("passed") is True or exception_only))
    checks["required_lengths"] = {r["length"] for r in numerical.get("lengths", [])} == set(LENGTHS)
    checks["all_numerical_rows_pass"] = all(r.get("passed") is True for r in numerical.get("lengths", []))
    checks["unchanged_limits"] = numerical.get("limits") == LIMITS
    generation = numerical.get("generation", [])
    checks["fixed_six_prompts"] = [r.get("prompt") for r in generation] == [p for p, _ in PROMPTS]
    checks["generation_pass"] = all((r.get("passed") is True or accepted_reference_error(r)) and r.get("completion", "").strip()
                                     and r.get("greedy_agreement", 0) >= LIMITS["greedy_per_prompt"]
                                     for r in generation)
    checks["overall_greedy_agreement"] = numerical.get("greedy_total_agreement", 0) >= LIMITS["greedy_total"]
    checks["native_complete"] = native.get("status") == "complete" and native.get("passed") is True
    checks["native_lengths"] = {r["length"] for r in native.get("lengths", [])} == set(LENGTHS)
    checks["native_precision_limit"] = native.get("max_abs_limit") == LIMITS["native_max_abs"]
    required_outputs = {"embedding", "final_norm", "logits", "last_logits"} | {f"layer_{i:02d}" for i in range(24)}
    checks["native_all_layers_all_logits"] = all(
        r.get("passed") is True and r.get("arithmetic") == "MPFR-128"
        and set(r.get("tensors", {})) == required_outputs
        and all(m.get("max_abs", float("inf")) <= LIMITS["native_max_abs"] for m in r["tensors"].values())
        for r in native.get("lengths", []))
    proofs = []
    authenticated_files = {}
    snapshot = Path(args.runtime_snapshot).resolve() if args.runtime_snapshot else None
    if snapshot:
        snapshot_manifest = read(snapshot / "manifest.json")
        snapshot_binary = digest(snapshot / "onnx-spain")
        snapshot_sources = {name: digest(snapshot / name)
                            for name in snapshot_manifest["source_sha256"]}
        evidence["runtime_snapshot"] = str(snapshot)
    for length, path in [(2, args.proof_seq2), (32, args.proof_seq32)]:
        directory = Path(path).resolve()
        report = read(directory / "proof_report.json")
        statement = read(directory / "statement.json")
        implementation = read(directory / "implementation_manifest.json")
        compiled = Path(implementation["compiled_artifact_dir"])
        meta = read(compiled / "meta.json")
        prefix = f"proof_seq{length}"
        if snapshot:
            checks[prefix + "_runtime_provenance"] = (
                snapshot_binary == snapshot_manifest["executable_sha256"] == implementation["executable_sha256"]
                and snapshot_sources == snapshot_manifest["source_sha256"]
                and all(implementation["source_sha256"].get(name) == value
                        for name, value in snapshot_sources.items()))
        model = onnx.load(compiled / "original_model.onnx", load_external_data=False)
        properties = {p.key: p.value for p in model.metadata_props}
        produced = {name for node in model.graph.node for name in node.output}
        checks[prefix + "_24_layers"] = (properties.get("spain.layers") == "24"
            and properties.get("spain.model") == "qwen3_5_text"
            and properties.get("spain.numerics") == "pade-v2"
            and all(f"layer_{i:02d}" in produced for i in range(24)))
        asset_ok = True
        for asset in meta["artifacts"]:
            filename = compiled / asset["path"]
            stat = filename.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity not in authenticated_files:
                authenticated_files[identity] = digest(filename)
            asset_ok &= (stat.st_size == asset["bytes"] and authenticated_files[identity] == asset["sha256"])
        canonical = json.dumps(meta["artifacts"], sort_keys=True, separators=(",", ":")).encode()
        layout = {k: v for k, v in meta.items() if k not in ("artifacts", "fingerprint_sha256")}
        checks[prefix + "_authenticated_assets"] = (asset_ok and read(compiled / "layout.json") == layout
            and hashlib.sha256(canonical).hexdigest() == meta["fingerprint_sha256"])
        checks[prefix + "_verified"] = report.get("verified") is True and report.get("input_tokens") == length
        checks[prefix + "_parameters"] = (report.get("scale_bits") == 75 and report.get("epsilon") == 2**-40
                                           and report.get("dark_chunks") == 256)
        fingerprint = meta["fingerprint_sha256"]
        checks[prefix + "_bound_statement"] = (statement.get("fingerprint_sha256") == fingerprint
            and report.get("fingerprint_sha256") == fingerprint
            and implementation.get("artifact_fingerprint") == fingerprint
            and len(statement.get("input_ids", [])) == length
            and statement.get("positions") == list(range(length)))
        claims = meta["public"]["claims"]
        checks[prefix + "_full_vocabulary"] = (len(claims) == 1 and claims[0]["name"] == "last_logits"
            and claims[0]["shape"] == [1, 1, 248320] and claims[0]["length"] == 248320
            and set(statement.get("claims_scaled", {})) == {"last_logits"}
            and len(statement["claims_scaled"]["last_logits"]) == 248320)
        checks[prefix + "_complete_witness"] = (directory / "primary_witness.i128le").stat().st_size == 16 * meta["num_witness_values"]
        checks[prefix + "_transcript"] = (directory / "transcript.jsonl").stat().st_size > 0 and report.get("messages", 0) > 50
        checks[prefix + "_resource_budget"] = report.get("peak_rss_kib", 2**63) <= 200 * 1024**2
        checks[prefix + "_compiler_provenance"] = bool(meta.get("source_graph_sha256") and meta.get("compiler_source_sha256"))
        forward = read(directory / "forward.json")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(compiled / "original_model.onnx"), options,
                                       providers=["CPUExecutionProvider"])
        expected = session.run(["last_logits"], {"input_ids": np.array([statement["input_ids"]], dtype=np.int64)})[0]
        actual = forward["outputs"]["last_logits"]
        actual = np.array(actual["values"], dtype=np.float64).reshape(actual["shape"])
        forward_metrics = tensor_error(actual, expected, logits=True)
        checks[prefix + "_actual_forward_accuracy"] = (forward.get("arithmetic") == "MPFR-128 ONNX"
            and forward.get("input_ids") == statement["input_ids"]
            and forward_metrics["max_abs"] <= LIMITS["native_max_abs"]
            and forward_metrics["margin_violations"] == 0)
        del session, expected, actual, forward
        proofs.append({"length": length, "directory": str(directory), "fingerprint_sha256": fingerprint,
                       "report_sha256": digest(directory / "proof_report.json"),
                       "transcript_sha256": digest(directory / "transcript.jsonl"),
                       "constraint_count": meta["num_constraints"], "peak_rss_kib": report.get("peak_rss_kib"),
                       "end_to_end_seconds": report.get("end_to_end_seconds"),
                       "actual_forward_versus_onnx": forward_metrics})
    evidence["proofs"] = proofs
    evidence["completions"] = [{k: row[k] for k in ("prompt", "completion", "greedy_agreement")} for row in generation]
    evidence["raw_capability_gate"] = {"passed": numerical.get("passed"), "failures": raw_failures}
    evidence["accepted_reference_limitation"] = {
        "enabled": args.accept_reference_arithmetic_error,
        "scope": "Only fixed 17 + 25 prompt: original HF and ONNX both answer 52 with identical greedy tokens",
        "authorization": "User explicitly chose faithful reproduction and recording the original arithmetic error as a known limitation",
        "applied": [r["prompt"] for r in generation if accepted_reference_error(r)]}
    result = {"schema_version": 1, "passed": all(checks.values()), "checks": checks,
              "failures": [name for name, value in checks.items() if not value], "evidence": evidence,
              "scope": "completed-run audit; separate unit/regression test logs accompany this report"}
    atomic_json(args.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", required=True)
    p.add_argument("--native-validation", required=True)
    p.add_argument("--proof-seq2", required=True)
    p.add_argument("--proof-seq32", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime-snapshot", help="Also authenticate the retained executable and source snapshot used by both proofs")
    p.add_argument("--accept-reference-arithmetic-error", action="store_true",
                   help="Apply the explicitly approved, narrowly scoped original-model arithmetic limitation")
    args = p.parse_args()
    result = audit(args)
    print("PASS" if result["passed"] else "FAIL: " + ", ".join(result["failures"]))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
