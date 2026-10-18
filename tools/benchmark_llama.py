#!/usr/bin/env python3
"""Serial, measured Llama/Spain sequence benchmarks (stdlib + tokenizers only).

Each attempt owns one foreground native child in a new session.  Completed
attempts are immutable; --resume accepts only matching, verified measurements.
The main proof-size column includes every P->V JSON payload, without JSONL
newlines.  The second proof-size column omits only public-range final openings,
matching SpainBroker; repeated secondary openings remain counted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
MIB = 1024 ** 2
MODEL_DIR = "/home/yliangh/models/meta-llama/Llama-3.2-1B"
TEXT = ("A small library stood near the river. Every morning, the librarian opened the doors "
        "and welcomed readers who came to learn about history, science, and distant places. "
        "The visitors chose their books carefully, sat beside the windows, and shared what "
        "they had learned with their friends before walking home through the quiet town. ")
NATIVE_CONFIG = {"threads": 8, "dark_chunks": 256, "scale_bits": 75, "epsilon": 2.0 ** -40,
                 "memory_budget_gib": 200, "disk_budget_gib": 60}
SECONDS_FIELDS = (
    "prover_seconds", "verifier_seconds", "prover_protocol_seconds", "verifier_protocol_seconds",
    "prover_circuit_setup_seconds", "verifier_circuit_setup_seconds", "prover_state_setup_seconds",
    "verifier_state_setup_seconds", "verifier_layout_seconds", "prover_serialization_seconds",
    "verifier_serialization_seconds", "prover_claim_seconds", "end_to_end_seconds",
    "proof_wall_seconds", "forward_seconds",
)
INTEGER_FIELDS = ("proof_size_bytes", "proof_size_excluding_public_bytes", "prover_message_bytes",
                  "verifier_message_bytes", "communication_bytes_including_public_statement",
                  "peak_rss_kib", "messages", "constraints", "dark_required_bases")
CSV_FIELDS = ("seq_len", "peak_rss_gib", "prover_seconds", "verifier_seconds", "proof_size_mib",
              "verified", "proof_size_excluding_public_mib", "repeat", "status", "resumed",
              "prover_protocol_seconds", "verifier_protocol_seconds", "forward_seconds",
              "end_to_end_seconds", "elapsed_seconds", "proof_wall_seconds", "dark_required_bases",
              "exit_code", "sampled_peak_rss_bytes", "observed_vm_hwm_bytes", "attempt_dir",
              "dark_cache_before_json", "dark_cache_after_json", "reason")


@dataclass(frozen=True)
class ResourceLimits:
    max_rss_bytes: int = 200 * GIB
    min_mem_available_bytes: int = 20 * GIB
    min_disk_free_bytes: int = 10 * GIB
    sample_interval_seconds: float = 0.25
    termination_grace_seconds: float = 5.0


class BenchmarkStop(BaseException):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"received signal {signum}")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest_json(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_identity(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def read_kib_fields(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition(":")
        parts = value.split()
        if separator and len(parts) >= 2 and parts[1] == "kB":
            result[key] = int(parts[0]) * 1024
    return result


def resource_snapshot(pid: int | None, directory: Path) -> dict:
    memory = read_kib_fields(Path("/proc/meminfo"))
    if "MemAvailable" not in memory:
        raise ValueError("Linux /proc/meminfo did not expose MemAvailable")
    process = {}
    if pid is not None:
        try:
            process = read_kib_fields(Path(f"/proc/{pid}/status"))
        except FileNotFoundError:
            pass  # The child can exit between poll() and this read.
    return {"rss_bytes": process.get("VmRSS"), "vm_hwm_bytes": process.get("VmHWM"),
            "mem_available_bytes": memory["MemAvailable"], "disk_free_bytes": shutil.disk_usage(directory).free}


def resource_violation(snapshot: dict, limits: ResourceLimits) -> str | None:
    if max(snapshot.get("rss_bytes") or 0, snapshot.get("vm_hwm_bytes") or 0) > limits.max_rss_bytes:
        return "process RSS exceeded the configured maximum"
    if snapshot["mem_available_bytes"] < limits.min_mem_available_bytes:
        return "MemAvailable fell below the configured minimum"
    if snapshot["disk_free_bytes"] < limits.min_disk_free_bytes:
        return "available disk space fell below the configured minimum"
    return None


def stop_owned_process(process: subprocess.Popen, grace_seconds: float) -> dict:
    """Signal only this Popen child, then always reap it; never signal other jobs."""
    sent = {"terminate_sent": False, "kill_sent": False}
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if process.poll() is None:
            process.terminate()
            sent["terminate_sent"] = True
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            sent["kill_sent"] = True
            process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return sent


def supervise(command: list[str], directory: Path, limits: ResourceLimits, on_progress=None) -> dict:
    """Run one foreground child, with an independent log and direct-PID samples."""
    record = {"status": "starting", "pid": None, "exit_code": None, "elapsed_seconds": 0.0,
              "sampled_peak_rss_bytes": 0, "observed_vm_hwm_bytes": 0, "samples": 0,
              "min_mem_available_bytes": None, "min_disk_free_bytes": None, "termination": {},
              "child_reaped": False}
    process = None
    started = None
    try:
        initial = resource_snapshot(None, directory)
        reason = resource_violation(initial, limits)
        record.update(min_mem_available_bytes=initial["mem_available_bytes"], min_disk_free_bytes=initial["disk_free_bytes"])
        if reason:
            record.update(status="resource_limit", reason=reason)
            return record
        with (directory / "native.log").open("wb") as log, (directory / "resource_samples.jsonl").open("w", buffering=1) as samples:
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
            record.update(status="running", pid=process.pid, started_unix_seconds=time.time())
            last_progress = -math.inf
            while process.poll() is None:
                snapshot = resource_snapshot(process.pid, directory)
                elapsed = time.monotonic() - started
                samples.write(json.dumps({"elapsed_seconds": elapsed, **snapshot}, separators=(",", ":")) + "\n")
                record["samples"] += 1
                record["elapsed_seconds"] = elapsed
                record["sampled_peak_rss_bytes"] = max(record["sampled_peak_rss_bytes"], snapshot["rss_bytes"] or 0)
                record["observed_vm_hwm_bytes"] = max(record["observed_vm_hwm_bytes"], snapshot["vm_hwm_bytes"] or 0)
                record["min_mem_available_bytes"] = min(record["min_mem_available_bytes"], snapshot["mem_available_bytes"])
                record["min_disk_free_bytes"] = min(record["min_disk_free_bytes"], snapshot["disk_free_bytes"])
                reason = resource_violation(snapshot, limits)
                if reason:
                    record.update(status="resource_limit", reason=reason)
                    break
                if on_progress and elapsed - last_progress >= 5:
                    on_progress(record)
                    last_progress = elapsed
                time.sleep(limits.sample_interval_seconds)
            if record["status"] == "running":
                record["status"] = "exited"
    except (KeyboardInterrupt, BenchmarkStop) as exc:
        record.update(status="interrupted", reason=str(exc) or "interrupted by user")
    except Exception as exc:
        record.update(status="supervision_error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None:
            if process.poll() is None:
                record["termination"] = stop_owned_process(process, limits.termination_grace_seconds)
            record["exit_code"] = process.wait()
            record["child_reaped"] = True
        if started is not None:
            record["elapsed_seconds"] = time.monotonic() - started
    return record


def transcript_counts(path: Path) -> dict:
    counts = {"messages": 0, "prover_message_bytes": 0, "verifier_message_bytes": 0,
              "proof_size_excluding_public_bytes": 0}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            digest.update(line)
            if not line.endswith(b"\n") or line == b"\n":
                raise ValueError("incomplete or empty transcript line")
            payload = line[:-1]
            message = json.loads(payload)
            counts["messages"] += 1
            # Native init_message originates at V; every response alternates.
            if index % 2 == 0:
                counts["verifier_message_bytes"] += len(payload)
            else:
                counts["prover_message_bytes"] += len(payload)
                if isinstance(message, dict) and "WitnessOpenings" in message:
                    openings = message["WitnessOpenings"]
                    if not isinstance(openings, list) or any(not isinstance(item, dict) or "range_index" not in item for item in openings):
                        raise ValueError("malformed WitnessOpenings payload")
                    discounted = {"WitnessOpenings": [item for item in openings if item["range_index"] != 0]}
                    length = len(json.dumps(discounted, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())
                else:
                    length = len(payload)
                counts["proof_size_excluding_public_bytes"] += length
    counts["sha256"] = digest.hexdigest()
    return counts


def require_number(report: dict, name: str, integer: bool = False) -> int | float:
    value = report.get(name)
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value < 0:
        raise ValueError(f"schema 2 requires non-negative finite {'integer' if integer else 'number'} {name}")
    return value


def validate_report(directory: Path, identity: dict) -> tuple[dict, dict]:
    report = json.loads((directory / "proof_report.json").read_text())
    if report.get("measurement_schema_version") != 2 or report.get("verified") is not True:
        raise ValueError("a verified measurement_schema_version=2 report is required")
    for name in SECONDS_FIELDS:
        require_number(report, name)
    for name in INTEGER_FIELDS:
        require_number(report, name, integer=True)
    for name in ("scale_bits", "epsilon", "dark_chunks"):
        if report.get(name) != identity["native_config"][name]:
            raise ValueError(f"native report configuration mismatch: {name}")
    if report.get("input_tokens") != len(identity["input_ids"]):
        raise ValueError("native report sequence length mismatch")
    for name, config_name in (("layers", "num_hidden_layers"), ("vocabulary", "vocab_size")):
        if report.get(name) != identity["model_dimensions"][config_name]:
            raise ValueError(f"native report model mismatch: {name}")
    for name in ("dark_cache_before", "dark_cache_after"):
        if not isinstance(report.get(name), list):
            raise ValueError(f"schema 2 requires {name}")
        for entry in report[name]:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError(f"malformed {name} entry")
            require_number(entry, "bytes", integer=True)
    prover_sum = sum(report[name] for name in ("forward_seconds", "prover_claim_seconds", "prover_circuit_setup_seconds",
                     "prover_state_setup_seconds", "prover_protocol_seconds", "prover_serialization_seconds"))
    verifier_sum = sum(report[name] for name in ("verifier_layout_seconds", "verifier_circuit_setup_seconds",
                       "verifier_state_setup_seconds", "verifier_protocol_seconds", "verifier_serialization_seconds"))
    if not math.isclose(prover_sum, report["prover_seconds"], rel_tol=1e-8, abs_tol=1e-6) or not math.isclose(verifier_sum, report["verifier_seconds"], rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("party timing components do not sum to the reported totals")
    if prover_sum + verifier_sum > report["end_to_end_seconds"] + 1e-5:
        raise ValueError("party active times exceed native end-to-end time")
    counts = transcript_counts(directory / "transcript.jsonl")
    for name in ("messages", "prover_message_bytes", "verifier_message_bytes", "proof_size_excluding_public_bytes"):
        if report[name] != counts[name]:
            raise ValueError(f"transcript direction/byte accounting mismatch: {name}")
    if report["proof_size_bytes"] != counts["prover_message_bytes"]:
        raise ValueError("proof_size_bytes must equal all P->V payload bytes")
    if report["communication_bytes_including_public_statement"] != counts["prover_message_bytes"] + counts["verifier_message_bytes"]:
        raise ValueError("bidirectional communication byte count mismatch")
    if not 0 <= report["proof_size_excluding_public_bytes"] <= report["proof_size_bytes"] or not counts["messages"]:
        raise ValueError("invalid proof byte counts or empty transcript")
    statement = json.loads((directory / "statement.json").read_text())
    if statement.get("input_ids") != identity["input_ids"] or statement.get("model") != identity["model_files"]:
        raise ValueError("statement input/checkpoint identity mismatch")
    if json.loads((directory / "input_ids.json").read_text()) != identity["input_ids"]:
        raise ValueError("saved input IDs do not match the measurement identity")
    hashes = {name: file_identity(directory / name)["sha256"] for name in ("proof_report.json", "statement.json", "input_ids.json")}
    hashes["transcript.jsonl"] = counts["sha256"]
    return report, hashes


def measurement_row(record: dict, report: dict | None = None) -> dict:
    row = {"seq_len": record["identity"]["seq_len"], "repeat": record["identity"]["repeat"],
           "status": record["status"], "verified": record["status"] == "verified", "resumed": False,
           "attempt_dir": record["attempt_dir"], "reason": record.get("reason", "")}
    for name in ("pid", "child_reaped", "exit_code", "elapsed_seconds", "sampled_peak_rss_bytes", "observed_vm_hwm_bytes"):
        row[name] = record.get(name)
    peak = max(record.get("sampled_peak_rss_bytes", 0), record.get("observed_vm_hwm_bytes", 0),
               (report or {}).get("peak_rss_kib", 0) * 1024)
    row["peak_rss_gib"] = peak / GIB if record.get("pid") is not None else None
    if report:
        row.update({name: report[name] for name in SECONDS_FIELDS + INTEGER_FIELDS})
        row["proof_size_mib"] = report["proof_size_bytes"] / MIB
        row["proof_size_excluding_public_mib"] = report["proof_size_excluding_public_bytes"] / MIB
        for name in ("dark_cache_before", "dark_cache_after"):
            row[name] = report[name]
            row[name + "_json"] = json.dumps(report[name], separators=(",", ":"))
    return row


def find_resumable(case_dir: Path, identity: dict) -> dict | None:
    for directory in sorted(case_dir.glob("attempt-*"), reverse=True):
        try:
            record = json.loads((directory / "run.json").read_text())
            if record.get("benchmark_schema_version") != 1 or record.get("status") != "verified" or record.get("exit_code") != 0 or record.get("identity") != identity:
                continue
            report, hashes = validate_report(directory, identity)
            if record.get("artifact_sha256") != hashes:
                continue
            row = measurement_row(record, report)
            row["resumed"] = True
            return row
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def fresh_attempt(case_dir: Path) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = case_dir / f"attempt-{index:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def native_budget_rejection(path: Path) -> str | None:
    """Recognize only the native executable's exact fatal budget diagnostics."""
    final_line = ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                final_line = line.strip()
    if (re.fullmatch(r"Error: estimated working memory exceeds budget: [0-9]+ bytes", final_line)
            or final_line == "Error: estimated artifacts exceed disk budget"):
        return final_line
    return None


def run_case(case_dir: Path, identity: dict, limits: ResourceLimits) -> dict:
    directory = fresh_attempt(case_dir)
    atomic_json(directory / "input_ids.json", identity["input_ids"])
    command = [identity["binary_path"], "prove", "--model-dir", identity["model_dir"],
               "--input-ids", str(directory / "input_ids.json"), "--output-dir", str(directory)]
    for name, value in identity["native_config"].items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    record = {"benchmark_schema_version": 1, "identity": identity, "identity_sha256": digest_json(identity),
              "attempt_dir": str(directory), "command": command, "status": "planned"}
    atomic_json(directory / "run.json", record)
    report = None
    try:
        if file_identity(Path(identity["binary_path"])) != identity["binary"]:
            raise ValueError("native binary changed after benchmark preparation")
        def progress(update):
            record.update(update)
            atomic_json(directory / "run.json", record)
        record.update(supervise(command, directory, limits, progress))
        if record["status"] == "exited":
            if record["exit_code"] != 0:
                reason = native_budget_rejection(directory / "native.log") if record["exit_code"] == 1 else None
                record.update(status="resource_limit" if reason else "failed",
                              reason=reason or f"native process exited with code {record['exit_code']}")
            else:
                report, hashes = validate_report(directory, identity)
                if report["peak_rss_kib"] * 1024 > limits.max_rss_bytes:
                    record.update(status="resource_limit", reason="native final peak RSS exceeded the configured maximum")
                else:
                    record.update(status="verified", artifact_sha256=hashes)
    except (KeyboardInterrupt, BenchmarkStop) as exc:
        record.update(status="interrupted", reason=str(exc) or "interrupted by user")
    except Exception as exc:
        record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    atomic_json(directory / "run.json", record)
    return measurement_row(record, report)


def build_inputs(model_dir: Path, lengths: list[int]) -> tuple[dict, dict[int, list[int]], dict]:
    from tokenizers import Tokenizer
    config = json.loads((model_dir / "config.json").read_text())
    if any(length > config["max_position_embeddings"] for length in lengths):
        raise ValueError("requested sequence length exceeds model context")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    repeat = 1
    while True:
        tokens = tokenizer.encode(TEXT * repeat, add_special_tokens=True).ids
        if len(tokens) >= max(lengths):
            break
        repeat *= 2
    if not tokens or tokens[0] != config.get("bos_token_id"):
        raise ValueError("the local tokenizer must insert exactly the configured initial BOS token")
    inputs = {length: tokens[:length] for length in lengths}
    return config, inputs, {"base_text": TEXT, "base_text_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
                            "text_repeats": repeat, "bos_token_id": config["bos_token_id"]}


def environment() -> dict:
    memory = read_kib_fields(Path("/proc/meminfo"))
    cpu_models = sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")})
    return {"platform": platform.platform(), "kernel": platform.release(), "machine": platform.machine(),
            "python": sys.version, "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_model_names": cpu_models, "mem_total_bytes": memory.get("MemTotal")}


def write_summary(root: Path, manifest: dict, rows: list[dict]) -> None:
    summary = {"benchmark_schema_version": 1, "manifest": manifest, "runs": rows,
               "complete": all(row["status"] not in ("planned", "not_run") for row in rows),
               "passed": bool(rows) and all(row.get("verified") is True for row in rows),
               "definitions": {
                   "peak_rss_gib": "Whole native process; maximum of direct-PID VmRSS samples, observed VmHWM, and native final VmHWM. GiB=2^30 bytes.",
                   "prover_seconds": "Native schema 2 party active wall time, including high-precision forward, claim quantization, assigned setup, protocol calls and outgoing JSON serialization.",
                   "verifier_seconds": "Native schema 2 party active wall time, including verifier layout/setup, init_message, protocol calls and outgoing JSON serialization.",
                   "proof_size_mib": "All P->V serialized JSON payloads, including public and secondary openings; excludes JSONL newlines. MiB=2^20 bytes.",
                   "proof_size_excluding_public_mib": "SpainBroker convention: only range 0 is removed from final WitnessOpenings before reserialization; repeated secondary openings are retained.",
                   "cache": "Caches are preserved; each native report records dark_cache_before/after. The runner performs no warmup or cache deletion.",
                   "failed_rows": "Missing proof timing/size fields remain empty; no extrapolation or replacement by Debug actor totals.",
               }}
    atomic_json(root / "summary.json", summary)
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "summary.csv")


@contextmanager
def runner_lock(root: Path):
    with (root / ".benchmark.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another benchmark runner owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(args) -> int:
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    limits = ResourceLimits(max_rss_bytes=int(args.max_rss_gib * GIB),
                            min_disk_free_bytes=int(args.min_disk_free_gib * GIB), sample_interval_seconds=args.sample_interval)
    with runner_lock(root):
        model_dir, binary = Path(args.model_dir).resolve(), Path(args.binary).resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"native executable is unavailable: {binary}")
        config, inputs, tokenization = build_inputs(model_dir, args.lengths)
        filenames = sorted(path for path in model_dir.iterdir() if path.name.endswith(".safetensors") or path.name in
                           {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors.index.json"})
        model_files = {path.name: file_identity(path) for path in filenames}
        base = {"binary_path": str(binary), "binary": file_identity(binary), "model_dir": str(model_dir),
                "model_files": model_files,
                "native_config": {**NATIVE_CONFIG, "memory_budget_gib": args.memory_budget_gib},
                "resource_limits": asdict(limits),
                "model_dimensions": {name: config[name] for name in ("num_hidden_layers", "vocab_size")},
                "base_text_sha256": tokenization["base_text_sha256"]}
        manifest = {**base, "environment": environment(), "lengths": args.lengths, "repeats": args.repeats,
                    "tokenization": tokenization, "inputs": {str(length): {"input_ids": ids, "sha256": digest_json(ids)} for length, ids in inputs.items()},
                    "config_path": str(model_dir / "config.json"), "tokenizer_path": str(model_dir / "tokenizer.json"),
                    "started_unix_seconds": time.time(), "dry_run": args.dry_run,
                    "continue_on_resource_limit": args.continue_on_resource_limit}
        if (root / "cache-preparation.json").exists():
            manifest["cache_preparation"] = file_identity(root / "cache-preparation.json")
        atomic_json(root / "benchmark_manifest.json", manifest)
        rows = [{"seq_len": length, "repeat": repeat, "status": "planned", "verified": False, "resumed": False}
                for length in args.lengths for repeat in range(1, args.repeats + 1)]
        write_summary(root, manifest, rows)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "output_dir": str(root), "cases": rows}, indent=2))
            return 0
        resource_limited = False
        for index, planned in enumerate(rows):
            length, repeat = planned["seq_len"], planned["repeat"]
            ids = inputs[length]
            identity = {**base, "seq_len": length, "repeat": repeat, "input_ids": ids, "input_sha256": digest_json(ids)}
            case_dir = root / f"length-{length:04d}" / f"repeat-{repeat:03d}"
            resumed = find_resumable(case_dir, identity) if args.resume else None
            if resumed is not None:
                rows[index] = resumed
                print(f"resume length={length} repeat={repeat}: verified matching attempt", flush=True)
            else:
                print(f"start length={length} repeat={repeat} threads=8", flush=True)
                rows[index] = run_case(case_dir, identity, limits)
                print(f"end length={length} repeat={repeat} status={rows[index]['status']} elapsed={rows[index].get('elapsed_seconds')}", flush=True)
            if not rows[index]["verified"]:
                resource_limited |= rows[index]["status"] == "resource_limit"
                safe_to_continue = rows[index].get("pid") is None or rows[index].get("child_reaped") is True
                if args.continue_on_resource_limit and rows[index]["status"] == "resource_limit" and safe_to_continue:
                    write_summary(root, manifest, rows)
                    continue
                for later in rows[index + 1:]:
                    later.update(status="not_run", reason=f"stopped after length={length}, repeat={repeat}: {rows[index]['status']}")
                write_summary(root, manifest, rows)
                return 130 if rows[index]["status"] == "interrupted" else 2 if resource_limited else 1
            write_summary(root, manifest, rows)
        return 2 if resource_limited else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--binary", default=str(ROOT / "target/release/llama-spain"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/llama/sequence-benchmark"))
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 8, 32])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="prepare manifests/inputs without starting any native process")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--min-disk-free-gib", type=float, default=10.0)
    parser.add_argument("--memory-budget-gib", type=int, default=200, help="native estimated-memory budget (positive integer GiB)")
    parser.add_argument("--max-rss-gib", type=float, default=200.0, help="supervised process RSS limit; the 20 GiB MemAvailable guard remains active")
    parser.add_argument("--continue-on-resource-limit", action="store_true", help="record resource-limited cases and continue after their child has been reaped")
    args = parser.parse_args(argv)
    if args.repeats <= 0 or any(length <= 0 for length in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths must be distinct positive integers and repeats must be positive")
    if not math.isfinite(args.sample_interval) or not 0 < args.sample_interval <= 10:
        raise ValueError("sample-interval must be finite and in (0, 10] seconds")
    if not math.isfinite(args.min_disk_free_gib) or args.min_disk_free_gib < 0:
        raise ValueError("min-disk-free-gib must be non-negative and finite")
    if args.memory_budget_gib <= 0:
        raise ValueError("memory-budget-gib must be a positive integer")
    if not math.isfinite(args.max_rss_gib * GIB) or args.max_rss_gib <= 0 or int(args.max_rss_gib * GIB) < 1:
        raise ValueError("max-rss-gib must be finite and positive (at least one byte)")
    previous = signal.signal(signal.SIGTERM, lambda signum, _frame: (_ for _ in ()).throw(BenchmarkStop(signum)))
    try:
        return run(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, BenchmarkStop):
        print("benchmark interrupted; no further cases started", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
