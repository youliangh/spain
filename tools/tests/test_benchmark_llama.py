"""Benchmark accounting and owned-child cleanup; no model forward or large proof."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_llama as benchmark


def fixture(directory):
    ids = [128000, 32]
    identity = {"seq_len": 2, "repeat": 1, "input_ids": ids,
                "binary": {"sha256": "binary-v1", "bytes": 10},
                "native_config": dict(benchmark.NATIVE_CONFIG),
                "model_dimensions": {"num_hidden_layers": 16, "vocab_size": 128256},
                "model_files": {"config.json": {"sha256": "checkpoint-v1", "bytes": 40}}}
    messages = [{"Setup": [1]}, {"Secondary": ["7", "-11"]}, "Challenge",
                {"Commitment": "abc"}, "RequestWitnessOpenings",
                {"WitnessOpenings": [{"range_index": 0, "width": 1, "values": ["42", "23"]},
                                     {"range_index": 2, "width": 1, "values": ["7", "-11"]}]}]
    payloads = [json.dumps(message, separators=(",", ":")).encode() for message in messages]
    (directory / "transcript.jsonl").write_bytes(b"\n".join(payloads) + b"\n")
    full = sum(map(len, payloads[1::2]))
    final_without_public = json.dumps({"WitnessOpenings": [messages[-1]["WitnessOpenings"][1]]}, separators=(",", ":")).encode()
    excluded = len(payloads[1]) + len(payloads[3]) + len(final_without_public)
    report = {name: 0.01 for name in benchmark.SECONDS_FIELDS}
    report.update({name: 1 for name in benchmark.INTEGER_FIELDS})
    report.update(measurement_schema_version=2, verified=True, scale_bits=75, epsilon=2.0**-40,
                  dark_chunks=256, input_tokens=2, layers=16, vocabulary=128256,
                  prover_seconds=.06, verifier_seconds=.05, end_to_end_seconds=.12,
                  proof_size_bytes=full, prover_message_bytes=full,
                  verifier_message_bytes=sum(map(len, payloads[::2])),
                  communication_bytes_including_public_statement=sum(map(len, payloads)),
                  proof_size_excluding_public_bytes=excluded, messages=len(payloads),
                  dark_cache_before=[{"name": "bases", "bytes": 123}],
                  dark_cache_after=[{"name": "bases", "bytes": 123}])
    benchmark.atomic_json(directory / "proof_report.json", report)
    benchmark.atomic_json(directory / "statement.json", {"input_ids": ids, "model": identity["model_files"]})
    benchmark.atomic_json(directory / "input_ids.json", ids)
    return identity, report


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = benchmark.fresh_attempt(self.root / "case")
        self.identity, self.report = fixture(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_schema2_and_directional_bytes(self):
        report, hashes = benchmark.validate_report(self.directory, self.identity)
        self.assertEqual(report["proof_size_bytes"], report["prover_message_bytes"])
        self.assertGreater(report["communication_bytes_including_public_statement"], report["proof_size_bytes"])
        self.assertLess(report["proof_size_excluding_public_bytes"], report["proof_size_bytes"])
        self.assertEqual(set(hashes), {"statement.json", "input_ids.json", "proof_report.json", "transcript.jsonl"})

    def test_rejects_direction_swap_and_incorrect_discount(self):
        for field, value in (("prover_message_bytes", self.report["verifier_message_bytes"]),
                             ("proof_size_excluding_public_bytes", 0)):
            with self.subTest(field=field):
                changed = {**self.report, field: value}
                benchmark.atomic_json(self.directory / "proof_report.json", changed)
                with self.assertRaisesRegex(ValueError, "accounting mismatch"):
                    benchmark.validate_report(self.directory, self.identity)

    def test_rejects_legacy_nonfinite_missing_and_bad_timing(self):
        changes = ({"measurement_schema_version": 1}, {"prover_seconds": float("nan")},
                   {"forward_seconds": None}, {"prover_seconds": 5.0}, {"end_to_end_seconds": .01})
        for change in changes:
            with self.subTest(change=change):
                (self.directory / "proof_report.json").write_text(json.dumps({**self.report, **change}))
                with self.assertRaises(ValueError):
                    benchmark.validate_report(self.directory, self.identity)

    def save_verified(self):
        _, hashes = benchmark.validate_report(self.directory, self.identity)
        record = {"benchmark_schema_version": 1, "identity": self.identity, "status": "verified",
                  "exit_code": 0, "pid": 123, "attempt_dir": str(self.directory), "artifact_sha256": hashes}
        benchmark.atomic_json(self.directory / "run.json", record)
        return record

    def test_resume_only_matching_verified_identity(self):
        self.save_verified()
        self.assertTrue(benchmark.find_resumable(self.root / "case", self.identity)["resumed"])
        for key, value in (("binary", {"sha256": "binary-v2", "bytes": 10}),
                           ("input_ids", [128000, 33]), ("model_files", {}),
                           ("native_config", {**benchmark.NATIVE_CONFIG, "dark_chunks": 128})):
            with self.subTest(key=key):
                changed = {**self.identity, key: value}
                self.assertIsNone(benchmark.find_resumable(self.root / "case", changed))

    def test_resume_rejects_tampered_artifact_and_failed_record(self):
        record = self.save_verified()
        original = (self.directory / "proof_report.json").read_text()
        (self.directory / "proof_report.json").write_text(original + " ")
        self.assertIsNone(benchmark.find_resumable(self.root / "case", self.identity))
        (self.directory / "proof_report.json").write_text(original)
        record["status"] = "failed"
        benchmark.atomic_json(self.directory / "run.json", record)
        self.assertIsNone(benchmark.find_resumable(self.root / "case", self.identity))

    def test_fresh_attempt_preserves_previous_files(self):
        self.save_verified()
        original = (self.directory / "run.json").read_bytes()
        new = benchmark.fresh_attempt(self.root / "case")
        self.assertNotEqual(new, self.directory)
        self.assertEqual((self.directory / "run.json").read_bytes(), original)

    def test_local_tokenizer_produces_shared_bos_prefix(self):
        benchmark.atomic_json(self.root / "config.json", {"max_position_embeddings": 128, "bos_token_id": 128000})
        tokenizer = types.SimpleNamespace(encode=lambda text, add_special_tokens: types.SimpleNamespace(ids=[128000] + list(range(1, len(text)))))
        module = types.SimpleNamespace(Tokenizer=types.SimpleNamespace(from_file=lambda path: tokenizer))
        with patch.dict(sys.modules, {"tokenizers": module}):
            _, inputs, info = benchmark.build_inputs(self.root, [1, 2, 8, 32])
        self.assertEqual(inputs[1], [128000])
        self.assertEqual(info["bos_token_id"], 128000)
        for length, ids in inputs.items():
            self.assertEqual(ids, inputs[32][:length])


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.limits = benchmark.ResourceLimits(max_rss_bytes=benchmark.GIB,
            min_mem_available_bytes=0, min_disk_free_bytes=0,
            sample_interval_seconds=.01, termination_grace_seconds=.1)

    def tearDown(self):
        self.temporary.cleanup()

    def snapshot(self, pid, directory):
        return {"rss_bytes": 1000 if pid else None, "vm_hwm_bytes": 2000 if pid else None,
                "mem_available_bytes": 30 * benchmark.GIB, "disk_free_bytes": 30 * benchmark.GIB}

    def assert_reaped(self, result):
        self.assertTrue(result["child_reaped"])
        with self.assertRaises(ProcessLookupError):
            os.kill(result["pid"], 0)

    def test_success_has_direct_session_pid_log_and_measurement(self):
        seen = []
        def inspect(pid, directory):
            if pid:
                seen.append(pid)
                self.assertEqual(os.getsid(pid), pid)
            return self.snapshot(pid, directory)
        with patch.object(benchmark, "resource_snapshot", side_effect=inspect):
            result = benchmark.supervise([sys.executable, "-c", "import time; print('native log'); time.sleep(.05)"], self.root, self.limits)
        self.assertEqual(result["status"], "exited")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(set(seen), {result["pid"]})
        self.assertEqual(result["observed_vm_hwm_bytes"], 2000)
        self.assertIn("native log", (self.root / "native.log").read_text())
        self.assert_reaped(result)

    def test_preflight_resource_limit_never_spawns(self):
        with patch.object(benchmark, "resource_snapshot", side_effect=self.snapshot), patch.object(benchmark.subprocess, "Popen") as popen:
            result = benchmark.supervise(["unstarted"], self.root, replace(self.limits, min_mem_available_bytes=40 * benchmark.GIB))
        popen.assert_not_called()
        self.assertEqual(result["status"], "resource_limit")
        self.assertIsNone(result["pid"])

    def test_live_rss_memory_and_disk_limits_terminate_and_reap(self):
        for index, field in enumerate(("rss_bytes", "mem_available_bytes", "disk_free_bytes")):
            directory = self.root / str(index)
            directory.mkdir()
            limits = replace(self.limits, min_mem_available_bytes=20 * benchmark.GIB, min_disk_free_bytes=10 * benchmark.GIB)
            def exceed(pid, path):
                value = self.snapshot(pid, path)
                if pid:
                    value[field] = 2 * benchmark.GIB if field == "rss_bytes" else 0
                return value
            with self.subTest(field=field), patch.object(benchmark, "resource_snapshot", side_effect=exceed):
                result = benchmark.supervise([sys.executable, "-c", "import time; time.sleep(60)"], directory, limits)
                self.assertEqual(result["status"], "resource_limit")
                self.assertTrue(result["termination"]["terminate_sent"])
                self.assertLess(result["exit_code"], 0)
                self.assert_reaped(result)

    def test_stubborn_child_is_killed_and_reaped(self):
        ready = self.root / "ready"
        code = "import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(" + repr(str(ready)) + ").touch(); time.sleep(60)"
        def exceed_when_ready(pid, directory):
            snapshot = self.snapshot(pid, directory)
            if pid and ready.exists():
                snapshot["rss_bytes"] = 2 * benchmark.GIB
            return snapshot
        with patch.object(benchmark, "resource_snapshot", side_effect=exceed_when_ready):
            result = benchmark.supervise([sys.executable, "-c", code], self.root, self.limits)
        self.assertEqual(result["status"], "resource_limit")
        self.assertTrue(result["termination"]["kill_sent"])
        self.assertEqual(result["exit_code"], -signal.SIGKILL)
        self.assert_reaped(result)

    def test_interrupt_reaps_owned_child(self):
        def interrupt(pid, directory):
            if pid:
                raise KeyboardInterrupt
            return self.snapshot(pid, directory)
        with patch.object(benchmark, "resource_snapshot", side_effect=interrupt):
            result = benchmark.supervise([sys.executable, "-c", "import time; time.sleep(60)"], self.root, self.limits)
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(result["termination"]["terminate_sent"])
        self.assert_reaped(result)


class ExtendedRunTests(unittest.TestCase):
    def test_cli_defaults_and_budget_overrides(self):
        for flags, expected in (([], (200, 200, False)),
                                (["--memory-budget-gib", "225", "--max-rss-gib", "224.5", "--continue-on-resource-limit"],
                                 (225, 224.5, True))):
            with self.subTest(flags=flags), patch.object(benchmark, "run", return_value=0) as run:
                self.assertEqual(benchmark.main(flags), 0)
                args = run.call_args.args[0]
                self.assertEqual((args.memory_budget_gib, args.max_rss_gib, args.continue_on_resource_limit), expected)
        self.assertEqual(benchmark.NATIVE_CONFIG["memory_budget_gib"], 200)
        self.assertEqual(benchmark.ResourceLimits().max_rss_bytes, 200 * benchmark.GIB)
        self.assertEqual(benchmark.ResourceLimits().min_mem_available_bytes, 20 * benchmark.GIB)
        self.assertEqual(benchmark.ResourceLimits().min_disk_free_bytes, 10 * benchmark.GIB)

    def test_invalid_budgets_do_not_start_runner(self):
        for flag, value in (("--memory-budget-gib", "0"), ("--memory-budget-gib", "-1"),
                            ("--memory-budget-gib", "nan"), ("--memory-budget-gib", "inf"),
                            ("--max-rss-gib", "0"), ("--max-rss-gib", "-1"),
                            ("--max-rss-gib", "nan"), ("--max-rss-gib", "inf"),
                            ("--max-rss-gib", "1e308")):
            with self.subTest(flag=flag, value=value), patch.object(benchmark, "run") as run, redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError, SystemExit)):
                    benchmark.main([flag, value])
                run.assert_not_called()

    def test_native_budget_rejections_and_ordinary_failures(self):
        diagnostics = [("Error: estimated working memory exceeds budget: 1000000 bytes", "resource_limit"),
                       ("Error: estimated artifacts exceed disk budget", "resource_limit"),
                       ("Error: invalid model weights", "failed"),
                       ("Error: estimated artifacts exceed disk budget\nError: unrelated failure", "failed")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            limits = benchmark.ResourceLimits(min_mem_available_bytes=0, min_disk_free_bytes=0,
                                              sample_interval_seconds=.01)
            for index, (diagnostic, expected) in enumerate(diagnostics):
                with self.subTest(diagnostic=diagnostic):
                    executable = root / f"fake-native-{index}"
                    executable.write_text(f"#!{sys.executable}\nimport sys\nprint({diagnostic!r}, file=sys.stderr)\nsys.exit(1)\n")
                    executable.chmod(0o700)
                    identity = {"seq_len": 64, "repeat": 1, "input_ids": [1] * 64,
                                "binary_path": str(executable), "binary": benchmark.file_identity(executable),
                                "model_dir": str(root), "native_config": dict(benchmark.NATIVE_CONFIG)}
                    row = benchmark.run_case(root / f"case-{index}", identity, limits)
                    self.assertEqual(row["status"], expected)
                    self.assertEqual(row["exit_code"], 1)
                    self.assertTrue(row["child_reaped"])
                    self.assertFalse(row["verified"])
                    self.assertNotIn("proof_size_mib", row)
                    if expected == "resource_limit":
                        self.assertEqual(row["reason"], diagnostic)

    def test_continue_policy_and_recorded_budgets(self):
        cases = [("resource_limit", True, 123, True, "verified", 2, 2),
                 ("resource_limit", False, 123, True, "verified", 1, 2),
                 ("resource_limit", True, 123, False, "verified", 1, 2),
                 ("resource_limit", True, None, False, "verified", 2, 2),
                 ("resource_limit", True, 123, True, "resource_limit", 2, 2),
                 ("resource_limit", True, 123, True, "failed", 2, 2),
                 ("resource_limit", True, 123, True, "interrupted", 2, 130),
                 ("failed", True, 123, True, "verified", 1, 1),
                 ("interrupted", True, 123, True, "verified", 1, 130)]
        for first_status, allow_continue, pid, reaped, second_status, expected_calls, expected_exit in cases:
            with self.subTest(status=first_status, allow=allow_continue, pid=pid, reaped=reaped), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = root / "model"
                model.mkdir()
                (model / "config.json").write_text("{}")
                output = root / "output"
                args = types.SimpleNamespace(output_dir=str(output), model_dir=str(model), binary=sys.executable,
                    lengths=[64, 128], repeats=1, resume=False, dry_run=False, sample_interval=.25,
                    min_disk_free_gib=10, memory_budget_gib=225, max_rss_gib=225,
                    continue_on_resource_limit=allow_continue)
                config = {"num_hidden_layers": 16, "vocab_size": 128256}
                inputs = {64: [1] * 64, 128: [1] * 128}
                calls = []
                def one_case(directory, identity, limits):
                    calls.append(identity["seq_len"])
                    self.assertEqual(identity["native_config"]["memory_budget_gib"], 225)
                    self.assertEqual(limits.max_rss_bytes, 225 * benchmark.GIB)
                    status = first_status if len(calls) == 1 else second_status
                    return {"seq_len": identity["seq_len"], "repeat": 1, "status": status,
                            "verified": status == "verified", "pid": pid, "child_reaped": reaped}
                with patch.object(benchmark, "build_inputs", return_value=(config, inputs, {"base_text_sha256": "text"})), \
                     patch.object(benchmark, "file_identity", return_value={"sha256": "file", "bytes": 1}), \
                     patch.object(benchmark, "environment", return_value={}), \
                     patch.object(benchmark, "run_case", side_effect=one_case), redirect_stdout(io.StringIO()):
                    self.assertEqual(benchmark.run(args), expected_exit)
                self.assertEqual(calls, [64, 128][:expected_calls])
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["manifest"]["native_config"]["memory_budget_gib"], 225)
                self.assertEqual(summary["manifest"]["resource_limits"]["max_rss_bytes"], 225 * benchmark.GIB)
                self.assertEqual(summary["manifest"]["resource_limits"]["min_mem_available_bytes"], 20 * benchmark.GIB)
                self.assertEqual(summary["manifest"]["continue_on_resource_limit"], allow_continue)
                self.assertEqual(summary["runs"][1]["status"], second_status if expected_calls == 2 else "not_run")
                self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
