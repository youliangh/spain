# Llama sequence-length benchmarks

This report collects the sequence-length tests for **1, 2, 8, 32, 64, and 128
input tokens, including BOS**, with one attempt per length. The five tested
lengths up to 64 completed native Spain prove/verify successfully; length 128 was rejected by the
native memory preflight before forward execution. Successful runs used the local
Llama-3.2-1B checkpoint, batch size one, all 16 decoder layers, final RMSNorm, and
all 128,256 logits at the last input position. Each measures a complete causal
prefix without a KV cache; it is not a per-token generation throughput test.

The fixed configuration is 8 Rayon threads, MPFR 128-bit arithmetic, denominator
`D = 2^75`, residual tolerance `epsilon = 2^-40`, and 256 DARK chunks. BF16
checkpoint values are loaded exactly. The numerical approximation and residual
bound have different meanings; see [NUMERICS.md](NUMERICS.md).

## Consolidated results

All attempts took place on **2026-10-18 (America/New_York)**. Values below are
rounded to two decimals; each successful row is a single observation.

| Input length | Status | Peak RSS (GiB) | Prover (s) | Verifier (s) | Full proof (MiB) | Excluding public (MiB) | Child elapsed (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Verified | 5.66 | 71.48 | 21.18 | 19.98 | 3.16 | 96.21 |
| 2 | Verified | 8.47 | 125.04 | 29.05 | 21.92 | 5.08 | 157.84 |
| 8 | Verified | 26.93 | 453.01 | 83.33 | 33.46 | 16.25 | 541.58 |
| 32 | Verified | 106.24 | 1847.98 | 319.79 | 89.80 | 71.11 | 2180.28 |
| 64 | Verified | 211.74 | 3747.33 | 677.28 | 189.12 | 168.50 | 4446.88 |
| 128 | Native memory preflight rejected | — | — | — | — | — | — |

Peak RSS covers the joint native process, including forward execution, prover,
verifier, and serialization. Prover time includes the forward pass and witness
construction; verifier time includes its own setup and protocol operations.
The party totals exclude shared model loading and explicit artifact I/O; child
elapsed measures the complete native process lifetime. Full proof counts every
prover-to-verifier JSON payload, including public and secondary openings.
The discounted column removes only the final public opening range. See
[Measurement definitions](#measurement-definitions) for the exact accounting.

The dashes for length 128 mean **no completed measurement**. Its native memory
estimate was **416.21 GiB**, exceeding both the configured 225 GiB budget and
the host's 247.4 GiB physical RAM. This estimate is not an observed memory peak.

The largest verified case, length 64, completed in **74.11 minutes** of native
process lifetime; its lowest sampled host `MemAvailable` was **31.03 GiB**.
Compared with length 32, its observed peak RSS, prover time, verifier time, and
full proof size were **1.99x, 2.03x, 2.12x, and 2.11x**, respectively. These are
single-run comparisons with the cache and budget differences recorded below;
they do not establish variance or an asymptotic scaling law.

Full-precision data for all six lengths are in the
[combined CSV](../artifacts/llama/sequence-benchmark-combined.csv) and
[combined JSON](../artifacts/llama/sequence-benchmark-combined.json). They retain
the source-summary paths, per-series memory limits, and loaded cache size.
Complete-proof metrics are empty for length 128.

The linked CSV/JSON evidence is included in Git. Other recorded local artifacts,
including witnesses, transcripts, resource samples, and DARK caches, are not
included; paths to those artifacts in the records refer to the benchmark host.

## Initial series: environment and cache

The initial series covered lengths 1, 2, 8, and 32 and started on
**2026-10-18 at 16:37:12 UTC** (12:37:12 EDT), from the manifest's
`started_unix_seconds`. All four fresh runs completed with exit code zero and
`verified: true`. Original observations and parameters remain in
[summary.csv](../artifacts/llama/sequence-benchmark/summary.csv) and
[summary.json](../artifacts/llama/sequence-benchmark/summary.json).

The recorded environment provides **8 available CPUs** (affinity 0–7) on an
AMD EPYC 9634 84-Core Processor host, with **247.4 GiB RAM**. It uses x86-64
Linux `7.0.0-38-generic`, glibc 2.43, and Python 3.14.4. The source baseline was
commit `f4042052cf712dc86eafa09cca2a4a7bc4c10a10`, with the benchmark
instrumentation in this change. The fixed native executable's SHA-256 is
`80d20d250106e027ff709663fd8fc58495588281730ba551460320c152ca3f35`;
the manifest also records the checkpoint file hashes.

[Cache preparation](../artifacts/llama/sequence-benchmark/cache-preparation.json)
took **162.67 seconds**, separately excluded from every table entry. It extended
an existing 262,144-base table to 1,048,576 bases. All four runs recorded the same
unchanged `dark_pip_base_20_30000` cache before and after execution. Single
observations do not establish variance or an asymptotic scaling law.

## Extended series: lengths 128 and 64

The extension started on **2026-10-18 at 17:38:45 UTC** (13:38:45 EDT), attempting length 128
first, then length 64, once each. It uses the same native executable SHA-256,
checkpoint file hashes, eight-thread numerical configuration, and token-prefix
construction as the initial series. The original inputs match prefixes of the
extended inputs, and the length-64 input is a prefix of the length-128 input.

The native estimated-memory budget and supervised RSS limit are each raised
from the default 200 GiB to **225 GiB** for this extension. The independent
**20 GiB minimum host `MemAvailable`** and 10 GiB minimum free disk space remain
active. Native memory estimates and observed process RSS are separate checks;
the host-availability guard can stop a process before it reaches the RSS limit.

Length 64 completed the full model proof and verification with native exit code
zero. The extension summary has `complete: true` and `passed: false`, and the
runner exits with code 2 because length 128 was resource-limited. Length 64's
own status remains `verified: true`.

For length 128, the native static estimate is **416.21 GiB**
(`446899900840` bytes), above the 225 GiB budget. The process exited with code 1
before forward execution or proof construction. This is an estimate, not a
measured peak. The raw attempt's roughly 0.25-second supervisor duration and
single low startup-RSS sample describe a rejected attempt; they are not complete
model or proof performance and do not supply the table's missing measurements.

The extension's [cache preparation](../artifacts/llama/sequence-benchmark-extended/cache-preparation.json)
extended the initial 1,048,576-base table to **2,097,152 bases**
(`dark_pip_base_21_30000`) in **220.527 seconds**, excluded from case timings.
This prepares the length-64 padded witness (`2^29` values with 256 chunks).
Length 128 would require 4,194,304 bases if it passed preflight; this attempt
never reached cache loading or extension. The initial four measurements retain
their original 1,048,576-base cache conditions.

Raw status and evidence: [summary.json](../artifacts/llama/sequence-benchmark-extended/summary.json),
[summary.csv](../artifacts/llama/sequence-benchmark-extended/summary.csv),
[resource_preflight.json](../artifacts/llama/sequence-benchmark-extended/resource_preflight.json),
the [length-64 proof report](../artifacts/llama/sequence-benchmark-extended/length-0064/repeat-001/attempt-0001/proof_report.json),
the [length-64 attempt record](../artifacts/llama/sequence-benchmark-extended/length-0064/repeat-001/attempt-0001/run.json),
and the [length-128 attempt record](../artifacts/llama/sequence-benchmark-extended/length-0128/repeat-001/attempt-0001/run.json).
The structural counts and extrapolations in `resource_preflight.json` are planning
data, separate from the completed length-64 measurement above.

## Inputs and execution

[tools/benchmark_llama.py](../tools/benchmark_llama.py) tokenizes its fixed English
passage with the checkpoint's local `tokenizer.json`, including the configured
initial BOS token, and takes exact prefixes of the resulting token IDs. Length
includes BOS: the length-1 input contains only BOS. The native executable receives
those IDs directly and inserts no additional tokens. The manifest saves the
passage, token IDs, hashes, checkpoint identity, binary identity, parameters,
CPU affinity, and machine information.

Cases run serially in separate native processes. Prover and verifier run inside
the same process and exchange typed messages in memory. Messages are also
serialized to JSON for transcript recording and byte accounting. There is no
network transport or incoming-message deserialization in this experiment.

Only a native process that exits successfully, reports `verified: true` with
`measurement_schema_version: 2`, and passes the runner's artifact/configuration
checks contributes a successful measurement. The runner independently recounts
message directions and payload bytes from the transcript. Failed, interrupted,
or resource-limited attempts retain their status and logs; missing proof metrics
remain empty. By default, the runner stops after an unsuccessful attempt.
`--continue-on-resource-limit`, enabled for the extension, permits later cases
after a resource-limited attempt only when any launched child has been reaped.
Other failures and interruptions still stop the series. Each subsequent case
checks available resources before launching, and the runner retains a nonzero
exit status if any case was resource-limited.

## Measurement definitions

All time fields are elapsed wall-clock seconds, not CPU seconds. The two party
totals include their assigned frontend work and outgoing JSON serialization:

```text
prover_seconds = forward_seconds
               + prover_claim_seconds
               + prover_circuit_setup_seconds
               + prover_state_setup_seconds
               + prover_protocol_seconds
               + prover_serialization_seconds

verifier_seconds = verifier_layout_seconds
                 + verifier_circuit_setup_seconds
                 + verifier_state_setup_seconds
                 + verifier_protocol_seconds
                 + verifier_serialization_seconds
```

Prover frontend work includes the high-precision forward pass, quantization of
the logit claim, and construction of its circuit/witness and protocol state.
Verifier frontend work includes its independently derived graph layout and its
circuit/protocol-state construction. Protocol time covers the corresponding
actor's message handlers; verifier time also includes `init_message`. Circuit
setup includes each executor's model-identity check. The shared model loading
and initial manifest/fingerprint hashing in `main` are outside the party timers.
Explicit artifact writes, transcript accounting, network transport, and incoming
message deserialization are also outside these totals. Reading/deserializing an
existing DARK cache inside a timed party operation remains included.

| Field | Meaning |
| --- | --- |
| `prover_seconds`, `verifier_seconds` | The component sums above; the primary party-time measurements. |
| `forward_seconds` | Native MPFR forward execution before circuit construction. |
| `proof_wall_seconds` | Protocol-loop elapsed time starting before verifier initialization; includes transcript handling, but excludes the preceding forward/circuit/state setup. |
| `end_to_end_seconds` | Native `run` elapsed time through report construction, including shared setup and preceding artifact writes. |
| Runner `elapsed_seconds` | Parent-observed native process lifetime, including startup and shutdown. |
| `peak_rss_kib` | Native process high-water RSS from Linux `VmHWM`, in KiB. |
| Runner `peak_rss_gib` | Maximum of sampled direct-process RSS, observed `VmHWM`, and the native final high-water mark, divided by `2^30`. |
| `proof_size_bytes` | All prover-to-verifier JSON payload bytes, including public and secondary openings; equals `prover_message_bytes`. |
| `proof_size_excluding_public_bytes` | SpainBroker convention: remove range 0 from the final `WitnessOpenings` message and reserialize it, leaving every other prover payload intact. |
| `communication_bytes_including_public_statement` | Sum of both directions' serialized payload bytes, including verifier challenges. |

Peak RSS covers the **whole native process**, including forward execution,
prover, verifier, and serialization. It cannot be attributed separately to the
prover or verifier. The Python supervisor is outside this process measurement.
The sampled RSS is supplementary to Linux's high-water mark, which captures
peaks between samples for successful native runs.

Proof sizes exclude JSONL newline delimiters and standalone artifact files such
as `primary_witness.i128le`. The discounted convention removes only the final
public range, including the randomness stored in that range. Secondary values
sent in both the bound-error claim and final opening are counted on both
transmissions; neither byte measure deduplicates them. Reported MiB and GiB use
`2^20` and `2^30` bytes, respectively.

The older `prover_statistics` and `verifier_statistics` debug actor totals omit
parts of frontend work and serialization. They must not replace schema-2 totals
or be treated as comparable end-to-end measurements.

## Shared cache preparation

For the initial series, prepare a DARK table for a padded witness of `2^28`
values and 256 chunks. This requires `2^28 / 256 = 1,048,576` bases. Its four
cases reuse the prepared cache; cache preparation time is recorded separately
and excluded from each case. The runner neither prepares nor deletes caches.

The standard build stores these tables under `dark/dark_pip_base_*_30000`.
The cache loader selects the smallest existing table large enough for a case,
or extends a smaller table. For the initial series the shared table was
`dark/dark_pip_base_20_30000`. Check the preparation report, each case's
`dark_cache_before`/`dark_cache_after`, and the cache-selection lines in
`native.log` when reproducing the shared-table setup. Other existing tables can
affect which table is selected. Preparation reports both required and actually
loaded bases; a larger existing table may be loaded in full.

These are measurements with a precomputed DARK table. Its per-process loading
cost is still part of the operation that loads it; they are not measurements
with all setup costs removed, nor cold-cache setup measurements.

## Reproduction

Run from the repository root with the local model available. See
[README.md](README.md) for the Rust/Python environment setup. Build once before
the series and keep that binary and checkpoint fixed:

```bash
bash tools/cargo-local.sh build --release -p llama-spain

target/release/llama-spain prepare-cache \
  --witness-log2 28 --dark-chunks 256 --threads 8 \
  --output artifacts/llama/sequence-benchmark/cache-preparation.json

.venv/bin/python -u tools/benchmark_llama.py \
  --model-dir /home/yliangh/models/meta-llama/Llama-3.2-1B \
  --binary target/release/llama-spain \
  --output-dir artifacts/llama/sequence-benchmark \
  --lengths 1 2 8 32 --repeats 1 --sample-interval 0.25
```

The runner fixes the eight-thread/numerical/chunk configuration above. Its
default native preflight budgets are 200 GiB memory and 60 GiB artifact space,
and its default supervised RSS limit is 200 GiB. While each native process runs,
it samples every 0.25 seconds by default and stops that child if RSS exceeds the
configured limit, host `MemAvailable` falls below 20 GiB, or free space on the
output filesystem falls below 10 GiB. These guards are sampled conditions, not
operating-system memory reservations.

The extension reuses the same native binary and runs with these commands:

```bash
target/release/llama-spain prepare-cache \
  --witness-log2 29 --dark-chunks 256 --threads 8 \
  --output artifacts/llama/sequence-benchmark-extended/cache-preparation.json

.venv/bin/python -u tools/benchmark_llama.py \
  --model-dir /home/yliangh/models/meta-llama/Llama-3.2-1B \
  --binary target/release/llama-spain \
  --output-dir artifacts/llama/sequence-benchmark-extended \
  --lengths 128 64 --repeats 1 --sample-interval 0.25 \
  --memory-budget-gib 225 --max-rss-gib 225 \
  --continue-on-resource-limit
```

The output root contains `benchmark_manifest.json`, `summary.json`, and
`summary.csv`, plus the separately created `cache-preparation.json`. Per-case
artifacts are placed in fresh directories of the form
`length-0008/repeat-001/attempt-0001/`. Each contains `run.json`, `native.log`,
`resource_samples.jsonl`, exact `input_ids.json`, and the native proof artifacts
described in [README.md](README.md#artifacts-and-acceptance).

Use `--dry-run` with a different output root to prepare a manifest without
launching native cases; this does not prepare the DARK cache. Add `--resume` to
reuse a completed matching attempt.
Resume requires matching model/binary/configuration/token identities, successful
exit and verification, and validated artifact hashes. Other attempts use a new
directory. To preserve an earlier series as a separate experiment, choose a new
output root because its top-level manifest and summaries are regenerated.

Each length has one observation, so this series does not estimate run-to-run
variance or confidence intervals. Only verified measurements belong in a result
table; resource estimates, failed-attempt partial timings, and extrapolations
are not substitutes for measurements at another sequence length.
