# Validation record

Final functional acceptance **passed** with the explicitly approved original-
model arithmetic limitation described below. Both complete Qwen proofs and all
eight numerical lengths have finished. The original strict capability report
retains its arithmetic failure; it has not been rewritten as a pass. The final
audit is `artifacts/qwen/final-audit.json`.
All 35 final audit checks passed, including re-execution of both actual proof
inputs in ONNX: their MPFR-versus-ONNX logit errors were `1.11e-12` (seq2) and
`1.75e-12` (seq32), with identical top-1.

The frozen original and `pade-v2` graphs have passed full 24-layer numerical
checks for seq=2 and seq=32 against the local HF reference. The full-vocabulary
logit maximum errors were `4.17e-5` and `3.15e-5`; top-1 agreed at every position.
The corresponding approximate-versus-original maxima were `1.64e-12` and
`6.08e-12`. All declared nonlinear domains were checked on these inputs.

Detailed source reports are in `artifacts/qwen/onnx-final-seq2` and
`artifacts/qwen/onnx-final-seq32`. The implemented ONNX generator answered
“中国的首都是哪里？请简短回答。” with “北京。” and stopped on EOS;
`artifacts/qwen/first-generation.json` records the complete token sequence.

Four Rust v2 integration tests pass, including the complete tiny protocol,
malformed-token/nonfinite rejection, authenticated artifact mutation, and
rejection of changed primary witness, token statement, logits, public embedding,
and secondary opening. The tiny fixture is stored in
`spain/tests/fixtures/onnx_v2` for offline tests.

The first real seq=2 proof exposed a constant-folding precision issue during
the exact residual check. Its failed log is retained at
`artifacts/qwen/proof-final-seq2.time`. Compiler v3 fixes this by folding only
exactly representable operations, materializing composite affine values, and
preserving nonbinary divisions as constraints.

The complete v3 seq=2 proof **verified** at the original scale and threshold:

| Measurement | Result |
|---|---:|
| Constraints | 14,674,643 |
| Primary witness values | 15,348,958 |
| Secondary witness values | 260,640 |
| Checked residual norm | 9.6451e-19 |
| End-to-end time | 198.05 seconds |
| MPFR forward time | 32.90 seconds |
| Peak whole-process RSS | 11,685,292 KiB |
| Prover communication, including public statement | 38,444,365 bytes |
| Prover communication, excluding public opening | 12,462,196 bytes |

The statement, full primary witness, transcript and successful report are in
`artifacts/qwen/proof-final-v3-seq2`. The authenticated compiled artifact is
`artifacts/qwen/compiled-final-v3-seq2`; its fingerprint is
`3178fd5dbc84a17703dd6d2031ec50218f30ee6381459e431540fa9128e25a32`.
These are measurements on this CPU environment, not paper benchmark claims.

The complete v3 seq=32 proof also **verified**, using the same scale, threshold
and 256 DARK chunks:

| Measurement | Result |
|---|---:|
| Constraints | 240,288,881 |
| Primary witness values | 252,916,816 |
| Secondary witness values | 3,479,040 |
| Checked residual norm | 3.8627e-18 |
| End-to-end time | 3,415.42 seconds |
| MPFR forward time | 543.12 seconds |
| Peak whole-process RSS | 178,420,004 KiB (170.15 GiB) |
| Protocol messages | 184 |
| Prover communication, including public statement | 217,094,560 bytes |
| Prover communication, excluding public opening | 160,673,038 bytes |

The complete report and retained witness are in
`artifacts/qwen/proof-final-v3-seq32`; the circuit fingerprint is
`7a4de71fd9135fb1b997f0dc076cd8d4e42e8c767ebd4daa39746c282afaedb0`.
The original executable and runtime/compiler/exporter sources used by both
complete proofs are retained at `artifacts/qwen/runtime-v3-snapshot`. A subsequent
driver-only provenance fix hashes `/proc/self/exe` on Linux, so a concurrent
build cannot make a long-running process report the hash of a replacement file.
That change does not alter model execution or the proof protocol; the full-model
measurements above refer to the retained executable, not a later build.

HF/ORT and native MPFR validation have passed all eight required sequence
lengths. Every native comparison covers embedding, all 24 hidden states, final
normalization, all-position vocabulary logits and last-position logits. All
positions have identical top-1 and zero margin violations.

| Length | Approximate ONNX vs HF: logit max abs | MPFR vs ONNX: all-tensor max abs | MPFR seconds |
|---:|---:|---:|---:|
| 1 | 2.6620e-5 | 2.7534e-12 | 26.278 |
| 2 | 2.7573e-5 | 2.7534e-12 | 37.911 |
| 8 | 1.3530e-5 | 2.7534e-12 | 132.680 |
| 32 | 1.6721e-5 | 3.4275e-12 | 609.013 |
| 63 | 1.6721e-5 | 4.3823e-12 | 1,596.482 |
| 64 | 1.6721e-5 | 4.3823e-12 | 1,595.632 |
| 65 | 1.6721e-5 | 4.3823e-12 | 1,925.871 |
| 128 | 1.6721e-5 | 5.6684e-12 | 3,217.173 |

These times include contention from parallel validation/proof jobs. All native
runs used the same executable snapshot. Source reports are
`artifacts/qwen/acceptance-validation.json` and
`artifacts/qwen/native-validation/validation.json`. The separate
`native-validation/weights-postrun-authentication.json` authenticates the shared
weights, graph references, executable and completed report after the run; it is
explicitly not represented as the historical validator source used by old rows.

The fixed arithmetic
prompt produced **52** for **17 + 25** in both the ONNX implementation and HF,
with 100% greedy token agreement. This is a recorded model capability failure;
the original strict validation JSON retains its failed expected-answer gate.
The user explicitly approved faithful reproduction with this arithmetic error
recorded as a known limitation. The final audit can apply only this narrow
exception with `--accept-reference-arithmetic-error`; numerical, greedy,
readability and proof requirements remain unchanged.

All six frozen prompts completed on EOS, with 63/63 greedy tokens matching HF.
Maximum logit error across generation steps was `1.9825e-5` and maximum row RMSE
was `4.2703e-6`. The responses were a Chinese self-introduction, “北京。”, “Paris”,
“52”, “早上好”, and a coherent description of a cat beside a window. The final
prompt requested two sentences, while both implementations produced one; this
is another observed instruction-following limitation of the checkpoint.

## Official GPT-2 compatibility regression

Both official artifacts from Zenodo record 21045357 were downloaded and matched
their publisher MD5 values. The tests retained their 12-layer architecture,
final-position output and the repository's original clip-Erf/rational-Softmax
approximation. Numerical comparisons cover every one of the 50,257 output logits.

| Comparison: maximum absolute difference | seq=2 | seq=32 |
|---|---:|---:|
| Exact MPFR versus exact FP64 | 1.71e-13 | 2.70e-13 |
| Artifact approximate MPFR versus corresponding ORT | 9.95e-14 | 2.56e-13 |
| Standard lowering versus actual legacy operator wiring | 5.68e-14 | 9.95e-14 |

The full seq=2 proof verified at scale `2^75`, epsilon `2^-40`, passes=1 and
256 DARK chunks, with 1,068,300 constraints. It took 39.15 seconds end to end,
including 10.51 seconds for MPFR forward, with peak RSS 2,186,016 KiB. The report
and transcript are at `artifacts/gpt2-regression/proof-seq2-v3`; the full
regression report is `artifacts/gpt2-regression/report.json`.

This is a compatibility regression of the original artifact approximation.
Its errors relative to the exact GPT-2 graph are separately reported, including
maxima 5.57 (seq2) and 35.58 (seq32); those are not executor errors and are not
presented as Qwen accuracy measurements. Download URLs, MD5/SHA256 and derived
graphs/weights remain available; the original large downloads were removed
after the completed tests to meet the disk budget.

## Regression tests

- Compiler: 7 tests, including exact constant arithmetic and fixed-point residual.
- Generic MPFR executor: 11 tests, including mapped nonfinite values and domains.
- ONNX v2 protocol integration: 4 tests.
- Existing public-statement binding: 12 tests.
- Existing synthetic protocol test: passed.
- Existing Llama frontend: 11 unit and 8 integration tests passed.

Logs are retained under `artifacts/qwen` and `artifacts/gpt2-regression`.
The final driver-only executable-hash fix also passed a full tiny CLI proof,
with its actual executable hash checked against the written manifest
(`artifacts/qwen/final-driver-tiny`).

## Resource accounting

The final conservative count of new physical disk blocks is **28.9474 GiB**,
below the 30 GiB budget at delivery. The count deduplicates hard links by device and
inode, includes generated artifacts, new/updated build caches, Python packages,
source files and task temporary files, and excludes existing checkpoint/environment
files. Native raw comparison outputs have been removed after successful checks.
The complete method and observations are in `artifacts/qwen/resource-audit.json`.

The initial artifact-only count missed newly generated Rust caches. Once those
were included, the conservative count reached **30.6124 GiB**, exceeding the
30 GiB limit during execution. This historical overrun is retained explicitly;
the delivery does not claim that disk use stayed below the limit throughout.
Two audited cleanups removed only newly created, reproducible build caches while
preserving the running executables and proof evidence. The largest proof's peak
RSS was **170.1546 GiB**, measured for its whole process including both protocol
actors and forward execution; this is not a machine-wide memory measurement.
