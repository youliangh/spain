# Qwen3.5-0.8B Dense through the Spain ONNX pipeline

This extension exports the complete 24-layer text model from a local Hugging Face
checkpoint to standard ONNX, compiles that graph to interactive R1CS, and runs
Spain's existing commitment, sum-check and DARK protocol. It follows the ONNX
route in the [Spain paper](https://www.usenix.org/conference/osdi26/presentation/destefano).
The text stack has 18 GatedDeltaNet blocks and six full-attention blocks. Vision
and multi-token prediction weights are outside this text-only statement.

One proof covers one fixed token prefix, zero initial recurrent state, all 24
blocks, final RMSNorm and the last position's **248,320 vocabulary logits**.
This matches the output boundary of the official GPT-2 artifacts. `passes=1`
means one input, not one generated token. Greedy generation repeatedly evaluates
the same approximate ONNX computation with a growing prefix; it does not produce
an autoregressive proof or a proof of a KV cache.

## Environment

The tested checkpoint is `/home/yliangh/models/Qwen/Qwen3.5-0.8B`. It is read only.
Exported FP64 weights occupy 6,019,144,192 bytes, shared by hard links among all
graphs and compiled circuits. Keep the artifact directories on one filesystem.
CPU inference is supported; CUDA is not required.

Run commands from the repository root. The pinned Python packages require
Python 3.12 or newer; this workspace was tested with Python 3.14. Create `.venv`
first on a fresh checkout (`uv venv --python 3.14 .venv`). Rust builds require
nightly and the native build dependencies described in the
[main README](../README.md#getting-started).

```bash
uv pip install --python .venv/bin/python -r tools/requirements-qwen.txt
bash tools/cargo-local.sh build --release -p spain --bin onnx-spain
bash tools/cargo-local.sh build --release -p model --bin onnx-forward
```

`cargo-local.sh` uses the prepared toolchain in this workspace's `.cache`.
On another machine with an installed nightly toolchain, replace
`bash tools/cargo-local.sh` with `cargo +nightly` in build and test commands.

## Generate text and validate the implementation

```bash
.venv/bin/python tools/qwen.py generate \
  --prompt '中国的首都是哪里？请简短回答。' \
  --max-new-tokens 64 --output artifacts/qwen/rerun-generation.json

.venv/bin/python tools/qwen.py validate \
  --lengths 1 2 8 32 63 64 65 128 --max-new-tokens 64 \
  --output artifacts/qwen/rerun-validation.json
```

The implemented generator uses ONNX Runtime FP64 on the approximate graph.
Transformers is used by `reference` and `validate` as an independent FP32
reference. Both use the checkpoint's tokenizer and non-thinking chat template,
greedy selection, and configured end-of-turn tokens. There is no reference-model
fallback in the ONNX generator.

The frozen suite exposes a checkpoint arithmetic error: both HF and ONNX answer
`17 + 25` with `52`. The strict validation report deliberately keeps this
failure and exits with status 1. For this delivery the user approved recording
the original-model error while requiring faithful reproduction. The final audit
has an explicit exception for that exact sample; it still requires identical
HF tokens and the original numerical limits. See [VALIDATION.md](VALIDATION.md).

To rerun the native MPFR comparisons while preserving the supplied native logs,
create a fresh workspace sharing the existing immutable graphs and weights by
hard link. These commands assume the supplied artifacts, or the completed
validation command above, are available on the same filesystem:

```bash
mkdir artifacts/qwen-native-rerun
cp -al artifacts/qwen/weights artifacts/qwen/seq-*-approx-debug \
  artifacts/qwen-native-rerun/
.venv/bin/python tools/qwen_native_validation.py \
  --artifact-dir artifacts/qwen-native-rerun \
  --validation artifacts/qwen/rerun-validation.json --threads 4
```

Its report is `artifacts/qwen-native-rerun/native-validation/validation.json`.
Choose another fresh directory for a further independent run.

## Export, compile and prove

Supply the public token IDs explicitly as a JSON array or an object containing
`input_ids`. The benchmark name checks their length. These commands reuse the
supplied final circuits and token inputs, avoiding another 16 GiB seq32 COO copy.
Every proof run needs a fresh output directory and space for its new witness and
transcript; the seq32 primary witness alone occupies 3.77 GiB.

```bash
.venv/bin/python run.py qwen3.5-0.8b-seq-2 --passes 1 \
  --compiled-dir artifacts/qwen/compiled-final-v3-seq2 \
  --input-ids artifacts/qwen/onnx-final-seq2/input_ids.json \
  --output-dir artifacts/qwen/proof-seq2-run1

.venv/bin/python run.py qwen3.5-0.8b-seq-32 --passes 1 \
  --compiled-dir artifacts/qwen/compiled-final-v3-seq32 \
  --input-ids artifacts/qwen/onnx-final-seq32/input_ids.json \
  --output-dir artifacts/qwen/proof-seq32-run1
```

The Rust driver authenticates the entire selected circuit manifest before
execution. To build from the checkpoint in a fresh workspace, omit
`--compiled-dir`, supply your token JSON, and select a fresh `--artifact-dir`;
the first call exports and compiles, and later calls reuse that circuit. Add
`--model-dir` when the checkpoint is outside the tested default location.
To perform the same MPFR forward computation without running the proof, use:

```bash
.venv/bin/python tools/qwen.py forward \
  --compiled-dir /path/to/circuit --input-ids /path/to/token-ids.json \
  --output-dir artifacts/qwen/native-run1
```

The driver defaults to four CPU workers, MPFR-128, fixed-point scale `2^75`,
constraint residual limit `2^-40`, and 256 DARK chunks. Large proofs run
sequentially. The resource preflight defaults to 200 GiB RAM and 30 GiB of proof
artifacts; existing shared weight files must be counted once when measuring disk
use. See [NUMERICS.md](NUMERICS.md) for the computation and proof boundaries and
[VALIDATION.md](VALIDATION.md) for acceptance evidence.

Each successful proof run writes `forward.json`, `statement.json`,
`primary_witness.i128le`, `transcript.jsonl`, `implementation_manifest.json`,
`resource_estimate.json`, and `proof_report.json`. A report with `verified: true`
is written only after both protocol actors complete. The statement fixes the
input IDs, circuit fingerprint and all claimed logits before challenges are
sampled. The transcript records an interactive local run; it is not a
noninteractive proof format with a separate replay verifier.

After both real proofs and all validation lengths have completed, aggregate and
authenticate their evidence with:

```bash
.venv/bin/python tools/qwen_acceptance.py \
  --validation artifacts/qwen/rerun-validation.json \
  --native-validation artifacts/qwen-native-rerun/native-validation/validation.json \
  --proof-seq2 artifacts/qwen/proof-seq2-run1 \
  --proof-seq32 artifacts/qwen/proof-seq32-run1 \
  --accept-reference-arithmetic-error \
  --output artifacts/qwen/rerun-audit.json
```

For the retained delivery evidence instead, use `acceptance-validation.json`,
`native-validation/validation.json`, and `proof-final-v3-seq2` / `proof-final-v3-seq32`
under `artifacts/qwen`. The delivery's `final-audit.json` records the completed,
passing audit; [VALIDATION.md](VALIDATION.md) records the measurements and known
limitations.
For those retained runs, add
`--runtime-snapshot artifacts/qwen/runtime-v3-snapshot` to also authenticate the
executable and runtime/compiler/exporter source snapshot against both proof
manifests. A new run should use a snapshot of its own executable, if requested.

The audit checks the full vocabulary and layer boundary, parameters, artifact
hashes, retained witness sizes, and actual proof forward logits against ONNX.
The original strict report is preserved. Timings are observations from the
recorded CPU runs, including contention from parallel validation, rather than
isolated performance benchmark claims.

## Source map and tests

- `tools/qwen_onnx.py`: architecture export, shared weights, nonlinear lowering.
- `circuit/onnx_v2.py`: generic ONNX-to-R1CS compiler and authenticated layouts.
- `model/src/onnx_v2.rs`: generic MPFR execution with lazy mapped weights.
- `spain/src/onnx_v2.rs`: public statement binding and two-stage witness adapter.
- `spain/src/bin/onnx-spain.rs`: complete existing Spain protocol orchestration.

```bash
.venv/bin/python -m unittest discover -s tools/tests -p 'test_qwen_onnx.py'
.venv/bin/python -m unittest discover -s circuit/tests
bash tools/cargo-local.sh test --release -p model --lib onnx_v2
bash tools/cargo-local.sh test --release -p spain --test onnx_v2 -- --test-threads=1
```
