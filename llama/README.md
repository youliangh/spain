# Llama 3.2 inference and Spain proofs

This frontend runs the complete local Llama model and proves a forward pass through
every decoder layer, final RMSNorm, and every vocabulary logit at the last input
position. The supplied checkpoint is the **base** Llama-3.2-1B model: use ordinary
text completion prompts. No layer truncation, weight quantization, or Hugging Face
forward fallback is used in the implemented inference path.

The Python implementation uses binary64 tensor kernels for iteration and generation.
The native implementation uses MPFR with 128-bit precision, including accumulation
in every matrix product, to produce Spain's witness. Both use the same bounded
Padé exponential and stable SiLU. Hugging Face FP32 eager execution is the independent
quality reference. Batch size is one; generation recomputes the complete causal
prefix without a KV cache.

## Environment and commands

The default checkpoint is `/home/yliangh/models/meta-llama/Llama-3.2-1B`.
All model loading is local. The model directory is read-only input.

```bash
# Python 3.14 on the validation machine. This creates a project-local environment.
bash tools/setup-llama-python.sh

# Alternatively seed an independent copy from an existing environment of the same
# Python version; the seed environment is not modified.
bash tools/setup-llama-python.sh --seed-venv /path/to/existing/venv

.venv/bin/python tools/llama.py generate \
  --prompt "The capital of France is" --max-new-tokens 64 \
  --output artifacts/llama/generation.json

.venv/bin/python tools/llama.py reference \
  --prompt "The capital of France is" --max-new-tokens 64 \
  --output artifacts/llama/reference.json

.venv/bin/python tools/llama.py validate \
  --max-new-tokens 64 --output artifacts/llama/validation.json
```

The Rust workspace requires nightly, GCC, make, m4 and protoc. The validation build
uses Rust `1.101.0-nightly (8006091eb 2026-10-17)`. `tools/cargo-local.sh` keeps Cargo,
Rustup and GMP caches under `.cache/`; it also accommodates locally extracted
m4/protoc under `.cache/native/usr/`. On GCC 15 the wrapper selects GNU C11 for
GMP's older configure tests. Install a project-local nightly before the first build:

```bash
CARGO_HOME="$PWD/.cache/cargo" RUSTUP_HOME="$PWD/.cache/rustup" \
  rustup toolchain install nightly --profile minimal --no-self-update
bash tools/cargo-local.sh build --release -p llama-spain

.venv/bin/python tools/llama.py prove \
  --prompt "Once upon a time, there was a little" \
  --output-dir artifacts/llama/proof
```

For explicit token IDs, use `--input-ids path.json` with a JSON array or an object
containing `input_ids`. This input is already tokenized: no additional BOS is added.
For high-precision execution without a proof:

```bash
target/release/llama-spain forward \
  --input-ids artifacts/llama/proof/input_ids.json \
  --output-dir artifacts/llama/high-precision
.venv/bin/python tools/llama.py generate --max-new-tokens 1 \
  --logits-json artifacts/llama/high-precision/forward.json \
  --output artifacts/llama/native-comparison.json
```

## Proof statement and numerical semantics

The frontend follows [Spain](https://www.usenix.org/system/files/osdi26-destefano.pdf),
especially approximate R1CS, I-R1CS and the Freivalds treatment of matrix products.
It uses the existing Spain sum-check and DARK implementation.

Fixed weights are projected after the primary witness commitment. For an HF
weight matrix stored as `[output,input]`, the verifier independently derives
`v = W^T r` and checks `X v = Y r`. Variable attention products have explicit
secondary projection/product constraints. The verifier builds its graph from the
local model configuration; it does not execute the model or accept a prover-defined
computation graph.

The public statement includes the model/config/tokenizer fingerprint, exact token
IDs, independently looked-up embeddings, and the complete claimed logit vector.
The logit claim is fixed before challenges, and logits also occur in the committed
primary witness. Public openings and the sampled randomness are checked explicitly.
Secondary witness values are disclosed and fixed before selecting the sum-check
field, and their final openings must match. This preserves the ordering required
by I-R1CS. This implementation does not claim zero knowledge.

The default denominator is `D = 2^75`, and the accepted residual norm is
`epsilon = 2^-40`. The verifier checks the integer squared residual against this
bound; I512 overflow, negative errors, and nonfinite parameters fail closed.
This residual bound is separate from approximation error relative to exact
transcendental functions, and from the empirical logit differences against HF.
See [NUMERICS.md](NUMERICS.md) for the approximation analysis.

The native CLI defaults to `--dark-chunks 256`, a power-of-two partition of the
complete primary witness commitment. For the 10-token validation input this
reduces the setup table from 4,194,304 bases (16 chunks) to 262,144 bases; all
witness values and all 26 DARK folding rounds remain present. The chunk count
is recorded in the proof report.

## Artifacts and acceptance

`prove` writes:

- `forward.json`: all last-position logits and all decoder layer hidden states,
  projected to binary64 for comparison, plus full graph coverage and timings.
- `statement.json`: model file hashes, token IDs, positions, exact scaled-integer
  logits, and fixed numerical parameters.
- `primary_witness.i128le`: signed little-endian i128 primary witness values.
- `circuit.json` and `resource_estimate.json`: dimensions, encoding and budget estimates.
- `transcript.jsonl` and `proof_report.json`: actual interactive messages, completion
  status, communication volume, integer residual, timings and peak RSS.

Use a fresh output directory for each proof. Existing proof artifacts are rejected
before execution so a failed rerun cannot retain a previous run's success report.

The transcript is a record of an interactive run, not a separately implemented
noninteractive proof format. Success is reported only after both protocol actors
complete and the verifier accepts. Text generation and a proof of one full forward
pass are separate acceptance requirements; this does not prove every generated token.

The proof backend retains Spain's research-artifact parameters, including its
deterministically generated benchmark RSA group. These experiments establish model
computation and protocol execution; they are not a production cryptographic security
assessment. See the research warning in the repository's main README.

The default resource estimates are checked against 200 GiB RAM and 60 GiB additional
artifact space; budgets are configurable. Full ONNX/COO weight expansion is not
required by this native frontend. Large diagnostic artifacts are permitted within
the machine's resource budget. `/tmp` is memory-backed on the validation machine,
so large proof artifacts should stay under the project directory.

```bash
.venv/bin/python -m unittest discover -s tools/tests -v
bash tools/cargo-local.sh test --release -p llama-spain -- --test-threads=1
bash tools/cargo-local.sh test --release -p spain --test statement_binding -- --test-threads=1
target/release/llama-spain tiny-test --output-dir artifacts/llama/new-tiny-proof
```

The small fixture tests exercise the same operators and proof backend. Final model
acceptance additionally requires the real checkpoint's full-layer numerical and
text suite and an actual complete-model Spain proof; tiny-model results alone do
not establish that acceptance.

The completed local-checkpoint results, generated text examples, full proof
measurements and test coverage are recorded in [VALIDATION.md](VALIDATION.md).

Sequence-length performance measurements and reproducible commands are described
in [BENCHMARKS.md](BENCHMARKS.md#consolidated-results), with a consolidated table
for input lengths 1, 2, 8, 32, 64, and 128. The five tested lengths up to 64 verified
successfully; length 128 was rejected by the native memory preflight. The schema-2
native report records separate
prover/verifier active times, whole-process peak RSS, and both full prover-message
size and proof size with public openings excluded. The benchmark runner validates
these counters against the saved transcript and writes JSON/CSV summaries.
