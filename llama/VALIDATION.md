# Llama-3.2-1B validation record

Measured on 2026-10-18 with the local checkpoint
`/home/yliangh/models/meta-llama/Llama-3.2-1B`: 1,235,814,400 parameters,
16 decoder layers, hidden width 2048, intermediate width 8192, 32 query heads,
8 KV heads, and 128256 vocabulary entries. The checkpoint's original BF16
values are decoded exactly; all layers and the tied vocabulary projection run.

The independent reference is Transformers 5.20.1, FP32 eager attention. The
implemented generation path uses FP64 and the native witness path uses MPFR-128.
Both implemented paths use the bounded rational exponential described in
[NUMERICS.md](NUMERICS.md). Commands and environment setup are in
[README.md](README.md).

## Numerical and text acceptance

| Check | Measured result | Required limit |
| --- | --- | --- |
| Full-layer comparison at lengths 1, 2, 8, 32, 128 | All five cases pass | Every decoder layer and final logits checked |
| Worst logit absolute error across these lengths | 1.378449e-5 | 0.01 |
| Worst per-position logit RMSE across these lengths | 4.770385e-6 | 0.001 |
| Five 64-token continuations | 320 of 320 same-prefix greedy choices match HF | At least 99% for each prompt |
| Worst generation logit absolute error | 3.819486e-5 | 0.01 |
| Worst generation logit RMSE | 1.120169e-5 | 0.001 |
| Native MPFR-128 vs implemented FP64 logits, 10-token input | 6.679102e-13 | 1e-7 |
| Native MPFR-128 vs FP64, maximum over all 16 layer arrays | 1.165290e-12 | 1e-7 |

The text suite took 1220.63 seconds while other task processes shared the CPU.
It is a correctness measurement, not a controlled throughput benchmark. The
initial run used a 95% greedy-agreement gate; every recorded token was then
independently recounted against the final 99% gate, without rerunning or selecting
different prompts. All five prompts have 100% agreement.

Examples from the implemented generator:

> The capital of France is Paris. It is the most populous city in France and the
> country’s capital and largest city.

> Once upon a time, there was a little girl who loved to play with her dolls.
> She had a lot of dolls, and they were all very pretty. One day, she decided to
> take her dolls out for a walk.

The water-temperature prompt produces a readable explanation beginning with
0 degrees Celsius. The hot/cold and French-flag prompts answer correctly but
repeat during continued greedy decoding, identically to HF. These are base-model
completions; this test is not an instruction-following or factual-accuracy benchmark.
The complete texts and every compared step remain in the report.

Evidence under `artifacts/llama/`:

- `validation.json` and `validation-summary.json`: all reference comparisons,
  generated text, per-step token IDs, acceptance gates and the final gate audit.
- `native_comparison.json`: high-precision comparison and exact equality recheck
  of the final run's inputs, logits, layer arrays and checkpoint identity.
- `real-proof-final/forward.json`: the final high-precision forward, measured at
  90.12 seconds. Its numerical arrays exactly match the independently checked run.

## Complete-model Spain proof

The final native executable completed the real interactive Spain protocol with
exit status zero. Both actors completed and the verifier accepted. This proves
one full forward pass for the 10-token input `Once upon a time, there was a little`
(including BOS), through all 16 decoder layers, final RMSNorm, and all 128256
last-position logits. It is separate from the 320-token generation comparison.

| Measurement | Result |
| --- | --- |
| Constraints | 38,438,816 |
| Primary witness values | 42,128,606 |
| Secondary witness values | 417,889 |
| Public values / random values | 148,779 / 542,976 |
| Fixed-point scale / epsilon | `2^75` / `2^-40` |
| Verified residual norm | `1.5133455523e-18` |
| DARK chunks / folding rounds | 256 / 26 |
| Interactive messages | 172 |
| Serialized communication, including public statement | 50,521,082 bytes |
| Protocol wall time | 467.25 seconds |
| Peak process RSS | 31.00 GiB |

Protocol time excludes native forward execution and initial circuit/witness
construction. Resource estimates are heuristics; actual peak RSS was measured
and stayed within the 200 GiB budget. The raw i128 primary witness is about
643 MiB. The native I-R1CS frontend projects fixed weights rather than expanding
all billion-plus coefficients into an ONNX/COO artifact.

The final record is `artifacts/llama/real-proof-final/proof_report.json`, with
`statement.json`, `transcript.jsonl`, `circuit.json`, `primary_witness.i128le`
and `implementation_manifest.json` alongside it. The manifest records the frozen
native executable, Rust sources and Cargo lockfile used for this run.
Earlier incomplete runs under other directories are not proof evidence.

These results retain the upstream research artifact's deterministic benchmark
RSA group, as explained in [README.md](README.md). They establish full model
execution and successful protocol execution with the stated parameters, not a
production cryptographic security assessment.

## Regression coverage

Thirty-nine Rust tests and twelve Python tests pass, plus the actual small-model
interactive proof with 256 DARK chunks. The Rust tests include every native
operator, causal masking, token/model/output tampering, public and secondary
witness binding, numerical overflow, malformed sum-check rounds, and DARK
integer/field opening consistency, bounds and final-round placement.

Logs are in `artifacts/llama/test-logs/` and `artifacts/llama/tests-python.log`.
Native stale-report protection and the forward resource-budget guard were also
checked. Existing ONNX tests requiring absent `circuit/export` fixtures were not
run; the old actor smoke test skips without those fixtures, and the GPT2 test
was already ignored. These skipped cases are not counted as successful tests.
