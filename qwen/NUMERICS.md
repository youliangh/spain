# Computation and numerical statement

The original ONNX graph is the dense text computation, without quantization or
layer truncation. BF16 checkpoint values are represented exactly in external
FP64 tensors. The LM head uses the token embedding weight, as specified by the
checkpoint's tied embeddings.

## Architecture details

Full-attention layers are 3, 7, 11, 15, 19 and 23 (zero based). Query projection
interleaves the query and output gate within each head. Query and key RMSNorm use
`1 + weight`. Partial RoPE rotates the first 64 dimensions with split halves and
base `1e7`; grouped-query attention repeats each of two KV heads four times.

The other 18 layers use a causal depthwise convolution of width four followed by
SiLU, Q/K L2 normalization, and gated delta recurrence. For each head:

```
beta_t = sigmoid(b_t)
alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias))
S'_t = alpha_t * S_(t-1)
U_t = beta_t * (v_t - k_t^T S'_t)
S_t = S'_t + k_t U_t^T
o_t = q_t^T S_t
```

Here `q_t` also includes the attention scale `1/sqrt(128)` after L2 normalization.

The exporter uses a mathematically equivalent triangular form for a static,
zero-state prefix. It computes `D[t,t]=1` and
`D[t,j]=alpha[t]*D[t-1,j]`, then solves the unit-triangular update by forward
substitution. It never divides by a product of small decay factors. Output is
`(D * causal(Q K^T)) U`, followed by gated RMSNorm and the output projection.
Gated RMSNorm uses the stored weight directly; the block/final RMSNorms use
`1 + weight`.

## Nonlinearities and arithmetic

The proof graph uses standard ONNX algebraic operators with declared input
domains. Its `pade-v2` profile uses range-reduced rational exponential,
20-term softplus series, bounded sigmoid doubling, and fused SiLU. Sqrt and Div
have explicit algebraic constraints. Square roots are constrained to be
nonnegative; domain membership is also constrained. Unsupported operations,
nonfinite values, invalid shapes, and invalid token IDs fail closed.

A Div equation `denominator * output = numerator` alone does not prove a
generic denominator is nonzero. In this Qwen graph, the relevant definitions
provide that property: normalization adds a positive epsilon, bounded Padé
denominators stay positive, sigmoid doubling uses a sum of two squares, and
softmax includes the shifted maximum's unit exponential term. The runtime also
rejects division by zero. These are properties of the supported model and
lowering, not a claim that arbitrary ONNX division domains are proved.

ONNX Runtime evaluates FP64 for fast inference and reference comparisons. The
proof witness runner evaluates the actual ONNX nodes using MPFR with 128-bit
precision, including products and accumulation. It maps the shared weights and
converts accessed values lazily, releasing dead activations. Witness values are
rounded to signed i128 at scale `2^75`. There is no FP64 witness fallback.

## Projection and statement binding

For a fixed right matrix `B`, the verifier samples `r` after primary commitment
and checks `X (B r) = Y r`. The verifier independently evaluates `B r` from the
authenticated weights. This avoids placing all 752 million text parameters in
COO matrices.

For variable MatMul, the secondary witness contains `u=B r` and scalar products
`p[i,l]=A[i,l]*u[l]`; explicit constraints check their sums against `C r`.
Secondary values are fixed before field selection and checked again at opening
by the existing Spain protocol.

Public values include the model/circuit fingerprint, token IDs, independently
looked-up embeddings, and complete claimed last-position logits. Shape-only
operations are aliases; the final `Slice` selects the last hidden position as an
alias of its source before multiplication by the tied embedding weight.
The manifest hashes the ONNX graphs, COO matrices, layout and external weights.

The checked residual is `sqrt(sum_i (A_i z * B_i z - C_i z)^2)`, at the
corresponding real-number scale. It measures the local algebraic constraint
residuals. It is **not** a global output-error bound against either the original
or approximate model; output accuracy is assessed by the separate numerical
validation.

## Acceptance thresholds

- HF FP32 versus each ONNX graph: all-position logits max absolute error ≤0.01,
  row RMSE ≤0.001, and each hidden output relative L2 error ≤0.001.
- MPFR versus approximate FP64 graph: max absolute error ≤`1e-7`.
- Greedy decisions on the same prefix: ≥99% overall and ≥95% for each of the six
  fixed prompts, with readable completions and the simple expected answers.
- Complete real-weight Spain proofs for seq=2 and seq=32: accepted at `2^-40`,
  with all layers and the full last-position vocabulary output.
