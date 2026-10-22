# ONNX I-R1CS artifact format v2

`circuit.v2.compile_model(graph, output_dir)` compiles a static standard ONNX
graph to the existing Spain commitment, sum-check, and DARK protocol. The
compiler calls the bounded nonlinear lowering in `tools/qwen_onnx.py` unless
`lower=False` is specified for an already lowered graph. It fails on unsupported
operations; no output is accepted as an unconstrained fallback.

```python
from circuit.v2 import estimate_model, compile_model
print(estimate_model("approximate_model.onnx", lower=False))
compile_model("approximate_model.onnx", "compiled", lower=False)
```

The compiler keeps immutable external weight files as memory mapped constants
and creates hard links inside the artifact directory. Affine expressions and
shape operations reuse symbolic variables. COO records are emitted in bounded
chunks after an exact structural counting pass. Compilation checks free space
for its matrices, the eventual primary witness, and one GiB of additional space.
Derived constants are folded only when exact rational arithmetic proves that
the result is representable as binary64. Inexact constants and composed affine
scalings remain explicit ONNX witness equations, preserving MPFR semantics.
ReduceMean uses `sum(x) = count * mean(x)` to avoid rounded reciprocal coefficients.

The executable artifacts are:

- `original_model.onnx`: the lowered numerical program, with its claimed outputs.
- `primary_model.onnx`: that program and constrained auxiliary values. Its
  outputs are observed during execution and written at metadata offsets.
- `secondary_model.onnx`: explicit challenge-dependent projections and products.
- `projection_model.onnx`: fixed-weight projections computed independently by
  both parties from the same weights and verifier challenges.
- `A.bin`, `B.bin`, `C.bin`: raw 16-byte COO records, without a header: little
  endian `u32 row`, `u32 column`, `f64 coefficient`. Indices are global.
- `layout.json`: witness layout, tensor mappings, and challenge coefficient slots.
- `meta.json`: that layout plus artifact hashes and a fingerprint.

Witness order is `[public, random, primary, secondary]`. Public values begin
with one, 32 fingerprint bytes, and model inputs. Embedding Gather values are
independently checked against the constant table and public token IDs; final
claims are fixed before any challenge. Gather with constant indices is a
variable alias. Intermediate primary and secondary records contain local
offsets within their respective witness segments.

Each challenge slot replaces consecutive COO coefficient values with a slice
of the random vector or a fixed-projection output. Zero placeholders retain the
same dimensions and sparsity pattern before commitment. For variable
`C=A B`, the secondary graph computes `u=B r` and `p_ij=A_ij u_j`; constraints
enforce these relations and `sum_j p_ij=sum_j C_ij r_j`. Thus challenge-dependent
values are actual witness variables rather than unchecked matrix coefficients.

All files that define the statement, including COO matrices and `layout.json`,
occur in `artifacts`. The fingerprint is SHA-256 of
`json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()`.
Verification checks these hashes and equality of `layout.json` with `meta.json`
after removing `artifacts` and `fingerprint_sha256`.
The authenticated layout includes hashes of the source ONNX and compiler source.
Adjacent exporter manifests and their referenced weight manifests are included
as authenticated artifact copies.

Sqrt constraints include the nonnegative branch. ReduceMax uses nonnegative
selectors summing to one, handles ties, and constrains nonnegative input gaps.
Declared `spain.domains` bounds receive square witnesses. These are approximate
algebraic constraints, so their endpoint tolerances inherit the proof residual
bound; they do not by themselves imply a global model logit-error theorem.
Public dynamic Gather currently requires a nonnegative axis; unsupported
negative-axis layouts fail the runtime shape consistency check.

Run the compiler/ONNX consistency and tamper checks with:

```bash
.venv/bin/python -m unittest circuit.tests.test_onnx_v2 -v
```
