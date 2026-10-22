"""Streaming ONNX -> approximate I-R1CS compiler.

The v2 format keeps ONNX as the executable program. Constants stay as mmap
views, fixed matrices are projected after commitment, and every variable
matrix product has explicit secondary witness equations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

COO_DTYPE = np.dtype([("row", "<u4"), ("col", "<u4"), ("value", "<f8")])
TAG = np.uint64(1 << 62)
MASK = np.uint64((1 << 62) - 1)
CHUNK = 262144


def exact_binary(op, a, b):
    """Fold only if the *real* operation on IEEE inputs is exactly binary64.

    ONNX witnesses execute at MPFR precision. Rounding a derived coefficient
    to binary64 here changes that program, even when its FP64 output looks
    indistinguishable. Large constant expressions conservatively stay in the
    circuit instead of creating Python objects for every weight.
    """
    a, b = np.broadcast_arrays(a, b)
    if a.size > 65536 or (op == "Div" and np.any(b == 0)):
        return None
    fn = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}[op]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = fn(a, b)
    if not np.isfinite(result).all():
        return None
    for av, bv, cv in zip(a.flat, b.flat, result.flat):
        x, y, z = (Fraction(float(v)) for v in (av, bv, cv))
        real = {"Add": lambda: x + y, "Sub": lambda: x - y,
                "Mul": lambda: x * y, "Div": lambda: x / y}[op]()
        if real != z:
            return None
    return result


@dataclass
class Ref:
    name: str
    values: np.ndarray
    constant: bool = False
    parts: tuple | None = None

    @property
    def shape(self):
        return self.values.shape


def affine(name, shape, parts):
    return Ref(name, np.broadcast_to(np.uint64(0), shape), False, tuple(parts))


def view(name, ref, transform):
    if ref.parts is None:
        return Ref(name, transform(ref.values), ref.constant)
    parts = []
    for atom, coefficient in ref.parts:
        # Both the symbolic values and their coefficients follow the view.
        atom_values = np.broadcast_to(atom.values, ref.shape)
        atom_ref = Ref(atom.name, atom_values, atom.constant, atom.parts)
        transformed = view(name, atom_ref, transform)
        coeff = transform(np.broadcast_to(coefficient, ref.shape))
        parts.append((transformed, coeff))
    return affine(name, transform(ref.values).shape, parts)


def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Compiler:
    def __init__(self, model, base_dir, output_dir, provenance=None):
        self.model = model
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.provenance = provenance or {}
        self.records = {"public": {}, "primary": {}, "secondary": {}}
        self.counts = {"public": 33, "primary": 0, "secondary": 0}
        self.refs = {}
        self.initializers = {t.name: t for t in model.graph.initializer}
        self.aux_nodes = []
        self.aux_initializers = []
        self.secondary_nodes = []
        self.secondary_initializers = []
        self.secondary_inputs = {}
        self.projection_nodes = []
        self.projection_initializers = []
        self.projection_outputs = []
        self.public = {"constant_offset": 0, "fingerprint_offset": 1,
                       "fingerprint_length": 32, "inputs": [], "gathers": [], "claims": []}
        self.random_count = 0
        self.random_specs = {}
        self.slots = []
        self.entries = {k: 0 for k in "ABC"}
        self.rows = 0
        self.emit = False
        self.handles = {}
        props = {p.key: p.value for p in model.metadata_props}
        self.domains = json.loads(props.get("spain.domains", "[]"))
        self.shapes = {}
        for tensor in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
            self.shapes[tensor.name] = tuple(d.dim_value for d in tensor.type.tensor_type.shape.dim)
        self.original_nodes = list(model.graph.node)

    def const_data(self, tensor):
        external = {e.key: e.value for e in tensor.external_data}
        if external:
            dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
            value = np.memmap(self.base_dir / external["location"], dtype=dtype, mode="r",
                              offset=int(external.get("offset", 0)), shape=tuple(tensor.dims))
        else:
            value = numpy_helper.to_array(tensor)
        flat = value.reshape(-1)
        for start in range(0, len(flat), CHUNK):
            if not np.isfinite(flat[start:start + CHUNK]).all():
                raise ValueError(f"nonfinite constant {tensor.name}")
        return value

    def alloc(self, name, shape, kind="primary", public_item=None):
        shape = tuple(int(v) for v in shape)
        length = math.prod(shape)
        if not self.emit:
            if name in self.records[kind]:
                raise ValueError(f"duplicate witness tensor {kind}:{name}")
            record = {"name": name, "shape": list(shape), "offset": self.counts[kind], "length": length}
            self.records[kind][name] = record
            self.counts[kind] += length
            if public_item:
                self.public[public_item].append(record.copy())
        record = self.records[kind][name]
        if record["shape"] != list(shape):
            raise ValueError(f"changed shape in second pass for {name}")
        tag = {"public": 0, "primary": 1, "secondary": 2}[kind]
        values = np.arange(record["offset"], record["offset"] + length, dtype=np.uint64)
        values |= np.uint64(tag) * TAG
        return Ref(name, values.reshape(shape))

    def global_columns(self, values):
        tags, local = values >> np.uint64(62), values & MASK
        starts = np.array([0, self.counts["public"] + self.random_count,
                           self.counts["public"] + self.random_count + self.counts["primary"]], dtype=np.uint64)
        global_values = local + starts[tags]
        if global_values.size and global_values.max() >= 2**32:
            raise ValueError("v2 COO u32 column limit exceeded")
        return global_values.astype(np.uint32)

    def write(self, matrix, rows, cols, coefficients=1., dynamic=None):
        if not self.emit:
            self.entries[matrix] += np.asarray(rows).size
            return
        rows = np.asarray(rows, dtype=np.uint64).reshape(-1)
        cols = np.broadcast_to(cols, rows.shape).reshape(-1)
        coefficients = np.broadcast_to(coefficients, rows.shape).reshape(-1)
        if rows.size and rows.max() >= 2**32:
            raise ValueError("v2 COO u32 row limit exceeded")
        if dynamic is not None:
            self.slots.append({"matrix": matrix, "entry_offset": self.entries[matrix],
                               "length": len(rows), **dynamic})
        for start in range(0, len(rows), CHUNK):
            end = start + CHUNK
            records = np.empty(len(rows[start:end]), dtype=COO_DTYPE)
            records["row"] = rows[start:end]
            records["col"] = self.global_columns(cols[start:end].astype(np.uint64))
            records["value"] = coefficients[start:end]
            if not np.isfinite(records["value"]).all():
                raise ValueError("nonfinite constraint coefficient")
            self.handles[matrix].write(records.tobytes())
        self.entries[matrix] += len(rows)

    def term(self, matrix, rows, ref, shape, coefficient=1.):
        if ref is not None and ref.parts is not None:
            for atom, scale in ref.parts:
                self.term(matrix, rows, atom, shape, np.broadcast_to(scale, ref.shape) * coefficient)
            return
        if not self.emit:
            self.entries[matrix] += math.prod(shape)
            return
        if ref is None:
            self.write(matrix, rows, np.uint64(0), np.broadcast_to(coefficient, shape).reshape(-1))
        elif ref.constant:
            values = np.broadcast_to(ref.values, shape).reshape(-1)
            self.write(matrix, rows, np.uint64(0), values * np.broadcast_to(coefficient, shape).reshape(-1))
        else:
            values = np.broadcast_to(ref.values, shape).reshape(-1)
            self.write(matrix, rows, values, np.broadcast_to(coefficient, shape).reshape(-1))

    def equation(self, shape, a, b, c):
        count = math.prod(shape)
        rows = np.arange(self.rows, self.rows + count, dtype=np.uint64) if self.emit else None
        for matrix, terms in zip("ABC", (a, b, c)):
            for ref, coefficient in terms:
                self.term(matrix, rows, ref, shape, coefficient)
        self.rows += count

    def auxiliary(self, name, op, inputs, shape, **attrs):
        ref = self.alloc(name, shape)
        if not self.emit:
            self.aux_nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
        return ref

    def materialize(self, ref):
        if ref.parts is None:
            return ref
        if ref.name in self.materialized:
            return self.materialized[ref.name]
        result = self.auxiliary(ref.name + ".spain_materialized", "Identity", [ref.name], ref.shape)
        self.equation(ref.shape, [(ref, 1.)], [(None, 1.)], [(result, 1.)])
        self.materialized[ref.name] = result
        return result

    def scalar(self, name, value, target="primary", dtype=np.float64):
        if not self.emit:
            tensor = numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)
            {"primary": self.aux_initializers, "secondary": self.secondary_initializers,
             "projection": self.projection_initializers}[target].append(tensor)
        return name

    def random_slice(self, name, shape, target):
        length = math.prod(shape)
        if not self.emit:
            offset = self.random_count
            self.random_count += length
            self.random_specs[name] = (offset, length)
            nodes = self.secondary_nodes if target == "secondary" else self.projection_nodes
            starts = self.scalar(name + ".starts", [offset], target, np.int64)
            ends = self.scalar(name + ".ends", [offset + length], target, np.int64)
            axes = self.scalar(name + ".axes", [0], target, np.int64)
            dimensions = self.scalar(name + ".shape", shape, target, np.int64)
            nodes.append(helper.make_node("Slice", ["R", starts, ends, axes], [name + ".flat"], name=name + ".slice"))
            nodes.append(helper.make_node("Reshape", [name + ".flat", dimensions], [name], name=name))
        return self.random_specs[name][0]

    def matmul(self, node, a, b):
        a, b = self.materialize(a), self.materialize(b)
        shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f"MatMul inner dimension mismatch {node.name}")
        if a.constant and b.constant:
            raise ValueError("constant MatMul must be folded by the exporter")
        if a.constant:
            raise ValueError("primary MatMul with fixed left operand is not supported; transpose it")
        out = self.alloc(node.output[0], shape)
        batch = shape[:-2]
        batches, m, k, n = math.prod(batch), shape[-2], a.shape[-1], shape[-1]
        av = np.broadcast_to(a.values, batch + (m, k)).reshape(batches, m, k)
        cv = out.values.reshape(batches, m, n)
        prefix = node.output[0] + ".spain"
        target = "projection" if b.constant else "secondary"
        offset = self.random_slice(prefix + ".r", batch + (n, 1), target)
        if b.constant:
            projection = prefix + ".fixed_projection"
            if not self.emit:
                self.projection_nodes.append(helper.make_node("MatMul", [b.name, prefix + ".r"], [projection], name=projection))
                self.projection_outputs.append({"name": projection, "shape": list(batch + (k, 1))})
            for bi in range(batches):
                for row in range(m):
                    rid = np.full(k, self.rows, dtype=np.uint64)
                    self.write("A", rid, av[bi, row], 0., {"source": "projection", "tensor": projection, "tensor_offset": bi * k})
                    self.write("B", [self.rows], np.uint64(0))
                    self.write("C", np.full(n, self.rows, dtype=np.uint64), cv[bi, row], 0.,
                               {"source": "random", "tensor_offset": offset + bi * n})
                    self.rows += 1
            return out
        bv = np.broadcast_to(b.values, batch + (k, n)).reshape(batches, k, n)
        u = self.alloc(prefix + ".br", batch + (k, 1), "secondary")
        p = self.alloc(prefix + ".products", batch + (m, k), "secondary")
        uv, pv = u.values.reshape(batches, k), p.values.reshape(batches, m, k)
        if not self.emit:
            for tensor in (a, b):
                self.secondary_inputs[tensor.name] = {"name": tensor.name, "shape": list(tensor.shape)}
            self.secondary_nodes.append(helper.make_node("MatMul", [b.name, prefix + ".r"], [u.name], name=u.name))
            perm = list(range(len(batch))) + [len(batch) + 1, len(batch)]
            self.secondary_nodes.append(helper.make_node("Transpose", [u.name], [prefix + ".br_t"], name=prefix + ".br_t", perm=perm))
            self.secondary_nodes.append(helper.make_node("Mul", [a.name, prefix + ".br_t"], [p.name], name=p.name))
        for bi in range(batches):
            for inner in range(k):
                self.write("A", np.full(n, self.rows, dtype=np.uint64), bv[bi, inner], 0.,
                           {"source": "random", "tensor_offset": offset + bi * n})
                self.write("B", [self.rows], np.uint64(0))
                self.write("C", [self.rows], uv[bi, inner])
                self.rows += 1
            rows = np.arange(self.rows, self.rows + m * k, dtype=np.uint64)
            self.write("A", rows, av[bi].reshape(-1))
            self.write("B", rows, np.broadcast_to(uv[bi], (m, k)).reshape(-1))
            self.write("C", rows, pv[bi].reshape(-1))
            self.rows += m * k
            for row in range(m):
                self.write("A", np.full(k, self.rows, dtype=np.uint64), pv[bi, row])
                self.write("B", [self.rows], np.uint64(0))
                self.write("C", np.full(n, self.rows, dtype=np.uint64), cv[bi, row], 0.,
                           {"source": "random", "tensor_offset": offset + bi * n})
                self.rows += 1
        return out

    def reduce(self, node, x, axes, keepdims, maximum=False, mean=False):
        x = self.materialize(x)
        axes = tuple(sorted(a % len(x.shape) for a in axes))
        outer = tuple(i for i in range(len(x.shape)) if i not in axes)
        output_shape = tuple(1 if i in axes else x.shape[i] for i in range(len(x.shape))) if keepdims else tuple(x.shape[i] for i in outer)
        if x.constant and maximum:
            # Max merely selects an existing exact IEEE constant.
            return Ref(node.output[0], np.max(x.values, axis=axes, keepdims=keepdims), True)
        out = self.alloc(node.output[0], output_shape)
        reduction = math.prod(x.shape[i] for i in axes)
        input_rows = np.transpose(x.values, outer + axes).reshape(-1, reduction)
        output_flat = out.values.reshape(-1)
        if not maximum:
            row_count = len(output_flat)
            rows = np.arange(self.rows, self.rows + row_count, dtype=np.uint64)
            if x.constant:
                self.write("A", np.repeat(rows, reduction), np.uint64(0), input_rows.reshape(-1))
            else:
                self.write("A", np.repeat(rows, reduction), input_rows.reshape(-1))
            self.write("B", rows, np.uint64(0))
            # sum(x) = count * mean(x) avoids a rounded 1/count coefficient.
            self.write("C", rows, output_flat, float(reduction) if mean else 1.)
            self.rows += row_count
            return out
        # A normalized selector is nonnegative, sums to one, and can have
        # support only at a maximum. This also handles duplicate maxima.
        prefix = out.name + ".spain_max"
        broad = output_shape if keepdims else tuple(1 if i in axes else x.shape[i] for i in range(len(x.shape)))
        max_ref = Ref(out.name, out.values.reshape(broad))
        expanded_name = out.name
        if not keepdims:
            dims = self.scalar(prefix + ".shape", broad, dtype=np.int64)
            expanded_name = prefix + ".expanded"
            if not self.emit:
                self.aux_nodes.append(helper.make_node("Reshape", [out.name, dims], [expanded_name], name=expanded_name))
        gap_name, equal_name, float_name, sum_name = (prefix + suffix for suffix in (".gap", ".equal", ".float", ".sum"))
        flags = self.alloc(prefix + ".flags", x.shape)
        gap_root = self.alloc(prefix + ".gap_root", x.shape)
        flag_root = self.alloc(prefix + ".flag_root", x.shape)
        if not self.emit:
            axes_name = self.scalar(prefix + ".axes", axes, dtype=np.int64)
            self.aux_nodes.extend([
                helper.make_node("Sub", [expanded_name, x.name], [gap_name], name=gap_name),
                helper.make_node("Sqrt", [gap_name], [gap_root.name], name=gap_root.name),
                helper.make_node("Equal", [expanded_name, x.name], [equal_name], name=equal_name),
                helper.make_node("Cast", [equal_name], [float_name], name=float_name, to=TensorProto.DOUBLE),
                helper.make_node("ReduceSum", [float_name, axes_name], [sum_name], name=sum_name, keepdims=1),
                helper.make_node("Div", [float_name, sum_name], [flags.name], name=flags.name),
                helper.make_node("Sqrt", [flags.name], [flag_root.name], name=flag_root.name),
            ])
        self.equation(x.shape, [(gap_root, 1.)], [(gap_root, 1.)], [(max_ref, 1.), (x, -1.)])
        self.equation(x.shape, [(flag_root, 1.)], [(flag_root, 1.)], [(flags, 1.)])
        self.equation(x.shape, [(flags, 1.)], [(max_ref, 1.), (x, -1.)], [])
        flag_rows = np.transpose(flags.values, outer + axes).reshape(-1, reduction)
        rows = np.arange(self.rows, self.rows + len(output_flat), dtype=np.uint64)
        self.write("A", np.repeat(rows, reduction), flag_rows.reshape(-1))
        self.write("B", rows, np.uint64(0))
        self.write("C", rows, np.uint64(0))
        self.rows += len(output_flat)
        return out

    def node(self, node):
        op, attrs = node.op_type, attributes(node)
        args = [self.refs[n] for n in node.input if n]
        name = node.output[0]
        if len(node.output) != 1:
            raise ValueError(f"v2 requires one output per primitive; lower {op}")
        if op == "Constant":
            return Ref(name, self.const_data(attrs["value"]), True)
        if op in ("Identity", "Cast"):
            if op == "Cast" and not args[0].constant and attrs.get("to") not in (TensorProto.DOUBLE, TensorProto.FLOAT):
                raise ValueError("only exact floating casts are supported for variables")
            return Ref(name, args[0].values, args[0].constant, args[0].parts)
        if op == "Transpose":
            return view(name, args[0], lambda values: np.transpose(values, attrs.get("perm")))
        if op == "Reshape":
            dims = list(int(v) for v in args[1].values.reshape(-1))
            if not attrs.get("allowzero", 0):
                dims = [args[0].shape[i] if d == 0 else d for i, d in enumerate(dims)]
            return view(name, args[0], lambda values: values.reshape(dims))
        if op in ("Unsqueeze", "Squeeze"):
            axes = tuple(int(v) for v in args[1].values.reshape(-1)) if len(args) > 1 else tuple(attrs.get("axes", []))
            fn = np.expand_dims if op == "Unsqueeze" else np.squeeze
            return view(name, args[0], lambda values: fn(values, axis=axes))
        if op == "Slice":
            starts, ends = args[1].values.reshape(-1), args[2].values.reshape(-1)
            axes = args[3].values.reshape(-1) if len(args) > 3 else range(len(starts))
            steps = args[4].values.reshape(-1) if len(args) > 4 else [1] * len(starts)
            slices = [slice(None)] * len(args[0].shape)
            for start, end, axis, step in zip(starts, ends, axes, steps):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            return view(name, args[0], lambda values: values[tuple(slices)])
        if op == "Concat":
            args = [self.materialize(arg) for arg in args]
            if len({a.constant for a in args}) != 1:
                # A zero/one padding constant must become a variable alias to
                # the public one row; a general mixed concat is materialized.
                shape = list(args[0].shape); axis = attrs["axis"]
                shape[axis] = sum(a.shape[axis] for a in args)
                out = self.alloc(name, shape)
                start = 0
                for arg in args:
                    selection = [slice(None)] * len(shape)
                    selection[axis] = slice(start, start + arg.shape[axis])
                    part = Ref(name, out.values[tuple(selection)])
                    self.equation(arg.shape, [(arg, 1.)], [(None, 1.)], [(part, 1.)])
                    start += arg.shape[axis]
                return out
            return Ref(name, np.concatenate([a.values for a in args], axis=attrs["axis"]), args[0].constant)
        if op == "Gather":
            table, indices = args
            axis = attrs.get("axis", 0)
            if indices.constant:
                return view(name, table, lambda values: np.take(values, indices.values.astype(np.int64), axis=axis))
            if not table.constant or indices.name not in {i.name for i in self.model.graph.input}:
                raise ValueError("dynamic Gather requires fixed table and a public input index tensor")
            shape = table.shape[:axis] + indices.shape + table.shape[axis + 1:]
            out = self.alloc(name, shape, "public", "gathers")
            if not self.emit:
                self.public["gathers"][-1].update(table=table.name, indices=indices.name, axis=axis)
            return out
        if op == "MatMul":
            return self.matmul(node, *args)
        if op in ("ReduceSum", "ReduceMean", "ReduceMax"):
            axes = tuple(int(a) for a in args[1].values.reshape(-1)) if len(args) > 1 else tuple(attrs.get("axes", range(len(args[0].shape))))
            return self.reduce(node, args[0], axes, attrs.get("keepdims", 1), op == "ReduceMax", op == "ReduceMean")
        if op in ("Add", "Sub", "Mul", "Div"):
            a, b = args
            both_constant = a.constant and b.constant
            if op == "Div" and b.constant and np.any(b.values == 0):
                raise ValueError("division by zero constant")
            if both_constant:
                folded = exact_binary(op, a.values, b.values)
                if folded is not None:
                    return Ref(name, folded, True)
            shape = np.broadcast_shapes(a.shape, b.shape)
            if not both_constant and op in ("Add", "Sub"):
                return affine(name, shape, [(a, 1.), (b, 1. if op == "Add" else -1.)])
            if not both_constant and op == "Mul" and (a.constant or b.constant):
                constant, variable = (a, b) if a.constant else (b, a)
                # Nested affine coefficients must not be multiplied in FP64.
                # Materialize the preceding ONNX value before scaling it.
                variable = self.materialize(variable)
                return affine(name, shape, [(variable, constant.values)])
            if not both_constant and op == "Div" and b.constant:
                reciprocal = exact_binary("Div", np.asarray(1.), b.values)
                if reciprocal is not None:
                    return affine(name, shape, [(self.materialize(a), reciprocal)])
            out = self.alloc(name, shape)
            if op in ("Add", "Sub"):
                self.equation(shape, [(a, 1.), (b, 1. if op == "Add" else -1.)], [(None, 1.)], [(out, 1.)])
            elif op == "Mul":
                self.equation(shape, [(a, 1.)], [(b, 1.)], [(out, 1.)])
            else:
                self.equation(shape, [(b, 1.)], [(out, 1.)], [(a, 1.)])
            return out
        if op == "Sqrt":
            x = args[0]
            if x.constant:
                if np.any(x.values < 0):
                    raise ValueError("negative constant square root")
                candidate = np.sqrt(x.values)
                squared = exact_binary("Mul", candidate, candidate)
                if squared is not None and np.array_equal(squared, x.values):
                    return Ref(name, candidate, True)
            out = self.alloc(name, x.shape)
            root = self.auxiliary(name + ".spain_positive_root", "Sqrt", [name], x.shape)
            self.equation(x.shape, [(out, 1.)], [(out, 1.)], [(x, 1.)])
            self.equation(x.shape, [(root, 1.)], [(root, 1.)], [(out, 1.)])
            return out
        raise ValueError(f"unsupported ONNX operation {op} at {node.name}; no unconstrained fallback")

    def pass_graph(self):
        self.materialized = {}
        self.refs = {name: Ref(name, self.const_data(tensor), True) for name, tensor in self.initializers.items()}
        for tensor in self.model.graph.input:
            if tensor.name not in self.initializers:
                self.refs[tensor.name] = self.alloc(tensor.name, self.shapes[tensor.name], "public", "inputs")
        uses = Counter(name for node in self.original_nodes for name in node.input)
        retained = {o.name for o in self.model.graph.output} | {d["tensor"] for d in self.domains}
        for node in self.original_nodes:
            self.refs[node.output[0]] = self.node(node)
            for name in node.input:
                uses[name] -= 1
                if uses[name] == 0 and name not in retained and name not in self.initializers:
                    self.refs.pop(name, None)
        for index, domain in enumerate(self.domains):
            x = self.refs[domain["tensor"]]
            for bound in ("lower", "upper"):
                if bound not in domain:
                    continue
                value = float(domain[bound]); sign = 1. if bound == "lower" else -1.
                prefix = x.name + f".spain_domain_{index}_{bound}"
                constant = self.scalar(prefix + ".bound", value)
                gap = prefix + ".gap"
                if not self.emit:
                    inputs = [x.name, constant] if bound == "lower" else [constant, x.name]
                    self.aux_nodes.append(helper.make_node("Sub", inputs, [gap], name=gap))
                root = self.auxiliary(prefix + ".root", "Sqrt", [gap], x.shape)
                self.equation(x.shape, [(root, 1.)], [(root, 1.)], [(x, sign), (None, -sign * value)])
        for tensor in self.model.graph.output:
            x = self.refs[tensor.name]
            claim = self.alloc("claim:" + tensor.name, x.shape, "public", "claims")
            if not self.emit:
                self.public["claims"][-1]["name"] = tensor.name
            self.equation(x.shape, [(x, 1.)], [(None, 1.)], [(claim, 1.)])

    def save_models(self):
        primary = onnx.ModelProto(); primary.CopyFrom(self.model)
        primary.graph.node.extend(self.aux_nodes)
        primary.graph.initializer.extend(self.aux_initializers)
        del primary.graph.output[:]
        outputs = {}
        for record in list(self.records["primary"].values()) + self.public["gathers"] + self.public["claims"] + list(self.secondary_inputs.values()):
            outputs[record["name"]] = record["shape"]
        for name, shape in outputs.items():
            primary.graph.output.append(helper.make_tensor_value_info(name, TensorProto.DOUBLE, shape))
        # Emit auxiliary nodes immediately after their dependencies. Observers
        # can then discard activations after their last actual consumer.
        producers = {name: node for node in primary.graph.node for name in node.output}
        ordered, visited = [], set()
        def visit(name):
            node = producers.get(name)
            if node is None or node.output[0] in visited:
                return
            visited.add(node.output[0])
            for dependency in node.input:
                visit(dependency)
            ordered.append(node)
        for name in outputs:
            visit(name)
        del primary.graph.node[:]
        primary.graph.node.extend(ordered)
        onnx.save_model(primary, self.output_dir / "primary_model.onnx")
        r = helper.make_tensor_value_info("R", TensorProto.DOUBLE, [self.random_count])
        secondary = helper.make_model(helper.make_graph(self.secondary_nodes, "Spain secondary witness", [r] + [helper.make_tensor_value_info(n, TensorProto.DOUBLE, v["shape"]) for n, v in self.secondary_inputs.items()], [helper.make_tensor_value_info(v["name"], TensorProto.DOUBLE, v["shape"]) for v in self.records["secondary"].values()], self.secondary_initializers), opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.save_model(secondary, self.output_dir / "secondary_model.onnx")
        # Fixed projections need only constant ancestors from the primary graph.
        constant_names = set(self.initializers)
        constant_nodes = []
        for node in self.original_nodes:
            if all(n in constant_names for n in node.input):
                constant_nodes.append(node)
                constant_names.update(node.output)
        projection = helper.make_model(helper.make_graph(constant_nodes + self.projection_nodes, "Spain fixed weight projections", [r], [helper.make_tensor_value_info(v["name"], TensorProto.DOUBLE, v["shape"]) for v in self.projection_outputs], list(self.model.graph.initializer) + self.projection_initializers), opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.save_model(projection, self.output_dir / "projection_model.onnx")

    def compile(self):
        self.pass_graph()
        planned_rows = self.rows
        planned_entries = self.entries.copy()
        estimated_coo_bytes = 16 * sum(planned_entries.values())
        estimated_witness_bytes = 16 * self.counts["primary"]
        available = shutil.disk_usage(self.output_dir).free
        if estimated_coo_bytes + estimated_witness_bytes + (1 << 30) > available:
            raise ValueError(f"insufficient artifact space: COO={estimated_coo_bytes}, primary={estimated_witness_bytes}, available={available}")
        if sum(self.counts.values()) + self.random_count >= 2**32:
            raise ValueError("v2 witness exceeds u32 indices")
        self.refs.clear()
        self.rows = 0
        self.entries = {k: 0 for k in "ABC"}
        self.emit = True
        try:
            self.handles = {k: open(self.output_dir / f"{k}.bin", "wb") for k in "ABC"}
            self.pass_graph()
        finally:
            for handle in self.handles.values():
                handle.close()
        assert planned_rows == self.rows
        assert planned_entries == self.entries, (planned_entries, self.entries)
        self.refs.clear()
        self.save_models()
        meta = {"schema_version": 2, "coo_format": "u32le,u32le,f64le; raw records; no header", "num_constraints": self.rows,
                "num_public_values": self.counts["public"], "num_random_values": self.random_count,
                "num_witness_values": self.counts["primary"], "num_secondary_witness_values": self.counts["secondary"],
                "num_secondary_constraint_variables": 0, "matrix_entries": self.entries,
                "primary_outputs": list(self.records["primary"].values()), "secondary_outputs": list(self.records["secondary"].values()),
                "secondary_inputs": list(self.secondary_inputs.values()), "projection_outputs": self.projection_outputs,
                "public": self.public, "challenge_slots": self.slots, **self.provenance}
        (self.output_dir / "layout.json").write_text(json.dumps(meta, indent=2) + "\n")
        asset_paths = {"original_model.onnx", "primary_model.onnx", "secondary_model.onnx", "projection_model.onnx", "layout.json", "A.bin", "B.bin", "C.bin"}
        asset_paths.update(p.name for p in self.output_dir.glob("source-*-manifest.json"))
        for tensor in self.model.graph.initializer:
            external = {e.key: e.value for e in tensor.external_data}
            if external:
                asset_paths.add(external["location"])
        artifacts = [{"path": p, "sha256": file_hash(self.output_dir / p), "bytes": (self.output_dir / p).stat().st_size} for p in sorted(asset_paths)]
        fingerprint = hashlib.sha256(json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        meta.update(artifacts=artifacts, fingerprint_sha256=fingerprint)
        (self.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta


def compile_v2(input_path, output_dir, *, lower=True):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "meta.json").exists():
        raise FileExistsError("choose a fresh v2 output directory")
    provenance = {"source_graph_sha256": file_hash(input_path),
                  "compiler_source_sha256": file_hash(Path(__file__))}
    manifests = set(input_path.parent.glob("*manifest*.json"))
    # Exporter manifests can point to a shared weight manifest in another
    # directory. Bind those small files too; weight bytes are hashed below.
    for manifest in list(manifests):
        document = json.loads(manifest.read_text())
        linked = document.get("weights_manifest")
        if linked:
            path = Path(linked)
            if not path.is_absolute():
                path = manifest.parent / path
            if path.is_file():
                manifests.add(path.resolve())
    for manifest in sorted(manifests):
        destination = output_dir / ("source-" + file_hash(manifest)[:16] + "-manifest.json")
        shutil.copyfile(manifest, destination)
    model = onnx.load(input_path, load_external_data=False)
    if lower:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from qwen_onnx import lower_nonlinearities
        model = lower_nonlinearities(model)
    # Keep a single immutable external weight file shared by all graph artifacts.
    # Relative paths are rewritten; no weight data is copied into protobufs.
    import os
    for tensor in model.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                source = (input_path.parent / item.value).resolve()
                filename = "external-" + hashlib.sha256(str(source).encode()).hexdigest()[:12] + "-" + source.name
                destination = output_dir / filename
                if not destination.exists():
                    os.link(source, destination)
                elif not os.path.samefile(source, destination):
                    raise ValueError("external data destination is not the shared immutable file")
                item.value = filename
    onnx.save_model(model, output_dir / "original_model.onnx")
    return Compiler(model, output_dir, output_dir, provenance).compile()


def estimate_v2(input_path, *, lower=True):
    """Exact structural counts without emitting matrix or witness artifacts."""
    input_path = Path(input_path).resolve()
    model = onnx.load(input_path, load_external_data=False)
    if lower:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from qwen_onnx import lower_nonlinearities
        model = lower_nonlinearities(model)
    compiler = Compiler(model, input_path.parent, input_path.parent)
    compiler.pass_graph()
    return {"num_constraints": compiler.rows, "num_public_values": compiler.counts["public"],
            "num_witness_values": compiler.counts["primary"], "num_secondary_witness_values": compiler.counts["secondary"],
            "num_random_values": compiler.random_count, "matrix_entries": compiler.entries,
            "coo_bytes": 16 * sum(compiler.entries.values()), "primary_bytes": 16 * compiler.counts["primary"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--already-lowered", action="store_true")
    args = parser.parse_args()
    result = compile_v2(args.input, args.output_dir, lower=not args.already_lowered)
    print(json.dumps({k: result[k] for k in ("num_constraints", "num_witness_values", "num_secondary_witness_values")}))
