#!/usr/bin/env python3
"""Prepare and compare the official Spain GPT-2 ONNX numerical regression.

The proof graph retains all 12 blocks and the official last-token 50257 logits.
Its Erf and Softmax functions equal the artifact approximations; LayerNorm and
Split are lowered to standard algebra/views accepted by the streaming compiler.
Original weights are promoted exactly from FLOAT to DOUBLE and shared on disk.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_softmax_coefficients():
    path = ROOT / "circuit/nodes/softmax.py"
    assignments = {}
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in {"P", "Q"}:
                assignments[name] = ast.literal_eval(statement.value)
    return assignments["P"], assignments["Q"]


def promote_and_externalize(model, directory, shared):
    shared.mkdir(parents=True, exist_ok=True)
    weights = shared / "weights.bin"
    index_path = shared / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    replacements = []
    changed_masks = 0
    with weights.open("ab") as out:
        for tensor in model.graph.initializer:
            array = numpy_helper.to_array(tensor)
            if array.dtype.kind == "f":
                if np.isnan(array).any() or np.isposinf(array).any():
                    raise ValueError(f"invalid source coefficient {tensor.name}")
                # Exactly the official circuit/script/preprocess_model.py rule.
                changed_masks += int(np.isneginf(array).sum())
                array = np.where(np.isneginf(array), -10, array).astype("<f8")
            array = np.ascontiguousarray(array) if array.ndim else array
            if array.nbytes >= 1 << 20:
                digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
                key = f"{array.dtype.str}:{list(array.shape)}:{digest}"
                if key not in index:
                    offset = out.tell()
                    out.write(memoryview(array).cast("B"))
                    index[key] = {"offset": offset, "length": array.nbytes}
                record = index[key]
                replacement = TensorProto(name=tensor.name, data_type=TensorProto.DOUBLE,
                                          dims=array.shape, data_location=TensorProto.EXTERNAL)
                for key, value in {"location": "weights.bin", **record}.items():
                    item = replacement.external_data.add(); item.key = key; item.value = str(value)
            else:
                replacement = numpy_helper.from_array(array, tensor.name)
            replacements.append(replacement)
    index_path.write_text(json.dumps(index, indent=2))
    destination = directory / "weights.bin"
    if not destination.exists():
        os.link(weights, destination)
    elif not os.path.samefile(weights, destination):
        raise ValueError(f"refusing to replace {destination}")
    del model.graph.initializer[:]; model.graph.initializer.extend(replacements)
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.type.tensor_type.elem_type == TensorProto.FLOAT:
            value.type.tensor_type.elem_type = TensorProto.DOUBLE
    return changed_masks


def lower(model, *, approximate):
    result = onnx.ModelProto(); result.CopyFrom(model)
    initializers = {t.name: t for t in result.graph.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(result.graph.input) + list(result.graph.value_info) + list(result.graph.output)}
    nodes = []
    counter = 0

    def const(value, dtype=np.float64):
        nonlocal counter
        counter += 1
        name = f"spain_gpt2_constant_{counter}"
        result.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=dtype), name))
        return name

    def op(kind, *inputs, output=None, **attrs):
        nonlocal counter
        counter += 1
        name = output or f"spain_gpt2_{counter}_{kind}"
        nodes.append(helper.make_node(kind, inputs, [name], name=name, **attrs))
        return name

    p, q = artifact_softmax_coefficients()
    for node in result.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        x, output = node.input[0], node.output[0]
        if node.op_type == "LayerNormalization":
            axis = attrs.get("axis", -1) % len(shapes[x])
            axes = const(list(range(axis, len(shapes[x]))), np.int64)
            centered = op("Sub", x, op("ReduceMean", x, axes, keepdims=1))
            variance = op("ReduceMean", op("Mul", centered, centered), axes, keepdims=1)
            std = op("Sqrt", op("Add", variance, const(attrs.get("epsilon", 1e-5))))
            # Preserve the artifact's reciprocal then multiply order.
            normalized = op("Mul", centered, op("Div", const(1.0), std))
            scaled = op("Mul", normalized, node.input[1])
            op("Add", scaled, node.input[2], output=output)
        elif node.op_type == "Split":
            axis = attrs.get("axis", 0) % len(shapes[x])
            sizes = numpy_helper.to_array(initializers[node.input[1]]).tolist()
            start = 0
            for name, size in zip(node.output, sizes):
                op("Slice", x, const([start], np.int64), const([start + size], np.int64),
                   const([axis], np.int64), output=name)
                start += size
            assert start == shapes[x][axis]
        elif approximate and node.op_type == "Erf":
            # Algebraic equality to artifact clip(x,-1,1), with positive Sqrt.
            plus, minus = op("Add", x, const(1.0)), op("Sub", x, const(1.0))
            absolute_plus = op("Sqrt", op("Mul", plus, plus))
            absolute_minus = op("Sqrt", op("Mul", minus, minus))
            op("Mul", op("Sub", absolute_plus, absolute_minus), const(0.5), output=output)
        elif approximate and node.op_type == "Softmax":
            axes = const([attrs.get("axis", -1)], np.int64)
            centered = op("Sub", x, op("ReduceMax", x, axes, keepdims=1))
            powers = [const(1.0), centered]
            for _ in range(2, len(p)):
                powers.append(op("Mul", powers[-1], centered))
            sums = []
            for coefficients in (p, q):
                terms = [op("Mul", power, const(coefficient)) for power, coefficient in zip(powers, coefficients)]
                total = terms[0]
                for term in terms[1:]:
                    total = op("Add", total, term)
                sums.append(total)
            exponential = op("Mul", sums[0], op("Div", const(1.0), sums[1]))
            op("Div", exponential, op("ReduceSum", exponential, axes, keepdims=1), output=output)
        else:
            nodes.append(node)
    del result.graph.node[:]; result.graph.node.extend(nodes)
    result = onnx.shape_inference.infer_shapes(result, strict_mode=True)
    helper.set_model_props(result, {
        "spain.architecture": "official-gpt2-artifact-regression",
        "spain.numerics": "official-artifact-clip-erf-rational-softmax" if approximate else "exact-fp64",
        "spain.gpt2.scope": "12 blocks, one prefill pass, last token all 50257 logits, no KV cache",
        "spain.gpt2.softmax_coefficients": json.dumps({"P": p, "Q": q}),
    })
    return result


def prepare(source, directory, shared):
    directory.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    original_ops = dict(Counter(n.op_type for n in model.graph.node))
    seq = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    masks = promote_and_externalize(model, directory, shared)
    exact, approximate = lower(model, approximate=False), lower(model, approximate=True)
    onnx.save_model(exact, directory / "exact_model.onnx")
    onnx.save_model(approximate, directory / "approximate_model.onnx")
    onnx.checker.check_model(str(directory / "exact_model.onnx"))
    onnx.checker.check_model(str(directory / "approximate_model.onnx"))
    ids = ([15496, 995] * ((seq + 1) // 2))[:seq]
    (directory / "input_ids.json").write_text(json.dumps(ids))
    (directory / "inputs.json").write_text(json.dumps({model.graph.input[0].name:
        {"shape": [1, seq], "dtype": "int64", "values": ids}}))
    report = {"source": str(source), "source_sha256": file_hash(source), "sequence_length": seq,
              "source_ops": original_ops, "negative_inf_masks_replaced_with_minus10": masks,
              "exact_model_sha256": file_hash(directory / "exact_model.onnx"),
              "approximate_model_sha256": file_hash(directory / "approximate_model.onnx"),
              "softmax_source_sha256": file_hash(ROOT / "circuit/nodes/softmax.py"),
              "erf_source_sha256": file_hash(ROOT / "circuit/nodes/erf.py"),
              "scope": "12 decoder blocks, last-token output1442[1,1,50257], passes=1, no KV cache"}
    (directory / "prepare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def reference(directory):
    import onnxruntime as ort
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.op_run import OpRun

    class Erf(OpRun):
        op_domain = ""

        def _run(self, x):
            # Upstream ONNX's Erf uses np.vectorize(..., otypes=["f"]),
            # silently computing float32 even for DOUBLE inputs. Keep libm's
            # double result here so this is an actual binary64 reference.
            return (np.vectorize(math.erf, otypes=[np.float64])(x).astype(x.dtype),)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ids = np.array([json.loads((directory / "input_ids.json").read_text())], dtype=np.int64)
    outputs, timings = {}, {}
    for profile in ("exact", "approximate"):
        path = str(directory / f"{profile}_model.onnx")
        # ORT CPU does not implement DOUBLE Erf. ONNX's reference evaluator
        # supplies the exact DOUBLE baseline without introducing float casts.
        if profile == "exact":
            session = ReferenceEvaluator(path, new_ops=[Erf])
            names, input_name = session.output_names, session.input_names[0]
        else:
            session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
            names, input_name = [v.name for v in session.get_outputs()], session.get_inputs()[0].name
        start = time.perf_counter()
        values = session.run(None, {input_name: ids})
        timings[profile] = time.perf_counter() - start
        outputs[profile] = values[0]
        np.savez(directory / f"{profile}_outputs.npz", **dict(zip(names, values)))
    difference = outputs["approximate"] - outputs["exact"]
    report = {"reference_seconds": timings,
              "exact_engine": "onnx.reference.ReferenceEvaluator DOUBLE with libm DOUBLE Erf override",
              "approximate_engine": "ONNX Runtime CPU DOUBLE",
              "approximation_max_abs": float(np.max(np.abs(difference))),
              "approximation_rmse": float(np.sqrt(np.mean(difference ** 2))),
              "exact_argmax": int(np.argmax(outputs["exact"][0, -1])),
              "approximate_argmax": int(np.argmax(outputs["approximate"][0, -1]))}
    source = Path(json.loads((directory / "prepare.json").read_text())["source"])
    if source.exists():
        session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
        original = session.run(None, {session.get_inputs()[0].name: ids})[0]
        np.savez(directory / "source_fp32_outputs.npz", **{"1442": original})
        difference = original - outputs["exact"]
        report["official_fp32_vs_exact_fp64"] = {
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "argmax": int(original[0, -1].argmax()),
        }
    (directory / "ort_reference.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def artifact_reference(directory):
    """Run the repository's actual legacy nonlinearity wiring for equivalence."""
    import onnxruntime as ort
    sys.path.insert(0, str(ROOT / "circuit"))
    from nodes.erf import wire_erf
    from nodes.softmax import wire_softmax

    model = onnx.load(directory / "exact_model.onnx", load_external_data=False)
    for node in list(model.graph.node):
        if node.op_type == "Erf":
            wire_erf(node, model.graph)
        elif node.op_type == "Softmax":
            wire_softmax(node, model.graph)
    # Legacy graph.convert_model_to_precision performs this conversion after
    # wiring. Avoid its eager materialization of all external initializers.
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t") and attribute.t.data_type == TensorProto.FLOAT:
                attribute.t.CopyFrom(numpy_helper.from_array(
                    numpy_helper.to_array(attribute.t).astype(np.float64), attribute.t.name))
    producers = {name: node for node in model.graph.node for name in node.output}
    ordered, visited = [], set()

    def visit(node):
        if node.output[0] in visited:
            return
        for name in node.input:
            if name in producers:
                visit(producers[name])
        visited.add(node.output[0]); ordered.append(node)

    for node in model.graph.node:
        visit(node)
    del model.graph.node[:]; model.graph.node.extend(ordered)
    path = directory / "legacy_wired_model.onnx"
    onnx.save_model(onnx.shape_inference.infer_shapes(model), path)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    ids = np.array([json.loads((directory / "input_ids.json").read_text())], dtype=np.int64)
    output = session.run(None, {session.get_inputs()[0].name: ids})[0]
    reference = np.load(directory / "approximate_outputs.npz")["1442"]
    difference = output - reference
    report = {"legacy_wiring_source": ["circuit/nodes/erf.py", "circuit/nodes/softmax.py"],
              "legacy_nodes": len(ordered), "max_abs": float(np.max(np.abs(difference))),
              "rmse": float(np.sqrt(np.mean(difference ** 2))), "argmax": int(output[0, -1].argmax())}
    (directory / "legacy_equivalence.json").write_text(json.dumps(report, indent=2))
    np.savez(directory / "legacy_wired_outputs.npz", **{"1442": output})
    print(json.dumps(report), flush=True)
    if report["max_abs"] > 1e-8:
        raise AssertionError("legacy wiring differs from proof graph")


def compare(directory):
    report = {}
    for profile in ("exact", "approximate"):
        raw = json.loads((directory / f"{profile}_mpfr.json").read_text())
        reference = np.load(directory / f"{profile}_outputs.npz")["1442"]
        actual = np.array(raw["outputs"]["1442"]["data"]).reshape(reference.shape)
        difference = actual - reference
        report[profile] = {
            "shape": list(actual.shape), "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "argmax_reference": int(reference[0, -1].argmax()),
            "argmax_mpfr": int(actual[0, -1].argmax()),
            "mpfr_seconds": raw["forward_seconds"], "nodes": raw["nodes_executed"],
            "all_finite": bool(np.isfinite(actual).all()),
        }
        if not report[profile]["all_finite"] or report[profile]["max_abs"] >= 1e-8:
            raise AssertionError(report[profile])
    source_path = directory / "source_mpfr.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
        exact = json.loads((directory / "exact_mpfr.json").read_text())
        a = np.array(source["outputs"]["1442"]["data"])
        b = np.array(exact["outputs"]["1442"]["data"])
        report["source_original_onnx"] = {
            "nodes": source["nodes_executed"], "seconds": source["forward_seconds"],
            "source_vs_lowered_exact_max_abs": float(np.max(np.abs(a - b))),
            "argmax": int(a.argmax()),
        }
    (directory / "mpfr_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("source", type=Path); prep.add_argument("directory", type=Path)
    prep.add_argument("--shared", type=Path, required=True)
    ref = commands.add_parser("reference"); ref.add_argument("directory", type=Path)
    legacy = commands.add_parser("legacy-reference"); legacy.add_argument("directory", type=Path)
    comp = commands.add_parser("compare"); comp.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.directory, args.shared)
    elif args.command == "reference":
        reference(args.directory)
    elif args.command == "legacy-reference":
        artifact_reference(args.directory)
    else:
        compare(args.directory)


if __name__ == "__main__":
    main()
