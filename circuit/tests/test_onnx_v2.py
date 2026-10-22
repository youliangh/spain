import json
from decimal import Decimal, localcontext
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from circuit.onnx_v2 import COO_DTYPE, compile_v2


def tiny_model(path):
    initializers = [
        numpy_helper.from_array(np.arange(32, dtype=np.float64).reshape(8, 4) / 32, "embedding"),
        numpy_helper.from_array(np.arange(12, dtype=np.float64).reshape(4, 3) / 16, "weight"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axis"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "last"),
        numpy_helper.from_array(np.array(1e-6, dtype=np.float64), "epsilon"),
        numpy_helper.from_array(np.array([0.1, 0.2, -0.3], dtype=np.float64), "bias"),
        numpy_helper.from_array(np.array([0.5, -0.75, 1.25], dtype=np.float64), "scale"),
    ]
    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["embedded"], name="embedding_lookup"),
        helper.make_node("MatMul", ["embedded", "weight"], ["x"], name="fixed_linear"),
        helper.make_node("Add", ["x", "bias"], ["biased"], name="bias_add"),
        helper.make_node("Mul", ["biased", "scale"], ["scaled"], name="affine_scale"),
        helper.make_node("Transpose", ["scaled"], ["xt"], name="transpose", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["scaled", "xt"], ["scores"], name="variable_matmul"),
        helper.make_node("ReduceMax", ["scores", "axis"], ["maximum"], name="maximum", keepdims=1),
        helper.make_node("Sub", ["scores", "maximum"], ["shifted"], name="subtract"),
        helper.make_node("Mul", ["shifted", "shifted"], ["squared"], name="square"),
        helper.make_node("ReduceMean", ["squared", "axis"], ["mean"], name="mean", keepdims=1),
        helper.make_node("Add", ["mean", "epsilon"], ["variance"], name="variance"),
        helper.make_node("Sqrt", ["variance"], ["root"], name="sqrt"),
        helper.make_node("Div", ["shifted", "root"], ["normalized"], name="divide"),
        helper.make_node("Gather", ["normalized", "last"], ["logits"], name="last_position", axis=1),
    ]
    graph = helper.make_graph(nodes, "v2 tiny", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 2])], [helper.make_tensor_value_info("logits", TensorProto.DOUBLE, [1, 1, 2])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    helper.set_model_props(model, {"spain.domains": json.dumps([{"tensor": "x", "lower": -16, "upper": 16}])})
    onnx.save_model(model, path)


def execute(directory, metadata, token_ids):
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    def run(name, inputs):
        session = ort.InferenceSession(str(directory / name), opts, providers=["CPUExecutionProvider"])
        return dict(zip([v.name for v in session.get_outputs()], session.run(None, inputs)))
    outputs = run("primary_model.onnx", {"input_ids": np.asarray([token_ids], dtype=np.int64)})
    randomness = np.random.default_rng(17).normal(size=metadata["num_random_values"])
    secondary = run("secondary_model.onnx", {"R": randomness, **{r["name"]: outputs[r["name"]] for r in metadata["secondary_inputs"]}})
    projection = run("projection_model.onnx", {"R": randomness})
    counts = [metadata[k] for k in ("num_public_values", "num_random_values", "num_witness_values", "num_secondary_witness_values")]
    z = np.zeros(sum(counts)); z[0] = 1
    z[1:33] = list(bytes.fromhex(metadata["fingerprint_sha256"]))
    for record in metadata["public"]["inputs"]:
        z[record["offset"]:record["offset"] + record["length"]] = token_ids
    for record in metadata["public"]["gathers"] + metadata["public"]["claims"]:
        z[record["offset"]:record["offset"] + record["length"]] = outputs[record["name"]].reshape(-1)
    z[counts[0]:sum(counts[:2])] = randomness
    for records, source, start in ((metadata["primary_outputs"], outputs, sum(counts[:2])), (metadata["secondary_outputs"], secondary, sum(counts[:3]))):
        for record in records:
            z[start + record["offset"]:start + record["offset"] + record["length"]] = source[record["name"]].reshape(-1)
    matrices = {m: np.fromfile(directory / f"{m}.bin", dtype=COO_DTYPE) for m in "ABC"}
    for slot in metadata["challenge_slots"]:
        source = randomness if slot["source"] == "random" else projection[slot["tensor"]].reshape(-1)
        matrices[slot["matrix"]]["value"][slot["entry_offset"]:slot["entry_offset"] + slot["length"]] = source[slot["tensor_offset"]:slot["tensor_offset"] + slot["length"]]
    return z, matrices


def residual(z, matrices, rows):
    products = []
    for m in "ABC":
        values = np.zeros(rows)
        np.add.at(values, matrices[m]["row"], matrices[m]["value"] * z[matrices[m]["col"]])
        products.append(values)
    return np.linalg.norm(products[0] * products[1] - products[2])


class V2Tests(unittest.TestCase):
    def test_inexact_constant_folding_and_affine_composition_match_high_precision(self):
        # FP64 execution cannot detect this bug: its witness and the broken
        # compiler both perform the same premature rounding. Evaluate the
        # actual ONNX arithmetic at 100 decimal digits instead.
        with tempfile.TemporaryDirectory() as temporary, localcontext() as context:
            context.prec = 100
            directory = Path(temporary)
            constants = {"a": np.array([0.1]), "b": np.array([0.2]),
                         "three": np.array([3.]), "two": np.array([2.]),
                         "vector": np.array([0.1, 0.2, 0.3]), "axes": np.array([0], dtype=np.int64)}
            operations = [("Add", ["a", "b"], "constant_add"),
                          ("Mul", ["a", "b"], "constant_mul"),
                          ("Div", ["a", "three"], "constant_div"),
                          ("Sqrt", ["two"], "constant_root"),
                          ("ReduceMean", ["vector", "axes"], "constant_mean"),
                          ("Mul", ["x", "a"], "scaled_once"),
                          ("Mul", ["scaled_once", "b"], "scaled_twice"),
                          ("Div", ["x", "three"], "variable_div")]
            nodes = [helper.make_node(op, inputs, [name], name=name) for op, inputs, name in operations]
            outputs = [helper.make_tensor_value_info(name, TensorProto.DOUBLE, [1]) for _, _, name in operations]
            model = helper.make_model(helper.make_graph(nodes, "rounding regression",
                [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [1])], outputs,
                [numpy_helper.from_array(array, name) for name, array in constants.items()]),
                opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
            onnx.save_model(model, directory / "model.onnx")
            meta = compile_v2(directory / "model.onnx", directory / "compiled", lower=False)
            primary = onnx.load(directory / "compiled/primary_model.onnx")
            values = {tensor.name: [Decimal.from_float(float(v)) for v in numpy_helper.to_array(tensor).flat]
                      for tensor in primary.graph.initializer}
            values["x"] = [Decimal(1000)]
            for node in primary.graph.node:
                operands = [values[name] for name in node.input]
                if node.op_type == "Identity": result = operands[0]
                elif node.op_type == "Sqrt": result = [v.sqrt() for v in operands[0]]
                elif node.op_type == "ReduceMean": result = [sum(operands[0]) / len(operands[0])]
                else:
                    a, b = operands
                    result = [{"Add": lambda: x + y, "Mul": lambda: x * y,
                               "Sub": lambda: x - y, "Div": lambda: x / y}[node.op_type]()
                              for x, y in zip(a, b)]
                values[node.output[0]] = result
            z = [Decimal(0)] * (meta["num_public_values"] + meta["num_witness_values"])
            z[0] = Decimal(1)
            for record in meta["public"]["inputs"] + meta["public"]["claims"]:
                z[record["offset"]:record["offset"] + record["length"]] = values[record["name"]]
            for record in meta["primary_outputs"]:
                start = meta["num_public_values"] + record["offset"]
                z[start:start + record["length"]] = values[record["name"]]
            products = []
            for matrix in "ABC":
                result = [Decimal(0)] * meta["num_constraints"]
                for entry in np.fromfile(directory / "compiled" / f"{matrix}.bin", dtype=COO_DTYPE):
                    result[int(entry["row"])] += Decimal.from_float(float(entry["value"])) * z[int(entry["col"])]
                products.append(result)
            errors = [abs(a * b - c) for a, b, c in zip(*products)]
            self.assertLess(max(errors), Decimal("1e-90"))
            scale = 1 << 75
            integer_z = [int((value * scale).to_integral_value()) for value in z]
            integer_products = []
            for matrix in "ABC":
                result = [0] * meta["num_constraints"]
                for entry in np.fromfile(directory / "compiled" / f"{matrix}.bin", dtype=COO_DTYPE):
                    coefficient = int((Decimal.from_float(float(entry["value"])) * scale).to_integral_value())
                    result[int(entry["row"])] += coefficient * integer_z[int(entry["col"])]
                integer_products.append(result)
            integer_errors = [a * b - c * scale**2 for a, b, c in zip(*integer_products)]
            self.assertLess(max(abs(v) for v in integer_errors), 1 << 255)
            norm = (Decimal(sum(v * v for v in integer_errors)) / Decimal(scale**8)).sqrt()
            self.assertLess(norm, Decimal(2) ** -40)
            names = {r["name"] for r in meta["primary_outputs"]}
            self.assertTrue({"constant_add", "constant_mul", "constant_div", "constant_root",
                             "constant_mean", "variable_div", "scaled_once.spain_materialized"} <= names)

    def test_onnx_primary_secondary_and_every_constraint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); tiny_model(directory / "tiny.onnx")
            output = directory / "compiled"
            meta = compile_v2(directory / "tiny.onnx", output, lower=False)
            for token_ids in ([1, 3], [2, 2]):  # include tied maxima
                z, matrices = execute(output, meta, token_ids)
                self.assertLess(residual(z, matrices, meta["num_constraints"]), 1e-10)
                primary_start = meta["num_public_values"] + meta["num_random_values"]
                for record in meta["primary_outputs"]:
                    wrong = z.copy(); wrong[primary_start + record["offset"]] += 0.01
                    self.assertGreater(residual(wrong, matrices, meta["num_constraints"]), 1e-9, record["name"])
                secondary_start = primary_start + meta["num_witness_values"]
                for record in meta["secondary_outputs"]:
                    wrong = z.copy(); wrong[secondary_start + record["offset"]] += 0.01
                    self.assertGreater(residual(wrong, matrices, meta["num_constraints"]), 1e-9, record["name"])

    def test_unknown_operator_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); tiny_model(directory / "tiny.onnx")
            model = onnx.load(directory / "tiny.onnx"); model.graph.node[1].op_type = "Unknown"
            onnx.save_model(model, directory / "bad.onnx")
            with self.assertRaisesRegex(ValueError, "unsupported ONNX"):
                compile_v2(directory / "bad.onnx", directory / "compiled", lower=False)

    def test_external_weights_remain_shared_and_fingerprint_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); tiny_model(directory / "tiny.onnx")
            model = onnx.load(directory / "tiny.onnx")
            onnx.save_model(model, directory / "external.onnx", save_as_external_data=True,
                            all_tensors_to_one_file=True, location="weights.bin", size_threshold=64)
            output = directory / "compiled"
            meta = compile_v2(directory / "external.onnx", output, lower=False)
            weights = [a for a in meta["artifacts"] if a["path"].startswith("external-")]
            self.assertEqual(len(weights), 1)
            self.assertTrue(os.path.samefile(directory / "weights.bin", output / weights[0]["path"]))
            z, matrices = execute(output, meta, [1, 3])
            self.assertLess(residual(z, matrices, meta["num_constraints"]), 1e-10)
            layout = json.loads((output / "layout.json").read_text())
            self.assertEqual(layout, {k: v for k, v in meta.items() if k not in ("artifacts", "fingerprint_sha256")})

    def test_negative_square_root_and_domain_forgery_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); tiny_model(directory / "tiny.onnx")
            output = directory / "compiled"
            meta = compile_v2(directory / "tiny.onnx", output, lower=False)
            z, matrices = execute(output, meta, [1, 3])
            start = meta["num_public_values"] + meta["num_random_values"]
            root = next(r for r in meta["primary_outputs"] if r["name"] == "root")
            forged = z.copy(); forged[start + root["offset"]] *= -1
            self.assertGreater(residual(forged, matrices, meta["num_constraints"]), 1e-6)
            for record in meta["primary_outputs"]:
                if ".spain_domain_" in record["name"]:
                    forged = z.copy(); forged[start + record["offset"]] = 0
                    self.assertGreater(residual(forged, matrices, meta["num_constraints"]), 1.)

    def test_nan_fixed_weight_fails_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); tiny_model(directory / "tiny.onnx")
            model = onnx.load(directory / "tiny.onnx")
            for tensor in model.graph.initializer:
                if tensor.name == "weight":
                    values = numpy_helper.to_array(tensor).copy(); values[0, 0] = np.nan
                    tensor.CopyFrom(numpy_helper.from_array(values, tensor.name))
            onnx.save_model(model, directory / "bad.onnx")
            with self.assertRaisesRegex(ValueError, "nonfinite constant weight"):
                compile_v2(directory / "bad.onnx", directory / "compiled", lower=False)

    def test_changed_embedding_and_final_claim_fail_constraints(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); tiny_model(directory / "tiny.onnx")
            output = directory / "compiled"
            meta = compile_v2(directory / "tiny.onnx", output, lower=False)
            z, matrices = execute(output, meta, [1, 3])
            for record in meta["public"]["gathers"] + meta["public"]["claims"]:
                forged = z.copy(); forged[record["offset"]] += 0.1
                self.assertGreater(residual(forged, matrices, meta["num_constraints"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
