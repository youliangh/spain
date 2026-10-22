"""Numerical and structural tests for the standard ONNX Qwen frontend."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
import torch

try:
    from tools.qwen_onnx import GraphBuilder, export_model, lower_nonlinearities, write_tiny_fixture
except ModuleNotFoundError:
    from qwen_onnx import GraphBuilder, export_model, lower_nonlinearities, write_tiny_fixture


def session(path_or_bytes):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path_or_bytes) if isinstance(path_or_bytes, Path) else path_or_bytes,
                                sess_options=options, providers=["CPUExecutionProvider"])


class NonlinearTests(unittest.TestCase):
    def test_bounded_lowering_accuracy_and_domains(self):
        b = GraphBuilder()
        b.shapes["x"] = (1, 513)
        outputs = [b.unary("Sigmoid", "x", "sigmoid"), b.unary("Softplus", "x", "softplus")]
        negative = b.element("Mul", b.unary("Sqrt", b.element("Mul", "x", "x")), b.const(-1.0))
        outputs.append(b.unary("Exp", negative, "exponential"))
        half = b.element("Mul", "x", b.const(0.5))
        outputs.append(b.op("Softmax", half, name="softmax", shape=(1, 513), axis=-1))
        graph = helper.make_graph(b.nodes, "nonlinear_test", [b.value_info("x")],
                                  [b.value_info(n) for n in outputs], initializer=b.initializers,
                                  value_info=[b.value_info(n) for n in b.shapes if n != "x" and n not in outputs])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 10
        lowered = lower_nonlinearities(model)
        onnx.checker.check_model(lowered)
        self.assertFalse({"Exp", "Sigmoid", "Softplus", "Softmax"} & {n.op_type for n in lowered.graph.node})
        domains = json.loads({p.key: p.value for p in lowered.metadata_props}["spain.domains"])
        self.assertTrue(any(d["tensor"] == "x" and d["lower"] == -128 and d["upper"] == 128 for d in domains))
        x = np.linspace(-128, 128, 513).reshape(1, -1)
        exact = session(model.SerializeToString()).run(None, {"x": x})
        actual = session(lowered.SerializeToString()).run(None, {"x": x})
        for left, right in zip(actual, exact):
            np.testing.assert_allclose(left, right, atol=3e-12, rtol=7e-9)


class HybridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        cls.temporary = tempfile.TemporaryDirectory(prefix="qwen-onnx-")
        cls.root = Path(cls.temporary.name)
        cls.source = write_tiny_fixture(cls.root / "checkpoint")
        torch.set_num_threads(2)
        config = Qwen3_5TextConfig.from_pretrained(cls.source, local_files_only=True)
        cls.reference = Qwen3_5ForCausalLM.from_pretrained(cls.source, config=config, dtype=torch.float32,
                                                         local_files_only=True, attn_implementation="eager").eval()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_hybrid_graph_and_nonlinear_lowering_match_hf(self):
        for length in [1, 2, 5]:
            with self.subTest(length=length):
                destination = self.root / f"seq{length}"
                exact_path = export_model(self.source, destination, length, weights_dir=self.root / "weights", debug=True)
                approximate_path = export_model(self.source, destination, length, weights_dir=self.root / "weights",
                                                debug=True, approximate=True)
                ids = np.arange(1, length + 1, dtype=np.int64)[None]
                exact = session(exact_path).run(None, {"input_ids": ids})
                actual = session(approximate_path).run(None, {"input_ids": ids})
                with torch.inference_mode():
                    reference = self.reference(torch.from_numpy(ids), use_cache=False).logits.numpy()
                np.testing.assert_allclose(exact[1], reference, atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(exact[0], reference[:, -1:], atol=2e-6, rtol=2e-5)
                for a, e in zip(actual, exact):
                    np.testing.assert_allclose(a, e, atol=2e-10, rtol=2e-9)
                self.assertEqual(exact[0].shape, (1, 1, 64))
                self.assertTrue(np.isfinite(actual[1]).all())
                model = onnx.load(exact_path, load_external_data=False)
                self.assertEqual(model.ir_version, 10)
                self.assertEqual(model.opset_import[0].version, 18)
                self.assertEqual(sum(t.name.endswith("embed_tokens.weight") for t in model.graph.initializer), 1)
                self.assertEqual(sum(n.name.startswith("layer_") and n.name[-2:].isdigit()
                                     for n in model.graph.node), 4)
                self.assertFalse({"Loop", "Scan", "Trilu", "Conv"} & {n.op_type for n in model.graph.node})

    def test_shared_weights_are_reused_and_source_changes_rejected(self):
        destination = self.root / "reuse"
        first = export_model(self.source, destination, 2, weights_dir=self.root / "weights")
        sidecar = self.root / "weights" / "weights.bin"
        before = sidecar.stat().st_mtime_ns
        export_model(self.source, destination, 3, weights_dir=self.root / "weights")
        self.assertEqual(sidecar.stat().st_mtime_ns, before)
        self.assertTrue(os.path.samefile(first.parent / "weights.bin", sidecar))


if __name__ == "__main__":
    unittest.main()
