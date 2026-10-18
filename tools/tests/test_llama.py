"""Small, deterministic regressions for the complete arithmetic path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llama import compare_native, metric_failures, reference_forward, run_prove, tensor_error
from llama_model import LlamaConfig, LlamaModel, apply_rope, causal_attention, negative_exp, rope_frequencies, silu


def tiny_model() -> LlamaModel:
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                        vocab_size=32, rms_norm_eps=1e-5, rope_theta=500000.0,
                        max_position_embeddings=128, tie_word_embeddings=True,
                        bos_token_id=1, eos_token_id=31,
                        rope_scaling={"rope_type": "llama3", "factor": 32.0,
                                      "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                      "original_max_position_embeddings": 8192})
    generator = torch.Generator().manual_seed(411)

    def normal(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64) * 0.1

    weights = {"model.embed_tokens.weight": normal(32, 16), "model.norm.weight": torch.ones(16, dtype=torch.float64)}
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        weights.update({prefix + name: value for name, value in {
            "input_layernorm.weight": torch.ones(16, dtype=torch.float64),
            "post_attention_layernorm.weight": torch.ones(16, dtype=torch.float64),
            "self_attn.q_proj.weight": normal(16, 16),
            "self_attn.k_proj.weight": normal(8, 16),
            "self_attn.v_proj.weight": normal(8, 16),
            "self_attn.o_proj.weight": normal(16, 16),
            "mlp.gate_proj.weight": normal(32, 16),
            "mlp.up_proj.weight": normal(32, 16),
            "mlp.down_proj.weight": normal(16, 32),
        }.items()})
    return LlamaModel(config, weights)


class ArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_negative_exp_and_silu_against_transcendentals(self):
        values = torch.linspace(-128, 0, 4097, dtype=torch.float64)
        reference = torch.exp(values)
        relative = (negative_exp(values) / reference - 1).abs().max()
        self.assertLess(float(relative), 1e-8)
        self.assertEqual(float(negative_exp(torch.tensor([0.0], dtype=torch.float64))), 1.0)
        signed = torch.linspace(-128, 128, 4097, dtype=torch.float64)
        torch.testing.assert_close(silu(signed), torch.nn.functional.silu(signed), rtol=1e-8, atol=1e-11)

    def test_exp_domain_rejects_instead_of_clamping(self):
        for value in (-128.001, 0.001, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                negative_exp(torch.tensor([value], dtype=torch.float64))

    def test_causal_attention_cannot_read_future_values(self):
        generator = torch.Generator().manual_seed(12)
        q, k, v = [torch.randn((2, 5, 4), generator=generator, dtype=torch.float64) for _ in range(3)]
        full = causal_attention(q, k, v)
        prefix = causal_attention(q[:, :3], k[:, :3], v[:, :3])
        torch.testing.assert_close(full[:, :3], prefix, rtol=1e-13, atol=1e-13)
        masked = (q @ k.transpose(-1, -2) / 2).masked_fill(~torch.ones(5, 5, dtype=torch.bool).tril(), -torch.inf)
        torch.testing.assert_close(full, masked.softmax(-1) @ v, rtol=1e-10, atol=1e-11)

    def test_rope_preserves_head_norm_and_uses_split_halves(self):
        model = tiny_model()
        inverse = rope_frequencies(model.config)
        angles = torch.arange(4, dtype=torch.float64)[:, None] * inverse
        angles = torch.cat((angles, angles), dim=-1)
        x = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
        rotated = apply_rope(x, angles.cos(), angles.sin())
        torch.testing.assert_close((rotated * rotated).sum(-1), (x * x).sum(-1))
        torch.testing.assert_close(rotated[:, 0], x[:, 0])
        self.assertAlmostEqual(float(rotated[0, 1, 0]), float(x[0, 1, 0] * angles[1, 0].cos() - x[0, 1, 2] * angles[1, 0].sin()))

    def test_full_decoder_matches_independent_transformers(self):
        from transformers import LlamaConfig as HFConfig, LlamaForCausalLM
        model = tiny_model()
        reference = LlamaForCausalLM(HFConfig(**asdict(model.config), attention_bias=False, mlp_bias=False)).float().eval()
        reference.config._attn_implementation = "eager"
        weights = {name: value.float() for name, value in model.weights.items()}
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        reference.load_state_dict(weights, strict=True)
        for ids in ([1], [1, 3], [1, 3, 5, 7, 9, 11, 13, 15]):
            actual = model.forward(ids, capture_layers=True)
            expected, layers = reference_forward(reference, ids, capture_layers=True)
            self.assertEqual(set(actual.layer_hidden_states), set(layers))
            torch.testing.assert_close(actual.logits, expected.double(), rtol=2e-5, atol=2e-6)
            for name in layers:
                torch.testing.assert_close(actual.layer_hidden_states[name], layers[name].double(), rtol=2e-5, atol=2e-6)

    def test_complete_model_prefix_and_last_logits_agree(self):
        model = tiny_model()
        tokens = [1, 4, 9, 7]
        full = model.forward(tokens)
        prefix = model.forward(tokens[:2])
        last = model.forward(tokens, last_logits_only=True)
        torch.testing.assert_close(full.logits[:2], prefix.logits, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(last.logits, full.logits[-1:], rtol=1e-12, atol=1e-12)

    def test_invalid_ids_and_missing_weights_fail(self):
        model = tiny_model()
        for ids in ([], [32], [-1], [True], [1.0], [1] * 129):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                model.forward(ids)
        weights = dict(model.weights)
        del weights["model.layers.1.mlp.up_proj.weight"]
        with self.assertRaisesRegex(ValueError, "missing checkpoint tensor"):
            LlamaModel(model.config, weights)

    def test_generation_stops_exactly_on_eos(self):
        model = tiny_model()
        token = int(model.forward([1]).logits[-1].argmax())
        object.__setattr__(model.config, "eos_token_id", token)
        result = model.generate([1], max_new_tokens=8)
        self.assertEqual(result["generated_ids"], [token])
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(len(result["steps"]), 1)

    def test_logit_gate_catches_a_bad_position(self):
        expected = torch.zeros(8, 32, dtype=torch.float64)
        actual = expected.clone()
        actual[7, 31] = 2.0
        metrics = tensor_error(actual, expected, logits=True)
        limits = {"max_abs": 0.01, "max_rmse": 0.001, "max_kl": 1e-5}
        self.assertTrue(metric_failures(metrics, limits, logits=True, name="corrupted"))
        self.assertGreater(metrics["max_row_rmse"], metrics["rmse"])

    def test_native_comparison_rejects_tampered_logits(self):
        model = tiny_model()
        ids = [1, 3]
        logits = model.forward(ids).logits[-1].tolist()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forward.json"
            path.write_text(json.dumps({"input_ids": ids, "logits": logits}))
            self.assertTrue(compare_native(str(path), model, 1e-9)["passed"])
            logits[2] += 0.5
            path.write_text(json.dumps({"input_ids": ids, "logits": logits}))
            self.assertFalse(compare_native(str(path), model, 1e-9)["passed"])

    def test_proof_delegation_cannot_accept_failed_or_missing_native_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps(asdict(tiny_model().config)))
            ids = directory / "tokens.json"
            ids.write_text("[1, 3]")
            args = argparse.Namespace(binary="/bin/false", model_dir=temporary, input_ids=str(ids),
                                      output_dir=str(directory / "proof"), threads=1,
                                      memory_budget_gib=1, disk_budget_gib=1)
            self.assertFalse(run_prove(args)["passed"])
            args.binary = "/bin/true"
            with self.assertRaisesRegex(ValueError, "without a verification report"):
                run_prove(args)

    def test_proof_output_guard_preserves_existing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps(asdict(tiny_model().config)))
            source = directory / "tokens.json"
            source.write_text("[1, 3]")
            destination = directory / "proof"
            destination.mkdir()
            previous_input = destination / "input_ids.json"
            previous_input.write_text("[9, 9]\n")
            (destination / "proof_report.json").write_text('{"verified": true}')
            args = argparse.Namespace(binary="/bin/true", model_dir=temporary, input_ids=str(source),
                                      output_dir=str(destination), threads=1,
                                      memory_budget_gib=1, disk_budget_gib=1)
            with self.assertRaisesRegex(ValueError, "fresh --output-dir"):
                run_prove(args)
            self.assertEqual(previous_input.read_text(), "[9, 9]\n")


if __name__ == "__main__":
    unittest.main()
