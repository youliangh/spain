"""Independent binary64 Llama inference with Spain's rational arithmetic profile.

Only checkpoint/tokenizer I/O uses Hugging Face file formats.  This module never
calls a Transformers model.  The complete decoder runs here, without a KV cache.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from tokenizers import Tokenizer


DEFAULT_MODEL_DIR = "/home/yliangh/models/meta-llama/Llama-3.2-1B"
PROFILE = "llama3-binary64-pade33-s1024-square10-v1"


@dataclass(frozen=True)
class LlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int | list[int]
    rope_scaling: dict | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "LlamaConfig":
        if raw.get("model_type", "llama") != "llama":
            raise ValueError("only Llama checkpoints are supported")
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError("this arithmetic profile requires SiLU")
        if raw.get("attention_bias", False) or raw.get("mlp_bias", False):
            raise ValueError("this arithmetic profile requires bias-free projections")
        values = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        values.setdefault("head_dim", raw["hidden_size"] // raw["num_attention_heads"])
        values.setdefault("num_key_value_heads", raw["num_attention_heads"])
        values.setdefault("tie_word_embeddings", False)
        values.setdefault("bos_token_id", 1)
        values.setdefault("eos_token_id", 2)
        values.setdefault("rope_scaling", None)
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "vocab_size", "max_position_embeddings"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must equal query heads times head dimension")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("GQA heads must divide query heads and RoPE head dimension must be even")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("RMSNorm epsilon must be positive and finite")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("RoPE theta must be positive and finite")
        if self.rope_scaling:
            scaling = self.rope_scaling
            if scaling.get("rope_type", scaling.get("type")) != "llama3":
                raise ValueError("only default RoPE and llama3 scaling are supported")
            for key in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"):
                if not math.isfinite(scaling[key]) or scaling[key] <= 0:
                    raise ValueError(f"invalid RoPE scaling {key}")
            if scaling["high_freq_factor"] <= scaling["low_freq_factor"]:
                raise ValueError("RoPE high frequency factor must exceed low frequency factor")

    @property
    def eos_ids(self) -> set[int]:
        return {self.eos_token_id} if isinstance(self.eos_token_id, int) else set(self.eos_token_id)


def negative_exp(x: torch.Tensor, stats: dict | None = None) -> torch.Tensor:
    """Pade [3/3](x / 1024), followed by ten squarings, for -128 <= x <= 0.

    Reject out-of-domain values instead of silently introducing a clamp into the
    program proved by Spain.  The same operation ordering is used by Rust.
    """
    if not bool(torch.isfinite(x).all()):
        raise ValueError("negative_exp received a non-finite argument")
    minimum, maximum = float(x.min()), float(x.max())
    if minimum < -128.0 or maximum > 0.0:
        raise ValueError(f"negative_exp domain violation: [{minimum}, {maximum}] outside [-128, 0]")
    if stats is not None:
        stats["exp_min_argument"] = min(stats.get("exp_min_argument", 0.0), minimum)
        stats["exp_max_argument"] = max(stats.get("exp_max_argument", -128.0), maximum)
    t = x / 1024.0
    t2 = t * t
    t3 = t2 * t
    numerator = 120.0 + 60.0 * t + 12.0 * t2 + t3
    denominator = 120.0 - 60.0 * t + 12.0 * t2 - t3
    result = numerator / denominator
    for _ in range(10):
        result = result * result
    return result


def silu(x: torch.Tensor, stats: dict | None = None) -> torch.Tensor:
    exponent = negative_exp(-torch.abs(x), stats)
    denominator = 1.0 + exponent
    return torch.where(x >= 0, x / denominator, x * exponent / denominator)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    # Division by sqrt matches the arithmetic circuit (rather than a fused rsqrt).
    denominator = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + epsilon)
    return x / denominator * weight


def rope_frequencies(config: LlamaConfig) -> torch.Tensor:
    inverse = config.rope_theta ** (-torch.arange(0, config.head_dim, 2, dtype=torch.float64) / config.head_dim)
    if config.rope_scaling:
        scaling = config.rope_scaling
        wavelength = 2.0 * math.pi / inverse
        original = scaling["original_max_position_embeddings"]
        low_wavelength = original / scaling["low_freq_factor"]
        high_wavelength = original / scaling["high_freq_factor"]
        smooth = (original / wavelength - scaling["low_freq_factor"]) / (
            scaling["high_freq_factor"] - scaling["low_freq_factor"])
        interpolated = (1.0 - smooth) * inverse / scaling["factor"] + smooth * inverse
        inverse = torch.where(wavelength > low_wavelength, inverse / scaling["factor"],
                              torch.where(wavelength < high_wavelength, inverse, interpolated))
    return inverse


def apply_rope(x: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cosine + rotated * sine


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     stats: dict | None = None) -> torch.Tensor:
    """Inputs are [heads, sequence, head_dim], with K/V already GQA-expanded."""
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    length = q.shape[-2]
    mask = torch.ones((length, length), dtype=torch.bool, device=q.device).tril()
    maximum = scores.masked_fill(~mask, -torch.inf).max(dim=-1, keepdim=True).values
    # Masked entries are not part of the exp computation/domain of the circuit.
    shifted = torch.where(mask, scores - maximum, 0.0)
    exponents = negative_exp(shifted, stats) * mask
    probability = exponents / exponents.sum(dim=-1, keepdim=True)
    return torch.matmul(probability, v)


@dataclass
class ForwardResult:
    logits: torch.Tensor
    layer_hidden_states: dict[str, torch.Tensor]
    arithmetic: dict


class LlamaModel:
    def __init__(self, config: LlamaConfig, weights: dict[str, torch.Tensor], tokenizer: Tokenizer | None = None):
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self._check_weights()
        self.inverse_frequency = rope_frequencies(config)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "LlamaModel":
        directory = Path(model_dir)
        config = LlamaConfig.from_dict(json.loads((directory / "config.json").read_text()))
        index = directory / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        weights = {}
        for filename in files:
            with safe_open(directory / filename, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in weights:
                        raise ValueError(f"duplicate checkpoint tensor: {name}")
                    weights[name] = handle.get_tensor(name).to(torch.float64)
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        return cls(config, weights, tokenizer)

    def _check_weights(self) -> None:
        cfg = self.config
        shapes = {"model.embed_tokens.weight": (cfg.vocab_size, cfg.hidden_size),
                  "model.norm.weight": (cfg.hidden_size,)}
        if not cfg.tie_word_embeddings:
            shapes["lm_head.weight"] = (cfg.vocab_size, cfg.hidden_size)
        for layer in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            shapes.update({prefix + key: value for key, value in {
                "input_layernorm.weight": (cfg.hidden_size,),
                "post_attention_layernorm.weight": (cfg.hidden_size,),
                "self_attn.q_proj.weight": (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
                "self_attn.k_proj.weight": (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
                "self_attn.v_proj.weight": (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
                "self_attn.o_proj.weight": (cfg.hidden_size, cfg.hidden_size),
                "mlp.gate_proj.weight": (cfg.intermediate_size, cfg.hidden_size),
                "mlp.up_proj.weight": (cfg.intermediate_size, cfg.hidden_size),
                "mlp.down_proj.weight": (cfg.hidden_size, cfg.intermediate_size),
            }.items()})
        for name, shape in shapes.items():
            if name not in self.weights:
                raise ValueError(f"missing checkpoint tensor: {name}")
            value = self.weights[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"invalid shape for {name}: {tuple(value.shape)} != {shape}")
            if value.dtype != torch.float64 or value.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU binary64 tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"non-finite checkpoint tensor: {name}")
        extras = set(self.weights) - set(shapes)
        if cfg.tie_word_embeddings and "lm_head.weight" in extras:
            if not torch.equal(self.weights["lm_head.weight"], self.weights["model.embed_tokens.weight"]):
                raise ValueError("tied lm_head is inconsistent with the token embedding")
            extras.remove("lm_head.weight")
        if extras:
            raise ValueError(f"unsupported checkpoint tensors: {sorted(extras)}")

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("model has no tokenizer")
        return self.tokenizer.encode(text, add_special_tokens=True).ids

    def decode(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            raise ValueError("model has no tokenizer")
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.inference_mode()
    def forward(self, input_ids: list[int], *, capture_layers: bool = False,
                last_logits_only: bool = False) -> ForwardResult:
        cfg = self.config
        if not input_ids or len(input_ids) > cfg.max_position_embeddings:
            raise ValueError("input sequence must be non-empty and fit max_position_embeddings")
        if any(type(token) is not int or token < 0 or token >= cfg.vocab_size for token in input_ids):
            raise ValueError("input token IDs must be integers inside the checkpoint vocabulary")
        hidden = self.weights["model.embed_tokens.weight"][input_ids]
        states = {"embedding": hidden.clone()} if capture_layers else {}
        angles = torch.arange(len(input_ids), dtype=torch.float64)[:, None] * self.inverse_frequency[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        statistics = {"profile": PROFILE, "layers_executed": cfg.num_hidden_layers,
                      "sequence_length": len(input_ids), "exp_min_argument": 0.0,
                      "exp_max_argument": -128.0}
        for layer in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            norm = rms_norm(hidden, self.weights[prefix + "input_layernorm.weight"], cfg.rms_norm_eps)
            projected = []
            for name, heads in (("q", cfg.num_attention_heads), ("k", cfg.num_key_value_heads), ("v", cfg.num_key_value_heads)):
                value = norm @ self.weights[prefix + f"self_attn.{name}_proj.weight"].T
                value = value.reshape(len(input_ids), heads, cfg.head_dim).transpose(0, 1)
                projected.append(apply_rope(value, cosine, sine) if name != "v" else value)
            query, key, value = projected
            repeats = cfg.num_attention_heads // cfg.num_key_value_heads
            attended = causal_attention(query, key.repeat_interleave(repeats, dim=0),
                                        value.repeat_interleave(repeats, dim=0), statistics)
            attended = attended.transpose(0, 1).reshape(len(input_ids), cfg.hidden_size)
            hidden = hidden + attended @ self.weights[prefix + "self_attn.o_proj.weight"].T
            norm = rms_norm(hidden, self.weights[prefix + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
            gate = norm @ self.weights[prefix + "mlp.gate_proj.weight"].T
            up = norm @ self.weights[prefix + "mlp.up_proj.weight"].T
            hidden = hidden + (silu(gate, statistics) * up) @ self.weights[prefix + "mlp.down_proj.weight"].T
            if not bool(torch.isfinite(hidden).all()):
                raise ValueError(f"non-finite hidden state at layer {layer}")
            if capture_layers:
                states[f"layer_{layer}"] = hidden.clone()
        hidden = rms_norm(hidden, self.weights["model.norm.weight"], cfg.rms_norm_eps)
        if capture_layers:
            states["final_norm"] = hidden.clone()
        output_weight = self.weights["model.embed_tokens.weight"] if cfg.tie_word_embeddings else self.weights["lm_head.weight"]
        logits = (hidden[-1:] if last_logits_only else hidden) @ output_weight.T
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("non-finite output logits")
        return ForwardResult(logits, states, statistics)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int = 64,
                 on_step: Callable[[list[int], ForwardResult, int], None] | None = None) -> dict:
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if len(input_ids) + max_new_tokens - 1 > self.config.max_position_embeddings:
            raise ValueError("requested generation exceeds max_position_embeddings")
        tokens, steps = list(input_ids), []
        minimum = 0.0
        for index in range(max_new_tokens):
            result = self.forward(tokens, last_logits_only=True)
            logits = result.logits[-1]
            chosen = int(logits.argmax())
            if on_step:
                on_step(tokens, result, chosen)
            values, choices = torch.topk(logits, min(5, self.config.vocab_size))
            steps.append({"step": index, "token_id": chosen,
                          "top5": [{"token_id": int(i), "logit": float(v)} for i, v in zip(choices, values)]})
            minimum = min(minimum, result.arithmetic["exp_min_argument"])
            tokens.append(chosen)
            if chosen in self.config.eos_ids:
                break
        generated = tokens[len(input_ids):]
        result = {"input_ids": list(input_ids), "generated_ids": generated, "token_ids": tokens,
                  "steps": steps, "stop_reason": "eos" if generated[-1] in self.config.eos_ids else "max_new_tokens",
                  "arithmetic": {"profile": PROFILE, "layers_executed": self.config.num_hidden_layers,
                                 "exp_min_argument": minimum, "kv_cache": False}}
        if self.tokenizer is not None:
            result.update(text=self.decode(tokens), completion=self.decode(generated))
        return result
