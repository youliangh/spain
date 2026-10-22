"""Complete Qwen3.5 Dense text models as standard ONNX, with shared FP64 weights."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, helper, numpy_helper

DEFAULT_MODEL_DIR = "/home/yliangh/models/Qwen/Qwen3.5-0.8B"
TEXT_PREFIX = "model.language_model."
EXPORT_VERSION = 1
NUMERICS = {
    "profile_version": 2,
    "exponential": "[3/3] Pade at x/1024, ten squarings",
    "sigmoid": "Pade sigmoid at x/1024, ten bounded sigmoid doublings",
    "softplus": "(x+sqrt(x*x))/2 + 20-term atanh log1p(exp(-abs(x)))",
    "softmax": "subtract row maximum; Pade negative exponential; normalize",
    "exponential_negative_domain": [-128.0, 0.0],
    "sigmoid_softplus_domain": [-128.0, 128.0],
    "log1p_absolute_tail_bound": 1.51e-21,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def export_shared_weights(model_dir: str | Path, weights_dir: str | Path) -> dict:
    """Stream each text checkpoint tensor once into a shared binary64 sidecar."""
    from safetensors import safe_open
    import torch
    source, destination = Path(model_dir).resolve(), Path(weights_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "weights_manifest.json"
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    names += sorted(p.name for p in source.glob("*.safetensors"))
    stamps = {name: [p.stat().st_size, p.stat().st_mtime_ns]
              for name in names if (p := source / name).exists()}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        sidecar = destination / manifest["external_file"]
        if (manifest.get("source_dir") != str(source) or manifest.get("source_stamps") != stamps
                or manifest.get("export_version") != EXPORT_VERSION):
            raise ValueError(f"existing weights export does not match checkpoint: {destination}")
        if not sidecar.is_file() or sidecar.stat().st_size != manifest["external_bytes"]:
            raise ValueError(f"missing or truncated shared weights: {sidecar}")
        return manifest
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors files in {source}")
    tensors, digest, offset = {}, hashlib.sha256(), 0
    temporary = destination / "weights.bin.tmp"
    with temporary.open("wb") as output:
        for file in files:
            with safe_open(str(file), framework="pt", device="cpu") as reader:
                for name in sorted(reader.keys()):
                    if not name.startswith(TEXT_PREFIX):
                        continue
                    if name in tensors:
                        raise ValueError(f"duplicate tensor {name}")
                    tensor = reader.get_tensor(name)
                    flat, length = tensor.reshape(-1), tensor.numel() * 8
                    tensors[name] = {"shape": list(tensor.shape), "dtype": "DOUBLE",
                                     "offset": offset, "length": length,
                                     "source_file": file.name, "source_dtype": str(tensor.dtype)}
                    for start in range(0, flat.numel(), 1 << 20):
                        values = flat[start:start + (1 << 20)].to(dtype=torch.float64).numpy()
                        if not np.isfinite(values).all():
                            raise ValueError(f"nonfinite checkpoint tensor {name}")
                        block = values.astype("<f8", copy=False).tobytes()
                        output.write(block); digest.update(block)
                    offset += length
    temporary.replace(destination / "weights.bin")
    manifest = {"schema_version": 1, "export_version": EXPORT_VERSION,
                "source_dir": str(source), "source_stamps": stamps,
                "sources": {name: sha256_file(source / name) for name in stamps},
                "external_file": "weights.bin", "external_bytes": offset,
                "external_sha256": digest.hexdigest(), "tensors": tensors}
    _json_write(manifest_path, manifest)
    return manifest


class GraphBuilder:
    """Static shapes and external references, without materializing large weights."""
    def __init__(self):
        self.nodes, self.initializers = [], []
        self.shapes, self.types, self._constants, self._transposes = {}, {}, {}, {}
        self._counter = 0

    def unique(self, stem):
        self._counter += 1
        return f"{stem}_{self._counter}"

    def const(self, value, name=None, *, integer=False):
        array = np.asarray(value, dtype=np.int64 if integer else np.float64)
        key = (array.dtype.str, array.shape, array.tobytes())
        if name is None and key in self._constants:
            return self._constants[key]
        name = name or self.unique("constant")
        self.initializers.append(numpy_helper.from_array(array, name))
        self.shapes[name], self.types[name] = array.shape, TensorProto.INT64 if integer else TensorProto.DOUBLE
        self._constants[key] = name
        return name

    def op(self, op, *inputs, name=None, shape=None, **attributes):
        name = name or self.unique(op.lower())
        self.nodes.append(helper.make_node(op, list(inputs), [name], name=name, **attributes))
        if shape is not None:
            self.shapes[name] = tuple(shape)
        self.types[name] = TensorProto.DOUBLE
        return name

    def element(self, op, left, right, name=None):
        return self.op(op, left, right, name=name,
                       shape=np.broadcast_shapes(self.shapes[left], self.shapes[right]))

    def unary(self, op, value, name=None):
        return self.op(op, value, name=name, shape=self.shapes[value])

    def reshape(self, value, shape, name=None):
        if math.prod(self.shapes[value]) != math.prod(shape):
            raise ValueError(f"invalid reshape {self.shapes[value]} -> {shape}")
        return self.op("Reshape", value, self.const(shape, integer=True), name=name, shape=shape)

    def transpose(self, value, perm, name=None):
        return self.op("Transpose", value, name=name, shape=[self.shapes[value][i] for i in perm], perm=perm)

    def slice(self, value, start, end, axis, name=None):
        shape = list(self.shapes[value]); axis %= len(shape)
        if not 0 <= start <= end <= shape[axis]:
            raise ValueError(f"invalid slice {start}:{end} axis {axis} shape {shape}")
        shape[axis] = end - start
        return self.op("Slice", value, self.const([start], integer=True), self.const([end], integer=True),
                       self.const([axis], integer=True), self.const([1], integer=True), name=name, shape=shape)

    def concat(self, values, axis, name=None):
        if len(values) == 1:
            return self.unary("Identity", values[0], name=name) if name else values[0]
        shape = list(self.shapes[values[0]]); axis %= len(shape)
        shape[axis] = sum(self.shapes[v][axis] for v in values)
        return self.op("Concat", *values, name=name, shape=shape, axis=axis)

    def reduce(self, op, value, axis, name=None):
        shape = list(self.shapes[value]); axis %= len(shape); shape[axis] = 1
        return self.op(op, value, self.const([axis], integer=True), name=name, shape=shape, keepdims=1)

    def matmul(self, left, right, name=None):
        a, b = self.shapes[left], self.shapes[right]
        if a[-1] != b[-2]:
            raise ValueError(f"matmul mismatch {a} {b}")
        shape = np.broadcast_shapes(a[:-2], b[:-2]) + (a[-2], b[-1])
        return self.op("MatMul", left, right, name=name, shape=shape)

    def linear(self, value, weight, name=None):
        if weight not in self._transposes:
            self._transposes[weight] = self.transpose(weight, [1, 0], weight + ".transpose")
        return self.matmul(value, self._transposes[weight], name=name)

    def norm(self, value, weight, epsilon, *, zero_centered=True, l2=False, name):
        square = self.element("Mul", value, value, name + ".square")
        mean = self.reduce("ReduceSum" if l2 else "ReduceMean", square, -1, name + ".variance")
        root = self.unary("Sqrt", self.element("Add", mean, self.const(epsilon)), name + ".root")
        normalized = self.element("Div", value, root, name + ".normalized")
        if weight is None:
            return normalized
        if zero_centered:
            weight = self.element("Add", weight, self.const(1.0), name + ".scale")
        return self.element("Mul", normalized, weight, name)

    def silu(self, value, name):
        return self.element("Mul", value, self.unary("Sigmoid", value), name)

    def value_info(self, name):
        return helper.make_tensor_value_info(name, self.types.get(name, TensorProto.DOUBLE), self.shapes[name])


def _external_initializers(builder, manifest):
    for name, entry in manifest["tensors"].items():
        tensor = TensorProto(name=name, data_type=TensorProto.DOUBLE, dims=entry["shape"])
        tensor.raw_data = b"\0"
        external_data_helper.set_external_data(tensor, location="weights.bin", offset=entry["offset"], length=entry["length"])
        tensor.ClearField("raw_data"); tensor.data_location = TensorProto.EXTERNAL
        builder.initializers.append(tensor)
        builder.shapes[name], builder.types[name] = tuple(entry["shape"]), TensorProto.DOUBLE


def _rope(b, value, config, seq, name):
    head_dim, parameters = config["head_dim"], config["rope_parameters"]
    rotary = int(head_dim * parameters.get("partial_rotary_factor", 1.0))
    if rotary == 0 or rotary % 2:
        raise ValueError("RoPE needs a positive even rotary dimension")
    inv = parameters["rope_theta"] ** (-np.arange(0, rotary, 2, dtype=np.float64) / rotary)
    frequency = np.arange(seq, dtype=np.float64)[:, None] * inv[None, :]
    frequency = np.concatenate([frequency, frequency], axis=-1)[None, None]
    cosine, sine = b.const(np.cos(frequency)), b.const(np.sin(frequency))
    rotating = b.slice(value, 0, rotary, -1)
    first, second = b.slice(rotating, 0, rotary // 2, -1), b.slice(rotating, rotary // 2, rotary, -1)
    rotated = b.concat([b.element("Mul", second, b.const(-1.0)), first], -1)
    output = b.element("Add", b.element("Mul", rotating, cosine), b.element("Mul", rotated, sine))
    if rotary < head_dim:
        output = b.concat([output, b.slice(value, rotary, head_dim, -1)], -1)
    return b.unary("Identity", output, name)


def _full_attention(b, x, c, seq, prefix):
    h, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    projected = b.reshape(b.linear(x, prefix + ".q_proj.weight"), (1, seq, h, 2 * d))
    query = b.slice(projected, 0, d, -1)
    gate = b.reshape(b.slice(projected, d, 2 * d, -1), (1, seq, h * d))
    key = b.reshape(b.linear(x, prefix + ".k_proj.weight"), (1, seq, kv, d))
    value = b.reshape(b.linear(x, prefix + ".v_proj.weight"), (1, seq, kv, d))
    query = b.norm(query, prefix + ".q_norm.weight", c["rms_norm_eps"], name=prefix + ".query_norm")
    key = b.norm(key, prefix + ".k_norm.weight", c["rms_norm_eps"], name=prefix + ".key_norm")
    query = _rope(b, b.transpose(query, [0, 2, 1, 3]), c, seq, prefix + ".query_rope")
    key = _rope(b, b.transpose(key, [0, 2, 1, 3]), c, seq, prefix + ".key_rope")
    value = b.transpose(value, [0, 2, 1, 3])
    if kv != h:
        key = b.concat([b.slice(key, i, i + 1, 1) for i in range(kv) for _ in range(h // kv)], 1)
        value = b.concat([b.slice(value, i, i + 1, 1) for i in range(kv) for _ in range(h // kv)], 1)
    scores = b.element("Mul", b.matmul(query, b.transpose(key, [0, 1, 3, 2])), b.const(d ** -0.5), prefix + ".scores")
    probabilities = []
    for t in range(seq):
        row = b.slice(b.slice(scores, t, t + 1, 2), 0, t + 1, 3)
        row = b.op("Softmax", row, shape=b.shapes[row], axis=-1, name=f"{prefix}.softmax_{t}")
        if t + 1 < seq:
            row = b.concat([row, b.const(np.zeros((1, h, 1, seq - t - 1)))], -1)
        probabilities.append(row)
    probabilities = b.concat(probabilities, 2, prefix + ".probabilities")
    attended = b.reshape(b.transpose(b.matmul(probabilities, value), [0, 2, 1, 3]), (1, seq, h * d))
    gated = b.element("Mul", attended, b.unary("Sigmoid", gate), prefix + ".gated")
    return b.linear(gated, prefix + ".o_proj.weight", prefix + ".output")


def _gated_delta(b, x, c, seq, prefix):
    kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
    qk_dim, v_dim = kh * kd, vh * vd
    width, channels = c["linear_conv_kernel_dim"], 2 * qk_dim + v_dim
    mixed = b.linear(x, prefix + ".in_proj_qkv.weight")
    conv, kernel = None, b.reshape(prefix + ".conv1d.weight", (channels, width))
    for lag in range(min(width, seq)):
        shifted = mixed if lag == 0 else b.concat([b.const(np.zeros((1, lag, channels))), b.slice(mixed, 0, seq - lag, 1)], 1)
        weight = b.reshape(b.slice(kernel, width - lag - 1, width - lag, 1), (channels,))
        term = b.element("Mul", shifted, weight)
        conv = term if conv is None else b.element("Add", conv, term)
    mixed = b.silu(conv, prefix + ".conv_silu")
    q = b.reshape(b.slice(mixed, 0, qk_dim, -1), (1, seq, kh, kd))
    k = b.reshape(b.slice(mixed, qk_dim, 2 * qk_dim, -1), (1, seq, kh, kd))
    v = b.reshape(b.slice(mixed, 2 * qk_dim, channels, -1), (1, seq, vh, vd))
    q = b.norm(q, None, 1e-6, l2=True, name=prefix + ".query_l2")
    k = b.norm(k, None, 1e-6, l2=True, name=prefix + ".key_l2")
    q = b.element("Mul", q, b.const(kd ** -0.5))
    q, k, v = [b.transpose(tensor, [0, 2, 1, 3]) for tensor in (q, k, v)]
    if vh != kh:
        q = b.concat([b.slice(q, i, i + 1, 1) for i in range(kh) for _ in range(vh // kh)], 1)
        k = b.concat([b.slice(k, i, i + 1, 1) for i in range(kh) for _ in range(vh // kh)], 1)
    beta = b.transpose(b.unary("Sigmoid", b.linear(x, prefix + ".in_proj_b.weight")), [0, 2, 1])
    a = b.element("Add", b.linear(x, prefix + ".in_proj_a.weight"), prefix + ".dt_bias")
    a_exp = b.unary("Exp", prefix + ".A_log", prefix + ".A_exp")
    g = b.element("Mul", b.unary("Softplus", a), b.element("Mul", a_exp, b.const(-1.0)))
    alpha = b.transpose(b.unary("Exp", g, prefix + ".decay"), [0, 2, 1])
    beta = b.reshape(beta, (1, vh, seq, 1))
    rows, previous = [], None
    for t in range(seq):
        diagonal = b.const(np.ones((1, vh, 1, 1)))
        if t:
            alpha_t = b.reshape(b.slice(alpha, t, t + 1, 2), (1, vh, 1, 1))
            row = b.concat([b.element("Mul", previous, alpha_t), diagonal], -1)
        else:
            row = diagonal
        previous = row
        if t + 1 < seq:
            row = b.concat([row, b.const(np.zeros((1, vh, 1, seq - t - 1)))], -1)
        rows.append(row)
    decay = b.concat(rows, 2, prefix + ".decay_products")
    kt = b.transpose(k, [0, 1, 3, 2])
    kk = b.matmul(k, kt, prefix + ".key_products")
    lower = b.element("Mul", b.element("Mul", b.element("Mul", beta, decay), kk),
                      b.const(np.tril(np.ones((seq, seq)), -1)), prefix + ".strict_lower")
    rhs = b.element("Mul", beta, v, prefix + ".beta_value")
    updates = None
    for t in range(seq):
        current = b.slice(rhs, t, t + 1, 2)
        if t:
            row = b.slice(b.slice(lower, t, t + 1, 2), 0, t, 3)
            current = b.element("Sub", current, b.matmul(row, updates), f"{prefix}.update_{t}")
            updates = b.concat([updates, current], 2)
        else:
            updates = current
    qk = b.matmul(q, kt, prefix + ".query_key_products")
    output = b.matmul(b.element("Mul", decay, qk), updates, prefix + ".core_output")
    output = b.transpose(output, [0, 2, 1, 3])
    output = b.norm(output, prefix + ".norm.weight", c["rms_norm_eps"], zero_centered=False, name=prefix + ".output_norm")
    z = b.reshape(b.linear(x, prefix + ".in_proj_z.weight"), (1, seq, vh, vd))
    output = b.element("Mul", output, b.silu(z, prefix + ".z_silu"))
    return b.linear(b.reshape(output, (1, seq, v_dim)), prefix + ".out_proj.weight", prefix + ".output")


def build_model(config, manifest, seq_len, *, debug=False, all_logits=False):
    """Build the whole Dense text graph, retaining external weight references."""
    c = config.get("text_config", config)
    if seq_len <= 0 or seq_len > c.get("max_position_embeddings", 262144):
        raise ValueError("sequence length outside model context")
    if c.get("hidden_act", "silu") != "silu" or c.get("attention_bias", False):
        raise ValueError("only bias-free SiLU Qwen3.5 Dense checkpoints are supported")
    if not c.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
        raise ValueError("this frontend requires tied Qwen3.5 token embeddings")
    if c["num_attention_heads"] % c["num_key_value_heads"] or c["linear_num_value_heads"] % c["linear_num_key_heads"]:
        raise ValueError("nonintegral head grouping")
    types = c.get("layer_types", ["full_attention" if (i + 1) % c.get("full_attention_interval", 4) == 0
                                  else "linear_attention" for i in range(c["num_hidden_layers"])])
    if len(types) != c["num_hidden_layers"] or set(types) - {"linear_attention", "full_attention"}:
        raise ValueError("invalid layer_types")
    b = GraphBuilder(); _external_initializers(b, manifest)
    b.shapes["input_ids"], b.types["input_ids"] = (1, seq_len), TensorProto.INT64
    x = b.op("Gather", TEXT_PREFIX + "embed_tokens.weight", "input_ids", axis=0,
             name="embedding", shape=(1, seq_len, c["hidden_size"]))
    debug_names = [x]
    for layer, layer_type in enumerate(types):
        prefix = f"{TEXT_PREFIX}layers.{layer}"
        normalized = b.norm(x, prefix + ".input_layernorm.weight", c["rms_norm_eps"], name=f"layer_{layer:02d}.input_norm")
        if layer_type == "linear_attention":
            mixed = _gated_delta(b, normalized, c, seq_len, prefix + ".linear_attn")
        else:
            mixed = _full_attention(b, normalized, c, seq_len, prefix + ".self_attn")
        x = b.element("Add", x, mixed, f"layer_{layer:02d}.attention_residual")
        normalized = b.norm(x, prefix + ".post_attention_layernorm.weight", c["rms_norm_eps"], name=f"layer_{layer:02d}.post_norm")
        gate = b.silu(b.linear(normalized, prefix + ".mlp.gate_proj.weight"), f"layer_{layer:02d}.mlp_silu")
        up = b.linear(normalized, prefix + ".mlp.up_proj.weight")
        down = b.linear(b.element("Mul", gate, up), prefix + ".mlp.down_proj.weight")
        x = b.element("Add", x, down, f"layer_{layer:02d}")
        debug_names.append(x)
    x = b.norm(x, TEXT_PREFIX + "norm.weight", c["rms_norm_eps"], name="final_norm")
    debug_names.append(x)
    last = b.slice(x, seq_len - 1, seq_len, 1, "last_hidden")
    logits = b.linear(last, TEXT_PREFIX + "embed_tokens.weight", "last_logits")
    output_names = [logits]
    if all_logits or debug:
        output_names.append(b.linear(x, TEXT_PREFIX + "embed_tokens.weight", "logits"))
    if debug:
        output_names.extend(debug_names)
    initializers = {t.name for t in b.initializers}
    info = [b.value_info(name) for name in b.shapes if name not in initializers | {"input_ids"} | set(output_names)]
    graph = helper.make_graph(b.nodes, "Qwen3.5-Dense-text", [b.value_info("input_ids")],
                              [b.value_info(n) for n in output_names], initializer=b.initializers, value_info=info)
    model = helper.make_model(graph, producer_name="spain-qwen3.5-onnx", opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    helper.set_model_props(model, {"spain.model": "qwen3_5_text", "spain.seq_len": str(seq_len),
                                  "spain.layers": str(c["num_hidden_layers"]),
                                  "spain.source_hashes": json.dumps(manifest["sources"], sort_keys=True),
                                  "spain.weights_sha256": manifest["external_sha256"], "spain.numerics": "exact",
                                  "spain.export_code_sha256": sha256_file(__file__)})
    return model


def lower_nonlinearities(model):
    """Lower nonlinearities to bounded algebra; compiler constrains spain.domains."""
    result = onnx.ModelProto(); result.CopyFrom(model)
    properties = {p.key: p.value for p in result.metadata_props}
    if properties.get("spain.numerics", "").startswith("pade-v"):
        if properties.get("spain.numerics") != "pade-v2":
            raise ValueError("refusing to relabel an older lowered graph; lower its exact original graph instead")
        return result
    b = GraphBuilder()
    for tensor in result.graph.initializer:
        b.shapes[tensor.name], b.types[tensor.name] = tuple(tensor.dims), tensor.data_type
    for value in list(result.graph.input) + list(result.graph.value_info) + list(result.graph.output):
        b.shapes[value.name] = tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
        b.types[value.name] = value.type.tensor_type.elem_type
    b._counter = 1000000
    domains = json.loads(properties.get("spain.domains", "[]"))
    known_domains = {(d["tensor"], d["lower"], d["upper"]) for d in domains}

    def domain(value, lower, upper):
        if (value, lower, upper) not in known_domains:
            domains.append({"tensor": value, "lower": lower, "upper": upper})
            known_domains.add((value, lower, upper))

    def pade_parts(value):
        z = b.element("Mul", value, b.const(1.0 / 1024.0))
        z2 = b.element("Mul", z, z); z3 = b.element("Mul", z2, z)
        even = b.element("Add", b.const(120.0), b.element("Mul", b.const(12.0), z2))
        odd = b.element("Add", b.element("Mul", b.const(60.0), z), z3)
        return b.element("Add", even, odd), b.element("Sub", even, odd)

    def exp_approx(value, constant=False):
        domain(value, -16.0 if constant else -128.0, 16.0 if constant else 0.0)
        p, q = pade_parts(value); exponential = b.element("Div", p, q)
        for _ in range(10):
            exponential = b.element("Mul", exponential, exponential)
        return exponential

    def sigmoid(value):
        domain(value, -128.0, 128.0)
        p, q = pade_parts(value); s = b.element("Div", p, b.element("Add", p, q))
        for _ in range(10):
            complement = b.element("Sub", b.const(1.0), s)
            square = b.element("Mul", s, s)
            s = b.element("Div", square, b.element("Add", square, b.element("Mul", complement, complement)))
        return s

    def softplus(value):
        domain(value, -128.0, 128.0)
        absolute = b.unary("Sqrt", b.element("Mul", value, value))
        e = exp_approx(b.element("Mul", absolute, b.const(-1.0)))
        u = b.element("Div", e, b.element("Add", b.const(2.0), e))
        u2 = b.element("Mul", u, u)
        polynomial = b.const(1.0 / 39.0)
        for j in range(18, -1, -1):
            polynomial = b.element("Add", b.const(1.0 / (2 * j + 1)), b.element("Mul", u2, polynomial))
        logarithm = b.element("Mul", b.const(2.0), b.element("Mul", u, polynomial))
        positive = b.element("Mul", b.const(0.5), b.element("Add", value, absolute))
        return b.element("Add", positive, logarithm)

    def silu(value):
        domain(value, -128.0, 128.0)
        absolute = b.unary("Sqrt", b.element("Mul", value, value))
        e = exp_approx(b.element("Mul", absolute, b.const(-1.0)))
        positive = b.element("Add", value, absolute)
        negative = b.element("Sub", value, absolute)
        numerator = b.element("Add", positive, b.element("Mul", negative, e))
        denominator = b.element("Mul", b.const(2.0), b.element("Add", b.const(1.0), e))
        return b.element("Div", numerator, denominator)

    # A fused algebraic SiLU avoids separately materializing the much larger
    # bounded sigmoid-doubling graph at every MLP/convolution element.
    producers = {output: node for node in result.graph.node for output in node.output}
    uses = {}
    for node in result.graph.node:
        for name in node.input:
            uses[name] = uses.get(name, 0) + 1
    public_outputs = {v.name for v in result.graph.output}
    fused_silu, skipped_sigmoids = {}, set()
    for node in result.graph.node:
        if node.op_type != "Mul":
            continue
        for x, candidate in [node.input, list(reversed(node.input))]:
            producer = producers.get(candidate)
            if (producer is not None and producer.op_type == "Sigmoid" and producer.input[0] == x
                    and uses[candidate] == 1 and candidate not in public_outputs):
                fused_silu[node.output[0]] = x
                skipped_sigmoids.add(candidate)
                break
    for node in result.graph.node:
        if node.output[0] in skipped_sigmoids:
            continue
        if node.output[0] in fused_silu:
            b.unary("Identity", silu(fused_silu[node.output[0]]), node.output[0])
            continue
        if node.op_type not in {"Exp", "Sigmoid", "Softplus", "Softmax"}:
            b.nodes.append(node)
            continue
        value = node.input[0]
        if node.op_type == "Exp":
            lowered = exp_approx(value, constant=value.endswith(".A_log"))
        elif node.op_type == "Sigmoid":
            lowered = sigmoid(value)
        elif node.op_type == "Softplus":
            lowered = softplus(value)
        else:
            axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
            centered = b.element("Sub", value, b.reduce("ReduceMax", value, axis))
            exponential = exp_approx(centered)
            lowered = b.element("Div", exponential, b.reduce("ReduceSum", exponential, axis))
        b.unary("Identity", lowered, node.output[0])
    del result.graph.node[:]; result.graph.node.extend(b.nodes)
    result.graph.initializer.extend(b.initializers)
    retained_info = [v for v in result.graph.value_info if v.name not in skipped_sigmoids]
    del result.graph.value_info[:]; result.graph.value_info.extend(retained_info)
    existing = {v.name for v in result.graph.value_info} | {v.name for v in result.graph.input} | {v.name for v in result.graph.output}
    initializers = {v.name for v in result.graph.initializer}
    result.graph.value_info.extend(b.value_info(name) for name in b.shapes if name not in existing | initializers | skipped_sigmoids)
    properties.update({"spain.numerics": "pade-v2", "spain.domains": json.dumps(domains, separators=(",", ":")),
                       "spain.approximation": json.dumps(NUMERICS, sort_keys=True)})
    helper.set_model_props(result, properties)
    return result


def export_model(model_dir, output_dir, seq_len, *, approximate=False, debug=False,
                 all_logits=False, weights_dir=None) -> Path:
    """Export the full model and manifest.  Return the standard ONNX model path."""
    source, destination = Path(model_dir).resolve(), Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    weights = Path(weights_dir).resolve() if weights_dir is not None else destination.parent / "weights"
    manifest = export_shared_weights(source, weights)
    sidecar, actual = destination / "weights.bin", weights / "weights.bin"
    if sidecar != actual:
        if sidecar.is_symlink() and sidecar.resolve() != actual:
            raise ValueError(f"weights symlink points to different export: {sidecar}")
        if sidecar.is_symlink():
            sidecar.unlink()
        if sidecar.exists() and not os.path.samefile(sidecar, actual):
            raise ValueError(f"refusing to replace independent weights file: {sidecar}")
        if not sidecar.exists():
            # ORT rejects symlinks escaping the graph directory.  Hard links
            # share the same physical data without weakening its path checks.
            os.link(actual, sidecar)
    config = json.loads((source / "config.json").read_text())
    model = build_model(config, manifest, seq_len, debug=debug, all_logits=all_logits)
    if approximate:
        model = lower_nonlinearities(model)
    path = destination / ("approximate_model.onnx" if approximate else "original_model.onnx")
    onnx.save_model(model, str(path))
    onnx.checker.check_model(str(path), full_check=False)
    graph_manifest = {"schema_version": 1, "export_version": EXPORT_VERSION,
                      "export_code_sha256": sha256_file(__file__),
                      "model_path": str(path), "model_sha256": sha256_file(path),
                      "model_dir": str(source), "seq_len": seq_len, "batch_size": 1,
                      "layers": config.get("text_config", config)["num_hidden_layers"],
                      "approximate": approximate, "debug": debug, "all_logits": all_logits or debug,
                      "weights_manifest": str(weights / "weights_manifest.json"),
                      "weights_sha256": manifest["external_sha256"], "sources": manifest["sources"],
                      "tokenizer_dir": str(source), "outputs": [v.name for v in model.graph.output],
                      "operator_counts": {op: sum(n.op_type == op for n in model.graph.node)
                                          for op in sorted({n.op_type for n in model.graph.node})},
                      "numerics": NUMERICS if approximate else "exact ONNX nonlinear operators"}
    _json_write(destination / ("approximate_manifest.json" if approximate else "graph_manifest.json"), graph_manifest)
    return path


def load_manifest(path):
    return json.loads(Path(path).read_text())


def write_tiny_fixture(directory, *, seed=7) -> Path:
    """Deterministic four-layer hybrid fixture for compiler/executor tests."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6, tie_word_embeddings=True,
        max_position_embeddings=128,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    )
    model = Qwen3_5ForCausalLM(config).eval()
    weights = {TEXT_PREFIX + key.removeprefix("model."): value.contiguous()
               for key, value in model.state_dict().items() if key.startswith("model.")}
    save_file(weights, str(destination / "model.safetensors"))
    _json_write(destination / "config.json", {"model_type": "qwen3_5", "tie_word_embeddings": True,
                                             "text_config": config.to_dict()})
    return destination
