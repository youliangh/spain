"""Public compiler entrypoint, retaining the standard ONNX artifact pipeline."""
from .onnx_v2 import compile_v2, estimate_v2


def compile_model(graph, output_dir, *, lower=True):
    return compile_v2(graph, output_dir, lower=lower)


def estimate_model(graph, *, lower=True):
    return estimate_v2(graph, lower=lower)
