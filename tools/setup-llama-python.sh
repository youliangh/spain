#!/usr/bin/env bash
# Create an independent project environment, optionally from an existing tested environment.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$#" -eq 2 ] && [ "$1" = "--seed-venv" ]; then
    seed_python="$2/bin/python"
    seed_site="$("$seed_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    python3 -m venv --without-pip "$project_root/.venv"
    target_site="$("$project_root/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    seed_version="$("$seed_python" -c 'import sys; print(sys.version_info[:2])')"
    target_version="$("$project_root/.venv/bin/python" -c 'import sys; print(sys.version_info[:2])')"
    [ "$seed_version" = "$target_version" ] || { echo 'Seed and target Python versions differ' >&2; exit 1; }
    cp -a --reflink=auto "$seed_site/." "$target_site/"
elif [ "$#" -eq 0 ]; then
    python3 -m venv "$project_root/.venv"
    "$project_root/.venv/bin/python" -m pip install -r "$project_root/tools/requirements-llama.txt"
else
    echo 'usage: setup-llama-python.sh [--seed-venv PATH]' >&2
    exit 1
fi
"$project_root/.venv/bin/python" -c 'import torch, transformers, safetensors, tokenizers; print("Ready:",torch.__version__,transformers.__version__,safetensors.__version__,tokenizers.__version__)'
