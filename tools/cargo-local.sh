#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
export CARGO_HOME="$project_root/.cache/cargo"
export RUSTUP_HOME="$project_root/.cache/rustup"
export PATH="$project_root/.cache/native/usr/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/.cache/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CFLAGS="${CFLAGS:--O2 -std=gnu11}"
export GMP_MPFR_SYS_CACHE="$project_root/.cache/gmp-mpfr"
exec cargo +nightly "$@"
