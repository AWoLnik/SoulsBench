#!/usr/bin/env bash
# Laya System One server on 127.0.0.1:8010 (the `laya` adapter's default). Needs the laya venv set
# up in ~/Sources/soulsbench-models (PyTorch cu126: driver 560.94 tops out at CUDA 12.6) and the
# zigcc wrapper for Triton's C compiler.
set -euo pipefail
M=${SOULSBENCH_MODELS:-$HOME/Sources/soulsbench-models}
cd "$M"
CC="$M/laya-venv/bin/zigcc" LAYA_HOST=127.0.0.1 LAYA_PORT=8010 LAYA_LOG_LEVEL=warning \
    exec laya-venv/bin/laya-serve
