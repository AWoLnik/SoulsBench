#!/usr/bin/env bash
# Kev System One server on 127.0.0.1:8009 (the `kev` adapter's default). 0.8B by default: Kev-4B
# does not fit next to DS3 on a 12 GB GPU. Needs flash-linear-attention (else ~10x slower).
set -euo pipefail
M=${SOULSBENCH_MODELS:-$HOME/Sources/soulsbench-models}
cd "$M/kev"
CC="$M/laya-venv/bin/zigcc" KEV_DTYPE=bf16 \
    exec .venv/bin/python -m kev.serve --run "${KEV_RUN:-jaredpalmer/kev-0.8b}" --host 127.0.0.1 --port 8009
