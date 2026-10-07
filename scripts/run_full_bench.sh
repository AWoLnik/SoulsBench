#!/usr/bin/env bash
# The full apples-to-apples benchmark: same scenario, seeds, prompt and SLA for every model; each
# model's best fight recorded with OBS. Resumable: re-run the script and every pass picks up where
# it stopped (finished passes are no-ops).
#
# Local models cannot share the 12 GB GPU with each other and DS3, so each gets its own pass with
# only its server loaded; the hosted models run interleaved by seed in one pass.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-50}
BENCH_NAME=${BENCH_NAME:-full50}  # not NAME: shells often set NAME to the hostname
LOGS=runs/bench_${BENCH_NAME}_logs
mkdir -p "$LOGS"
RUN=(uv run python -m soulsbench.bench --seeds "$SEEDS" --record-best)

wait_for() {  # wait_for <url> <name>
  for _ in $(seq 1 120); do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' -X POST "$1" -H 'content-type: application/json' -d '{}' || true)
    [[ "$code" =~ ^(200|400|422)$ ]] && { echo "$2 is up"; return 0; }
    sleep 3
  done
  echo "$2 did not come up" >&2; return 1
}

stop_servers() {  # Kev ignores SIGTERM: wait, then SIGKILL, and never start a second copy
  local pids
  pids=$(ps -eo pid=,args= | awk '/[k]ev\.serve --run|[l]aya-serve/ {print $1}')
  [ -z "$pids" ] && return 0
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 10); do
    pids=$(ps -eo pid=,args= | awk '/[k]ev\.serve --run|[l]aya-serve/ {print $1}')
    [ -z "$pids" ] && return 0
    sleep 1
  done
  kill -9 $pids 2>/dev/null || true
  sleep 2
  pids=$(ps -eo pid=,args= | awk '/[k]ev\.serve --run|[l]aya-serve/ {print $1}')
  [ -z "$pids" ] || { echo "could not stop model servers: $pids" >&2; return 1; }
}

local_pass() {  # local_pass <adapter> <serve script> <url>
  stop_servers
  nohup "scripts/$2" > "$LOGS/$1-server.log" 2>&1 &
  wait_for "$3" "$1"
  "${RUN[@]}" --name "$BENCH_NAME-$1" --adapters "$1" | tee -a "$LOGS/$1.log"
  stop_servers
}

local_pass kev serve_kev.sh http://127.0.0.1:8009/v1/systemone
local_pass laya serve_laya.sh http://127.0.0.1:8010/v1/systemone
"${RUN[@]}" --name "$BENCH_NAME-hosted" --adapters openai,jev,pplx | tee -a "$LOGS/hosted.log"

uv run python -m soulsbench.report runs/bench_"$BENCH_NAME"-*/ --out "runs/${BENCH_NAME}_comparison.md"
echo "FULL BENCH COMPLETE: runs/${BENCH_NAME}_comparison.md"
