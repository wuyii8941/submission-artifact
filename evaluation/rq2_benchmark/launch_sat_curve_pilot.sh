#!/bin/bash
# Saturation-curve pilot: 1 seed (42) × 4 tools × 4h, 4 GPUs in parallel.
# Worker now saves sat_counts in each timeseries entry.
# Random uses restart-on-crash (CUDA illegal-mem-access is sporadic in random).
set -u
cd "$(dirname "$0")"

OUT=results_sat_pilot
mkdir -p "$OUT" ${TMPDIR:-/tmp/kernelfuzz}
PYBIN=python3
WORKER=$(pwd)/_benchmark_worker.py
BUDGET=14400
WHITEFOX_DIR=${WHITEFOX_TEST_DIR}
SEED=42

run_worker() {
  local mode=$1 gpu=$2
  local ckpt="$OUT/${mode}_seed${SEED}.json"
  local log="$OUT/${mode}_seed${SEED}.log"
  local cache=./cache/torchinductor_pt220_sat_seed${SEED}_${mode}
  local tcache=./cache/triton_pt220_sat_seed${SEED}_${mode}
  mkdir -p "$cache" "$tcache"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TORCHINDUCTOR_CACHE_DIR="$cache" \
  TRITON_CACHE_DIR="$tcache" \
  TMPDIR=${TMPDIR:-/tmp/kernelfuzz} \
  WHITEFOX_TEST_DIR="$WHITEFOX_DIR" \
  "$PYBIN" "$WORKER" --mode "$mode" --seed "$SEED" --time-budget "$BUDGET" \
    --device cuda --checkpoint "$ckpt" >> "$log" 2>&1
}

restart_until_done() {
  local mode=$1 gpu=$2
  local ckpt="$OUT/${mode}_seed${SEED}.json"
  local attempt=0
  while true; do
    attempt=$((attempt+1))
    local cur=0
    if [ -f "$ckpt" ]; then
      cur=$(python3 -c "import json; print(int(json.load(open('$ckpt')).get('elapsed', 0)))" 2>/dev/null || echo 0)
    fi
    if [ "$cur" -ge "$BUDGET" ]; then
      echo "[$(date +%H:%M:%S)] $mode COMPLETE (elapsed=${cur}s)"
      return
    fi
    echo "[$(date +%H:%M:%S)] $mode seed=$SEED GPU=$gpu attempt #${attempt} (elapsed=${cur}/${BUDGET}s)"
    run_worker "$mode" "$gpu"
    echo "[$(date +%H:%M:%S)] $mode attempt #${attempt} exited rc=$?"
    sleep 5
  done
}

# Launch all 4 in parallel, each with restart-on-crash (only random typically crashes,
# but the wrapper is cheap for the others).
(restart_until_done targeted  0) &
(restart_until_done random    1) &
(restart_until_done nnsmith   2) &
(restart_until_done whitefox  3) &
wait
echo "[$(date +%H:%M:%S)] PILOT DONE"
