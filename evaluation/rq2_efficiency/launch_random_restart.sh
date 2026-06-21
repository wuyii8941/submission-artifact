#!/bin/bash
# Restart-on-crash wrapper for random mode (CUDA illegal-memory-access is sporadic
# in random-generated programs; worker uses checkpoint to resume).
# Each seed runs on its own GPU; restarts until time-budget elapsed.
set -u
cd "$(dirname "$0")"

OUT=results_template
PYBIN=python3
WORKER=$(pwd)/_benchmark_worker.py
BUDGET=14400
WHITEFOX_DIR=${WHITEFOX_TEST_DIR}

restart_until_done() {
  local seed=$1 gpu=$2
  local ckpt="$OUT/random_seed${seed}.json"
  local log="$OUT/random_seed${seed}.log"
  local cache=./cache/torchinductor_pt220_template_seed${seed}_random
  local tcache=./cache/triton_pt220_template_seed${seed}_random
  mkdir -p "$cache" "$tcache"

  local attempt=0
  while true; do
    attempt=$((attempt+1))
    local cur=0
    if [ -f "$ckpt" ]; then
      cur=$(python3 -c "import json; print(int(json.load(open('$ckpt')).get('elapsed', 0)))" 2>/dev/null || echo 0)
    fi
    if [ "$cur" -ge "$BUDGET" ]; then
      echo "[$(date +%H:%M:%S)] random seed=$seed COMPLETE (elapsed=${cur}s)"
      return
    fi
    echo "[$(date +%H:%M:%S)] random seed=$seed GPU=$gpu attempt #${attempt} (elapsed=${cur}/${BUDGET}s)"
    CUDA_VISIBLE_DEVICES="$gpu" \
    TORCHINDUCTOR_CACHE_DIR="$cache" \
    TRITON_CACHE_DIR="$tcache" \
    TMPDIR=${TMPDIR:-/tmp/deepopfuzz} \
    WHITEFOX_TEST_DIR="$WHITEFOX_DIR" \
    "$PYBIN" "$WORKER" --mode random --seed "$seed" --time-budget "$BUDGET" \
      --device cuda --checkpoint "$ckpt" >> "$log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] random seed=$seed attempt #${attempt} exited rc=$rc"
    sleep 5
  done
}

# Use GPUs 1, 4, 5, 6, 7 (avoid 0/2/3 which batch 3 is occupying).
restart_until_done 42   1 &
restart_until_done 123  4 &
restart_until_done 456  5 &
restart_until_done 789  6 &
restart_until_done 1024 7 &
wait
echo "[$(date +%H:%M:%S)] ALL RANDOM DONE"
