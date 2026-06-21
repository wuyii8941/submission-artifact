#!/bin/bash
# Run after batch 1 (seeds 42, 123) finishes.
# Batch 2: seeds 456, 789 across 4 tools = 8 jobs on 8 GPUs.
# Batch 3: seed 1024 across 4 tools = 4 jobs on GPUs 0-3.
set -u
cd "$(dirname "$0")"

OUT=results_template
mkdir -p "$OUT" ${TMPDIR:-/tmp/deepopfuzz}
PYBIN=python3
WORKER=$(pwd)/_benchmark_worker.py
BUDGET=14400
WHITEFOX_DIR=${WHITEFOX_TEST_DIR}

launch() {
  local mode=$1 seed=$2 gpu=$3
  local ckpt="$OUT/${mode}_seed${seed}.json"
  local log="$OUT/${mode}_seed${seed}.log"
  local cache=./cache/torchinductor_pt220_template_seed${seed}_${mode}
  local tcache=./cache/triton_pt220_template_seed${seed}_${mode}
  mkdir -p "$cache" "$tcache"
  echo "[$(date +%H:%M:%S)] $mode seed=$seed GPU=$gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TORCHINDUCTOR_CACHE_DIR="$cache" \
  TRITON_CACHE_DIR="$tcache" \
  TMPDIR=${TMPDIR:-/tmp/deepopfuzz} \
  WHITEFOX_TEST_DIR="$WHITEFOX_DIR" \
  nohup "$PYBIN" "$WORKER" \
    --mode "$mode" --seed "$seed" --time-budget "$BUDGET" \
    --device cuda --checkpoint "$ckpt" > "$log" 2>&1 &
}

# Wait for batch 1 (any seed 42 or 123 worker) to finish.
echo "[$(date +%H:%M:%S)] Waiting for batch 1 (seeds 42, 123) to finish..."
while pgrep -af "_benchmark_worker.py --mode .* --seed (42|123) " > /dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] Batch 1 done."

# Batch 2: seeds 456, 789
echo "[$(date +%H:%M:%S)] Launching batch 2: seeds 456, 789"
launch targeted 456  0
launch random   456  1
launch nnsmith  456  2
launch whitefox 456  3
launch targeted 789  4
launch random   789  5
launch nnsmith  789  6
launch whitefox 789  7
wait
echo "[$(date +%H:%M:%S)] Batch 2 done."

# Batch 3: seed 1024
echo "[$(date +%H:%M:%S)] Launching batch 3: seed 1024"
launch targeted 1024 0
launch random   1024 1
launch nnsmith  1024 2
launch whitefox 1024 3
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
