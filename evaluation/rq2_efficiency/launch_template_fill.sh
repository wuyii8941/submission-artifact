#!/bin/bash
# Fill in the 4 missing batch-1 jobs (nnsmith, whitefox for seed 42 and 123)
# on idle GPUs 2, 3, 6, 7. The other 4 (targeted/random seed 42,123) are
# already running from the prior launch on GPUs 0, 1, 4, 5.
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
  echo "  PID=$!"
}

# Remove any partial JSON from the crashed earlier attempts
rm -f $OUT/nnsmith_seed42.json $OUT/nnsmith_seed123.json
rm -f $OUT/whitefox_seed42.json $OUT/whitefox_seed123.json

launch nnsmith  42  2
launch whitefox 42  3
launch nnsmith  123 6
launch whitefox 123 7

wait
echo "[$(date +%H:%M:%S)] Fill batch done."
