#!/bin/bash
# Launch full WhiteFox 5-seed × 4h run on PyTorch 2.2.0 with kernel-capture patch.
# Each seed runs on its own GPU + own cache dirs to avoid contention.
set -u

SEEDS=(42 123 456 789 1024)
GPUS=(0 1 2 3 4)
BUDGET=14400  # 4 hours
WORKER=${ARTIFACT_ROOT}/kernelfuzz/evaluation/rq2_benchmark/_benchmark_worker.py
PYBIN=python3
WHITEFOX_DIR=${WHITEFOX_TEST_DIR}
OUT=${ARTIFACT_ROOT}/kernelfuzz/evaluation/rq2_benchmark/whitefox_kernel_patch/results

mkdir -p "$OUT"

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  ckpt="$OUT/whitefox_seed${seed}.json"
  log="$OUT/whitefox_seed${seed}.log"
  errlog="$OUT/whitefox_seed${seed}.err"

  echo "Launching seed=$seed gpu=$gpu -> $ckpt"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TORCHINDUCTOR_CACHE_DIR=./cache/torchinductor_pt220_seed${seed} \
  TRITON_CACHE_DIR=./cache/triton_pt220_seed${seed} \
  TMPDIR=${TMPDIR:-/tmp/kernelfuzz} \
  WHITEFOX_TEST_DIR="$WHITEFOX_DIR" \
  nohup "$PYBIN" "$WORKER" \
    --mode whitefox --seed "$seed" --time-budget "$BUDGET" --device cuda \
    --checkpoint "$ckpt" \
    >"$log" 2>"$errlog" &
  echo "  PID=$!"
done

wait
echo "All 5 runs completed."
