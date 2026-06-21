#!/bin/bash
# Launch 5 seeds x 4 tools across 8 GPUs for template_sample_counts.
# Seed-major order so one full seed completes per batch.
#   Batch 1: seeds 42+123 (8 jobs)
#   Batch 2: seeds 456+789 (8 jobs)
#   Batch 3: seed 1024 (4 jobs, GPUs 0-3)
#
# Per-job env: TMPDIR + per-seed Inductor/Triton cache dirs + WHITEFOX_TEST_DIR.
# Skips a job if its checkpoint JSON is currently being updated by an alive worker
# (so this can be re-run safely after partial progress).

set -u
cd "$(dirname "$0")"

OUT=results_template
mkdir -p "$OUT"
mkdir -p ${TMPDIR:-/tmp/deepopfuzz}

PYBIN=python3
WORKER=$(pwd)/_benchmark_worker.py
BUDGET=14400  # 4 hours
WHITEFOX_DIR=${WHITEFOX_TEST_DIR}

NGPU=8

# Job list: seed-major order (within each seed: targeted, random, nnsmith, whitefox)
JOBS=(
  "targeted 42"  "random 42"  "nnsmith 42"  "whitefox 42"
  "targeted 123" "random 123" "nnsmith 123" "whitefox 123"
  "targeted 456" "random 456" "nnsmith 456" "whitefox 456"
  "targeted 789" "random 789" "nnsmith 789" "whitefox 789"
  "targeted 1024" "random 1024" "nnsmith 1024" "whitefox 1024"
)

# Helper: launch one job on a given GPU (sets per-seed cache + mode-specific env).
launch_job() {
  local mode=$1
  local seed=$2
  local gpu=$3
  local ckpt="$OUT/${mode}_seed${seed}.json"
  local log="$OUT/${mode}_seed${seed}.log"

  # Skip if a worker for this (mode, seed) is already alive.
  if pgrep -af "_benchmark_worker.py --mode $mode --seed $seed" > /dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $mode seed=$seed (already running)"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] Launch $mode seed=$seed on GPU $gpu -> $log"
  local cache=./cache/torchinductor_pt220_template_seed${seed}_${mode}
  local triton_cache=./cache/triton_pt220_template_seed${seed}_${mode}
  mkdir -p "$cache" "$triton_cache"

  CUDA_VISIBLE_DEVICES="$gpu" \
  TORCHINDUCTOR_CACHE_DIR="$cache" \
  TRITON_CACHE_DIR="$triton_cache" \
  TMPDIR=${TMPDIR:-/tmp/deepopfuzz} \
  WHITEFOX_TEST_DIR="$WHITEFOX_DIR" \
  nohup "$PYBIN" "$WORKER" \
    --mode "$mode" --seed "$seed" --time-budget "$BUDGET" \
    --device cuda --checkpoint "$ckpt" > "$log" 2>&1 &
  echo "  PID=$!"
  eval "GPU_PIDS_$gpu=$!"
}

# Wait for a GPU to be free (its assigned PID, if any, to exit).
wait_gpu() {
  local gpu=$1
  local var="GPU_PIDS_$gpu"
  local pid=${!var:-}
  if [ -n "$pid" ]; then
    wait "$pid" 2>/dev/null || true
    eval "GPU_PIDS_$gpu="
  fi
}

# Also wait for any external workers already running for this (mode, seed).
wait_existing() {
  local mode=$1
  local seed=$2
  while pgrep -af "_benchmark_worker.py --mode $mode --seed $seed" > /dev/null; do
    sleep 30
  done
}

# Initialise GPU_PIDS_* tracking (we use them to serialize per-GPU batches).
for g in $(seq 0 $((NGPU-1))); do
  eval "GPU_PIDS_$g="
done

# Account for already-running workers occupying GPUs from the prior partial launch.
# Map alive (mode, seed) -> the GPU they were assigned to by the original launcher
# (which used round-robin index = job_index % NGPU on the original ordering).
# We don't have a record, so we conservatively just wait for them via wait_existing
# when their slot comes up.

i=0
for spec in "${JOBS[@]}"; do
  mode=$(echo "$spec" | cut -d' ' -f1)
  seed=$(echo "$spec" | cut -d' ' -f2)
  gpu=$((i % NGPU))

  # Wait for previous job on this GPU
  wait_gpu "$gpu"
  # And wait for any externally-running worker for this (mode, seed)
  wait_existing "$mode" "$seed"
  launch_job "$mode" "$seed" "$gpu"
  i=$((i + 1))
done

# Final wait
for g in $(seq 0 $((NGPU-1))); do
  wait_gpu "$g"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
