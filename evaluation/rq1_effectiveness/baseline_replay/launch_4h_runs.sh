#!/bin/bash
# Launch the 4-hour baseline replay used for Table II (RQ1 baseline coverage of
# 30 PyTorch/Inductor root causes).
#
# Runs (one seed each, 4 hours wall-clock):
#   DeepOpFuzz, Random, FreeFuzz, NNSmith, WhiteFox
#
# Output layout under $OUT_ROOT:
#   deepopfuzz/graph-oracle/{potential-bug,crash}/<trigger>/*.py
#   <baseline>_ckpt/triggers/<baseline>_seed42/trigger_*.py
#   <baseline>_4h.log
#
# Requirements:
#   - DOF_PT_PYTHON pointing at a PyTorch 2.11 + Triton 3.6 env (with NNSmith,
#     ml_dtypes, hydra-core installed for the NNSmith mode)
#   - KF_FREEFUZZ_POOL pointing at a directory of pre-generated FreeFuzz programs
#   - WhiteFox tests under ${WHITEFOX_TEST_DIR} (default:
#     ./whitefox-baseline/starcoder-generated/torch-inductor-generated;
#     override via the env var or edit the worker's `test_dir` constant)
#   - 5 CUDA GPUs available (one per run)
#
# Single seed by design: this is a 4h budget audit, not a statistical comparison.
# Reviewers wanting variance estimates should re-run with multiple seeds.

set -e

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/runs_4h}"
TIME_BUDGET="${TIME_BUDGET:-14400}"
SEED="${SEED:-42}"

DOF_PT_PYTHON="${DOF_PT_PYTHON:?need PyTorch 2.11 env path; e.g. export DOF_PT_PYTHON=/path/to/pt211/bin/python}"
KF_FREEFUZZ_POOL="${KF_FREEFUZZ_POOL:?need FreeFuzz pool path; e.g. export KF_FREEFUZZ_POOL=/path/to/freefuzz_pool}"

# Locate kernelfuzz and the rq2_efficiency worker (the same worker is used by
# RQ2; here we drive it with a single seed and on PyTorch 2.11 instead of 2.2.0).
KERNELFUZZ_SRC="$ARTIFACT_ROOT/kernelfuzz"
RQ2_WORKER="$ARTIFACT_ROOT/evaluation/rq2_efficiency/_benchmark_worker.py"
FREEFUZZ_SRC="${FREEFUZZ_SRC:-$ARTIFACT_ROOT/evaluation/baselines/freefuzz_baseline/src}"

mkdir -p "$OUT_ROOT/deepopfuzz"
for mode in random freefuzz nnsmith whitefox; do
  mkdir -p "$OUT_ROOT/${mode}_ckpt"
done

echo "[$(date)] launching 5 runs into $OUT_ROOT (4h each, single seed=$SEED)"

# ── DeepOpFuzz (targeted fuzzer) on GPU 0 ────────────────────────────────────
PYTHONPATH="$KERNELFUZZ_SRC:$FREEFUZZ_SRC" \
CUDA_VISIBLE_DEVICES=0 \
"$DOF_PT_PYTHON" "$KERNELFUZZ_SRC/fuzzer.py" \
    --v4 --isolated \
    -o "$OUT_ROOT/deepopfuzz" \
    -n 400000 -t 240 \
    > "$OUT_ROOT/deepopfuzz_4h.log" 2>&1 &
echo "  DeepOpFuzz PID=$! on GPU 0"

# ── Four baselines on GPUs 1-4 ───────────────────────────────────────────────
for i in 0 1 2 3; do
  modes=(random freefuzz nnsmith whitefox)
  gpu=$((i+1))
  mode=${modes[$i]}
  ckpt="$OUT_ROOT/${mode}_ckpt/ckpt.json"
  PYTHONPATH="$KERNELFUZZ_SRC:$FREEFUZZ_SRC" \
  KF_FREEFUZZ_POOL="$KF_FREEFUZZ_POOL" \
  CUDA_VISIBLE_DEVICES=$gpu \
  "$DOF_PT_PYTHON" "$RQ2_WORKER" \
      --mode "$mode" --seed "$SEED" \
      --time-budget "$TIME_BUDGET" --device cuda \
      --checkpoint "$ckpt" \
      > "$OUT_ROOT/${mode}_4h.log" 2>&1 &
  echo "  $mode PID=$! on GPU $gpu"
done

wait
echo "[$(date)] all 5 runs finished. Run: python attribute_table2.py --runs-root $OUT_ROOT"
