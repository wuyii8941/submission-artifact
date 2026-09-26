#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$HERE/../../evaluation/rq2_efficiency/_benchmark_worker.py"
export PYTHONPATH="$HERE/../../kernelfuzz${PYTHONPATH:+:$PYTHONPATH}"
OUT=${OUT:-$HERE/data}/rq2_runs
PY=${PT_PY:-python3}
BUDGET=${BUDGET:-14400}
IFS=, read -ra G <<< "${GPUS:-0,1}"
mkdir -p "$OUT"
i=0
for seed in 42 123 456; do
  for mode in targeted random; do
    gpu=${G[$((i % ${#G[@]}))]}
    mkdir -p "$OUT/cache/tmp_${mode}_${seed}"
    CUDA_VISIBLE_DEVICES="$gpu" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/ind_${mode}_${seed}" \
    TRITON_CACHE_DIR="$OUT/cache/tri_${mode}_${seed}" TMPDIR="$OUT/cache/tmp_${mode}_${seed}" \
      "$PY" "$WORKER" --mode "$mode" --seed "$seed" --time-budget "$BUDGET" --device cuda \
      --checkpoint "$OUT/${mode}_seed${seed}.ckpt" > "$OUT/${mode}_seed${seed}.log" 2>&1 &
    i=$((i + 1))
    if (( i % ${#G[@]} == 0 )); then wait; fi
  done
done
wait
for seed in 42 123 456; do
  for mode in targeted random; do
    tail -n 1 "$OUT/${mode}_seed${seed}.log" > "$OUT/${mode}_seed${seed}.json"
  done
done
rm -rf "$OUT/cache"
