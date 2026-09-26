#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
OUT=${1:-results/runs}
SECONDS_BUDGET=${SECONDS_BUDGET:-7200}
for seed in 42 123 456; do
  for arm in pattern_uniform pass_uniform hit_feedback; do
    out=$OUT/formal_primary/${arm}_seed${seed}
    if [ -e "$out" ]; then echo "exists: $out" >&2; exit 1; fi
    mkdir -p "$(dirname "$out")"
    "$PY" a3_run.py --arm "$arm" --seed "$seed" --targets targets_primary.json \
      --seconds "$SECONDS_BUDGET" --output "$out" > /dev/null 2>&1 &
  done
done
wait
