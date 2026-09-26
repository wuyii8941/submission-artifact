#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MODE=$1; BUILD=${2:-nightly}
PY=$(cd "$HERE" && python3 -c "import oracle; print(oracle.PY['$BUILD'])")
OUT=$HERE/runs/$([ "$BUILD" = nightly ] && echo "$MODE" || echo "$BUILD/$MODE"); mkdir -p "$OUT"
IFS=, read -ra G <<< "${GPUS:-0,1,2,3}"
i=0
while read -r b; do
  [ -z "$b" ] && continue
  echo "CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} $PY $HERE/runner.py $PY $REPO/data/confirmed_bugs/pytorch/$b $MODE $OUT/$b.json"
  i=$((i + 1))
done < <(cd "$HERE" && python3 -c "import oracle, os, sys; m, b = sys.argv[1:3]; print(*(sorted(d for d in os.listdir(os.path.join(os.path.dirname(oracle.HERE), os.pardir, 'data', 'confirmed_bugs', 'pytorch')) if d.startswith('bug_')) if m == 'baseline' else [k for k, v in oracle.ORACLE.items() if v[0] == b]), sep=chr(10))" "$MODE" "$BUILD") | xargs -P "${#G[@]}" -I{} bash -c "{}"
