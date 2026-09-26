#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PT_PY=${PT_PY:-python3}
TF_PY=${TF_PY:-python3}
TF_GPU_LIBS=${TF_GPU_LIBS:-$("$PT_PY" -c 'import glob, site; print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))')}
GPUS=${GPUS:-0}
ARTIFACT=${ARTIFACT:-$(cd "$HERE/../.." && pwd)}
OUT=${OUT:-$HERE/build_out}
CACHE=${CACHE:-/dev/shm/target_build_$$}
export ARTIFACT

mkdir -p "$OUT/logs" "$OUT/tmp" "$CACHE"
cd "$OUT"
export TMPDIR="$OUT/tmp" XDG_CACHE_HOME="$CACHE/xdg" CUDA_CACHE_PATH="$CACHE/nv" TF_CPP_MIN_LOG_LEVEL=3 \
       TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1
IFS=, read -ra GPU <<< "$GPUS"
NSH=${#GPU[@]}
: > timings.tsv
step() {
  local name=$1 dev=$2; shift 2
  local t0=$(date +%s)
  echo "[$(date +%T)] $name"
  CUDA_VISIBLE_DEVICES=$dev TORCHINDUCTOR_CACHE_DIR="$CACHE/ind_$name" TRITON_CACHE_DIR="$CACHE/tri_$name" \
    "$@" > "logs/$name.log" 2>&1 || { echo "  FAILED, see $OUT/logs/$name.log"; tail -5 "logs/$name.log"; exit 1; }
  printf '%s\t%s\n' "$name" $(( $(date +%s) - t0 )) >> timings.tsv
  grep -v -i 'warn' "logs/$name.log" | tail -2 | sed 's/^/  /'
}

step 01_pt_candidates  ""        "$PT_PY" "$HERE/pt_enumerate_candidates.py" pt_candidates.json
step 02_pt_families    ""        "$PT_PY" "$HERE/pt_apply_criteria.py" pt_candidates.json
( step 06_pt_conditions  "${GPU[0]}" "$PT_PY" "$HERE/pt_probe.py" conditions pt_conditions.json
  step 07_pt_composition "${GPU[0]}" "$PT_PY" "$HERE/pt_probe.py" composition pt_composition.json
  step 08_pt_dtypes      "${GPU[0]}" "$PT_PY" "$HERE/pt_probe.py" dtypes pt_conditions.json pt_dtypes.json ) &
pids=($!)
for i in "${!GPU[@]}"; do
  step "03_pt_entries_$i" "${GPU[$i]}" "$PT_PY" "$HERE/pt_probe.py" entries "$i" "$NSH" "pt_entries_$i.json" &
  pids+=($!)
done
step 04_pt_entry_metadata "" "$PT_PY" "$HERE/pt_entry_metadata.py" pt_entry_metadata.json
for p in "${pids[@]}"; do wait "$p" || { echo "a GPU step failed"; exit 1; }; done
step 05_pt_entry_pools ""        "$PT_PY" "$HERE/pt_select_entries.py"
step 09_pt_targets     ""        "$PT_PY" "$HERE/pt_form_targets.py"
step 07b_pt_templates  "${GPU[0]}" "$PT_PY" "$HERE/pt_probe.py" templates pt_composition_templates.json

TFENV=(env LD_LIBRARY_PATH="$TF_GPU_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
step 10_tf_registry    "${GPU[0]}" "${TFENV[@]}" "$TF_PY" "$HERE/tf_enumerate_candidates.py" tf_candidates.json
step 11_tf_trace       ""          "$TF_PY" "$HERE/tf_probe.py" trace tf_trace.json
step 12_tf_branches    ""          "$TF_PY" "$HERE/tf_shape_branches.py" "$("$TF_PY" -c 'import site;print(site.getsitepackages()[0])')" tf_shape_branches.json
step 13_tf_families    ""          "$TF_PY" "$HERE/tf_apply_criteria.py" families
step 15_tf_catalog     ""          "$TF_PY" "$HERE/tf_apply_criteria.py" catalog
step 15b_tf_targets    ""          "$TF_PY" "$HERE/tf_form_targets.py"

step 16_compare        ""          "$PT_PY" "$HERE/compare_with_artifact.py"

rm -rf "$CACHE"
rm -rf "$OUT/tmp"
echo "done: catalogs pt_entry_pools.json, pt_conditions.json, pt_composition.json, pt_composition_templates.json, pt_target_states.json, tf_targets.json;"
echo "      comparison REPORT.json; step times timings.tsv (all in $OUT)"
