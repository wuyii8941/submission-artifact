#!/bin/bash
# Run TF 1800s experiments in parallel across GPUs
# targeted on GPU 3-4, random on GPU 5-6, nnsmith on GPU 7-11

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TF_PYTHON="${ARTIFACT_ROOT}/tf_nightly_env/bin/python3"
RESULTS="$SCRIPT_DIR/results"
mkdir -p "$RESULTS"

BUDGET=1800
SEEDS=(42 123 456 789 1024)

echo "=== TF 1800s Parallel Experiment ==="
echo "Budget: ${BUDGET}s, Seeds: ${SEEDS[*]}"

# --- Targeted (5 seeds on GPU 3-4) ---
run_targeted() {
    for i in "${!SEEDS[@]}"; do
        seed=${SEEDS[$i]}
        gpu=$((3 + i % 2))
        echo "[targeted] seed=$seed gpu=$gpu"
        CUDA_VISIBLE_DEVICES=$gpu $TF_PYTHON "$SCRIPT_DIR/tf_time_controlled.py" \
            --single targeted "$seed" "$BUDGET" "$RESULTS/_tmp_targeted_${seed}_1800.json" \
            2>"$RESULTS/_log_targeted_${seed}.txt" &
    done
    wait
    echo "[targeted] All seeds done"
}

# --- Random (5 seeds on GPU 5-6) ---
run_random() {
    for i in "${!SEEDS[@]}"; do
        seed=${SEEDS[$i]}
        gpu=$((5 + i % 2))
        echo "[random] seed=$seed gpu=$gpu"
        CUDA_VISIBLE_DEVICES=$gpu $TF_PYTHON "$SCRIPT_DIR/tf_time_controlled.py" \
            --single random "$seed" "$BUDGET" "$RESULTS/_tmp_random_${seed}_1800.json" \
            2>"$RESULTS/_log_random_${seed}.txt" &
    done
    wait
    echo "[random] All seeds done"
}

# --- NNSmith (5 seeds on GPU 7-11) ---
run_nnsmith() {
    for i in "${!SEEDS[@]}"; do
        seed=${SEEDS[$i]}
        gpu=$((7 + i))
        echo "[nnsmith] seed=$seed gpu=$gpu"
        CUDA_VISIBLE_DEVICES=$gpu $TF_PYTHON "$SCRIPT_DIR/_nnsmith_tf_worker.py" \
            --seed "$seed" --max-budget 50000 --min-budget 100 --patience 500 \
            --device cuda --time-budget "$BUDGET" \
            >"$RESULTS/_tmp_nnsmith_${seed}_1800.json" \
            2>"$RESULTS/_log_nnsmith_${seed}.txt" &
    done
    wait
    echo "[nnsmith] All seeds done"
}

# Run all three groups in parallel
run_targeted &
PID_T=$!
run_random &
PID_R=$!
run_nnsmith &
PID_N=$!

echo "Waiting for all groups (PIDs: $PID_T $PID_R $PID_N)..."
wait $PID_T $PID_R $PID_N
echo "All experiments complete."

# --- Merge results ---
$TF_PYTHON -c "
import json, os, sys
import numpy as np
from pathlib import Path

results_dir = Path('$RESULTS')
seeds = $( python3 -c "print([42,123,456,789,1024])" )

all_runs = []
for mode in ['targeted', 'random']:
    for seed in seeds:
        p = results_dir / f'_tmp_{mode}_{seed}_1800.json'
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            all_runs.append(r)
        else:
            print(f'WARNING: missing {p}', file=sys.stderr)

# Aggregate targeted + random
agg = {}
for mode in ['targeted', 'random']:
    runs = [r for r in all_runs if r['mode'] == mode]
    if not runs:
        continue
    findings = [r['n_findings'] for r in runs]
    crash_counts = [sum(1 for f in r['findings'] if 'crash' in f['type']) for r in runs]
    diff_counts = [sum(1 for f in r['findings'] if 'crash' not in f['type']) for r in runs]
    iters = [r['iterations'] for r in runs]
    gd_vals = [r.get('graph_diversity', 0) for r in runs]
    agg[mode] = {
        'total_findings': {'mean': round(float(np.mean(findings)),1), 'std': round(float(np.std(findings)),1), 'values': findings},
        'crash_findings': {'mean': round(float(np.mean(crash_counts)),1), 'std': round(float(np.std(crash_counts)),1), 'values': crash_counts},
        'diff_findings': {'mean': round(float(np.mean(diff_counts)),1), 'std': round(float(np.std(diff_counts)),1), 'values': diff_counts},
        'graph_diversity': {'mean': round(float(np.mean(gd_vals)),1), 'std': round(float(np.std(gd_vals)),1), 'values': gd_vals},
        'iterations': {'mean': round(float(np.mean(iters)),1), 'values': iters},
    }

# NNSmith
nn_runs = []
for seed in seeds:
    p = results_dir / f'_tmp_nnsmith_{seed}_1800.json'
    if p.exists():
        with open(p) as f:
            r = json.load(f)
        nn_runs.append({
            'seed': seed,
            'iters': r.get('budget_used', 0),
            'total': r.get('bugs_found', 0),
            'crashes': sum(1 for b in r.get('bugs',[]) if 'crash' in b.get('bug_type','')),
            'diffs': sum(1 for b in r.get('bugs',[]) if 'crash' not in b.get('bug_type','')),
            'gd': r.get('graph_diversity', 0),
            'time': r.get('elapsed_sec', $BUDGET),
        })
    else:
        print(f'WARNING: missing {p}', file=sys.stderr)

nn_agg = {}
if nn_runs:
    nn_agg = {
        'total': {'mean': round(float(np.mean([r['total'] for r in nn_runs])),1), 'std': round(float(np.std([r['total'] for r in nn_runs])),1)},
        'crashes': {'mean': round(float(np.mean([r['crashes'] for r in nn_runs])),1), 'std': round(float(np.std([r['crashes'] for r in nn_runs])),1)},
        'diffs': {'mean': round(float(np.mean([r['diffs'] for r in nn_runs])),1), 'std': round(float(np.std([r['diffs'] for r in nn_runs])),1)},
        'iters': {'mean': round(float(np.mean([r['iters'] for r in nn_runs])),1)},
        'gd': {'mean': round(float(np.mean([r['gd'] for r in nn_runs])),1), 'std': round(float(np.std([r['gd'] for r in nn_runs])),1)},
    }

output = {
    'experiment': 'tf_1800s',
    'time_budget': $BUDGET,
    'seeds': seeds,
    'per_run': all_runs,
    'aggregated': agg,
    'nnsmith': {'per_seed': nn_runs, 'aggregated': nn_agg},
}

out_path = results_dir / 'tf_1800s_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'Saved to {out_path}')

# Summary
for mode in ['targeted', 'random']:
    if mode in agg:
        a = agg[mode]
        print(f'{mode}: total={a[\"total_findings\"][\"mean\"]}±{a[\"total_findings\"][\"std\"]} crash={a[\"crash_findings\"][\"mean\"]}±{a[\"crash_findings\"][\"std\"]} diff={a[\"diff_findings\"][\"mean\"]}±{a[\"diff_findings\"][\"std\"]}')
if nn_agg:
    print(f'nnsmith: total={nn_agg[\"total\"][\"mean\"]}±{nn_agg[\"total\"][\"std\"]} crash={nn_agg[\"crashes\"][\"mean\"]}±{nn_agg[\"crashes\"][\"std\"]} diff={nn_agg[\"diffs\"][\"mean\"]}±{nn_agg[\"diffs\"][\"std\"]}')
"

echo "=== Done ==="
