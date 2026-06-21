#!/usr/bin/env python3
"""Merge original 3-seed + supplementary 2-seed ablation results into 5-seed file."""

import json
import os
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
with open(os.path.join(EVAL_DIR, 'ablation_v2', 'ablation_v2_results.json')) as f:
    orig = json.load(f)

with open(os.path.join(EVAL_DIR, 'ablation_v2', 'results', 'ablation_v2_results.json')) as f:
    new = json.load(f)

merged_per_run = orig['per_run'] + new['per_run']
all_seeds = sorted(set(r['seed'] for r in merged_per_run))

print('=== MERGED 5-SEED ABLATION SUMMARY ===')
for r in merged_per_run:
    print(f'  {r["mode"]:>10s} seed={r["seed"]:>5d}: KSS={r["kss_hits"]:>2d}/46 '
          f'L2={r["l2_count"]:>3d} bugs={r["bugs_found"]:>3d} '
          f'trivial={r["trivial_crashes"]:>3d} '
          f'budget={r["budget_used"]:>4d} sat@{r["saturated_at"]}')

print()
for mode in ['targeted', 'random']:
    runs = [r for r in merged_per_run if r['mode'] == mode]
    kss = np.array([r['kss_hits'] for r in runs])
    bugs = np.array([r['bugs_found'] for r in runs])
    triv = np.array([r['trivial_crashes'] for r in runs])
    budget = np.array([r['budget_used'] for r in runs])
    l2 = np.array([r['l2_count'] for r in runs])
    fam = np.array([r['n_families'] for r in runs])
    wall = np.array([r.get('wall_sec', r.get('elapsed_sec', 0)) for r in runs])

    print(f'{mode} (n={len(runs)}):')
    print(f'  KSS: {kss.mean():.1f} +/- {kss.std():.1f}  range=[{kss.min()},{kss.max()}]')
    print(f'  L2: {l2.mean():.1f} +/- {l2.std():.1f}')
    print(f'  bugs: {bugs.mean():.1f} +/- {bugs.std():.1f}  trivial: {triv.mean():.1f} +/- {triv.std():.1f}')
    print(f'  budget: {budget.mean():.1f} +/- {budget.std():.1f}')
    print(f'  families: {fam.mean():.1f} +/- {fam.std():.1f}')
    print(f'  wall: {wall.mean():.1f} +/- {wall.std():.1f}')
    print()

# Recompute aggregated
metrics_keys = ['kss_hits', 'kss_coverage', 'l2_count', 'l3_count',
                'bugs_found', 'trivial_crashes', 'n_families', 'budget_used']

aggregated = {}
for mode in ['targeted', 'random']:
    runs = [r for r in merged_per_run if r['mode'] == mode]
    mode_agg = {}
    for k in metrics_keys:
        vals = [r[k] for r in runs]
        mode_agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    aggregated[mode] = mode_agg

# Save merged
merged = {
    'timestamp': f'{orig["timestamp"]} + {new["timestamp"]} (merged)',
    'config': orig['config'],
    'seeds': all_seeds,
    'per_run': merged_per_run,
    'aggregated': aggregated,
}

out_path = os.path.join(EVAL_DIR, 'results', 'ablation_5seed_merged.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(merged, f, indent=2, default=str)
print(f'Saved merged 5-seed results to: {out_path}')
