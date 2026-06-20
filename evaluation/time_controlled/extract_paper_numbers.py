#!/usr/bin/env python3
"""Extract paper-ready numbers from time_controlled_results.json."""

import json
import numpy as np
from scipy.stats import mannwhitneyu

import os
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'time_controlled_results.json')

with open(RESULTS_PATH) as f:
    data = json.load(f)

per_run = data['per_run']

modes = ['targeted', 'random', 'nnsmith']
for mode in modes:
    runs = [r for r in per_run if r['mode'] == mode]
    kd = [r['kernel_diversity'] for r in runs]
    bugs = [r['bugs_found'] for r in runs]
    iters = [r['iterations_completed'] for r in runs]

    print(f"\n{mode.upper()} ({len(runs)} seeds):")
    print(f"  Kernel Diversity: {np.mean(kd):.1f} ± {np.std(kd):.1f}  values={kd}")
    print(f"  Programs/5min:    {np.mean(iters):.0f} ± {np.std(iters):.0f}  values={iters}")
    print(f"  Bug candidates:   {np.mean(bugs):.1f} ± {np.std(bugs):.1f}  values={bugs}")

    # Family reachability from kss_per_family
    families_reached = set()
    for r in runs:
        for fam, info in r.get('kss_per_family', {}).items():
            if info['hits'] > 0:
                families_reached.add(fam)
    print(f"  Families reached: {len(families_reached)} = {sorted(families_reached)}")

    # split_scan
    ss_reached = sum(1 for r in runs
                     if r.get('kss_per_family', {}).get('split_scan', {}).get('hits', 0) > 0)
    print(f"  split_scan: {ss_reached}/{len(runs)}")

# Statistical tests
print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)

t_runs = [r for r in per_run if r['mode'] == 'targeted']
r_runs = [r for r in per_run if r['mode'] == 'random']
n_runs = [r for r in per_run if r['mode'] == 'nnsmith']

t_kd = [r['kernel_diversity'] for r in t_runs]
r_kd = [r['kernel_diversity'] for r in r_runs]
n_kd = [r['kernel_diversity'] for r in n_runs]

# Mann-Whitney U: targeted > random
stat, p = mannwhitneyu(t_kd, r_kd, alternative='greater')
print(f"\nMann-Whitney U (targeted > random): U={stat:.1f}, p={p:.6f}")

# Cohen's d
pooled_std = np.sqrt((np.std(t_kd)**2 + np.std(r_kd)**2) / 2)
d = (np.mean(t_kd) - np.mean(r_kd)) / pooled_std if pooled_std > 0 else float('inf')
print(f"Cohen's d: {d:.2f}")

# Ratio
ratio = np.mean(t_kd) / np.mean(r_kd)
print(f"Ratio (targeted/random): {ratio:.2f}x")

# Mann-Whitney U: targeted > nnsmith
stat2, p2 = mannwhitneyu(t_kd, n_kd, alternative='greater')
print(f"\nMann-Whitney U (targeted > nnsmith): U={stat2:.1f}, p={p2:.6f}")
ratio2 = np.mean(t_kd) / np.mean(n_kd)
print(f"Ratio (targeted/nnsmith): {ratio2:.1f}x")

# Paper-ready LaTeX values
print("\n" + "="*60)
print("LATEX TABLE VALUES")
print("="*60)
print(f"Kernel diversity: \\textbf{{{int(np.mean(t_kd))}}}$\\pm${int(np.std(t_kd))}$^{{**}}$ & "
      f"{int(np.mean(r_kd))}$\\pm${int(np.std(r_kd))} & "
      f"{int(np.mean(n_kd))}$\\pm${int(np.std(n_kd))}")
print(f"Programs / 5 min: \\textbf{{{int(np.mean([r['iterations_completed'] for r in t_runs]))}}} & "
      f"{int(np.mean([r['iterations_completed'] for r in r_runs]))} & "
      f"{int(np.mean([r['iterations_completed'] for r in n_runs]))}")
print(f"Bug candidates: {int(np.mean([r['bugs_found'] for r in t_runs]))} & "
      f"{int(np.mean([r['bugs_found'] for r in r_runs]))} & "
      f"{int(np.mean([r['bugs_found'] for r in n_runs]))}")
