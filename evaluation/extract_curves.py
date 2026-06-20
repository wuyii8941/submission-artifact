#!/usr/bin/env python3
"""
Extract time-to-bug and KSS coverage curves from ablation and NNSmith results.
Produces JSON data suitable for LaTeX/pgfplots figures.
"""

import json
import os
import sys

import numpy as np


def interpolate_timeseries(all_ts, x_key='iter', y_keys=None, max_x=None):
    """Interpolate multiple timeseries onto a common x-axis.

    Returns dict of {y_key: {'x': [...], 'mean': [...], 'std': [...]}}.
    """
    if y_keys is None:
        y_keys = ['kss', 'bugs']

    # Find common x range
    all_x_max = [ts[-1][x_key] for ts in all_ts]
    if max_x is None:
        max_x = max(all_x_max)

    # Create uniform x grid
    x_grid = np.arange(0, max_x + 1, 10)

    result = {}
    for y_key in y_keys:
        curves = []
        for ts in all_ts:
            xs = np.array([t[x_key] for t in ts], dtype=float)
            ys = np.array([t[y_key] for t in ts], dtype=float)
            # Extend to max_x with last value
            interp = np.interp(x_grid, xs, ys)
            curves.append(interp)

        curves = np.array(curves)
        result[y_key] = {
            'x': x_grid.tolist(),
            'mean': curves.mean(axis=0).tolist(),
            'std': curves.std(axis=0).tolist(),
            'min': curves.min(axis=0).tolist(),
            'max': curves.max(axis=0).tolist(),
            'n_seeds': len(all_ts),
        }

    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load ablation results (prefer 5-seed merged)
    merged_path = os.path.join(script_dir, 'results', 'ablation_5seed_merged.json')
    fallback_path = os.path.join(script_dir, 'ablation_v2', 'ablation_v2_results.json')
    ablation_path = merged_path if os.path.exists(merged_path) else fallback_path
    print(f"Using ablation data: {ablation_path}")
    with open(ablation_path) as f:
        ablation = json.load(f)

    # Load NNSmith results (prefer 5-seed merged)
    nnsmith_merged = os.path.join(script_dir, 'results', 'nnsmith_5seed_merged.json')
    nnsmith_fallback = os.path.join(script_dir, 'nnsmith_baseline', 'nnsmith_baseline_results.json')
    nnsmith_path = nnsmith_merged if os.path.exists(nnsmith_merged) else nnsmith_fallback
    print(f"Using NNSmith data: {nnsmith_path}")
    with open(nnsmith_path) as f:
        nnsmith = json.load(f)

    # Group timeseries by mode
    ts_by_mode = {}
    for ps in ablation['per_run']:
        mode = ps['mode']
        if mode not in ts_by_mode:
            ts_by_mode[mode] = []
        ts_by_mode[mode].append(ps['timeseries'])

    ts_by_mode['nnsmith'] = [ps['timeseries'] for ps in nnsmith['per_seed']]

    # Find common max budget for fair comparison
    max_budget = 1500  # enough to show saturation for all modes

    y_keys = ['kss', 'l2', 'bugs', 'trivial']
    curves = {}
    for mode, ts_list in ts_by_mode.items():
        curves[mode] = interpolate_timeseries(ts_list, y_keys=y_keys, max_x=max_budget)

    # Print summary
    print("=" * 70)
    print("  COVERAGE & BUG DISCOVERY CURVES")
    print("=" * 70)

    for mode in ['targeted', 'random', 'nnsmith']:
        c = curves[mode]
        kss = c['kss']
        bugs = c['bugs']

        # Find iter where KSS reaches 50%, 90% of final
        kss_final = kss['mean'][-1]
        kss_arr = np.array(kss['mean'])
        x_arr = np.array(kss['x'])

        idx_50 = np.searchsorted(kss_arr, kss_final * 0.5)
        idx_90 = np.searchsorted(kss_arr, kss_final * 0.9)
        iter_50 = x_arr[min(idx_50, len(x_arr) - 1)]
        iter_90 = x_arr[min(idx_90, len(x_arr) - 1)]

        # Time to first bug
        bugs_arr = np.array(bugs['mean'])
        first_bug_idx = np.searchsorted(bugs_arr, 0.5)
        iter_first_bug = x_arr[min(first_bug_idx, len(x_arr) - 1)]

        print(f"\n  {mode:12s} ({kss['n_seeds']} seeds)")
        print(f"    KSS final:    {kss_final:.1f}/46")
        print(f"    KSS 50%:      iter {iter_50}")
        print(f"    KSS 90%:      iter {iter_90}")
        print(f"    Bugs (raw):   {bugs['mean'][-1]:.1f}")
        print(f"    First bug at: iter {iter_first_bug}")

    # Milestone comparison
    print(f"\n{'='*70}")
    print(f"  MILESTONE COMPARISON")
    print(f"{'='*70}")
    milestones = [100, 200, 500, 1000, 1500]
    print(f"{'Iter':>6s} ", end='')
    for mode in ['targeted', 'random', 'nnsmith']:
        print(f"  {mode+' KSS':>14s} {mode+' bugs':>14s}", end='')
    print()
    print("-" * 100)
    for m in milestones:
        idx = min(m // 10, len(curves['targeted']['kss']['x']) - 1)
        print(f"{m:>6d} ", end='')
        for mode in ['targeted', 'random', 'nnsmith']:
            c = curves[mode]
            kss_m = c['kss']['mean'][idx]
            kss_s = c['kss']['std'][idx]
            bugs_m = c['bugs']['mean'][idx]
            bugs_s = c['bugs']['std'][idx]
            print(f"  {kss_m:>5.1f}±{kss_s:<5.1f} {bugs_m:>6.1f}±{bugs_s:<5.1f}", end='')
        print()

    # Save curve data
    out_path = os.path.join(script_dir, 'results', 'coverage_curves.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(curves, f, indent=2)
    print(f"\nCurve data saved to: {out_path}")

    # Also save pgfplots-friendly CSV
    for mode in curves:
        csv_path = os.path.join(script_dir, 'results', f'curve_{mode}.csv')
        c = curves[mode]
        with open(csv_path, 'w') as f:
            f.write('iter,kss_mean,kss_std,l2_mean,l2_std,bugs_mean,bugs_std\n')
            n = len(c['kss']['x'])
            for i in range(n):
                f.write(f"{c['kss']['x'][i]:.0f},"
                        f"{c['kss']['mean'][i]:.2f},{c['kss']['std'][i]:.2f},"
                        f"{c['l2']['mean'][i]:.2f},{c['l2']['std'][i]:.2f},"
                        f"{c['bugs']['mean'][i]:.2f},{c['bugs']['std'][i]:.2f}\n")
        print(f"CSV saved: {csv_path}")


if __name__ == '__main__':
    main()
