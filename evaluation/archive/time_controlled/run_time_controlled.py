#!/usr/bin/env python3
"""
Time-Controlled Experiment: Targeted vs Random vs NNSmith under same wall-clock.

Purpose: answer the reviewer question "given the same time budget, what does
each method achieve?"

Design:
  - Fixed wall-clock budget (default: 300s, matching targeted's saturation time)
  - 5 seeds × 3 modes = 15 subprocess invocations (sequential)
  - Same oracle, same KSS tracking, same hardware
  - Output: per-run JSON + aggregated comparison table

Expected outcome:
  - Targeted saturates KSS within budget, reaches split_scan
  - Random gets more iterations (faster per-program) but lower KSS
  - NNSmith gets fewest iterations (slow generation) and lowest KSS
  - Bug-bearing family coverage: targeted ≥ random > nnsmith
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SEEDS = [42, 123, 456, 789, 1024]
MODES = ['targeted', 'random', 'nnsmith']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, '_time_worker.py')

PYTHON = os.environ.get('KERNELFUZZ_PYTHON', sys.executable)
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def run_single(mode, seed, time_budget, device):
    cmd = [
        PYTHON, WORKER,
        '--mode', mode,
        '--seed', str(seed),
        '--time-budget', str(time_budget),
        '--device', device,
    ]

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} | seed={seed} | time_budget={time_budget}s")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=time_budget + 120  # grace period
    )
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}, wall={wall:.0f}s)")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"  {line}")
        return None

    # Show last few progress lines
    if result.stderr:
        lines = result.stderr.strip().split('\n')
        for line in lines[-3:]:
            print(f"  {line}")

    # Parse JSON from stdout
    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                data = json.loads(line)
                data['actual_wall_sec'] = round(wall, 1)
                return data
            except json.JSONDecodeError:
                pass

    print(f"  No JSON output found")
    return None


def aggregate_results(all_results):
    """Compute mean ± std across seeds for each mode."""
    agg = {}
    for mode in MODES:
        mode_results = [r for r in all_results if r and r['mode'] == mode]
        if not mode_results:
            continue

        metrics = {}
        for key in ['kernel_diversity', 'bugs_found', 'trivial_crashes',
                    'n_families', 'iterations_completed', 'elapsed_sec',
                    'gen_failures']:
            vals = [r[key] for r in mode_results if key in r]
            if vals:
                metrics[key] = {
                    'mean': round(float(np.mean(vals)), 2),
                    'std': round(float(np.std(vals)), 2),
                    'values': vals,
                }

        # Per-family KSS
        family_agg = {}
        for r in mode_results:
            for fam, info in r.get('kss_per_family', {}).items():
                if fam not in family_agg:
                    family_agg[fam] = {'hits': [], 'total': info['total']}
                family_agg[fam]['hits'].append(info['hits'])
        for fam in family_agg:
            vals = family_agg[fam]['hits']
            family_agg[fam]['mean_hits'] = round(float(np.mean(vals)), 1)
            family_agg[fam]['std_hits'] = round(float(np.std(vals)), 1)
        metrics['kss_per_family'] = family_agg

        # split_scan reachability
        split_scan_hits = [
            r.get('kss_per_family', {}).get('split_scan', {}).get('hits', 0)
            for r in mode_results
        ]
        metrics['split_scan_reached'] = sum(1 for h in split_scan_hits if h > 0)
        metrics['split_scan_total_seeds'] = len(mode_results)

        # Families reached (intersection / union)
        families_all = [set(r.get('families_seen', [])) for r in mode_results]
        if families_all:
            metrics['families_intersection'] = sorted(
                set.intersection(*families_all))
            metrics['families_union'] = sorted(set.union(*families_all))

        agg[mode] = metrics

    return agg


def print_comparison(agg, time_budget):
    """Print comparison table for paper."""
    if not agg:
        return

    print(f"\n{'='*76}")
    print(f"  TIME-CONTROLLED COMPARISON (budget={time_budget}s, "
          f"mean ± std across seeds)")
    print(f"{'='*76}")

    header = f"{'Metric':<30s}"
    for mode in MODES:
        if mode in agg:
            header += f" {mode:>14s}"
    print(header)
    print("-" * 76)

    display_keys = [
        ('Kernel Diversity', 'kernel_diversity'),
        ('Bug candidates', 'bugs_found'),
        ('Trivial crashes', 'trivial_crashes'),
        ('Kernel families', 'n_families'),
        ('Programs generated', 'iterations_completed'),
    ]

    for label, key in display_keys:
        row = f"{label:<30s}"
        for mode in MODES:
            if mode in agg and key in agg[mode]:
                m = agg[mode][key]
                row += f" {m['mean']:>6.1f}±{m['std']:<5.1f}"
            else:
                row += f" {'N/A':>14s}"
        print(row)

    # split_scan reachability
    print()
    row = f"{'split_scan reached (seeds)':<30s}"
    for mode in MODES:
        if mode in agg:
            n = agg[mode].get('split_scan_reached', 0)
            t = agg[mode].get('split_scan_total_seeds', 0)
            row += f" {n}/{t:>11d}"
        else:
            row += f" {'N/A':>14s}"
    print(row)

    # Per-family breakdown
    print()
    print("Per-family KSS (mean_hits / total):")
    all_families = set()
    for mode in MODES:
        if mode in agg:
            all_families.update(agg[mode].get('kss_per_family', {}).keys())

    header = f"  {'Family':<25s}"
    for mode in MODES:
        if mode in agg:
            header += f" {mode:>14s}"
    print(header)

    for fam in sorted(all_families):
        row = f"  {fam:<25s}"
        for mode in MODES:
            if mode in agg:
                info = agg[mode].get('kss_per_family', {}).get(fam)
                if info:
                    row += f" {info['mean_hits']:>4.1f}±{info['std_hits']:<3.1f}/{info['total']}"
                else:
                    row += f" {'—':>14s}"
        print(row)

    # Statistical test (targeted vs random on Kernel Diversity)
    print()
    if 'targeted' in agg and 'random' in agg:
        t_vals = agg['targeted'].get('kernel_diversity', {}).get('values', [])
        r_vals = agg['random'].get('kernel_diversity', {}).get('values', [])
        if len(t_vals) >= 3 and len(r_vals) >= 3:
            from scipy.stats import mannwhitneyu
            try:
                stat, p = mannwhitneyu(t_vals, r_vals, alternative='greater')
                print(f"  Mann-Whitney U (targeted > random KD): "
                      f"U={stat:.1f}, p={p:.4f}")
                # Effect size
                import numpy as np
                pooled = np.sqrt((np.std(t_vals)**2 + np.std(r_vals)**2) / 2)
                d = (np.mean(t_vals) - np.mean(r_vals)) / pooled if pooled > 0 else 0
                print(f"  Cohen's d = {d:.2f}")
            except Exception as e:
                print(f"  Mann-Whitney U: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Time-controlled experiment: same wall-clock, compare methods')
    parser.add_argument('--time-budget', type=int, default=300,
                        help='Wall-clock seconds per (mode, seed) run (default: 300)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 42,123,456,789,1024)')
    parser.add_argument('--modes', type=str, default=None,
                        help='Comma-separated modes (default: targeted,random,nnsmith)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='CUDA_VISIBLE_DEVICES (default: 0)')
    args = parser.parse_args()

    global SEEDS, MODES
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(',')]
    if args.modes:
        MODES = [m.strip() for m in args.modes.split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f"Time-Controlled Experiment")
    print(f"  Time budget: {args.time_budget}s per (mode, seed)")
    print(f"  Seeds: {SEEDS} ({len(SEEDS)} seeds)")
    print(f"  Modes: {MODES}")
    print(f"  Device: {args.device}, GPU: {args.gpu}")
    print(f"  Total expected time: ~{len(SEEDS) * len(MODES) * args.time_budget / 60:.0f} min")
    print()

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    all_results = []
    total_t0 = time.time()

    for seed in SEEDS:
        for mode in MODES:
            result = run_single(mode, seed, args.time_budget, args.device)
            if result:
                all_results.append(result)
                print(f"  -> {mode} seed={seed}: "
                      f"KD={result['kernel_diversity']} "
                      f"bugs={result['bugs_found']} "
                      f"fam={result['n_families']} "
                      f"iters={result['iterations_completed']}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal experiment time: {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")

    # Aggregate
    agg = aggregate_results(all_results)
    print_comparison(agg, args.time_budget)

    # Save
    out_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'time_controlled_results.json')

    output = {
        'experiment': 'time_controlled',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'time_budget_sec': args.time_budget,
            'seeds': SEEDS,
            'modes': MODES,
            'device': args.device,
            'gpu': args.gpu,
        },
        'per_run': all_results,
        'aggregated': agg,
        'total_wall_time': round(total_elapsed, 1),
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
