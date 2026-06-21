#!/usr/bin/env python3
"""
Ablation V2: Targeted vs Random, multi-seed, saturation-aware.

Runs 3 seeds × 2 modes (6 subprocess invocations, sequential).
Each (mode, seed) runs until KSS coverage saturates.

Formal metrics tracked per the paper's definitions:
  L1 = KSS: |hit states| / 46,  s = (family, dtype, persistent, n_fused)
  L2 = |unique fusion patterns|
  L3 = |unique kernel hashes|

Output: results/ablation_v2_results.json
  - per-seed, per-mode timeseries
  - aggregated mean ± std across seeds
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np


SEEDS = [42, 123, 456]
MODES = ['targeted', 'random']


def run_single(mode, seed, max_budget, min_budget, patience, device):
    """Run one (mode, seed) in a subprocess. Returns parsed JSON or None."""
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '_ablation_worker_v2.py')
    python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'pt_nightly_env', 'bin', 'python3')
    if not os.path.exists(python):
        python = sys.executable

    cmd = [
        python, worker,
        '--mode', mode,
        '--seed', str(seed),
        '--max-budget', str(max_budget),
        '--min-budget', str(min_budget),
        '--patience', str(patience),
        '--device', device,
    ]

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} | seed={seed} | max_budget={max_budget}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=7200)  # 2h max per run
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}, wall={wall:.0f}s)")
        if result.stderr:
            print(result.stderr[-800:])
        return None

    # Show stderr progress
    if result.stderr:
        for line in result.stderr.strip().split('\n')[-8:]:
            print(f"  {line}")

    # Parse JSON from stdout (last line starting with '{')
    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                data = json.loads(line)
                data['wall_sec'] = round(wall, 1)
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
        for key in ['kss_hits', 'kss_coverage', 'l2_count', 'l3_count',
                     'bugs_found', 'trivial_crashes', 'n_families',
                     'budget_used', 'elapsed_sec']:
            vals = [r[key] for r in mode_results if key in r]
            if vals:
                metrics[key] = {
                    'mean': round(float(np.mean(vals)), 2),
                    'std': round(float(np.std(vals)), 2),
                    'values': vals,
                }

        # Per-family KSS breakdown (mean across seeds)
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

        # Families consistently reached (across all seeds)
        families_all = [set(r.get('families_seen', [])) for r in mode_results]
        if families_all:
            metrics['families_intersection'] = sorted(
                set.intersection(*families_all))
            metrics['families_union'] = sorted(set.union(*families_all))

        agg[mode] = metrics

    return agg


def print_comparison(agg):
    """Print a clean comparison table."""
    if not agg:
        return

    print(f"\n{'='*70}")
    print(f"  AGGREGATED COMPARISON (mean ± std across {len(SEEDS)} seeds)")
    print(f"{'='*70}")

    header = f"{'Metric':<30s}"
    for mode in MODES:
        if mode in agg:
            header += f" {mode:>18s}"
    print(header)
    print("-" * 70)

    display_keys = [
        ('KSS hits (/46)', 'kss_hits'),
        ('KSS coverage', 'kss_coverage'),
        ('L2 fusion patterns', 'l2_count'),
        ('L3 kernel hashes', 'l3_count'),
        ('Genuine bugs', 'bugs_found'),
        ('Trivial crashes', 'trivial_crashes'),
        ('Kernel families', 'n_families'),
        ('Budget used', 'budget_used'),
        ('Time (sec)', 'elapsed_sec'),
    ]

    for label, key in display_keys:
        row = f"{label:<30s}"
        for mode in MODES:
            if mode in agg and key in agg[mode]:
                m = agg[mode][key]
                row += f" {m['mean']:>8.1f} ± {m['std']:>5.1f}"
            else:
                row += f" {'N/A':>18s}"
        print(row)

    # Per-family breakdown
    print()
    print("Per-family KSS (mean hits / total):")
    all_families = set()
    for mode in MODES:
        if mode in agg:
            all_families.update(agg[mode].get('kss_per_family', {}).keys())

    header = f"  {'Family':<25s}"
    for mode in MODES:
        header += f" {mode:>18s}"
    print(header)

    for fam in sorted(all_families):
        row = f"  {fam:<25s}"
        for mode in MODES:
            if mode in agg:
                info = agg[mode].get('kss_per_family', {}).get(fam)
                if info:
                    row += f" {info['mean_hits']:>5.1f}±{info['std_hits']:<4.1f}/{info['total']}"
                else:
                    row += f" {'—':>18s}"
        print(row)

    # Families reached
    print()
    for mode in MODES:
        if mode in agg:
            fam_inter = agg[mode].get('families_intersection', [])
            fam_union = agg[mode].get('families_union', [])
            print(f"  {mode} families (all seeds): {fam_inter}")
            if set(fam_union) - set(fam_inter):
                print(f"  {mode} families (some seeds): "
                      f"{sorted(set(fam_union) - set(fam_inter))}")


def main():
    parser = argparse.ArgumentParser(
        description='Ablation V2: multi-seed saturation experiment')
    parser.add_argument('--max-budget', type=int, default=5000)
    parser.add_argument('--min-budget', type=int, default=500)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 42,123,456)')
    args = parser.parse_args()

    global SEEDS
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(',')]

    print(f"Ablation V2: Targeted vs Random Generator")
    print(f"Seeds: {SEEDS}")
    print(f"Max budget: {args.max_budget} per (mode, seed)")
    print(f"Min budget: {args.min_budget}, patience: {args.patience}")
    print(f"Device: {args.device}")

    import torch
    print(f"PyTorch: {torch.__version__}")

    all_results = []
    total_t0 = time.time()

    for seed in SEEDS:
        for mode in MODES:
            result = run_single(mode, seed, args.max_budget,
                                args.min_budget, args.patience, args.device)
            if result:
                all_results.append(result)
                print(f"  -> KSS={result['kss_hits']}/{result['kss_total']} "
                      f"L2={result['l2_count']} L3={result['l3_count']} "
                      f"bugs={result['bugs_found']} "
                      f"trivial={result['trivial_crashes']} "
                      f"budget={result['budget_used']} "
                      f"sat@{result['saturated_at']}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")

    # Aggregate
    agg = aggregate_results(all_results)
    print_comparison(agg)

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'ablation_v2_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seeds': SEEDS,
            'max_budget': args.max_budget,
            'min_budget': args.min_budget,
            'patience': args.patience,
            'device': args.device,
        },
        'per_run': all_results,
        'aggregated': agg,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
