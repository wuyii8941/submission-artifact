#!/usr/bin/env python3
"""
Extended Coverage Experiment (RQ3): D(t) growth curves for all three methods.

Runs targeted, random, and NNSmith with a 1800s (30 min) budget per seed,
5 seeds each. Produces timeseries data for plotting coverage growth curves
and verifying saturation claims.

Also serves as extended budget evidence for the "practically unreachable"
claim: if random never reaches split_scan even at 30 min, the 300s result
is not a budget artifact.

Output: results/extended_coverage_results.json
  - per_run[].timeseries: D(t) at every 10 iterations
  - aggregated: mean/std per mode for final metrics
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
    print(f"  {mode.upper()} | seed={seed} | budget={time_budget}s")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=time_budget + 300,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0')},
    )
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}, wall={wall:.0f}s)")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f"  {line}")
        return None

    if result.stderr:
        for line in result.stderr.strip().split('\n')[-3:]:
            print(f"  {line}")

    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                data = json.loads(line)
                data['actual_wall_sec'] = round(wall, 1)
                return data
            except json.JSONDecodeError:
                continue

    print(f"  WARNING: no JSON output found")
    return None


def aggregate(results):
    agg = {}
    for mode in MODES:
        runs = [r for r in results if r['mode'] == mode]
        if not runs:
            continue

        agg[mode] = {
            'kernel_diversity': {
                'mean': round(np.mean([r['kernel_diversity'] for r in runs]), 1),
                'std': round(np.std([r['kernel_diversity'] for r in runs]), 2),
                'values': [r['kernel_diversity'] for r in runs],
            },
            'bugs_found': {
                'mean': round(np.mean([r['bugs_found'] for r in runs]), 1),
                'std': round(np.std([r['bugs_found'] for r in runs]), 2),
                'values': [r['bugs_found'] for r in runs],
            },
            'n_families': {
                'mean': round(np.mean([r['n_families'] for r in runs]), 1),
                'values': [r['n_families'] for r in runs],
            },
            'iterations_completed': {
                'mean': round(np.mean([r['iterations_completed'] for r in runs]), 1),
                'values': [r['iterations_completed'] for r in runs],
            },
            'split_scan_reached': sum(
                1 for r in runs if 'split_scan' in r.get('families_seen', [])
            ),
            'split_scan_total_seeds': len(runs),
        }
    return agg


def main():
    parser = argparse.ArgumentParser(description='Extended coverage experiment (RQ3)')
    parser.add_argument('--time-budget', type=int, default=1800,
                        help='Wall-clock seconds per run (default: 1800 = 30 min)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--modes', nargs='+', default=MODES,
                        choices=MODES)
    parser.add_argument('--gpu', default='0',
                        help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    results = []
    total_runs = len(args.modes) * len(args.seeds)
    done = 0

    for mode in args.modes:
        for seed in args.seeds:
            done += 1
            print(f"\n[{done}/{total_runs}] Starting {mode} seed={seed}")
            data = run_single(mode, seed, args.time_budget, args.device)
            if data is not None:
                results.append(data)
                print(f"  -> KD={data['kernel_diversity']} "
                      f"bugs={data['bugs_found']} "
                      f"families={data['n_families']} "
                      f"iters={data['iterations_completed']}")

    agg = aggregate(results)

    output = {
        'experiment': 'extended_coverage',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'time_budget_sec': args.time_budget,
            'seeds': args.seeds,
            'modes': args.modes,
            'device': args.device,
        },
        'per_run': results,
        'aggregated': agg,
    }

    os.makedirs(os.path.join(SCRIPT_DIR, 'results'), exist_ok=True)
    # Save per-mode file (safe for parallel runs)
    for mode in args.modes:
        mode_path = os.path.join(SCRIPT_DIR, 'results', f'extended_{mode}.json')
        mode_runs = [r for r in results if r['mode'] == mode]
        mode_out = {
            'experiment': 'extended_coverage',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {**output['config'], 'modes': [mode]},
            'per_run': mode_runs,
            'aggregated': {mode: agg[mode]} if mode in agg else {},
        }
        with open(mode_path, 'w') as f:
            json.dump(mode_out, f, indent=2, default=str)
    # Also save combined file
    out_path = os.path.join(SCRIPT_DIR, 'results', 'extended_coverage_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results written to {out_path}")
    print(f"{'='*60}")

    for mode in args.modes:
        if mode in agg:
            a = agg[mode]
            print(f"\n{mode.upper()}:")
            print(f"  KD = {a['kernel_diversity']['mean']} ± {a['kernel_diversity']['std']}")
            print(f"  Bugs = {a['bugs_found']['mean']} ± {a['bugs_found']['std']}")
            print(f"  Families = {a['n_families']['mean']}")
            print(f"  Iters = {a['iterations_completed']['mean']}")
            print(f"  split_scan: {a['split_scan_reached']}/{a['split_scan_total_seeds']}")


if __name__ == '__main__':
    main()
