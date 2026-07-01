#!/usr/bin/env python3
"""
NNSmith Native Oracle Experiment (RQ2 supplement): runs NNSmith with its
own multi-backend differential oracle (CPU vs CUDA torchjit).

This complements the main generation comparison (Table III, tab:budget-split) where NNSmith
uses DeepOpFuzz's oracle. Here NNSmith runs in its native setting:
  - Generate model via SMT-guided graph construction
  - Execute on torchjit-CPU (oracle) and torchjit-CUDA (test backend)
  - Differential comparison using NNSmith's assert_allclose

This measures NNSmith's bug-finding capability under its own oracle, so
the comparison is method-vs-method rather than generation-strategy-only.

Output: results/nnsmith_native_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SEEDS = [42, 123, 456, 789, 1024]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, '_nnsmith_native_worker.py')

PYTHON = os.environ.get('KERNELFUZZ_PYTHON', sys.executable)
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def run_single(seed, time_budget):
    cmd = [
        PYTHON, WORKER,
        '--seed', str(seed),
        '--time-budget', str(time_budget),
    ]

    print(f"\n{'='*60}")
    print(f"  NNSmith-native | seed={seed} | budget={time_budget}s")
    print(f"{'='*60}")

    t0 = time.time()
    env = os.environ.copy()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=time_budget + 300,
        env=env,
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
    return None


def main():
    parser = argparse.ArgumentParser(
        description='NNSmith native oracle experiment')
    parser.add_argument('--time-budget', type=int, default=300,
                        help='Wall-clock seconds per run (default: 300)')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()

    results = []
    for i, seed in enumerate(args.seeds):
        print(f"\n[{i+1}/{len(args.seeds)}] Starting seed={seed}")
        data = run_single(seed, args.time_budget)
        if data is not None:
            results.append(data)
            print(f"  -> bugs={data['bugs_found']} "
                  f"crashes={data.get('crashes', 0)} "
                  f"inconsistencies={data.get('inconsistencies', 0)} "
                  f"iters={data['iterations']}")

    agg = {}
    if results:
        agg = {
            'bugs_found': {
                'mean': round(np.mean([r['bugs_found'] for r in results]), 1),
                'std': round(np.std([r['bugs_found'] for r in results]), 2),
                'values': [r['bugs_found'] for r in results],
            },
            'inconsistencies': {
                'mean': round(np.mean([r.get('inconsistencies', 0)
                                       for r in results]), 1),
                'values': [r.get('inconsistencies', 0) for r in results],
            },
            'crashes': {
                'mean': round(np.mean([r.get('crashes', 0)
                                       for r in results]), 1),
                'values': [r.get('crashes', 0) for r in results],
            },
            'iterations': {
                'mean': round(np.mean([r['iterations'] for r in results]), 1),
                'values': [r['iterations'] for r in results],
            },
        }

    output = {
        'experiment': 'nnsmith_native_oracle',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'time_budget_sec': args.time_budget,
            'seeds': args.seeds,
            'oracle': 'torchjit-cpu vs torchjit-cuda',
        },
        'per_run': results,
        'aggregated': agg,
    }

    os.makedirs(os.path.join(SCRIPT_DIR, 'results'), exist_ok=True)
    out_path = os.path.join(SCRIPT_DIR, 'results', 'nnsmith_native_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults written to {out_path}")
    if agg:
        print(f"Bugs: {agg['bugs_found']['mean']} ± {agg['bugs_found']['std']}")
        print(f"Inconsistencies: {agg['inconsistencies']['mean']}")
        print(f"Crashes: {agg['crashes']['mean']}")


if __name__ == '__main__':
    main()
