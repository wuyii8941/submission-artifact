#!/usr/bin/env python3
"""
NNSmith baseline experiment: runs NNSmith model generation through
torch.compile/Inductor with our differential oracle and KSS tracking.

3 seeds × 1 mode (nnsmith), same protocol as ablation V2.
Compares against targeted/random results from ablation_v2_results.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SEEDS = [42, 123, 456]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(SCRIPT_DIR, '_nnsmith_worker.py')
PYTHON = os.path.join(SCRIPT_DIR, '..', 'pt_nightly_env', 'bin', 'python3')
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def run_one(seed, max_budget, min_budget, patience, device):
    cmd = [
        PYTHON, WORKER,
        '--seed', str(seed),
        '--max-budget', str(max_budget),
        '--min-budget', str(min_budget),
        '--patience', str(patience),
        '--device', device,
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')

    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=7200)

    # Parse JSON from last line of stdout
    stdout_lines = result.stdout.strip().split('\n')
    stderr_lines = result.stderr.strip().split('\n') if result.stderr else []

    for line in stderr_lines:
        if line.strip():
            print(f"  {line.strip()}", file=sys.stderr)

    for line in reversed(stdout_lines):
        line = line.strip()
        if line.startswith('{'):
            data = json.loads(line)
            summary = (f"KSS={data['kss_hits']}/{data['kss_total']} "
                       f"L2={data['l2_count']} L3={data['l3_count']} "
                       f"bugs={data['bugs_found']} trivial={data['trivial_crashes']} "
                       f"fail={data.get('gen_failures', 0)} "
                       f"budget={data['budget_used']} "
                       f"sat@{data['saturated_at']}")
            print(f"-> {summary}", file=sys.stderr)
            return data

    print(f"ERROR: no JSON output from worker", file=sys.stderr)
    print(f"STDOUT: {result.stdout[-500:]}", file=sys.stderr)
    print(f"STDERR: {result.stderr[-500:]}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-budget', type=int, default=5000)
    parser.add_argument('--min-budget', type=int, default=500)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 42,123,456)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS

    import torch
    print(f"NNSmith Baseline Experiment")
    print(f"Seeds: {seeds}")
    print(f"Max budget: {args.max_budget} per seed")
    print(f"Min budget: {args.min_budget}, patience: {args.patience}")
    print(f"Device: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    all_results = []
    t0 = time.time()

    for seed in seeds:
        print(f"{'='*60}")
        print(f"  NNSMITH | seed={seed} | max_budget={args.max_budget}")
        print(f"{'='*60}")

        data = run_one(seed, args.max_budget, args.min_budget,
                       args.patience, args.device)
        if data:
            all_results.append(data)
        print()

    total_time = time.time() - t0
    print(f"Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print()

    if not all_results:
        print("ERROR: No results collected")
        return

    # Aggregate
    metrics = {
        'kss_hits': [r['kss_hits'] for r in all_results],
        'kss_coverage': [r['kss_coverage'] for r in all_results],
        'l2_count': [r['l2_count'] for r in all_results],
        'l3_count': [r['l3_count'] for r in all_results],
        'bugs_found': [r['bugs_found'] for r in all_results],
        'trivial_crashes': [r['trivial_crashes'] for r in all_results],
        'n_families': [r['n_families'] for r in all_results],
        'budget_used': [r['budget_used'] for r in all_results],
        'elapsed_sec': [r['elapsed_sec'] for r in all_results],
        'gen_failures': [r.get('gen_failures', 0) for r in all_results],
    }

    print(f"{'='*60}")
    print(f"  NNSMITH AGGREGATED (mean ± std across {len(seeds)} seeds)")
    print(f"{'='*60}")
    for metric, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        print(f"  {metric:25s}: {arr.mean():.1f} ± {arr.std():.1f}")

    # Per-family breakdown
    print(f"\nPer-family KSS (mean hits / total):")
    families = all_results[0]['kss_per_family'].keys()
    for fam in sorted(families):
        hits = [r['kss_per_family'][fam]['hits'] for r in all_results]
        total = all_results[0]['kss_per_family'][fam]['total']
        arr = np.array(hits, dtype=float)
        print(f"  {fam:25s}: {arr.mean():.1f}±{arr.std():.1f} /{total}")

    # Compare with ablation V2 if available
    ablation_path = os.path.join(SCRIPT_DIR, 'results', 'ablation_v2_results.json')
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)
        print(f"\n{'='*70}")
        print(f"  COMPARISON: Targeted vs Random vs NNSmith")
        print(f"{'='*70}")

        agg = ablation.get('aggregated', {})
        targeted = agg.get('targeted', {})
        random_ = agg.get('random', {})

        def fmt(d, key):
            if key in d:
                v = d[key]
                if isinstance(v, dict) and 'mean' in v:
                    return f"{v['mean']:.1f} ± {v['std']:.1f}"
            return "N/A"

        nn_kss = np.array(metrics['kss_hits'])
        nn_l2 = np.array(metrics['l2_count'])
        nn_bugs = np.array(metrics['bugs_found'])
        nn_fam = np.array(metrics['n_families'])
        nn_budget = np.array(metrics['budget_used'])
        nn_time = np.array(metrics['elapsed_sec'])

        print(f"{'Metric':30s} {'targeted':>18s} {'random':>18s} {'nnsmith':>18s}")
        print("-" * 86)
        print(f"{'KSS hits (/46)':30s} {fmt(targeted, 'kss_hits'):>18s} "
              f"{fmt(random_, 'kss_hits'):>18s} "
              f"{nn_kss.mean():>7.1f} ± {nn_kss.std():>5.1f}")
        print(f"{'L2 fusion patterns':30s} {fmt(targeted, 'l2_count'):>18s} "
              f"{fmt(random_, 'l2_count'):>18s} "
              f"{nn_l2.mean():>7.1f} ± {nn_l2.std():>5.1f}")
        print(f"{'Genuine bugs':30s} {fmt(targeted, 'bugs_found'):>18s} "
              f"{fmt(random_, 'bugs_found'):>18s} "
              f"{nn_bugs.mean():>7.1f} ± {nn_bugs.std():>5.1f}")
        print(f"{'Kernel families':30s} {fmt(targeted, 'n_families'):>18s} "
              f"{fmt(random_, 'n_families'):>18s} "
              f"{nn_fam.mean():>7.1f} ± {nn_fam.std():>5.1f}")
        print(f"{'Budget used':30s} {fmt(targeted, 'budget_used'):>18s} "
              f"{fmt(random_, 'budget_used'):>18s} "
              f"{nn_budget.mean():>7.1f} ± {nn_budget.std():>5.1f}")

    # Save
    out_path = os.path.join(SCRIPT_DIR, 'results', 'nnsmith_baseline_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_data = {
        'experiment': 'nnsmith_baseline',
        'seeds': seeds,
        'config': {
            'max_budget': args.max_budget,
            'min_budget': args.min_budget,
            'patience': args.patience,
            'device': args.device,
        },
        'per_seed': all_results,
        'aggregated': {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in metrics.items()
        },
        'total_wall_time': total_time,
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
