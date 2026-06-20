#!/usr/bin/env python3
"""
NNSmith TF-XLA baseline: runs NNSmith model generation through TF's
3-way oracle (eager vs Grappler vs XLA).

Aligned with PyTorch NNSmith baseline (run_nnsmith_baseline.py):
  - 3 seeds × 1 mode
  - Same budget/saturation protocol
  - Graph diversity as coverage proxy (TF analogue of KSS)
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
WORKER = os.path.join(SCRIPT_DIR, '_nnsmith_tf_worker.py')
TF_PYTHON = os.environ.get('TF_PYTHON', 'python3')


def run_one(seed, max_budget, min_budget, patience, device, gpu_id='0'):
    cmd = [
        TF_PYTHON, WORKER,
        '--seed', str(seed),
        '--max-budget', str(max_budget),
        '--min-budget', str(min_budget),
        '--patience', str(patience),
        '--device', device,
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'

    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=7200)

    stdout_lines = result.stdout.strip().split('\n')
    stderr_lines = result.stderr.strip().split('\n') if result.stderr else []

    for line in stderr_lines:
        if line.strip() and not line.startswith(('I0000', 'W0000')):
            print(f"  {line.strip()}", file=sys.stderr)

    for line in reversed(stdout_lines):
        line = line.strip()
        if line.startswith('{'):
            data = json.loads(line)
            summary = (f"GD={data['graph_diversity']} "
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
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 42,123,456)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS

    print(f"NNSmith TF-XLA Baseline Experiment")
    print(f"Seeds: {seeds}")
    print(f"Max budget: {args.max_budget} per seed")
    print(f"Min budget: {args.min_budget}, patience: {args.patience}")
    print(f"Device: {args.device}, GPU: {args.gpu}")
    print(f"TF Python: {TF_PYTHON}")
    print()

    all_results = []
    t0 = time.time()

    for seed in seeds:
        print(f"{'='*60}")
        print(f"  NNSMITH-TF | seed={seed} | max_budget={args.max_budget}")
        print(f"{'='*60}")

        data = run_one(seed, args.max_budget, args.min_budget,
                       args.patience, args.device, args.gpu)
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
        'graph_diversity': [r['graph_diversity'] for r in all_results],
        'bugs_found': [r['bugs_found'] for r in all_results],
        'trivial_crashes': [r['trivial_crashes'] for r in all_results],
        'budget_used': [r['budget_used'] for r in all_results],
        'elapsed_sec': [r['elapsed_sec'] for r in all_results],
        'gen_failures': [r.get('gen_failures', 0) for r in all_results],
        'total_ok': [r.get('total_ok', 0) for r in all_results],
    }

    print(f"{'='*60}")
    print(f"  NNSMITH-TF AGGREGATED (mean ± std across {len(seeds)} seeds)")
    print(f"{'='*60}")
    for metric, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        print(f"  {metric:25s}: {arr.mean():.1f} ± {arr.std():.1f}")

    # Compare with TF random baseline if available
    tf_results_path = os.path.join(SCRIPT_DIR, 'results', 'tf_time_controlled_results.json')
    if os.path.exists(tf_results_path):
        with open(tf_results_path) as f:
            tf_data = json.load(f)
        tf_agg = tf_data.get('aggregated', {})
        tf_random = tf_agg.get('random', {})
        tf_targeted = tf_agg.get('targeted', {})

        print(f"\n{'='*70}")
        print(f"  COMPARISON: TF Targeted vs Random vs NNSmith")
        print(f"{'='*70}")

        def fmt(d, key):
            if key in d:
                v = d[key]
                if isinstance(v, dict) and 'mean' in v:
                    return f"{v['mean']:.1f} ± {v['std']:.1f}"
            return 'N/A'

        nn_gd = np.array(metrics['graph_diversity'])
        nn_bugs = np.array(metrics['bugs_found'])
        nn_budget = np.array(metrics['budget_used'])

        print(f"{'Metric':30s} {'targeted':>18s} {'random':>18s} {'nnsmith':>18s}")
        print("-" * 86)
        print(f"{'Graph diversity':30s} "
              f"{fmt(tf_targeted, 'graph_diversity'):>18s} "
              f"{fmt(tf_random, 'graph_diversity'):>18s} "
              f"{nn_gd.mean():>7.1f} ± {nn_gd.std():>5.1f}")
        print(f"{'Findings':30s} "
              f"{fmt(tf_targeted, 'total_findings'):>18s} "
              f"{fmt(tf_random, 'total_findings'):>18s} "
              f"{nn_bugs.mean():>7.1f} ± {nn_bugs.std():>5.1f}")
        print(f"{'Budget used':30s} "
              f"{'N/A':>18s} "
              f"{'N/A':>18s} "
              f"{nn_budget.mean():>7.1f} ± {nn_budget.std():>5.1f}")

    # Save
    out_path = os.path.join(SCRIPT_DIR, 'results', 'nnsmith_tf_baseline_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_data = {
        'experiment': 'nnsmith_tf_baseline',
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
