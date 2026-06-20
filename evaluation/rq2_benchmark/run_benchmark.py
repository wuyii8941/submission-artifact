#!/usr/bin/env python3
"""
RQ2 Benchmark: Run KernelFuzz vs baselines on PyTorch 2.2.0.
Measures bugs found over time against 17 known ground-truth bugs.
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
WORKER = os.path.join(SCRIPT_DIR, '_benchmark_worker.py')

PYTHON = os.environ.get('KERNELFUZZ_PYTHON', sys.executable)


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
    )
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}, wall={wall:.0f}s)")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
        return None

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


def aggregate_results(all_results, time_budget):
    agg = {}
    time_points = list(range(0, time_budget + 1, 30))

    for mode in MODES:
        runs = [r for r in all_results if r and r['mode'] == mode]
        if not runs:
            continue

        iters = [r['iterations_completed'] for r in runs]
        bugs = [r['bugs_found'] for r in runs]
        raw = [r['raw_triggers'] for r in runs]
        kd = [r.get('kernel_diversity', 0) for r in runs]

        bug_curves = []
        kd_curves = []
        for run in runs:
            ts = run.get('timeseries', [])
            b_curve = []
            k_curve = []
            for tp in time_points:
                bval = 0
                kval = 0
                for pt in ts:
                    if pt['time_sec'] <= tp:
                        bval = pt['bugs']
                        kval = pt.get('kernel_diversity', 0)
                    else:
                        break
                b_curve.append(bval)
                k_curve.append(kval)
            bug_curves.append(b_curve)
            kd_curves.append(k_curve)

        bug_curves = np.array(bug_curves)
        kd_curves = np.array(kd_curves)
        agg[mode] = {
            'n_seeds': len(runs),
            'iterations': {
                'mean': round(float(np.mean(iters)), 1),
                'std': round(float(np.std(iters)), 1),
                'values': iters,
            },
            'bugs': {
                'mean': round(float(np.mean(bugs)), 1),
                'std': round(float(np.std(bugs)), 1),
                'values': bugs,
            },
            'raw_triggers': {
                'mean': round(float(np.mean(raw)), 1),
                'std': round(float(np.std(raw)), 1),
            },
            'kernel_diversity': {
                'mean': round(float(np.mean(kd)), 1),
                'std': round(float(np.std(kd)), 1),
                'values': kd,
            },
            'bug_curve': {
                'time_points': time_points,
                'mean': [round(float(x), 2) for x in bug_curves.mean(axis=0)],
                'std': [round(float(x), 2) for x in bug_curves.std(axis=0)],
            },
            'kd_curve': {
                'time_points': time_points,
                'mean': [round(float(x), 2) for x in kd_curves.mean(axis=0)],
                'std': [round(float(x), 2) for x in kd_curves.std(axis=0)],
            },
        }

    return agg


def print_summary(agg):
    if not agg:
        return

    print(f"\n{'='*70}")
    print(f"  BENCHMARK SUMMARY (mean +/- std over seeds)")
    print(f"{'='*70}")

    header = f"{'Metric':<25s}"
    for mode in MODES:
        if mode in agg:
            header += f" {mode:>16s}"
    print(header)
    print("-" * 70)

    for label, key in [
        ('Programs executed', 'iterations'),
        ('Kernel diversity', 'kernel_diversity'),
        ('Bugs found (deduped)', 'bugs'),
        ('Raw triggers', 'raw_triggers'),
    ]:
        row = f"{label:<25s}"
        for mode in MODES:
            if mode in agg:
                m = agg[mode].get(key, {})
                mean = m.get('mean', 0)
                std = m.get('std', 0)
                row += f" {mean:>8.1f}±{std:<5.1f}"
        print(row)


def export_csv(agg, out_dir):
    for mode in MODES:
        if mode not in agg:
            continue
        b_curve = agg[mode]['bug_curve']
        k_curve = agg[mode]['kd_curve']
        csv_path = os.path.join(out_dir, f'efficiency_{mode}.csv')
        with open(csv_path, 'w') as f:
            f.write('time,bugs_mean,bugs_upper,bugs_lower,'
                    'kd_mean,kd_upper,kd_lower\n')
            for i, tp in enumerate(b_curve['time_points']):
                bm = b_curve['mean'][i]
                bs = b_curve['std'][i]
                km = k_curve['mean'][i]
                ks = k_curve['std'][i]
                f.write(f"{tp},{bm},{bm+bs},{max(0,bm-bs)},"
                        f"{km},{km+ks},{max(0,km-ks)}\n")
        print(f"  CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='RQ2 Benchmark on PyTorch 2.2.0')
    parser.add_argument('--time-budget', type=int, default=14400,
                        help='Wall-clock seconds per run (default: 14400 = 4h)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--modes', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    global SEEDS, MODES
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(',')]
    if args.modes:
        MODES = [m.strip() for m in args.modes.split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    print(f"RQ2 Benchmark Experiment")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Budget: {args.time_budget}s per run")
    print(f"  Seeds: {SEEDS}")
    print(f"  Modes: {MODES}")
    print(f"  Total: ~{len(SEEDS)*len(MODES)*args.time_budget/60:.0f} min")
    print()

    all_results = []
    total_t0 = time.time()

    for seed in SEEDS:
        for mode in MODES:
            result = run_single(mode, seed, args.time_budget, args.device)
            if result:
                all_results.append(result)
                print(f"  -> {mode} seed={seed}: "
                      f"iters={result['iterations_completed']} "
                      f"bugs={result['bugs_found']} "
                      f"raw={result['raw_triggers']}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed/60:.0f}min)")

    agg = aggregate_results(all_results, args.time_budget)
    print_summary(agg)

    out_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'benchmark_results.json')
    output = {
        'experiment': 'rq2_benchmark',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'time_budget_sec': args.time_budget,
            'seeds': SEEDS,
            'modes': MODES,
            'device': args.device,
        },
        'per_run': all_results,
        'aggregated': agg,
        'total_wall_time': round(total_elapsed, 1),
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults: {out_path}")
    export_csv(agg, out_dir)


if __name__ == '__main__':
    main()
