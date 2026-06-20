#!/usr/bin/env python3
"""
RQ2 Coverage Experiment: Inductor source code coverage over time.

Standard efficiency metric (NNSmith, Tzer, NeuRI, Fuzz4All):
  line/branch coverage of compiler source over wall-clock time.

Design:
  - Fixed wall-clock budget (default: 14400s = 4h, following NNSmith/Tzer)
  - 5 seeds x 3 modes = 15 subprocess invocations (sequential)
  - Same oracle, same hardware, coverage.py on torch._inductor
  - Output: per-run JSON + aggregated coverage curves
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
WORKER = os.path.join(SCRIPT_DIR, '_coverage_worker.py')

PYTHON = os.environ.get('KERNELFUZZ_PYTHON', sys.executable)
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def run_single(mode, seed, time_budget, device, sample_interval):
    cmd = [
        PYTHON, WORKER,
        '--mode', mode,
        '--seed', str(seed),
        '--time-budget', str(time_budget),
        '--device', device,
        '--sample-interval', str(sample_interval),
    ]

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} | seed={seed} | budget={time_budget}s")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=time_budget + 300,  # grace period
    )
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}, wall={wall:.0f}s)")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
        return None

    if result.stderr:
        lines = result.stderr.strip().split('\n')
        for line in lines[-3:]:
            print(f"  {line}")

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


def interpolate_timeseries(all_runs, time_points):
    """Interpolate coverage values at common time points across seeds."""
    interpolated = []
    for tp in time_points:
        values = []
        for run in all_runs:
            ts = run['coverage_timeseries']
            times = [s['time_sec'] for s in ts]
            lines = [s['lines_covered'] for s in ts]
            branches = [s['branches_covered'] for s in ts]

            if tp <= times[0]:
                values.append({'lines': lines[0], 'branches': branches[0]})
            elif tp >= times[-1]:
                values.append({'lines': lines[-1], 'branches': branches[-1]})
            else:
                for j in range(len(times) - 1):
                    if times[j] <= tp <= times[j + 1]:
                        frac = (tp - times[j]) / (times[j + 1] - times[j]) \
                            if times[j + 1] > times[j] else 0
                        l = lines[j] + frac * (lines[j + 1] - lines[j])
                        b = branches[j] + frac * (branches[j + 1] - branches[j])
                        values.append({'lines': l, 'branches': b})
                        break

        if values:
            l_vals = [v['lines'] for v in values]
            b_vals = [v['branches'] for v in values]
            interpolated.append({
                'time': tp,
                'lines_mean': round(float(np.mean(l_vals)), 1),
                'lines_std': round(float(np.std(l_vals)), 1),
                'branches_mean': round(float(np.mean(b_vals)), 1),
                'branches_std': round(float(np.std(b_vals)), 1),
            })

    return interpolated


def aggregate_results(all_results, time_budget):
    """Compute aggregated coverage curves per mode."""
    agg = {}
    time_points = list(range(0, time_budget + 1, 30))

    for mode in MODES:
        runs = [r for r in all_results if r and r['mode'] == mode]
        if not runs:
            continue

        curve = interpolate_timeseries(runs, time_points)

        final_lines = [r['final_coverage']['lines_covered'] for r in runs]
        final_branches = [r['final_coverage']['branches_covered'] for r in runs]
        final_files = [r['final_coverage']['files_touched'] for r in runs]
        iters = [r['iterations_completed'] for r in runs]
        bugs = [r['bugs_found'] for r in runs]
        kd = [r['kernel_diversity'] for r in runs]

        agg[mode] = {
            'n_seeds': len(runs),
            'coverage_curve': curve,
            'final': {
                'lines': {
                    'mean': round(float(np.mean(final_lines)), 1),
                    'std': round(float(np.std(final_lines)), 1),
                    'values': final_lines,
                },
                'branches': {
                    'mean': round(float(np.mean(final_branches)), 1),
                    'std': round(float(np.std(final_branches)), 1),
                    'values': final_branches,
                },
                'files': {
                    'mean': round(float(np.mean(final_files)), 1),
                    'std': round(float(np.std(final_files)), 1),
                },
                'iterations': {
                    'mean': round(float(np.mean(iters)), 1),
                    'std': round(float(np.std(iters)), 1),
                },
                'bugs': {
                    'mean': round(float(np.mean(bugs)), 1),
                    'std': round(float(np.std(bugs)), 1),
                },
                'kernel_diversity': {
                    'mean': round(float(np.mean(kd)), 1),
                    'std': round(float(np.std(kd)), 1),
                },
            },
        }

    return agg


def print_summary(agg):
    """Print comparison table."""
    if not agg:
        return

    print(f"\n{'='*76}")
    print(f"  COVERAGE COMPARISON SUMMARY (final values, mean +/- std)")
    print(f"{'='*76}")

    header = f"{'Metric':<25s}"
    for mode in MODES:
        if mode in agg:
            header += f" {mode:>16s}"
    print(header)
    print("-" * 76)

    metrics = [
        ('Lines covered', 'lines'),
        ('Branches covered', 'branches'),
        ('Files touched', 'files'),
        ('Programs executed', 'iterations'),
        ('Bugs found', 'bugs'),
        ('Kernel diversity', 'kernel_diversity'),
    ]

    for label, key in metrics:
        row = f"{label:<25s}"
        for mode in MODES:
            if mode in agg:
                m = agg[mode]['final'].get(key, {})
                mean = m.get('mean', 0)
                std = m.get('std', 0)
                row += f" {mean:>8.1f}±{std:<5.1f}"
            else:
                row += f" {'N/A':>16s}"
        print(row)

    # Coverage advantage
    print()
    if 'targeted' in agg and 'random' in agg:
        t_lines = agg['targeted']['final']['lines']['mean']
        r_lines = agg['random']['final']['lines']['mean']
        ratio = t_lines / r_lines if r_lines > 0 else float('inf')
        print(f"  Targeted/Random line coverage ratio: {ratio:.2f}x")

        t_br = agg['targeted']['final']['branches']['mean']
        r_br = agg['random']['final']['branches']['mean']
        ratio_br = t_br / r_br if r_br > 0 else float('inf')
        print(f"  Targeted/Random branch coverage ratio: {ratio_br:.2f}x")


def export_csv(agg, out_dir):
    """Export coverage curves as CSV for pgfplots."""
    for mode in MODES:
        if mode not in agg:
            continue
        curve = agg[mode]['coverage_curve']
        csv_path = os.path.join(out_dir, f'coverage_{mode}.csv')
        with open(csv_path, 'w') as f:
            f.write('time,lines_mean,lines_upper,lines_lower,'
                    'branches_mean,branches_upper,branches_lower\n')
            for pt in curve:
                f.write(f"{pt['time']},"
                        f"{pt['lines_mean']},"
                        f"{pt['lines_mean'] + pt['lines_std']},"
                        f"{max(0, pt['lines_mean'] - pt['lines_std'])},"
                        f"{pt['branches_mean']},"
                        f"{pt['branches_mean'] + pt['branches_std']},"
                        f"{max(0, pt['branches_mean'] - pt['branches_std'])}\n")
        print(f"  CSV exported: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='RQ2 Coverage Experiment: compiler code coverage over time')
    parser.add_argument('--time-budget', type=int, default=14400,
                        help='Wall-clock seconds per run (default: 14400 = 4h)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 42,123,456,789,1024)')
    parser.add_argument('--modes', type=str, default=None,
                        help='Comma-separated modes (default: targeted,random,nnsmith)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='CUDA_VISIBLE_DEVICES (default: 0)')
    parser.add_argument('--sample-interval', type=int, default=30,
                        help='Coverage sampling interval in seconds (default: 30)')
    args = parser.parse_args()

    global SEEDS, MODES
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(',')]
    if args.modes:
        MODES = [m.strip() for m in args.modes.split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f"RQ2 Coverage Experiment")
    print(f"  Budget: {args.time_budget}s ({args.time_budget/3600:.1f}h) per run")
    print(f"  Seeds: {SEEDS}")
    print(f"  Modes: {MODES}")
    print(f"  Sample interval: {args.sample_interval}s")
    print(f"  Total expected: ~{len(SEEDS) * len(MODES) * args.time_budget / 3600:.1f}h")
    print()

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    all_results = []
    total_t0 = time.time()

    for seed in SEEDS:
        for mode in MODES:
            result = run_single(
                mode, seed, args.time_budget, args.device, args.sample_interval
            )
            if result:
                all_results.append(result)
                fc = result['final_coverage']
                print(f"  -> {mode} seed={seed}: "
                      f"lines={fc['lines_covered']} "
                      f"branches={fc['branches_covered']} "
                      f"files={fc['files_touched']} "
                      f"iters={result['iterations_completed']}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")

    agg = aggregate_results(all_results, args.time_budget)
    print_summary(agg)

    out_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'coverage_results.json')
    output = {
        'experiment': 'rq2_coverage',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'time_budget_sec': args.time_budget,
            'seeds': SEEDS,
            'modes': MODES,
            'sample_interval': args.sample_interval,
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
