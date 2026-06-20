#!/usr/bin/env python3
"""
Plot D(t) coverage growth curves from extended coverage experiment results.
Produces a publication-ready figure for RQ3.

Usage:
  python plot_coverage_curves.py results/extended_coverage_results.json
  python plot_coverage_curves.py results/time_controlled_results.json  # also works
"""

import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

COLORS = {
    'targeted': '#2166ac',
    'random': '#b2182b',
    'nnsmith': '#4dac26',
}
LABELS = {
    'targeted': 'KernelFuzz (targeted)',
    'random': 'Random baseline',
    'nnsmith': 'NNSmith',
}


def extract_timeseries(data):
    """Extract per-mode timeseries: list of (time_sec, kernel_diversity) per seed."""
    series = {}
    for run in data['per_run']:
        mode = run['mode']
        ts = run.get('timeseries', [])
        if not ts:
            continue
        times = [p['time_sec'] for p in ts]
        divs = [p['kernel_diversity'] for p in ts]
        series.setdefault(mode, []).append((times, divs))
    return series


def interpolate_to_grid(series_list, grid):
    """Interpolate multiple timeseries onto a common time grid."""
    interp = []
    for times, divs in series_list:
        vals = np.interp(grid, times, divs, left=divs[0], right=divs[-1])
        interp.append(vals)
    return np.array(interp)


def plot(data, output_path, max_time=None):
    series = extract_timeseries(data)
    budget = data['config']['time_budget_sec']
    if max_time is None:
        max_time = budget

    grid = np.linspace(0, max_time, 200)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for mode in ['targeted', 'random', 'nnsmith']:
        if mode not in series:
            continue
        interp = interpolate_to_grid(series[mode], grid)
        mean = interp.mean(axis=0)
        std = interp.std(axis=0)

        ax.plot(grid, mean, color=COLORS[mode], label=LABELS[mode], linewidth=2)
        ax.fill_between(grid, mean - std, mean + std,
                        color=COLORS[mode], alpha=0.15)

    ax.set_xlabel('Wall-clock time (seconds)', fontsize=11)
    ax.set_ylabel('Kernel diversity $\\mathcal{D}$', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, max_time)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Also save a version with family-first-reached annotations
    # (if the data has families_seen timeseries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='JSON results file')
    parser.add_argument('-o', '--output', default='coverage_curves.pdf')
    parser.add_argument('--max-time', type=float, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    plot(data, args.output, args.max_time)


if __name__ == '__main__':
    main()
