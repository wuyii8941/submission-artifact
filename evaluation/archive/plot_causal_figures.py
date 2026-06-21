#!/usr/bin/env python3
"""
Generate two figures for paper revision:
1. Time-alignment: bug discoveries overlaid on kernel diversity curve
2. Time-to-first-bug: cumulative bugs over time (targeted vs random vs NNSmith)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

RESULTS_DIR = Path(__file__).parent / "time_controlled" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "tex" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    targeted = json.loads((RESULTS_DIR / "extended_targeted.json").read_text())
    random_ = json.loads((RESULTS_DIR / "extended_random.json").read_text())
    nnsmith = json.loads((RESULTS_DIR / "extended_nnsmith.json").read_text())
    return targeted, random_, nnsmith


def extract_bug_times(data):
    """Extract bug discovery times per seed."""
    all_bugs = []
    for run in data["per_run"]:
        bugs = [(b["time_sec"], b["bug_type"], b.get("meta", {}).get("family", "unknown"))
                for b in run["bugs"]]
        all_bugs.append(bugs)
    return all_bugs


def extract_diversity_curve(data):
    """Extract kernel diversity over time from extended coverage results."""
    coverage_file = RESULTS_DIR / "extended_coverage_results.json"
    if coverage_file.exists():
        cov = json.loads(coverage_file.read_text())
        curves = {}
        for mode in ["targeted", "random", "nnsmith"]:
            if mode in cov:
                curves[mode] = cov[mode]
        if curves:
            return curves

    # Fallback: reconstruct from per-run data (approximate)
    # Use iteration-proportional time to estimate diversity growth
    return None


def plot_family_bug_alignment(targeted_data, random_data):
    """
    Figure 1: Two panels showing D → bug causal link.
    (a) Targeted trigger distribution by family (pie/donut) — shows split_scan dominance
    (b) Family reachability × confirmed bugs — shows unreachable family harbors real bug
    """
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4), gridspec_kw={'width_ratios': [1.1, 1]})

    families_order = ['split_scan', 'pointwise', 'extern_conv',
                      'extern_gemm', 'persistent_reduction', 'reduction']
    family_labels = ['split_scan', 'pointwise', 'extern_conv',
                     'extern_gemm', 'per_red', 'reduction']

    # (a) Targeted trigger distribution by family
    t_fam = {f: 0 for f in families_order}
    for run in targeted_data["per_run"]:
        for b in run["bugs"]:
            if "diff" in b.get("bug_type", "").lower():
                fam = b.get("meta", {}).get("family", "unknown")
                if fam in t_fam:
                    t_fam[fam] += 1

    vals = [t_fam[f] for f in families_order]
    colors = ['tab:red'] + ['tab:blue'] * 5  # split_scan in red
    explode = [0.08] + [0] * 5

    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        vals, labels=None, autopct=lambda pct: f'{pct:.0f}%' if pct > 4 else '',
        colors=colors, explode=explode, startangle=90,
        pctdistance=0.75, textprops={'fontsize': 6.5})

    for at in autotexts:
        at.set_fontsize(6)
        at.set_color('white')
        at.set_fontweight('bold')

    ax.set_title('(a) Diff triggers by family\n(targeted, all seeds)', fontsize=7.5)

    # Manual legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='tab:red', label=f'split_scan ({vals[0]})', alpha=0.9),
        Patch(facecolor='tab:blue', label=f'other 5 ({sum(vals[1:])})', alpha=0.9),
    ]
    ax.legend(handles=legend_elems, fontsize=6, loc='lower left',
              bbox_to_anchor=(-0.1, -0.05))

    # (b) Family reachability vs confirmed bugs (from Table 7 in paper)
    ax2 = axes[1]
    families_b = ['pw', 'red', 'per_red', 'split', 'gemm', 'conv']
    confirmed = [5, 1, 0, 1, 1, 0]  # from paper Table 7
    reached_by_random = [True, True, True, False, True, True]

    x = np.arange(len(families_b))
    bar_colors = ['tab:blue' if r else 'tab:red' for r in reached_by_random]
    bars = ax2.bar(x, confirmed, 0.65, color=bar_colors, alpha=0.85)

    # Add reachability markers below bars
    for i, reached in enumerate(reached_by_random):
        marker = 'o' if reached else 'x'
        color = 'green' if reached else 'red'
        ax2.plot(i, -0.4, marker=marker, color=color, markersize=7,
                 markeredgewidth=1.8, clip_on=False)

    ax2.set_xticks(x)
    ax2.set_xticklabels(families_b, fontsize=6.5, rotation=30, ha='right')
    ax2.set_ylabel('Confirmed bugs', fontsize=7.5)
    ax2.set_ylim(-0.2, 6)
    ax2.set_title('(b) Bugs vs reachability', fontsize=7.5)

    legend_elems2 = [
        Patch(facecolor='tab:blue', alpha=0.85, label='Random reaches'),
        Patch(facecolor='tab:red', alpha=0.85, label='Targeted only'),
    ]
    ax2.legend(handles=legend_elems2, fontsize=5.5, loc='upper right')

    plt.tight_layout(w_pad=0.5)
    plt.savefig(FIGURES_DIR / "family_bug_alignment.pdf")
    plt.savefig(FIGURES_DIR / "family_bug_alignment.png")
    print(f"Saved: {FIGURES_DIR / 'family_bug_alignment.pdf'}")
    plt.close()


def plot_time_to_first_bug(targeted_data, random_data, nnsmith_data):
    """
    Figure 2: Cumulative DIFFERENTIAL triggers over time.
    Filters out crash/trivial triggers to show quality bug discovery rate.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))

    methods = [
        ("Targeted", targeted_data, 'tab:blue'),
        ("Random", random_data, 'tab:orange'),
        ("NNSmith", nnsmith_data, 'tab:green'),
    ]

    time_grid = np.linspace(0, 1800, 1000)

    for name, data, color in methods:
        cum_curves = []
        for run in data["per_run"]:
            # Filter: only differential triggers (SUSPICIOUS_diff / CATASTROPHIC_diff)
            diff_bugs = [b for b in run["bugs"] if "diff" in b.get("bug_type", "").lower()]
            bug_times = sorted([b["time_sec"] for b in diff_bugs])
            cum = np.zeros_like(time_grid)
            for t_bug in bug_times:
                cum[time_grid >= t_bug] += 1
            cum_curves.append(cum)

        if not cum_curves:
            continue

        cum_curves = np.array(cum_curves)
        mean_cum = cum_curves.mean(axis=0)
        std_cum = cum_curves.std(axis=0)

        ax.plot(time_grid, mean_cum, color=color, linewidth=1.3, label=name)
        ax.fill_between(time_grid, mean_cum - std_cum, mean_cum + std_cum,
                        alpha=0.15, color=color)

    ax.set_xlabel('Wall-clock time (seconds)')
    ax.set_ylabel('Cumulative differential triggers')
    ax.set_xlim(0, 1800)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(FIGURES_DIR / "time_to_first_bug.pdf")
    plt.savefig(FIGURES_DIR / "time_to_first_bug.png")
    print(f"Saved: {FIGURES_DIR / 'time_to_first_bug.pdf'}")
    plt.close()


def compute_stats(targeted_data, random_data):
    """Print key statistics for the paper text."""
    print("\n=== Statistics for Paper ===\n")

    # Time to first DIFFERENTIAL trigger
    print("--- Time to first DIFFERENTIAL trigger ---")
    for name, data in [("Targeted", targeted_data), ("Random", random_data)]:
        first_diff_times = []
        for run in data["per_run"]:
            diff_bugs = [b for b in run["bugs"] if "diff" in b.get("bug_type", "").lower()]
            if diff_bugs:
                first_diff_times.append(diff_bugs[0]["time_sec"])
            else:
                first_diff_times.append(float('inf'))
        arr = np.array([t for t in first_diff_times if t != float('inf')])
        if len(arr):
            print(f"  {name}: {arr.mean():.1f} ± {arr.std():.1f}s (n={len(arr)} seeds)")

    # Differential triggers at 300s and 1800s
    print("\n--- Cumulative differential triggers at key time points ---")
    for name, data in [("Targeted", targeted_data), ("Random", random_data)]:
        at_300 = []
        at_1800 = []
        for run in data["per_run"]:
            diff_bugs = [b for b in run["bugs"] if "diff" in b.get("bug_type", "").lower()]
            at_300.append(sum(1 for b in diff_bugs if b["time_sec"] <= 300))
            at_1800.append(sum(1 for b in diff_bugs if b["time_sec"] <= 1800))
        print(f"  {name} @300s:  {np.mean(at_300):.1f} ± {np.std(at_300):.1f}")
        print(f"  {name} @1800s: {np.mean(at_1800):.1f} ± {np.std(at_1800):.1f}")

    # Bug type breakdown
    print("\n--- Bug type breakdown ---")
    for name, data in [("Targeted", targeted_data), ("Random", random_data)]:
        types = defaultdict(int)
        for run in data["per_run"]:
            for b in run["bugs"]:
                types[b["bug_type"]] += 1
        total = sum(types.values())
        print(f"  {name} ({total} total):")
        for k, v in sorted(types.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v} ({100*v/total:.0f}%)")

    # Bug family distribution (targeted only)
    print("\n--- Bug family distribution (targeted, all seeds) ---")
    family_counts = defaultdict(int)
    for run in targeted_data["per_run"]:
        for b in run["bugs"]:
            fam = b.get("meta", {}).get("family", "unknown")
            family_counts[fam] += 1
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {fam}: {cnt}")

    # Correlation: diversity growth rate vs bug rate (for time-alignment claim)
    print("\n--- Temporal correlation (per-seed, 60s windows) ---")
    from scipy import stats as sp_stats
    for run in targeted_data["per_run"][:1]:
        diff_bugs = [b for b in run["bugs"] if "diff" in b.get("bug_type", "").lower()]
        bug_times = [b["time_sec"] for b in diff_bugs]
        # Bin into 60s windows
        n_bins = 30
        bins = np.linspace(0, 1800, n_bins + 1)
        bug_per_window, _ = np.histogram(bug_times, bins=bins)
        # Approximate diversity growth per window from iteration count
        iters = run["iterations_completed"]
        iter_per_window = iters / n_bins  # uniform approximation
        # Use bug_per_window as proxy (can't get exact D per window without timeseries)
        print(f"  Seed {run['seed']}: bugs/window distribution:")
        print(f"    First 10 windows (0-600s): {bug_per_window[:10].tolist()}")
        print(f"    Last 10 windows (1200-1800s): {bug_per_window[-10:].tolist()}")


def main():
    targeted, random_, nnsmith = load_results()

    print("Loaded data:")
    print(f"  Targeted: {len(targeted['per_run'])} runs, {sum(r['bugs_found'] for r in targeted['per_run'])} total bugs")
    print(f"  Random:   {len(random_['per_run'])} runs, {sum(r['bugs_found'] for r in random_['per_run'])} total bugs")
    print(f"  NNSmith:  {len(nnsmith['per_run'])} runs, {sum(r['bugs_found'] for r in nnsmith['per_run'])} total bugs")

    plot_family_bug_alignment(targeted, random_)
    plot_time_to_first_bug(targeted, random_, nnsmith)
    compute_stats(targeted, random_)

    print("\nDone. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
