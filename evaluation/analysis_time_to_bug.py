#!/usr/bin/env python3
"""
Extract time-to-first-bug and cumulative finding curves from existing 1800s data.
Produces LaTeX-ready numbers and CSV for plotting.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────
# PyTorch data
# ─────────────────────────────────────────────────────────────────────
def load_pytorch():
    v4_file = BASE / "time_controlled/results/extended_coverage_v4_results.json"
    nnsmith_file = BASE / "time_controlled/results/extended_nnsmith.json"

    runs = {"targeted": [], "random": [], "nnsmith": []}

    with open(v4_file) as f:
        d = json.load(f)
    for r in d["per_run"]:
        runs[r["mode"]].append(r)

    with open(nnsmith_file) as f:
        d = json.load(f)
    for r in d["per_run"]:
        runs["nnsmith"].append(r)

    return runs


def load_tf():
    results_dir = BASE / "rq5_cross_framework/results"
    seeds = [42, 123, 456, 789, 1024]
    runs = {"targeted": [], "random": [], "nnsmith": []}

    for mode in ["targeted", "random"]:
        for seed in seeds:
            p = results_dir / f"_tmp_{mode}_{seed}_1800.json"
            if p.exists():
                with open(p) as f:
                    runs[mode].append(json.load(f))

    for seed in seeds:
        p = results_dir / f"_tmp_nnsmith_{seed}_1800.json"
        if p.exists():
            with open(p) as f:
                runs["nnsmith"].append(json.load(f))

    return runs


# ─────────────────────────────────────────────────────────────────────
# Time-to-first-bug analysis
# ─────────────────────────────────────────────────────────────────────
def time_to_first_bug_pytorch(runs):
    """Extract time to first bug for each method."""
    results = {}
    for mode, run_list in runs.items():
        times = []
        for r in run_list:
            bugs = r.get("bugs", [])
            if bugs:
                first_time = min(b["time_sec"] for b in bugs if b.get("time_sec") is not None)
                times.append(first_time)
            else:
                times.append(float("inf"))
        results[mode] = times
    return results


def time_to_first_bug_tf(runs):
    """Extract time to first finding for each TF method."""
    results = {}
    for mode, run_list in runs.items():
        times = []
        for r in run_list:
            if mode == "nnsmith":
                bugs = r.get("bugs", [])
                if bugs:
                    # NNSmith bugs don't have time_sec, estimate from iteration
                    elapsed = r.get("elapsed_sec", 1800)
                    budget = r.get("budget_used", 1)
                    first_idx = min(b["idx"] for b in bugs)
                    estimated_time = elapsed * first_idx / max(budget, 1)
                    times.append(estimated_time)
                else:
                    times.append(float("inf"))
            else:
                findings = r.get("findings", [])
                if findings:
                    first_time = min(f["time_sec"] for f in findings)
                    times.append(first_time)
                else:
                    times.append(float("inf"))
        results[mode] = times
    return results


# ─────────────────────────────────────────────────────────────────────
# Cumulative curves (time-binned)
# ─────────────────────────────────────────────────────────────────────
def cumulative_curves_pytorch(runs, time_bins):
    """Build cumulative bug count curves over time."""
    curves = {}
    for mode, run_list in runs.items():
        seed_curves = []
        for r in run_list:
            bugs = r.get("bugs", [])
            bug_times = sorted(b["time_sec"] for b in bugs if b.get("time_sec") is not None)
            cum = []
            for t in time_bins:
                count = sum(1 for bt in bug_times if bt <= t)
                cum.append(count)
            seed_curves.append(cum)
        curves[mode] = np.array(seed_curves) if seed_curves else np.zeros((1, len(time_bins)))
    return curves


def cumulative_curves_tf(runs, time_bins):
    """Build cumulative finding count curves over time."""
    curves = {}
    for mode, run_list in runs.items():
        seed_curves = []
        for r in run_list:
            if mode == "nnsmith":
                bugs = r.get("bugs", [])
                elapsed = r.get("elapsed_sec", 1800)
                budget = r.get("budget_used", 1)
                bug_times = sorted(elapsed * b["idx"] / max(budget, 1) for b in bugs)
            else:
                findings = r.get("findings", [])
                bug_times = sorted(f["time_sec"] for f in findings)
            cum = []
            for t in time_bins:
                count = sum(1 for bt in bug_times if bt <= t)
                cum.append(count)
            seed_curves.append(cum)
        curves[mode] = np.array(seed_curves) if seed_curves else np.zeros((1, len(time_bins)))
    return curves


# ─────────────────────────────────────────────────────────────────────
# Kernel diversity curves (from timeseries)
# ─────────────────────────────────────────────────────────────────────
def diversity_curves_pytorch(runs, time_bins):
    """Build kernel diversity curves over time from timeseries."""
    curves = {}
    for mode, run_list in runs.items():
        seed_curves = []
        for r in run_list:
            ts = r.get("timeseries", [])
            if not ts:
                seed_curves.append([0] * len(time_bins))
                continue
            ts_times = [p["time_sec"] for p in ts]
            if mode == "nnsmith":
                ts_vals = [p.get("kernel_diversity", p.get("graph_diversity", 0)) for p in ts]
            else:
                ts_vals = [p["kernel_diversity"] for p in ts]
            cum = []
            for t in time_bins:
                idx = np.searchsorted(ts_times, t, side="right") - 1
                idx = max(0, min(idx, len(ts_vals) - 1))
                cum.append(ts_vals[idx])
            seed_curves.append(cum)
        curves[mode] = np.array(seed_curves)
    return curves


# ─────────────────────────────────────────────────────────────────────
# Family/scheduling curves (from timeseries)
# ─────────────────────────────────────────────────────────────────────
def family_curves_pytorch(runs, time_bins):
    """Build family count curves over time."""
    curves = {}
    for mode, run_list in runs.items():
        seed_curves = []
        for r in run_list:
            ts = r.get("timeseries", [])
            if not ts or "families" not in ts[0]:
                seed_curves.append([0] * len(time_bins))
                continue
            ts_times = [p["time_sec"] for p in ts]
            ts_vals = [p["families"] for p in ts]
            cum = []
            for t in time_bins:
                idx = np.searchsorted(ts_times, t, side="right") - 1
                idx = max(0, min(idx, len(ts_vals) - 1))
                cum.append(ts_vals[idx])
            seed_curves.append(cum)
        curves[mode] = np.array(seed_curves)
    return curves


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("TIME-TO-FIRST-BUG & CUMULATIVE CURVE ANALYSIS")
    print("=" * 70)

    # ── PyTorch ──
    pt_runs = load_pytorch()
    print(f"\nPyTorch runs: targeted={len(pt_runs['targeted'])}, "
          f"random={len(pt_runs['random'])}, nnsmith={len(pt_runs['nnsmith'])}")

    ttfb_pt = time_to_first_bug_pytorch(pt_runs)
    print("\n── PyTorch: Time to First Bug (seconds) ──")
    for mode in ["targeted", "random", "nnsmith"]:
        times = ttfb_pt[mode]
        finite = [t for t in times if t < float("inf")]
        if finite:
            print(f"  {mode:12s}: mean={np.mean(finite):.1f}s, "
                  f"median={np.median(finite):.1f}s, "
                  f"min={min(finite):.1f}s, max={max(finite):.1f}s, "
                  f"seeds_with_bug={len(finite)}/{len(times)}")
        else:
            print(f"  {mode:12s}: NO BUGS across {len(times)} seeds")

    # Time bins for curves
    pt_bins = np.arange(0, 1801, 10)  # every 10s

    pt_bug_curves = cumulative_curves_pytorch(pt_runs, pt_bins)
    pt_div_curves = diversity_curves_pytorch(pt_runs, pt_bins)
    pt_fam_curves = family_curves_pytorch(pt_runs, pt_bins)

    print("\n── PyTorch: Cumulative Bugs at Key Timepoints ──")
    for tp in [60, 300, 600, 900, 1200, 1800]:
        idx = tp // 10
        print(f"  t={tp:5d}s:", end="")
        for mode in ["targeted", "random", "nnsmith"]:
            c = pt_bug_curves[mode]
            print(f"  {mode}={np.mean(c[:, idx]):.1f}±{np.std(c[:, idx]):.1f}", end="")
        print()

    print("\n── PyTorch: Kernel Diversity at Key Timepoints ──")
    for tp in [60, 300, 600, 900, 1200, 1800]:
        idx = tp // 10
        print(f"  t={tp:5d}s:", end="")
        for mode in ["targeted", "random", "nnsmith"]:
            c = pt_div_curves[mode]
            print(f"  {mode}={np.mean(c[:, idx]):.0f}±{np.std(c[:, idx]):.0f}", end="")
        print()

    print("\n── PyTorch: Families Reached at Key Timepoints ──")
    for tp in [60, 300, 600, 900, 1200, 1800]:
        idx = tp // 10
        print(f"  t={tp:5d}s:", end="")
        for mode in ["targeted", "random", "nnsmith"]:
            c = pt_fam_curves[mode]
            if c.max() > 0:
                print(f"  {mode}={np.mean(c[:, idx]):.1f}±{np.std(c[:, idx]):.1f}", end="")
            else:
                print(f"  {mode}=N/A", end="")
        print()

    # ── TensorFlow ──
    tf_runs = load_tf()
    print(f"\n{'='*70}")
    print(f"TensorFlow runs: targeted={len(tf_runs['targeted'])}, "
          f"random={len(tf_runs['random'])}, nnsmith={len(tf_runs['nnsmith'])}")

    ttfb_tf = time_to_first_bug_tf(tf_runs)
    print("\n── TF: Time to First Finding (seconds) ──")
    for mode in ["targeted", "random", "nnsmith"]:
        times = ttfb_tf[mode]
        finite = [t for t in times if t < float("inf")]
        if finite:
            print(f"  {mode:12s}: mean={np.mean(finite):.1f}s, "
                  f"median={np.median(finite):.1f}s, "
                  f"min={min(finite):.1f}s, max={max(finite):.1f}s, "
                  f"seeds_with_finding={len(finite)}/{len(times)}")
        else:
            print(f"  {mode:12s}: NO FINDINGS across {len(times)} seeds")

    tf_bins = np.arange(0, 1801, 10)
    tf_bug_curves = cumulative_curves_tf(tf_runs, tf_bins)

    print("\n── TF: Cumulative Findings at Key Timepoints ──")
    for tp in [60, 300, 600, 900, 1200, 1800]:
        idx = tp // 10
        print(f"  t={tp:5d}s:", end="")
        for mode in ["targeted", "random", "nnsmith"]:
            c = tf_bug_curves[mode]
            print(f"  {mode}={np.mean(c[:, idx]):.1f}±{np.std(c[:, idx]):.1f}", end="")
        print()

    # ── Targeted advantage ratio over time ──
    print("\n── Targeted/Random Advantage Ratio Over Time ──")
    print("  PyTorch (cumulative bugs):")
    for tp in [60, 300, 600, 900, 1200, 1800]:
        idx = tp // 10
        t_mean = np.mean(pt_bug_curves["targeted"][:, idx])
        r_mean = np.mean(pt_bug_curves["random"][:, idx])
        ratio = t_mean / max(r_mean, 0.1)
        print(f"    t={tp:5d}s: targeted={t_mean:.1f}, random={r_mean:.1f}, ratio={ratio:.1f}x")

    print("  TF (cumulative findings):")
    for tp in [60, 300, 600, 900, 1200, 1800]:
        idx = tp // 10
        t_mean = np.mean(tf_bug_curves["targeted"][:, idx])
        r_mean = np.mean(tf_bug_curves["random"][:, idx])
        ratio = t_mean / max(r_mean, 0.1)
        print(f"    t={tp:5d}s: targeted={t_mean:.1f}, random={r_mean:.1f}, ratio={ratio:.1f}x")

    # ── Save CSV for plotting ──
    out_dir = BASE / "analysis_results"
    out_dir.mkdir(exist_ok=True)

    # PyTorch curves
    with open(out_dir / "pt_cumulative_bugs.csv", "w") as f:
        f.write("time_sec,targeted_mean,targeted_std,random_mean,random_std,nnsmith_mean,nnsmith_std\n")
        for i, t in enumerate(pt_bins):
            vals = []
            for mode in ["targeted", "random", "nnsmith"]:
                c = pt_bug_curves[mode]
                vals.extend([np.mean(c[:, i]), np.std(c[:, i])])
            f.write(f"{t},{','.join(f'{v:.2f}' for v in vals)}\n")

    with open(out_dir / "pt_diversity.csv", "w") as f:
        f.write("time_sec,targeted_mean,targeted_std,random_mean,random_std,nnsmith_mean,nnsmith_std\n")
        for i, t in enumerate(pt_bins):
            vals = []
            for mode in ["targeted", "random", "nnsmith"]:
                c = pt_div_curves[mode]
                vals.extend([np.mean(c[:, i]), np.std(c[:, i])])
            f.write(f"{t},{','.join(f'{v:.2f}' for v in vals)}\n")

    # TF curves
    with open(out_dir / "tf_cumulative_findings.csv", "w") as f:
        f.write("time_sec,targeted_mean,targeted_std,random_mean,random_std,nnsmith_mean,nnsmith_std\n")
        for i, t in enumerate(tf_bins):
            vals = []
            for mode in ["targeted", "random", "nnsmith"]:
                c = tf_bug_curves[mode]
                vals.extend([np.mean(c[:, i]), np.std(c[:, i])])
            f.write(f"{t},{','.join(f'{v:.2f}' for v in vals)}\n")

    print(f"\nCSV files saved to {out_dir}/")

    # ── Summary for paper ──
    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)
    print("\nPyTorch time-to-first-bug:")
    for mode in ["targeted", "random"]:
        finite = [t for t in ttfb_pt[mode] if t < float("inf")]
        if finite:
            print(f"  {mode}: {np.mean(finite):.0f}s (median {np.median(finite):.0f}s)")
    print(f"  nnsmith: no bugs across 5 seeds")

    print("\nTF time-to-first-finding:")
    for mode in ["targeted", "random", "nnsmith"]:
        finite = [t for t in ttfb_tf[mode] if t < float("inf")]
        if finite:
            print(f"  {mode}: {np.mean(finite):.1f}s (median {np.median(finite):.1f}s)")


if __name__ == "__main__":
    main()
