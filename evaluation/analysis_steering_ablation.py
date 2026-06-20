#!/usr/bin/env python3
"""
Analyze steering ablation results: targeted vs targeted_blind.
targeted_blind uses the same family-aware generators but with
frequency-weighted random family selection instead of systematic
state rotation.
"""

import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
SEEDS = [42, 123, 456, 789, 1024]


def load_runs(mode, results_dir):
    runs = []
    for seed in SEEDS:
        p = results_dir / f"{mode}_{seed}_1800.json"
        ckpt = results_dir / f".ckpt_blind_{seed}.json"
        if p.exists() and p.stat().st_size > 0:
            with open(p) as f:
                runs.append(json.load(f))
        elif ckpt.exists():
            with open(ckpt) as f:
                d = json.load(f)
            runs.append({
                "mode": mode,
                "seed": seed,
                "kernel_diversity": len(d.get("kernel_templates", [])),
                "families_seen": sorted(d.get("families_seen", [])),
                "n_families": len(d.get("families_seen", [])),
                "bugs_found": len(d.get("bugs", [])),
                "bugs": d.get("bugs", []),
                "iterations_completed": d.get("iteration", 0),
                "elapsed_sec": d.get("elapsed", 0),
                "total_ok": d.get("total_ok", 0),
                "gen_failures": d.get("gen_failures", 0),
                "timeseries": d.get("timeseries", []),
                "kernel_templates": d.get("kernel_templates", []),
                "raw_triggers": d.get("raw_triggers", 0),
                "trivial_crashes": d.get("trivial_crashes", 0),
                "regions_seen": d.get("regions_seen", []),
            })
    return runs


def load_targeted_v4():
    v4_file = BASE / "time_controlled/results/extended_coverage_v4_results.json"
    with open(v4_file) as f:
        d = json.load(f)
    return [r for r in d["per_run"] if r["mode"] == "targeted"]


def main():
    ablation_dir = BASE / "time_controlled/results/steering_ablation"

    blind_runs = load_runs("targeted_blind", ablation_dir)
    targeted_runs = load_targeted_v4()

    print(f"Targeted runs: {len(targeted_runs)}")
    print(f"Targeted_blind runs: {len(blind_runs)}")

    if not blind_runs:
        print("No targeted_blind results found!")
        return

    print("\n" + "=" * 70)
    print("STEERING ABLATION: targeted vs targeted_blind")
    print("(Same generators, different state selection policy)")
    print("=" * 70)

    # Key metrics
    for label, runs in [("targeted", targeted_runs), ("targeted_blind", blind_runs)]:
        kds = [r["kernel_diversity"] for r in runs]
        bugs = [r["bugs_found"] for r in runs]
        families = [r["n_families"] for r in runs]
        iters = [r["iterations_completed"] for r in runs]

        print(f"\n── {label} ──")
        print(f"  Kernel diversity:   {np.mean(kds):.0f} ± {np.std(kds):.0f}")
        print(f"  Bugs found:         {np.mean(bugs):.1f} ± {np.std(bugs):.1f}")
        print(f"  Families reached:   {np.mean(families):.1f} ± {np.std(families):.1f}")
        print(f"  Iterations:         {np.mean(iters):.0f} ± {np.std(iters):.0f}")

        # Family breakdown
        all_fams = set()
        for r in runs:
            all_fams.update(r.get("families_seen", []))
        print(f"  Unique families (union): {sorted(all_fams)}")

        # split_scan reachability
        split_scan_count = sum(1 for r in runs if "split_scan" in r.get("families_seen", []))
        print(f"  split_scan reached: {split_scan_count}/{len(runs)} seeds")

    # Time-to-first-bug
    print("\n── Time to First Bug ──")
    for label, runs in [("targeted", targeted_runs), ("targeted_blind", blind_runs)]:
        times = []
        for r in runs:
            bugs = r.get("bugs", [])
            if bugs:
                first = min(b["time_sec"] for b in bugs if b.get("time_sec") is not None)
                times.append(first)
            else:
                times.append(float("inf"))
        finite = [t for t in times if t < float("inf")]
        if finite:
            print(f"  {label:20s}: mean={np.mean(finite):.1f}s, "
                  f"median={np.median(finite):.1f}s, "
                  f"seeds_with_bug={len(finite)}/{len(times)}")
        else:
            print(f"  {label:20s}: NO BUGS")

    # Cumulative curves at key timepoints
    time_bins = np.arange(0, 1801, 10)
    print("\n── Cumulative Bugs at Key Timepoints ──")
    for label, runs in [("targeted", targeted_runs), ("targeted_blind", blind_runs)]:
        curves = []
        for r in runs:
            bugs = r.get("bugs", [])
            bug_times = sorted(b["time_sec"] for b in bugs if b.get("time_sec") is not None)
            cum = [sum(1 for bt in bug_times if bt <= t) for t in time_bins]
            curves.append(cum)
        curves = np.array(curves)
        print(f"  {label}:")
        for tp in [60, 300, 600, 900, 1200, 1800]:
            idx = tp // 10
            print(f"    t={tp:5d}s: {np.mean(curves[:, idx]):.1f} ± {np.std(curves[:, idx]):.1f}")

    # Kernel diversity over time
    print("\n── Kernel Diversity at Key Timepoints ──")
    for label, runs in [("targeted", targeted_runs), ("targeted_blind", blind_runs)]:
        curves = []
        for r in runs:
            ts = r.get("timeseries", [])
            if not ts:
                continue
            ts_times = [p["time_sec"] for p in ts]
            ts_vals = [p["kernel_diversity"] for p in ts]
            cum = []
            for t in time_bins:
                idx = np.searchsorted(ts_times, t, side="right") - 1
                idx = max(0, min(idx, len(ts_vals) - 1))
                cum.append(ts_vals[idx])
            curves.append(cum)
        curves = np.array(curves)
        print(f"  {label}:")
        for tp in [60, 300, 600, 900, 1200, 1800]:
            idx = tp // 10
            print(f"    t={tp:5d}s: {np.mean(curves[:, idx]):.0f} ± {np.std(curves[:, idx]):.0f}")

    # Bug type breakdown
    print("\n── Bug Type Distribution ──")
    for label, runs in [("targeted", targeted_runs), ("targeted_blind", blind_runs)]:
        type_counts = {}
        total = 0
        for r in runs:
            for b in r.get("bugs", []):
                bt = b.get("bug_type", "unknown")
                type_counts[bt] = type_counts.get(bt, 0) + 1
                total += 1
        print(f"  {label} (total={total} across {len(runs)} seeds):")
        for bt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {bt}: {cnt} ({100*cnt/max(total,1):.1f}%)")

    # Scheduling decision distribution from checkpoint data
    print("\n── Scheduling Decisions (from kernel templates) ──")
    for label, runs in [("targeted", targeted_runs), ("targeted_blind", blind_runs)]:
        if label == "targeted":
            ckpt_dir = BASE / "time_controlled/results/logs_4h_v4"
            ckpts = sorted(ckpt_dir.glob("*.ckpt.json"))
            all_templates = []
            for cp in ckpts:
                with open(cp) as f:
                    d = json.load(f)
                tpls = d.get("kernel_templates", [])
                if "targeted" in str(cp):
                    all_templates.extend(tpls)
        else:
            all_templates = []
            for seed in SEEDS:
                cp = BASE / f"time_controlled/results/steering_ablation/.ckpt_blind_{seed}.json"
                if cp.exists():
                    with open(cp) as f:
                        d = json.load(f)
                    all_templates.extend(d.get("kernel_templates", []))

        sched_counts = {"pointwise": 0, "reduction": 0, "persistent_reduction": 0, "split_scan": 0, "other": 0}
        for t in all_templates:
            if t.startswith("triton_poi"):
                sched_counts["pointwise"] += 1
            elif t.startswith("triton_red"):
                sched_counts["reduction"] += 1
            elif t.startswith("triton_per"):
                sched_counts["persistent_reduction"] += 1
            elif t.startswith("triton_spl"):
                sched_counts["split_scan"] += 1
            else:
                sched_counts["other"] += 1

        total = sum(sched_counts.values())
        print(f"  {label} ({total} templates):")
        for k, v in sched_counts.items():
            if total > 0:
                print(f"    {k:25s}: {v:5d} ({100*v/total:.1f}%)")

    # Statistical test
    print("\n── Statistical Significance (Mann-Whitney U) ──")
    from scipy.stats import mannwhitneyu
    t_kd = [r["kernel_diversity"] for r in targeted_runs]
    b_kd = [r["kernel_diversity"] for r in blind_runs]
    t_bugs = [r["bugs_found"] for r in targeted_runs]
    b_bugs = [r["bugs_found"] for r in blind_runs]

    stat, p = mannwhitneyu(t_kd, b_kd, alternative="greater")
    print(f"  Kernel diversity: targeted > blind? U={stat}, p={p:.4f}")
    stat, p = mannwhitneyu(t_bugs, b_bugs, alternative="greater")
    print(f"  Bug count: targeted > blind? U={stat}, p={p:.4f}")


if __name__ == "__main__":
    main()
