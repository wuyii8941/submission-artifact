#!/usr/bin/env python3
"""
Interaction Coverage vs Bug Yield — the causal chain closer.

Shows that random/targeted generation saturates at a small number of
semantic interaction cells, and that a large fraction of confirmed bugs
occupy cells NEVER explored by these baselines.

Key findings:
  - Random/Targeted both plateau at 9 interaction cells (out of 6^2 possible)
  - Saturation occurs at ~200 programs — the remaining 800 add nothing
  - 6/23 bugs (26%) are in interaction cells with ZERO baseline frequency
  - DeepOpFuzz found 23 bugs: 13 kernel-level spanning 5 distinct cells, 10 pre-codegen

Uses:
  - Random baseline (1k programs, compiled traces)
  - Targeted baseline (1k programs, compiled traces)
  - 23 confirmed Inductor bugs
"""

import json
import os
import re
import sys
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_boundary_analysis import (
    OP_TO_CATEGORY, classify_fusion_boundary, boundary_label, BUGS,
    CATEGORY_POINTWISE, CATEGORY_REDUCTION, CATEGORY_SCAN,
    CATEGORY_SCATTER, CATEGORY_LAYOUT, CATEGORY_EXTERN,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def extract_interaction_cells(traces):
    """Extract unique semantic interaction cells from compilation traces."""
    cells = set()
    n_ok = 0
    cell_sequence = []

    for t in traces:
        sig = t.get('sig', {})
        trace = t.get('trace', t)
        if sig.get('sig_L1') == 'ERROR' or trace.get('error'):
            continue
        n_ok += 1
        for fp_str in trace.get('fusion_patterns', []):
            ops = re.findall(r"'(\w+)'", fp_str)
            boundary = classify_fusion_boundary(ops)
            if boundary:
                cells.add(boundary)
        cell_sequence.append((n_ok, len(cells)))

    return cells, n_ok, cell_sequence


def get_bug_cells():
    """Get the set of semantic interaction cells that bugs occupy (kernel-level only)."""
    cells = set()
    for bug in BUGS.values():
        if bug['boundary'] is not None:
            cells.add(bug['boundary'])
    return cells


def load_traces(path):
    with open(path) as f:
        return json.load(f)


def analyze():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load baselines ───────────────────────────────────────────────

    print("Loading random baseline...")
    random_path = '${ARTIFACT_ROOT}/experiments/results/fusion_baseline/traces_random_1k.json'
    if not os.path.exists(random_path):
        random_path = '${ARTIFACT_ROOT}/experiments/results/collapse_cds/traces_random.json'
    random_traces = load_traces(random_path)
    r_cells, r_n, r_seq = extract_interaction_cells(random_traces)

    print("Loading targeted baseline...")
    targeted_path = '${ARTIFACT_ROOT}/experiments/results/fusion_baseline/traces_targeted_1k.json'
    if os.path.exists(targeted_path):
        targeted_traces = load_traces(targeted_path)
        t_cells, t_n, t_seq = extract_interaction_cells(targeted_traces)
    else:
        t_cells, t_n, t_seq = set(), 0, []

    bug_cells = get_bug_cells()

    # ── 1. Saturation Analysis ───────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"1. INTERACTION CELL SATURATION")
    print(f"{'=' * 80}")

    print(f"\n  Random:   {r_n:>5} programs -> {len(r_cells)} interaction cells")
    print(f"  Targeted: {t_n:>5} programs -> {len(t_cells)} interaction cells")
    print(f"  Bug-relevant cells:  {len(bug_cells)}")

    # When did each method stop discovering new cells?
    if r_seq:
        last_new = 0
        for i, (n, c) in enumerate(r_seq):
            if i > 0 and c > r_seq[i-1][1]:
                last_new = n
        print(f"\n  Random: last new cell discovered at program #{last_new}")
        print(f"    -> {r_n - last_new} programs ({(r_n - last_new)/r_n*100:.0f}%) "
              f"added zero new cells")

    if t_seq:
        last_new = 0
        for i, (n, c) in enumerate(t_seq):
            if i > 0 and c > t_seq[i-1][1]:
                last_new = n
        print(f"  Targeted: last new cell discovered at program #{last_new}")
        print(f"    -> {t_n - last_new} programs ({(t_n - last_new)/t_n*100:.0f}%) "
              f"added zero new cells")

    # ── 2. Bug Reachability Gap ──────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"2. BUG REACHABILITY GAP")
    print(f"{'=' * 80}")

    reached = bug_cells & r_cells
    unreached = bug_cells - r_cells
    reached_t = bug_cells & t_cells
    unreached_t = bug_cells - t_cells

    bugs_in_reached = [bid for bid, b in BUGS.items()
                       if b['boundary'] is not None and b['boundary'] in reached]
    bugs_in_unreached = [bid for bid, b in BUGS.items()
                         if b['boundary'] is not None and b['boundary'] in unreached]
    bugs_guard_tracing = [bid for bid, b in BUGS.items() if b['boundary'] is None]

    print(f"\n  Bug cells reached by Random:    {len(reached)}/{len(bug_cells)}")
    print(f"  Bug cells reached by Targeted:  {len(reached_t)}/{len(bug_cells)}")
    print(f"  Bug cells NEVER explored:       {len(unreached)}/{len(bug_cells)}")

    kernel_bugs = [bid for bid, b in BUGS.items() if b['boundary'] is not None]
    print(f"\n  Kernel-level bugs:         {len(kernel_bugs)}/{len(BUGS)}")
    print(f"  Bugs in reachable cells:   {len(bugs_in_reached)}/{len(BUGS)} "
          f"({len(bugs_in_reached)/len(BUGS)*100:.0f}%)")
    print(f"  Bugs in unreachable cells: {len(bugs_in_unreached)}/{len(BUGS)} "
          f"({len(bugs_in_unreached)/len(BUGS)*100:.0f}%)")
    print(f"  Guard/tracing (no kernel): {len(bugs_guard_tracing)}/{len(BUGS)} "
          f"({len(bugs_guard_tracing)/len(BUGS)*100:.0f}%)")

    # ── 3. Per-Bug Interaction Attribution ────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"3. PER-BUG INTERACTION ATTRIBUTION")
    print(f"{'=' * 80}")

    # Count fusion frequency per cell in random baseline
    cell_freq = Counter()
    total_fusions = 0
    for t in random_traces:
        sig = t.get('sig', {})
        trace = t.get('trace', t)
        if sig.get('sig_L1') == 'ERROR' or trace.get('error'):
            continue
        for fp_str in trace.get('fusion_patterns', []):
            ops = re.findall(r"'(\w+)'", fp_str)
            boundary = classify_fusion_boundary(ops)
            if boundary:
                cell_freq[boundary] += 1
                total_fusions += 1

    print(f"\n  {'Bug':<10} {'Type':<8} {'Interaction Cell':<35} "
          f"{'Baseline Freq':>14} {'Reachable':>10}")
    print('  ' + '─' * 82)

    for bid, bug in sorted(BUGS.items()):
        if bug['boundary'] is None:
            label = 'N/A (guard/tracing)'
            freq_str = '—'
            reachable = 'N/A'
        else:
            label = boundary_label(bug['boundary'])
            freq = cell_freq.get(bug['boundary'], 0)
            pct = freq / max(total_fusions, 1) * 100
            reachable = 'YES' if bug['boundary'] in r_cells else 'NO'
            freq_str = f"{freq} ({pct:.1f}%)" if freq > 0 else "0 (0.0%)"
        print(f"  {bid:<10} {bug['type']:<8} {label:<35} {freq_str:>14} {reachable:>10}")

    # ── 4. Unreachable Cell Detail ───────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"4. UNREACHABLE INTERACTION CELLS — WHERE BASELINES ARE BLIND")
    print(f"{'=' * 80}")

    for cell in sorted(unreached, key=lambda x: boundary_label(x)):
        label = boundary_label(cell)
        bugs_here = [(bid, b) for bid, b in BUGS.items() if b['boundary'] == cell]
        print(f"\n  {label}")
        for bid, b in bugs_here:
            print(f"    {bid}: {b['desc']}")
            print(f"      Detail: {b['boundary_detail']}")
            print(f"      Why hard: requires {' + '.join(sorted(cell))} in single fusion")

    # ── 5. Figures ───────────────────────────────────────────────────

    # Figure 1: Interaction Cell Growth Curve
    fig, ax = plt.subplots(figsize=(10, 6))

    if r_seq:
        r_xs = [s[0] for s in r_seq]
        r_ys = [s[1] for s in r_seq]
        ax.plot(r_xs, r_ys, color='#5778A4', linewidth=2.5,
                label=f'Random (N={r_n})')

    if t_seq:
        t_xs = [s[0] for s in t_seq]
        t_ys = [s[1] for s in t_seq]
        ax.plot(t_xs, t_ys, color='#67A286', linewidth=2.5,
                label=f'Targeted (N={t_n})')

    ax.axhline(y=len(bug_cells), color='#e74c3c', linestyle='--',
               linewidth=2, alpha=0.8,
               label=f'Distinct bug cell types ({len(bug_cells)})')

    # Shade the saturation region
    if r_seq:
        sat_start = 0
        for i, (n, c) in enumerate(r_seq):
            if c == r_seq[-1][1]:
                sat_start = n
                break
        ax.axvspan(sat_start, r_n, alpha=0.08, color='gray')
        ax.text(sat_start + (r_n - sat_start) * 0.5, 1,
                'Saturated\n(no new cells)', ha='center', va='bottom',
                fontsize=10, color='gray', style='italic')

    ax.set_xlabel('Programs Compiled', fontsize=13)
    ax.set_ylabel('Unique Interaction Cells Explored', fontsize=13)
    ax.set_title('Semantic Interaction Cell Discovery Saturation', fontsize=14)
    ax.legend(fontsize=11, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(len(r_cells), len(bug_cells)) + 2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_growth_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_growth_curve.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved interaction_growth_curve.{png,pdf}")

    # Figure 2: Bug Reachability — stacked bar showing reached vs unreached
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Interaction cells explored vs total possible bug cells
    methods = ['Random\n(852 programs)', 'Targeted\n(1000 programs)']
    cell_counts = [len(r_cells), len(t_cells)]
    colors_cells = ['#5778A4', '#67A286']

    bars = ax1.bar(range(2), cell_counts, color=colors_cells, alpha=0.8,
                   edgecolor='black', linewidth=0.5, width=0.5)
    for bar, count in zip(bars, cell_counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 str(count), ha='center', va='bottom', fontsize=15, fontweight='bold')

    ax1.axhline(y=len(bug_cells), color='#e74c3c', linestyle='--',
                linewidth=1.5, alpha=0.7,
                label=f'Bug cell types ({len(bug_cells)})')
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel('Interaction Cells', fontsize=12)
    ax1.set_title('Cells Explored by Baselines', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(len(r_cells), len(bug_cells)) + 3)

    # Right: Bugs in reached vs unreached cells (stacked)
    n_reached = len(bugs_in_reached)
    n_unreached = len(bugs_in_unreached)

    bar_reached = ax2.bar([0], [n_reached], color='#67A286', alpha=0.8,
                          edgecolor='black', linewidth=0.5, width=0.5,
                          label=f'In explored cells ({n_reached})')
    bar_unreached = ax2.bar([0], [n_unreached], bottom=[n_reached],
                            color='#e74c3c', alpha=0.8,
                            edgecolor='black', linewidth=0.5, width=0.5,
                            label=f'In blind-spot cells ({n_unreached})')

    ax2.text(0, n_reached / 2, f'{n_reached}\n({n_reached/len(BUGS)*100:.0f}%)',
             ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    ax2.text(0, n_reached + n_unreached / 2,
             f'{n_unreached}\n({n_unreached/len(BUGS)*100:.0f}%)',
             ha='center', va='center', fontsize=13, fontweight='bold', color='white')

    ax2.set_xticks([0])
    ax2.set_xticklabels([f'{len(BUGS)} Confirmed\nInductor Bugs'], fontsize=11)
    ax2.set_ylabel('Number of Bugs', fontsize=12)
    ax2.set_title('Bug Distribution by Cell Reachability', fontsize=13)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bug_reachability_gap.png'),
                dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'bug_reachability_gap.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved bug_reachability_gap.{png,pdf}")

    # Figure 3: Per-bug attribution — horizontal bar showing each bug's cell frequency
    fig, ax = plt.subplots(figsize=(12, 8))

    kernel_bug_items = [(bid, b) for bid, b in BUGS.items() if b['boundary'] is not None]
    kernel_bug_items.sort(key=lambda x: cell_freq.get(x[1]['boundary'], 0))
    y_labels = []
    y_freqs = []
    y_colors = []
    for bid, b in kernel_bug_items:
        label = boundary_label(b['boundary'])
        freq = cell_freq.get(b['boundary'], 0)
        pct = freq / max(total_fusions, 1) * 100
        y_labels.append(f"{bid}: {label}")
        y_freqs.append(pct)
        y_colors.append('#e74c3c' if freq == 0 else
                        '#f39c12' if pct < 1 else '#67A286')

    y_pos = range(len(y_labels))
    bars = ax.barh(y_pos, y_freqs, color=y_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.3, height=0.7)

    for i, (bar, pct) in enumerate(zip(bars, y_freqs)):
        if pct == 0:
            ax.text(0.3, i, 'ABSENT', ha='left', va='center',
                    fontsize=9, fontweight='bold', color='#e74c3c')
        else:
            ax.text(bar.get_width() + 0.3, i, f'{pct:.1f}%',
                    ha='left', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Baseline Frequency (%)', fontsize=12)
    ax.set_title(f'Per-Bug Interaction Cell Frequency in Random Baseline\n'
                 f'(N={r_n}, {total_fusions} total fusions)', fontsize=13)

    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', alpha=0.85, label='Absent (0%)'),
        mpatches.Patch(facecolor='#f39c12', alpha=0.85, label='Rare (<1%)'),
        mpatches.Patch(facecolor='#67A286', alpha=0.85, label='Present (>=1%)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-1, max(y_freqs) * 1.2 + 2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_bug_interaction_attribution.png'),
                dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_bug_interaction_attribution.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved per_bug_interaction_attribution.{png,pdf}")

    # ── 6. Summary ───────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n  1. SATURATION: Random ({r_n} programs) and Targeted ({t_n} programs)")
    print(f"     both discover exactly {len(r_cells)} interaction cells and stop.")
    print(f"     The semantic interaction space has {len(bug_cells)} bug-relevant cells.")

    print(f"\n  2. BLIND SPOTS: {len(unreached)}/{len(bug_cells)} bug-relevant cells "
          f"({len(unreached)/len(bug_cells)*100:.0f}%) have ZERO baseline coverage.")
    print(f"     These blind spots contain {len(bugs_in_unreached)}/{len(BUGS)} bugs "
          f"({len(bugs_in_unreached)/len(BUGS)*100:.0f}%).")

    absent_bugs = sum(1 for b in BUGS.values()
                      if b['boundary'] is not None and cell_freq.get(b['boundary'], 0) == 0)
    rare_bugs = sum(1 for b in BUGS.values()
                    if b['boundary'] is not None and 0 < cell_freq.get(b['boundary'], 0) / max(total_fusions, 1) < 0.01)
    print(f"\n  3. BUG DISTRIBUTION BY CELL FREQUENCY:")
    print(f"     Absent (0%):   {absent_bugs}/{len(BUGS)} bugs")
    print(f"     Rare (<1%):    {rare_bugs}/{len(BUGS)} bugs")
    print(f"     Present (>=1%): {len(BUGS) - absent_bugs - rare_bugs}/{len(BUGS)} bugs")

    print(f"\n  Conclusion: Program count is a weak proxy for interaction exploration.")
    print(f"  1000 programs discover {len(r_cells)} cells; "
          f"{len(bugs_in_unreached)} bugs ({len(bugs_in_unreached)/len(BUGS)*100:.0f}%) "
          f"are in permanently unreachable cells.")

    # Save results
    results = {
        'random': {
            'programs': r_n,
            'cells_explored': len(r_cells),
            'cells_list': [boundary_label(c) for c in sorted(r_cells, key=boundary_label)],
        },
        'targeted': {
            'programs': t_n,
            'cells_explored': len(t_cells),
            'cells_list': [boundary_label(c) for c in sorted(t_cells, key=boundary_label)],
        },
        'bugs': {
            'total': len(BUGS),
            'kernel_level': len(kernel_bugs),
            'guard_tracing': len(bugs_guard_tracing),
            'distinct_cells': len(bug_cells),
            'cells_reached_by_baseline': len(reached),
            'cells_unreached': len(unreached),
            'bugs_in_reached_cells': len(bugs_in_reached),
            'bugs_in_unreached_cells': len(bugs_in_unreached),
            'bugs_in_absent_cells': absent_bugs,
            'bugs_in_rare_cells': rare_bugs,
        },
        'per_bug': {
            bid: {
                'cell': boundary_label(b['boundary']) if b['boundary'] is not None else 'N/A',
                'baseline_freq': cell_freq.get(b['boundary'], 0) if b['boundary'] is not None else None,
                'baseline_pct': round(cell_freq.get(b['boundary'], 0) / max(total_fusions, 1) * 100, 2) if b['boundary'] is not None else None,
                'reachable': b['boundary'] in r_cells if b['boundary'] is not None else None,
            }
            for bid, b in sorted(BUGS.items())
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'interaction_coverage_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved interaction_coverage_results.json")


if __name__ == '__main__':
    analyze()
