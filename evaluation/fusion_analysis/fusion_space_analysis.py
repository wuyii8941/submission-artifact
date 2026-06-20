#!/usr/bin/env python3
"""
Fusion Space Analysis — the core experiment for the paper.

Produces:
  1. Fusion op frequency distribution (random baseline)
  2. Bug-op vs baseline frequency table
  3. Rare-op reachability comparison (random vs KernelFuzz corpus)
  4. Fusion distribution shift visualization
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Bug metadata
BUGS = {
    'bug_001': {'desc': 'ReflectionPad + view wrong output', 'ops': ['reflection_pad3d', 'view', 'split'], 'type': 'silent'},
    'bug_004': {'desc': 'bf16 cast elided in fusion', 'ops': ['to_bf16', 'sigmoid', 'mul', 'matmul'], 'type': 'silent'},
    'bug_005': {'desc': 'fp64 persistent_reduction wrong', 'ops': ['softmax'], 'type': 'silent'},
    'bug_006': {'desc': 'adaptive_avg_pool + flatten + sum fusion wrong', 'ops': ['adaptive_avg_pool2d', 'flatten', 'sum'], 'type': 'silent'},
    'bug_007': {'desc': 'slice_scatter backward zeroes grad', 'ops': ['slice_scatter'], 'type': 'silent'},
    'bug_011': {'desc': 'cumsum SplitScan crash', 'ops': ['cumsum', 'sigmoid', 'matmul'], 'type': 'crash'},
    'bug_012': {'desc': 'nan_to_num + chunk + cat wrong', 'ops': ['nan_to_num', 'chunk', 'cat'], 'type': 'silent'},
    'bug_015': {'desc': 'associative_scan dynamic crash', 'ops': ['associative_scan'], 'type': 'crash'},
    'bug_019': {'desc': 'create_graph grad metadata dropped', 'ops': ['autograd_grad'], 'type': 'silent'},
    'bug_020': {'desc': 'inference_mode batched matmul crash', 'ops': ['bmm', 'inference_mode'], 'type': 'crash'},
    'bug_021': {'desc': 'complex abs/angle backward crash', 'ops': ['abs', 'angle', 'fft'], 'type': 'crash'},
    'bug_022': {'desc': 'diagonal_scatter backward zeroes grad', 'ops': ['diagonal_scatter'], 'type': 'silent'},
    'bug_025': {'desc': 'select_scatter backward zeroes grad', 'ops': ['select_scatter'], 'type': 'silent'},
    'bug_026': {'desc': 'index_copy/fill backward crash', 'ops': ['index_copy', 'index_fill'], 'type': 'crash'},
    'bug_027': {'desc': 'Embedding sparse backward crash', 'ops': ['embedding'], 'type': 'crash'},
    'bug_028': {'desc': 'SDPA non-pow2 head_dim crash', 'ops': ['sdpa'], 'type': 'crash'},
    'bug_030': {'desc': 'while_loop backward stride crash', 'ops': ['while_loop'], 'type': 'crash'},
}

# Map bug ops to fusion-level op names (as they appear in kernel names)
BUG_OP_TO_FUSION = {
    'reflection_pad3d': 'reflection',
    'to_bf16': None,  # type cast doesn't appear in fusion names
    'sigmoid': 'sigmoid',
    'mul': 'mul',
    'matmul': None,  # extern kernel
    'softmax': 'softmax',
    'adaptive_avg_pool2d': 'adaptive',
    'flatten': None,
    'sum': 'sum',
    'slice_scatter': 'scatter',
    'cumsum': 'cumsum',
    'nan_to_num': 'nan',
    'chunk': 'chunk',
    'cat': 'cat',
    'associative_scan': 'scan',
    'autograd_grad': None,
    'bmm': None,  # extern kernel
    'inference_mode': None,
    'abs': 'abs',
    'angle': 'angle',
    'fft': None,
    'diagonal_scatter': 'diagonal',
    'select_scatter': 'select',
    'index_copy': 'index',
    'index_fill': 'index',
    'embedding': 'embedding',
    'sdpa': None,
    'while_loop': None,
}


def load_random_baseline():
    """Load 1k random baseline."""
    path = '${ARTIFACT_ROOT}/experiments/results/fusion_baseline/traces_random_1k.json'
    if not os.path.exists(path):
        path = '${ARTIFACT_ROOT}/experiments/results/collapse_cds/traces_random.json'
        print(f"  Using fallback: {path}")
    with open(path) as f:
        return json.load(f)


def extract_op_frequency(traces):
    """Count fusion op frequencies across all compiled programs."""
    op_counter = Counter()
    program_has_op = defaultdict(int)
    n_ok = 0

    for t in traces:
        sig = t.get('sig', {})
        trace = t.get('trace', t)
        if sig.get('sig_L1') == 'ERROR' or trace.get('error'):
            continue
        n_ok += 1

        fp_list = trace.get('fusion_patterns', [])
        program_ops = set()
        for fp_str in fp_list:
            ops = re.findall(r"'(\w+)'", fp_str)
            for op in ops:
                op_counter[op] += 1
                program_ops.add(op)

        for op in program_ops:
            program_has_op[op] += 1

    return op_counter, program_has_op, n_ok


def load_kernelfuzz_corpus(union_all=True):
    """Load KernelFuzz corpus programs. If union_all, merge all campaigns."""
    if union_all:
        seen_names = set()
        entries = []
        for i in range(1, 10):
            path = f'${ARTIFACT_ROOT}/kernelfuzz_campaign_{i:02d}/graph-oracle/corpus_checkpoint.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for e in data.get('entries', []):
                    if e['name'] not in seen_names:
                        seen_names.add(e['name'])
                        entries.append(e)
        return entries
    else:
        corpus_path = '${ARTIFACT_ROOT}/kernelfuzz_campaign_09/graph-oracle/corpus_checkpoint.json'
        with open(corpus_path) as f:
            data = json.load(f)
        return data.get('entries', [])


def compute_saturation_curve(traces):
    """Compute unique-op count as a function of programs seen."""
    seen_ops = set()
    curve = []
    n = 0
    for t in traces:
        sig = t.get('sig', {})
        trace = t.get('trace', t)
        if sig.get('sig_L1') == 'ERROR' or trace.get('error'):
            continue
        n += 1
        for fp_str in trace.get('fusion_patterns', []):
            ops = re.findall(r"'(\w+)'", fp_str)
            seen_ops.update(ops)
        curve.append((n, len(seen_ops)))
    return curve


def analyze():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading random baseline...")
    random_traces = load_random_baseline()
    op_freq, program_has_op, n_random = extract_op_frequency(random_traces)
    print(f"  {n_random} compiled programs, {len(op_freq)} unique ops in fusions")

    # ── 0. Saturation curve ──────────────────────────────────────────

    random_curve = compute_saturation_curve(random_traces)

    fig, ax = plt.subplots(figsize=(10, 5))
    xs, ys = zip(*random_curve)
    ax.plot(xs, ys, color='#3498db', linewidth=2, label='Random')

    # Load targeted if available
    targeted_path = '${ARTIFACT_ROOT}/experiments/results/fusion_baseline/traces_targeted_1k.json'
    if os.path.exists(targeted_path):
        with open(targeted_path) as f:
            targeted_traces = json.load(f)
        targeted_curve = compute_saturation_curve(targeted_traces)
        txs, tys = zip(*targeted_curve)
        ax.plot(txs, tys, color='#27ae60', linewidth=2, label='Targeted')

    ax.set_xlabel('Programs Compiled', fontsize=12)
    ax.set_ylabel('Unique Fusion Ops Discovered', fontsize=12)
    ax.set_title('Fusion Op Discovery Saturation', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'op_saturation_curve.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'op_saturation_curve.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved op_saturation_curve.{png,pdf}")

    # ── 1. Op frequency distribution ──────────────────────────────────

    fig, ax = plt.subplots(figsize=(14, 6))
    ops_sorted = op_freq.most_common()
    names = [o[0] for o in ops_sorted]
    counts = [o[1] for o in ops_sorted]

    # Color bars: red for bug-relevant, blue for common
    bug_fusion_ops = set()
    for bug in BUGS.values():
        for op in bug['ops']:
            fop = BUG_OP_TO_FUSION.get(op)
            if fop:
                bug_fusion_ops.add(fop)

    colors = ['#e74c3c' if n in bug_fusion_ops else '#3498db' for n in names]
    bars = ax.bar(range(len(names)), counts, color=colors, alpha=0.8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, ha='right', fontsize=8)
    ax.set_ylabel('Frequency in Random Baseline', fontsize=12)
    ax.set_title(f'Fusion Op Frequency Distribution (N={n_random} random programs)\n'
                 f'Red = bug-relevant ops, Blue = non-bug ops', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='Bug-relevant ops'),
                       Patch(facecolor='#3498db', alpha=0.8, label='Non-bug ops')]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fusion_op_frequency.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fusion_op_frequency.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fusion_op_frequency.{png,pdf}")

    # ── 2. Bug-op vs baseline frequency table ─────────────────────────

    print(f"\n{'=' * 80}")
    print(f"BUG-OP vs BASELINE FREQUENCY")
    print(f"{'=' * 80}")

    total_fusions = sum(op_freq.values())
    bug_table = []
    for bug_id, bug in sorted(BUGS.items()):
        for op in bug['ops']:
            fusion_op = BUG_OP_TO_FUSION.get(op)
            if fusion_op:
                freq = op_freq.get(fusion_op, 0)
                prog_pct = program_has_op.get(fusion_op, 0) / max(n_random, 1) * 100
            else:
                freq = 0
                prog_pct = 0.0
            bug_table.append({
                'bug_id': bug_id,
                'op': op,
                'fusion_op': fusion_op or '(extern/context)',
                'freq': freq,
                'freq_pct': freq / max(total_fusions, 1) * 100,
                'prog_pct': prog_pct,
                'type': bug['type'],
            })

    header = f"{'Bug':<10} {'Op':<22} {'Fusion Op':<15} {'Freq':>6} {'%Fusions':>9} {'%Programs':>10} {'Type':<8}"
    print(header)
    print('─' * len(header))

    n_rare = 0
    n_absent = 0
    for row in bug_table:
        rarity = ''
        if row['freq'] == 0:
            rarity = ' *** ABSENT'
            n_absent += 1
            n_rare += 1
        elif row['prog_pct'] < 5:
            rarity = ' ** RARE'
            n_rare += 1

        print(f"{row['bug_id']:<10} {row['op']:<22} {row['fusion_op']:<15} "
              f"{row['freq']:>6} {row['freq_pct']:>8.1f}% {row['prog_pct']:>9.1f}%"
              f" {row['type']:<8}{rarity}")

    print(f"\nTotal bug-ops: {len(bug_table)}")
    print(f"Absent from baseline: {n_absent}/{len(bug_table)} ({n_absent/len(bug_table)*100:.0f}%)")
    print(f"Rare or absent: {n_rare}/{len(bug_table)} ({n_rare/len(bug_table)*100:.0f}%)")

    # ── 3. Bug density by op rarity ───────────────────────────────────

    # Classify each bug by the rarity of its FUSION ops (ignore extern/context)
    # Use program-coverage percentage thresholds (scale-independent)
    rare_threshold = 0.02   # < 2% of programs
    uncommon_threshold = 0.10  # < 10% of programs

    bug_per_rarity = Counter()
    bug_classifications = {}
    for bug_id, bug in sorted(BUGS.items()):
        fusion_ops_for_bug = []
        has_any_fusion_op = False
        for op in bug['ops']:
            fop = BUG_OP_TO_FUSION.get(op)
            if fop:
                has_any_fusion_op = True
                fusion_ops_for_bug.append((fop, program_has_op.get(fop, 0)))

        if not has_any_fusion_op:
            bug_per_rarity['extern_only'] += 1
            bug_classifications[bug_id] = 'extern_only'
            continue

        # Use the rarest fusion op to classify this bug
        min_prog_pct = min(cnt / max(n_random, 1) for _, cnt in fusion_ops_for_bug)
        if min_prog_pct == 0:
            bug_per_rarity['absent'] += 1
            bug_classifications[bug_id] = 'absent'
        elif min_prog_pct < rare_threshold:
            bug_per_rarity['rare'] += 1
            bug_classifications[bug_id] = 'rare'
        elif min_prog_pct < uncommon_threshold:
            bug_per_rarity['uncommon'] += 1
            bug_classifications[bug_id] = 'uncommon'
        else:
            bug_per_rarity['common'] += 1
            bug_classifications[bug_id] = 'common'

    print(f"\n{'=' * 80}")
    print(f"BUG DENSITY BY FUSION OP RARITY")
    print(f"{'=' * 80}")
    for rarity in ['absent', 'rare', 'uncommon', 'common', 'extern_only']:
        n_bugs = bug_per_rarity.get(rarity, 0)
        label = {
            'absent': f'absent    (0% of programs)',
            'rare': f'rare      (<{rare_threshold*100:.0f}% of programs)',
            'uncommon': f'uncommon  (<{uncommon_threshold*100:.0f}% of programs)',
            'common': f'common    (>={uncommon_threshold*100:.0f}% of programs)',
            'extern_only': f'extern    (no fusion ops, e.g. matmul/sdpa)',
        }[rarity]
        print(f"  {label}: {n_bugs} bugs")
    n_sparse = bug_per_rarity.get('absent', 0) + bug_per_rarity.get('rare', 0)
    n_with_fusion = sum(v for k, v in bug_per_rarity.items() if k != 'extern_only')
    print(f"\n  Sparse (absent+rare) fusion bugs: {n_sparse}/{n_with_fusion} "
          f"({n_sparse/max(n_with_fusion,1)*100:.0f}% of bugs with fusion ops)")

    # ── 4. Rare-op reachability: KernelFuzz corpus vs random ──────────

    print(f"\n{'=' * 80}")
    print(f"RARE-OP REACHABILITY: KernelFuzz CORPUS vs RANDOM")
    print(f"{'=' * 80}")

    kf_corpus = load_kernelfuzz_corpus()
    kf_names = [e['name'].lower() for e in kf_corpus]

    rare_op_keywords = {
        'scatter': ['scatter'],
        'diagonal': ['diagonal'],
        'scan/cummax': ['scan', 'cummax', 'cummin', 'associative'],
        'index_ops': ['index_select', 'index_copy', 'gather', 'masked_scatter'],
        'embedding': ['embedding'],
        'attention/sdpa': ['attention', 'sdpa', 'flex_attention'],
        'as_strided': ['as_strided', 'strided'],
        'einsum': ['einsum'],
        'while_loop': ['while_loop', 'while'],
    }

    header = f"{'Rare Op Group':<20} {'Random (N={n_random})':>18} {'KF Corpus (N={len(kf_corpus)})':>20}"
    print(header)
    print('─' * len(header))

    for group, keywords in rare_op_keywords.items():
        # Random: count programs with this op in their fusions
        random_count = 0
        for t in random_traces:
            sig = t.get('sig', {})
            trace = t.get('trace', t)
            if sig.get('sig_L1') == 'ERROR' or trace.get('error'):
                continue
            fusions_str = ' '.join(trace.get('fusion_patterns', []))
            if any(kw in fusions_str.lower() for kw in keywords):
                random_count += 1

        # KernelFuzz: count corpus entries with this keyword
        kf_count = sum(1 for n in kf_names if any(kw in n for kw in keywords))

        r_pct = random_count / max(n_random, 1) * 100
        k_pct = kf_count / max(len(kf_corpus), 1) * 100

        marker = ''
        if random_count == 0 and kf_count > 0:
            marker = ' <<<'

        print(f"  {group:<20} {random_count:>5} ({r_pct:>5.1f}%)       {kf_count:>5} ({k_pct:>5.1f}%){marker}")

    # ── 5. Distribution shift visualization ───────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Random: op frequency distribution (log scale)
    ops_sorted = op_freq.most_common(30)
    names = [o[0] for o in ops_sorted]
    counts = [o[1] for o in ops_sorted]
    colors = ['#e74c3c' if n in bug_fusion_ops else '#3498db' for n in names]

    ax1.barh(range(len(names)), counts, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Frequency', fontsize=11)
    ax1.set_title(f'Random Baseline: Top 30 Fusion Ops\n(N={n_random})', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Bug ops that are absent from random
    absent_bug_ops = []
    for bug in BUGS.values():
        for op in bug['ops']:
            fop = BUG_OP_TO_FUSION.get(op)
            if fop and op_freq.get(fop, 0) == 0:
                absent_bug_ops.append(fop)
    absent_bug_ops = sorted(set(absent_bug_ops))

    if absent_bug_ops:
        y_pos = range(len(absent_bug_ops))
        ax2.barh(y_pos, [0] * len(absent_bug_ops), color='#e74c3c', alpha=0.3)
        for i, op in enumerate(absent_bug_ops):
            ax2.text(0.5, i, f'{op} (0 occurrences in {n_random} programs)',
                     va='center', fontsize=10, color='#e74c3c', fontweight='bold')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(absent_bug_ops, fontsize=10, color='#e74c3c')
        ax2.set_xlabel('Frequency', fontsize=11)
        ax2.set_title('Bug-Relevant Ops ABSENT from Random Baseline', fontsize=12)
        ax2.invert_yaxis()
        ax2.set_xlim(0, max(counts) * 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fusion_distribution_shift.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fusion_distribution_shift.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fusion_distribution_shift.{png,pdf}")

    # ── 6. Bug rarity summary figure ──────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5))
    rarity_order = ['absent', 'rare', 'uncommon', 'common', 'extern_only']
    rarity_labels = ['Absent\n(0%)', f'Rare\n(<{rare_threshold*100:.0f}%)',
                     f'Uncommon\n(<{uncommon_threshold*100:.0f}%)',
                     f'Common\n(≥{uncommon_threshold*100:.0f}%)',
                     'Extern-only\n(no fusion ops)']
    rarity_colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#95a5a6']
    rarity_counts = [bug_per_rarity.get(r, 0) for r in rarity_order]

    bars = ax.bar(range(len(rarity_order)), rarity_counts,
                  color=rarity_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, rarity_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(rarity_order)))
    ax.set_xticklabels(rarity_labels, fontsize=10)
    ax.set_ylabel('Number of Confirmed Bugs', fontsize=12)
    ax.set_title(f'Bug Distribution by Fusion Op Rarity\n'
                 f'({len(BUGS)} bugs, baseline = {n_random} random programs)',
                 fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bug_rarity_distribution.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'bug_rarity_distribution.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved bug_rarity_distribution.{png,pdf}")

    # ── 7. Targeted baseline comparison (if available) ──────────────

    targeted_path = '${ARTIFACT_ROOT}/experiments/results/fusion_baseline/traces_targeted_1k.json'
    if os.path.exists(targeted_path):
        print(f"\n{'=' * 80}")
        print(f"TARGETED vs RANDOM FUSION DISTRIBUTION")
        print(f"{'=' * 80}")

        with open(targeted_path) as f:
            targeted_traces = json.load(f)
        t_op_freq, t_prog_has_op, n_targeted = extract_op_frequency(targeted_traces)
        print(f"  Targeted: {n_targeted} compiled programs, {len(t_op_freq)} unique ops")

        # Compare bug-relevant op coverage
        print(f"\n  {'Bug Fusion Op':<18} {'Random %':>10} {'Targeted %':>12}")
        print(f"  {'─'*18} {'─'*10} {'─'*12}")
        for fop in sorted(bug_fusion_ops):
            r_pct = program_has_op.get(fop, 0) / max(n_random, 1) * 100
            t_pct = t_prog_has_op.get(fop, 0) / max(n_targeted, 1) * 100
            marker = ' ✗' if r_pct == 0 and t_pct == 0 else ''
            print(f"  {fop:<18} {r_pct:>9.1f}% {t_pct:>11.1f}%{marker}")

        # Distribution shift figure: random vs targeted
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        top_n = 30
        r_sorted = op_freq.most_common(top_n)
        r_names = [o[0] for o in r_sorted]
        r_counts = [o[1] for o in r_sorted]
        r_colors = ['#e74c3c' if n in bug_fusion_ops else '#3498db' for n in r_names]

        ax1.barh(range(len(r_names)), r_counts, color=r_colors, alpha=0.8)
        ax1.set_yticks(range(len(r_names)))
        ax1.set_yticklabels(r_names, fontsize=9)
        ax1.set_xlabel('Frequency', fontsize=11)
        ax1.set_title(f'Random: Top {top_n} Fusion Ops (N={n_random})', fontsize=12)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        t_sorted = t_op_freq.most_common(top_n)
        t_names = [o[0] for o in t_sorted]
        t_counts = [o[1] for o in t_sorted]
        t_colors = ['#e74c3c' if n in bug_fusion_ops else '#27ae60' for n in t_names]

        ax2.barh(range(len(t_names)), t_counts, color=t_colors, alpha=0.8)
        ax2.set_yticks(range(len(t_names)))
        ax2.set_yticklabels(t_names, fontsize=9)
        ax2.set_xlabel('Frequency', fontsize=11)
        ax2.set_title(f'Targeted: Top {top_n} Fusion Ops (N={n_targeted})', fontsize=12)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='Bug-relevant ops')]
        ax1.legend(handles=legend_elements, fontsize=9, loc='lower right')
        ax2.legend(handles=legend_elements, fontsize=9, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'random_vs_targeted_distribution.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, 'random_vs_targeted_distribution.pdf'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved random_vs_targeted_distribution.{png,pdf}")

    # ── Summary ───────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Random baseline: {n_random} programs, {len(op_freq)} unique fusion ops")
    print(f"  Bugs analyzed: {len(BUGS)}")
    n_sparse = bug_per_rarity.get('absent', 0) + bug_per_rarity.get('rare', 0)
    n_with_fusion = sum(v for k, v in bug_per_rarity.items() if k != 'extern_only')
    print(f"  Bugs in sparse fusion regions: {n_sparse}/{n_with_fusion} "
          f"({n_sparse/max(n_with_fusion,1)*100:.0f}%)")
    print(f"  Bug-relevant ops absent from baseline: "
          f"{sum(1 for fop in bug_fusion_ops if op_freq.get(fop, 0) == 0)}/{len(bug_fusion_ops)}")

    # Save all data
    results = {
        'n_random_programs': n_random,
        'n_unique_ops': len(op_freq),
        'op_frequency': dict(op_freq.most_common()),
        'program_has_op': dict(program_has_op),
        'bug_table': bug_table,
        'bug_per_rarity': dict(bug_per_rarity),
        'bug_classifications': bug_classifications,
    }
    with open(os.path.join(OUTPUT_DIR, 'fusion_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    analyze()
