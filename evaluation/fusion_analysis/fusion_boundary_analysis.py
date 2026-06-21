#!/usr/bin/env python3
"""
Cross-Semantic Fusion Boundary Analysis.

Uses PyTorch Inductor's own IR node type hierarchy as the canonical
operator categorization:

  - Pointwise  (ir.py:1126)  — elementwise, no reduction dimensions
  - Reduction  (ir.py:1276)  — has reduction dimensions
  - Scan       (ir.py:2442)  — sequential scan dependency
  - Scatter    (ir.py:1167)  — non-contiguous writes
  - ExternKernel (ir.py:6243) — opaque external kernels

Fusion kernel prefixes map to these via codegen/triton.py:5514:
  poi → Pointwise, red/per/coo → Reduction, spl → Scan

This script classifies each fusion pattern by which semantic categories
it crosses, then correlates boundary type with bug incidence.
"""

import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# ═══════════════════════════════════════════════════════════════════════
# Inductor IR Category Mapping
#
# Each fusion op name (as it appears in Triton kernel names) is mapped
# to its Inductor IR node type. This mapping is derived from
# torch/_inductor/lowering.py which registers each aten op to an IR type.
# ═══════════════════════════════════════════════════════════════════════

CATEGORY_POINTWISE = 'pointwise'
CATEGORY_REDUCTION = 'reduction'
CATEGORY_SCAN = 'scan'
CATEGORY_SCATTER = 'scatter'
CATEGORY_EXTERN = 'extern'
CATEGORY_LAYOUT = 'layout'  # view/reshape — fused via indexing, not computation

# Ops as they appear in kernel fused names → Inductor IR category
# Source: torch/_inductor/lowering.py register_lowering calls
OP_TO_CATEGORY = {
    # ── Pointwise (ir.Pointwise) ──────────────────────────────────────
    # Activation functions — all elementwise, lowered via pointwise
    'relu': CATEGORY_POINTWISE,
    'sigmoid': CATEGORY_POINTWISE,
    'tanh': CATEGORY_POINTWISE,
    'silu': CATEGORY_POINTWISE,
    'gelu': CATEGORY_POINTWISE,
    'elu': CATEGORY_POINTWISE,
    'leaky': CATEGORY_POINTWISE,  # leaky_relu

    # Unary math — elementwise
    'abs': CATEGORY_POINTWISE,
    'neg': CATEGORY_POINTWISE,
    'exp': CATEGORY_POINTWISE,
    'log': CATEGORY_POINTWISE,
    'log1p': CATEGORY_POINTWISE,
    'sqrt': CATEGORY_POINTWISE,
    'reciprocal': CATEGORY_POINTWISE,
    'sign': CATEGORY_POINTWISE,
    'sin': CATEGORY_POINTWISE,
    'cos': CATEGORY_POINTWISE,
    'erf': CATEGORY_POINTWISE,
    'erfc': CATEGORY_POINTWISE,
    'ceil': CATEGORY_POINTWISE,
    'floor': CATEGORY_POINTWISE,
    'pow': CATEGORY_POINTWISE,

    # Binary elementwise
    'mul': CATEGORY_POINTWISE,
    'add': CATEGORY_POINTWISE,
    'sub': CATEGORY_POINTWISE,
    'where': CATEGORY_POINTWISE,
    'ge': CATEGORY_POINTWISE,
    'le': CATEGORY_POINTWISE,
    'gt': CATEGORY_POINTWISE,
    'lt': CATEGORY_POINTWISE,
    'eq': CATEGORY_POINTWISE,
    'ne': CATEGORY_POINTWISE,
    'clamp': CATEGORY_POINTWISE,
    'minimum': CATEGORY_POINTWISE,
    'maximum': CATEGORY_POINTWISE,

    # Type / misc pointwise
    'to': CATEGORY_POINTWISE,       # dtype cast
    'copy': CATEGORY_POINTWISE,     # copy_
    'fill': CATEGORY_POINTWISE,
    'scalar': CATEGORY_POINTWISE,   # scalar_tensor
    'tensor': CATEGORY_POINTWISE,
    'nan': CATEGORY_POINTWISE,      # nan_to_num
    'unsafe': CATEGORY_POINTWISE,   # unsafe_view (treated as pointwise reindex)

    # ── Reduction (ir.Reduction) ──────────────────────────────────────
    # Standard reductions
    'sum': CATEGORY_REDUCTION,
    'mean': CATEGORY_REDUCTION,
    'prod': CATEGORY_REDUCTION,
    'max': CATEGORY_REDUCTION,
    'min': CATEGORY_REDUCTION,
    'amax': CATEGORY_REDUCTION,
    'amin': CATEGORY_REDUCTION,
    'var': CATEGORY_REDUCTION,
    'std': CATEGORY_REDUCTION,
    'any': CATEGORY_REDUCTION,

    # Norm reductions
    'norm': CATEGORY_REDUCTION,
    'linalg': CATEGORY_REDUCTION,   # linalg.vector_norm
    'vector': CATEGORY_REDUCTION,   # part of linalg.vector_norm
    'layer': CATEGORY_REDUCTION,    # layer_norm
    'native': CATEGORY_REDUCTION,   # native_layer_norm / native_batch_norm

    # Softmax (OnlineSoftmaxReduction, ir.py:2176)
    'softmax': CATEGORY_REDUCTION,
    'online': CATEGORY_REDUCTION,   # online softmax
    'prepare': CATEGORY_REDUCTION,  # softmax prepare

    # Pooling reductions
    'pool2d': CATEGORY_REDUCTION,
    'adaptive': CATEGORY_REDUCTION, # adaptive_avg_pool2d
    'avg': CATEGORY_REDUCTION,      # avg_pool2d
    'indices': CATEGORY_REDUCTION,  # max_pool2d_with_indices
    'with': CATEGORY_REDUCTION,     # part of max_pool2d_with_indices

    # ── Scan (ir.Scan) ────────────────────────────────────────────────
    'cumsum': CATEGORY_SCAN,
    'cumprod': CATEGORY_SCAN,
    'scan': CATEGORY_SCAN,          # associative_scan

    # ── Scatter (ir.Scatter) ──────────────────────────────────────────
    'scatter': CATEGORY_SCATTER,
    'gather': CATEGORY_SCATTER,
    'index': CATEGORY_SCATTER,      # index_put, index_copy, etc.
    'select': CATEGORY_SCATTER,     # select_scatter
    'diagonal': CATEGORY_SCATTER,   # diagonal_scatter

    # ── Layout (view/reshape — fused via reindexing) ──────────────────
    'view': CATEGORY_LAYOUT,
    'reshape': CATEGORY_LAYOUT,
    'permute': CATEGORY_LAYOUT,
    'cat': CATEGORY_LAYOUT,
    'chunk': CATEGORY_LAYOUT,
    'split': CATEGORY_LAYOUT,
    'flatten': CATEGORY_LAYOUT,
    'unsqueeze': CATEGORY_LAYOUT,
    'squeeze': CATEGORY_LAYOUT,
    'reflection': CATEGORY_LAYOUT,  # reflection_pad

    # ── Extern (ir.ExternKernel) ──────────────────────────────────────
    'mm': CATEGORY_EXTERN,          # matmul
    'bmm': CATEGORY_EXTERN,
    'addmm': CATEGORY_EXTERN,
    'convolution': CATEGORY_EXTERN,
    'embedding': CATEGORY_EXTERN,
    'sdpa': CATEGORY_EXTERN,
    'attention': CATEGORY_EXTERN,
    'fft': CATEGORY_EXTERN,
    'angle': CATEGORY_EXTERN,       # complex angle
}


def get_category(op_name):
    """Map a fusion op name to its Inductor IR category."""
    return OP_TO_CATEGORY.get(op_name, 'unknown')


def classify_fusion_boundary(ops):
    """
    Given a set of ops in a fusion, return which semantic categories
    are present and the boundary type.
    """
    categories = set()
    for op in ops:
        cat = get_category(op)
        if cat != 'unknown':
            categories.add(cat)
    return frozenset(categories)


def boundary_label(categories):
    """Human-readable label for a boundary type."""
    if len(categories) == 0:
        return 'empty'
    if len(categories) == 1:
        return f'{list(categories)[0]}-only'
    return ' × '.join(sorted(categories))


# ═══════════════════════════════════════════════════════════════════════
# Bug metadata with explicit semantic boundary classification
# ═══════════════════════════════════════════════════════════════════════

BUGS = {
    # ── Fusion composition (2) ──────────────────────────────────────
    '#anon-1': {
        'desc': 'bf16 cast elided in pointwise fusion',
        'ops': ['to_bf16', 'sigmoid', 'to_fp32', 'mul'],
        'boundary': frozenset({CATEGORY_POINTWISE}),
        'boundary_detail': 'pointwise-only (dtype cast elision in fused kernel)',
        'type': 'silent',
    },
    '#anon-2': {
        'desc': 'adaptive_avg_pool + flatten + sum fusion stride error',
        'ops': ['adaptive_avg_pool2d', 'flatten', 'sum'],
        'boundary': frozenset({CATEGORY_REDUCTION, CATEGORY_LAYOUT}),
        'boundary_detail': 'reduction(pool+sum) × layout(flatten)',
        'type': 'silent',
    },
    # ── Backward codegen (8) ────────────────────────────────────────
    '#anon-3': {
        'desc': 'slice_scatter backward zeroes grad',
        'ops': ['slice_scatter'],
        'boundary': frozenset({CATEGORY_SCATTER}),
        'boundary_detail': 'scatter-only (backward codegen)',
        'type': 'silent',
    },
    '#anon-4': {
        'desc': 'slice_scatter backward variant (includes index 0)',
        'ops': ['slice_scatter'],
        'boundary': frozenset({CATEGORY_SCATTER}),
        'boundary_detail': 'scatter-only (backward codegen)',
        'type': 'silent',
    },
    '#anon-5': {
        'desc': 'slice_scatter backward variant (excludes index 0)',
        'ops': ['slice_scatter'],
        'boundary': frozenset({CATEGORY_SCATTER}),
        'boundary_detail': 'scatter-only (backward codegen)',
        'type': 'silent',
    },
    '#anon-6': {
        'desc': 'diagonal_scatter backward zeroes grad',
        'ops': ['diagonal_scatter'],
        'boundary': frozenset({CATEGORY_SCATTER}),
        'boundary_detail': 'scatter-only (backward codegen)',
        'type': 'silent',
    },
    '#anon-7': {
        'desc': 'while_loop backward stride mismatch crash',
        'ops': ['while_loop'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: backward compilation crash)',
        'type': 'crash',
    },
    '#anon-8': {
        'desc': 'addcmul_/addcdiv_ backward missing autograd formula',
        'ops': ['addcmul_'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: missing prims.fma formula)',
        'type': 'crash',
    },
    '#anon-9': {
        'desc': 'complex literal backward .get_name() crash',
        'ops': ['mul', 'complex'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: raw complex in IR)',
        'type': 'crash',
    },
    '#anon-10': {
        'desc': 'complex abs/angle backward stride assertion crash',
        'ops': ['abs', 'angle'],
        'boundary': frozenset({CATEGORY_POINTWISE}),
        'boundary_detail': 'pointwise-only (stride assertion in backward kernel)',
        'type': 'crash',
    },
    # ── Split-scan boundary (3) ─────────────────────────────────────
    '#anon-11': {
        'desc': 'cumsum SplitScan codegen crash at dim>=129',
        'ops': ['cumsum'],
        'boundary': frozenset({CATEGORY_SCAN}),
        'boundary_detail': 'scan-only (split_scan boundary crash)',
        'type': 'crash',
    },
    '#anon-12': {
        'desc': 'associative_scan dynamic crash ndim>=2',
        'ops': ['associative_scan'],
        'boundary': frozenset({CATEGORY_SCAN}),
        'boundary_detail': 'scan-only (dynamic shape crash)',
        'type': 'crash',
    },
    '#anon-13': {
        'desc': 'cumprod backward crash at scan dim>=8193',
        'ops': ['cumprod'],
        'boundary': frozenset({CATEGORY_SCAN}),
        'boundary_detail': 'scan-only (backward split_scan crash)',
        'type': 'crash',
    },
    # ── Scatter / indexing (2) ──────────────────────────────────────
    '#anon-14': {
        'desc': 'index_add on expanded tensors wrong result',
        'ops': ['index_add'],
        'boundary': frozenset({CATEGORY_SCATTER}),
        'boundary_detail': 'scatter-only (stride-0 expansion)',
        'type': 'silent',
    },
    '#anon-15': {
        'desc': 'randperm pattern matcher missing slice_shape',
        'ops': ['randperm'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: pattern matcher crash)',
        'type': 'crash',
    },
    # ── Context / higher-order (4) ──────────────────────────────────
    '#anon-16': {
        'desc': 'scheduler prologue fusion copy-paste typo',
        'ops': [],
        'boundary': None,
        'boundary_detail': 'N/A (scheduler-level guard bug)',
        'type': 'crash',
    },
    '#anon-17': {
        'desc': 'inference_mode batched matmul crash',
        'ops': ['bmm'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: inference_mode guard)',
        'type': 'crash',
    },
    '#anon-18': {
        'desc': 'AOTAutograd drops higher-order grad metadata',
        'ops': ['autograd_grad'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: AOTAutograd metadata)',
        'type': 'silent',
    },
    '#anon-19': {
        'desc': 'NestedTensor layer_norm guard builder crash',
        'ops': ['layer_norm'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: guard builder)',
        'type': 'crash',
    },
    # ── Codegen / lowering (4) ──────────────────────────────────────
    '#anon-20': {
        'desc': 'fractional_max_pool2d wrong result under compile',
        'ops': ['fractional_max_pool2d'],
        'boundary': frozenset({CATEGORY_REDUCTION}),
        'boundary_detail': 'reduction-only (wrong pooling indices)',
        'type': 'silent',
    },
    '#anon-21': {
        'desc': 'float8 cast Triton codegen unsupported type crash',
        'ops': ['to_float8'],
        'boundary': None,
        'boundary_detail': 'N/A (pre-codegen: arch-unsupported dtype)',
        'type': 'crash',
    },
    '#anon-22': {
        'desc': 'flip 0-d tensor IndexError in lowering',
        'ops': ['flip'],
        'boundary': None,
        'boundary_detail': 'N/A (lowering crash)',
        'type': 'crash',
    },
    '#anon-23': {
        'desc': 'foreach_sub silently ignores alpha parameter',
        'ops': ['_foreach_sub'],
        'boundary': frozenset({CATEGORY_POINTWISE}),
        'boundary_detail': 'pointwise-only (missing alpha in lowering)',
        'type': 'silent',
    },
}


def load_traces(path):
    with open(path) as f:
        return json.load(f)


def extract_fusion_boundaries(traces):
    """Extract fusion boundary types from trace data."""
    boundary_counter = Counter()
    n_ok = 0

    for t in traces:
        sig = t.get('sig', {})
        trace = t.get('trace', t)
        if sig.get('sig_L1') == 'ERROR' or trace.get('error'):
            continue
        n_ok += 1

        for fp_str in trace.get('fusion_patterns', []):
            ops = re.findall(r"'(\w+)'", fp_str)
            boundary = classify_fusion_boundary(ops)
            boundary_counter[boundary] += 1

    return boundary_counter, n_ok


def analyze():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load random baseline
    print("Loading random baseline...")
    random_path = '${ARTIFACT_ROOT}/experiments/results/fusion_baseline/traces_random_1k.json'
    if not os.path.exists(random_path):
        random_path = '${ARTIFACT_ROOT}/experiments/results/collapse_cds/traces_random.json'
    random_traces = load_traces(random_path)
    r_boundary, n_random = extract_fusion_boundaries(random_traces)
    print(f"  {n_random} compiled programs, {sum(r_boundary.values())} fusion patterns")

    # ── 1. Boundary Type Frequency ────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"FUSION BOUNDARY TYPE FREQUENCY (Random Baseline)")
    print(f"{'=' * 80}")

    total_fusions = sum(r_boundary.values())
    homogeneous = 0
    cross_boundary = 0

    boundary_data = []
    for boundary, count in r_boundary.most_common():
        cats = sorted(boundary)
        is_cross = len(boundary) > 1
        if is_cross:
            cross_boundary += count
        else:
            homogeneous += count
        label = boundary_label(boundary)
        pct = count / total_fusions * 100
        boundary_data.append({
            'boundary': boundary,
            'label': label,
            'count': count,
            'pct': pct,
            'is_cross': is_cross,
            'n_categories': len(boundary),
        })
        print(f"  {label:<45} {count:>5} ({pct:>5.1f}%)"
              f"  {'CROSS' if is_cross else ''}")

    print(f"\n  Homogeneous fusions: {homogeneous}/{total_fusions} ({homogeneous/total_fusions*100:.1f}%)")
    print(f"  Cross-boundary fusions: {cross_boundary}/{total_fusions} ({cross_boundary/total_fusions*100:.1f}%)")

    # ── 2. Bug Boundary Classification ────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"BUG BOUNDARY CLASSIFICATION")
    print(f"{'=' * 80}")

    bug_boundary_counter = Counter()
    bug_cross_count = 0
    bug_homo_count = 0

    for bug_id, bug in sorted(BUGS.items()):
        boundary = bug['boundary']
        is_cross = len(boundary) > 1
        label = boundary_label(boundary)
        bug_boundary_counter[boundary] += 1

        if is_cross:
            bug_cross_count += 1
        else:
            bug_homo_count += 1

        # Find frequency in random baseline
        freq = r_boundary.get(boundary, 0)
        freq_pct = freq / max(total_fusions, 1) * 100

        print(f"  {bug_id}: {bug['desc']}")
        print(f"    Boundary: {label}")
        print(f"    Detail:   {bug['boundary_detail']}")
        print(f"    Baseline: {freq}/{total_fusions} ({freq_pct:.1f}%)")
        print()

    print(f"  Cross-boundary bugs: {bug_cross_count}/{len(BUGS)} ({bug_cross_count/len(BUGS)*100:.0f}%)")
    print(f"  Homogeneous bugs:    {bug_homo_count}/{len(BUGS)} ({bug_homo_count/len(BUGS)*100:.0f}%)")

    # ── 3. Semantic Boundary Matrix ───────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"SEMANTIC BOUNDARY MATRIX")
    print(f"{'=' * 80}")

    all_cats = [CATEGORY_POINTWISE, CATEGORY_REDUCTION, CATEGORY_SCAN,
                CATEGORY_SCATTER, CATEGORY_LAYOUT, CATEGORY_EXTERN]

    # Count pair frequencies in random baseline
    pair_freq = Counter()
    for boundary, count in r_boundary.items():
        cats = sorted(boundary)
        if len(cats) == 1:
            pair_freq[(cats[0], cats[0])] += count
        else:
            for c1, c2 in combinations(cats, 2):
                pair_freq[(c1, c2)] += count

    # Count pair frequencies in bugs
    pair_bugs = Counter()
    for bug in BUGS.values():
        cats = sorted(bug['boundary'])
        if len(cats) == 1:
            pair_bugs[(cats[0], cats[0])] += 1
        else:
            for c1, c2 in combinations(cats, 2):
                pair_bugs[(c1, c2)] += 1

    # Print matrix
    header = f"{'':>15}" + ''.join(f'{c:>12}' for c in all_cats)
    print(header)
    print('─' * len(header))

    for c1 in all_cats:
        row = f'{c1:>15}'
        for c2 in all_cats:
            key = tuple(sorted([c1, c2]))
            freq = pair_freq.get(key, 0)
            bugs = pair_bugs.get(key, 0)
            if freq > 0 or bugs > 0:
                row += f'{freq:>8}({bugs}b)'
            else:
                if c1 <= c2:
                    row += f'{"0":>12}'
                else:
                    row += f'{"":>12}'
        print(row)

    # ── 4. Boundary Frequency vs Bug Density Figure ───────────────────

    # Aggregate: for each boundary type, count fusions and bugs
    boundary_types = []
    for bd in sorted(set(list(r_boundary.keys()) +
                        [b['boundary'] for b in BUGS.values()])):
        label = boundary_label(bd)
        freq = r_boundary.get(bd, 0)
        bugs = bug_boundary_counter.get(bd, 0)
        if freq > 0 or bugs > 0:
            boundary_types.append({
                'boundary': bd,
                'label': label,
                'freq': freq,
                'freq_pct': freq / max(total_fusions, 1) * 100,
                'bugs': bugs,
                'is_cross': len(bd) > 1,
            })

    # Sort by frequency descending
    boundary_types.sort(key=lambda x: -x['freq'])

    # Figure: two-panel — frequency distribution and bug density
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Top boundary types by frequency
    top_n = 20
    bt_top = boundary_types[:top_n]
    # Also include any bug-relevant types not in top
    bug_bt = [bt for bt in boundary_types if bt['bugs'] > 0 and bt not in bt_top]
    display = bt_top + bug_bt

    labels = [bt['label'] for bt in display]
    freqs = [bt['freq'] for bt in display]
    colors = ['#e74c3c' if bt['bugs'] > 0 else
              ('#f39c12' if bt['is_cross'] else '#3498db')
              for bt in display]

    y_pos = range(len(labels))
    ax1.barh(y_pos, freqs, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Frequency in Random Baseline', fontsize=11)
    ax1.set_title(f'Fusion Boundary Types by Frequency\n'
                  f'(N={n_random}, {total_fusions} total fusions)', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, label='Homogeneous (single category)'),
        Patch(facecolor='#f39c12', alpha=0.8, label='Cross-boundary (multi-category)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Bug-relevant boundary'),
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc='lower right')

    # Right: Bug count by boundary class
    class_labels = ['Homogeneous\n(single semantic)', 'Cross-boundary\n(multi-semantic)']
    class_counts = [bug_homo_count, bug_cross_count]
    class_colors = ['#3498db', '#e74c3c']

    bars = ax2.bar(range(2), class_counts, color=class_colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5, width=0.5)
    for bar, count in zip(bars, class_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 str(count), ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax2.set_xticks(range(2))
    ax2.set_xticklabels(class_labels, fontsize=11)
    ax2.set_ylabel('Number of Confirmed Bugs', fontsize=12)
    ax2.set_title(f'Bugs by Semantic Boundary Type\n'
                  f'({len(BUGS)} total bugs)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boundary_frequency_vs_bugs.png'),
                dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'boundary_frequency_vs_bugs.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved boundary_frequency_vs_bugs.{png,pdf}")

    # ── 5. Semantic Interaction Heatmap ───────────────────────────────

    cat_labels = ['pointwise', 'reduction', 'scan', 'scatter', 'layout', 'extern']
    n_cats = len(cat_labels)

    freq_matrix = np.zeros((n_cats, n_cats))
    bug_matrix = np.zeros((n_cats, n_cats))

    for i, c1 in enumerate(cat_labels):
        for j, c2 in enumerate(cat_labels):
            key = tuple(sorted([c1, c2]))
            freq_matrix[i, j] = pair_freq.get(key, 0)
            bug_matrix[i, j] = pair_bugs.get(key, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Frequency heatmap (log scale)
    freq_log = np.log10(freq_matrix + 1)
    im1 = ax1.imshow(freq_log, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(n_cats))
    ax1.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_yticks(range(n_cats))
    ax1.set_yticklabels(cat_labels, fontsize=10)
    ax1.set_title(f'Fusion Interaction Frequency\n(log10, N={n_random})', fontsize=12)

    for i in range(n_cats):
        for j in range(n_cats):
            val = int(freq_matrix[i, j])
            if val > 0:
                ax1.text(j, i, str(val), ha='center', va='center',
                         fontsize=9, color='white' if freq_log[i, j] > 1.5 else 'black')

    plt.colorbar(im1, ax=ax1, shrink=0.8, label='log10(count+1)')

    # Bug heatmap
    im2 = ax2.imshow(bug_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=max(bug_matrix.max(), 1))
    ax2.set_xticks(range(n_cats))
    ax2.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_yticks(range(n_cats))
    ax2.set_yticklabels(cat_labels, fontsize=10)
    ax2.set_title(f'Bug Incidence by Interaction\n({len(BUGS)} bugs)', fontsize=12)

    for i in range(n_cats):
        for j in range(n_cats):
            val = int(bug_matrix[i, j])
            if val > 0:
                ax2.text(j, i, str(val), ha='center', va='center',
                         fontsize=12, fontweight='bold',
                         color='white' if bug_matrix[i, j] > 2 else 'black')

    plt.colorbar(im2, ax=ax2, shrink=0.8, label='bug count')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'semantic_interaction_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'semantic_interaction_heatmap.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved semantic_interaction_heatmap.{png,pdf}")

    # ── 6. KernelFuzz boundary coverage comparison ────────────────────

    print(f"\n{'=' * 80}")
    print(f"RANDOM vs KERNELFUZZ: BOUNDARY COVERAGE")
    print(f"{'=' * 80}")

    # Load KernelFuzz corpus
    kf_entries = []
    seen = set()
    for i in range(1, 10):
        path = f'${ARTIFACT_ROOT}/kernelfuzz_campaign_{i:02d}/graph-oracle/corpus_checkpoint.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for e in data.get('entries', []):
                if e['name'] not in seen:
                    seen.add(e['name'])
                    kf_entries.append(e)

    kf_names = [e['name'].lower() for e in kf_entries]

    # Map boundary types to keyword patterns for corpus search
    boundary_keywords = {
        'scan × *': ['scan', 'cumsum', 'cumprod', 'cummax', 'cummin', 'associative'],
        'scatter × *': ['scatter', 'index_copy', 'index_put', 'masked_scatter',
                        'diagonal_scatter', 'select_scatter', 'slice_scatter'],
        'layout × *': ['chunk', 'cat', 'split', 'narrow', 'as_strided',
                       'reflection_pad', 'replication_pad'],
        'extern (rare)': ['embedding', 'sdpa', 'attention', 'flex_attention',
                          'while_loop', 'einsum', 'fft'],
    }

    header = f"{'Boundary Region':<20} {'Random (N={n_random})':>20} {'KF (N={len(kf_entries)})':>18}"
    print(header)
    print('─' * len(header))

    for group, keywords in boundary_keywords.items():
        # Random: check fusion patterns
        random_count = 0
        for t in random_traces:
            sig = t.get('sig', {})
            trace = t.get('trace', t)
            if sig.get('sig_L1') == 'ERROR' or trace.get('error'):
                continue
            fusions_str = ' '.join(trace.get('fusion_patterns', []))
            if any(kw in fusions_str.lower() for kw in keywords):
                random_count += 1

        kf_count = sum(1 for n in kf_names if any(kw in n for kw in keywords))

        r_pct = random_count / max(n_random, 1) * 100
        k_pct = kf_count / max(len(kf_entries), 1) * 100
        marker = ' <<<' if random_count == 0 and kf_count > 0 else ''
        print(f"  {group:<20} {random_count:>5} ({r_pct:>5.1f}%)     {kf_count:>5} ({k_pct:>5.1f}%){marker}")

    # ── 7. Summary ────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: CROSS-SEMANTIC FUSION BOUNDARY ANALYSIS")
    print(f"{'=' * 80}")

    print(f"\n  Random baseline: {n_random} programs, {total_fusions} fusions")
    print(f"  Homogeneous fusions: {homogeneous}/{total_fusions} ({homogeneous/total_fusions*100:.0f}%)")
    print(f"  Cross-boundary fusions: {cross_boundary}/{total_fusions} ({cross_boundary/total_fusions*100:.0f}%)")
    print(f"\n  Bugs: {len(BUGS)} total")
    print(f"  Cross-boundary bugs: {bug_cross_count}/{len(BUGS)} ({bug_cross_count/len(BUGS)*100:.0f}%)")
    print(f"  Homogeneous bugs: {bug_homo_count}/{len(BUGS)} ({bug_homo_count/len(BUGS)*100:.0f}%)")

    # Key insight stats
    dominant = r_boundary.most_common(1)[0]
    print(f"\n  Most common boundary: {boundary_label(dominant[0])} "
          f"({dominant[1]}/{total_fusions}, {dominant[1]/total_fusions*100:.0f}%)")

    # Bugs in boundaries with <1% baseline frequency
    rare_boundary_bugs = sum(1 for bug in BUGS.values()
                             if r_boundary.get(bug['boundary'], 0) / max(total_fusions, 1) < 0.01)
    print(f"  Bugs in <1% frequency boundaries: {rare_boundary_bugs}/{len(BUGS)} "
          f"({rare_boundary_bugs/len(BUGS)*100:.0f}%)")

    # Save results
    results = {
        'n_random': n_random,
        'total_fusions': total_fusions,
        'homogeneous_fusions': homogeneous,
        'cross_boundary_fusions': cross_boundary,
        'boundary_frequency': {boundary_label(k): v for k, v in r_boundary.most_common()},
        'bug_cross_count': bug_cross_count,
        'bug_homo_count': bug_homo_count,
        'pair_frequency': {f'{k[0]}×{k[1]}': v for k, v in pair_freq.most_common()},
        'pair_bugs': {f'{k[0]}×{k[1]}': v for k, v in pair_bugs.most_common()},
    }
    with open(os.path.join(OUTPUT_DIR, 'boundary_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  All results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    analyze()
