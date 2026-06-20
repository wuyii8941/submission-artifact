#!/usr/bin/env python3
"""
Generate RQ1 Table V2: Bug density by decision-state region.

Two-dimensional classification:
  Fused interaction × Schedule feature

Key findings:
  - Random produces only 4 schedule features (pointwise, std/pers reduction, extern)
  - 5 schedule features with bugs have ZERO random coverage
  - scan-only interaction reached by random (0.6%) but compiled as reduction,
    not scan kernel — schedule-level gap invisible in interaction-only table
"""

import json
import re
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_boundary_analysis import (
    OP_TO_CATEGORY, classify_fusion_boundary, boundary_label,
)


# ═════════════════════════════════════════════════════════════════════════
# Per-bug classification: (issue, fused_interaction, schedule_feature, cat)
#
# Aligned with appendix tab:pt-issues (30 bugs):
#   Fusion(2) + Backward(10) + Split-scan(3) + Scatter(2) + Context(5) + Codegen(8)
# ═════════════════════════════════════════════════════════════════════════

SF_POINTWISE  = 'pointwise tile/mask'
SF_RED        = 'standard/persistent reduction'
SF_LAYOUT_RED = 'layout-sensitive reduction'
SF_SPLIT_SCAN = 'local/split scan'
SF_SCATTER    = 'scatter/stride-sensitive'
SF_EXTERN     = 'extern dispatch'
SF_FOREACH    = 'foreach dispatch'
SF_CONTEXT    = 'guard/autograd/context'

BUGS = [
    # Fusion composition (2)
    ('#179561', 'pointwise-only',       SF_POINTWISE,  'codegen'),
    ('#179931', 'layout × reduction',   SF_LAYOUT_RED, 'codegen'),

    # Backward codegen (10)
    ('#180164', 'scatter-only',         SF_SCATTER,    'codegen'),
    ('#180771', 'scatter-only',         SF_SCATTER,    'codegen'),
    ('#180772', 'scatter-only',         SF_SCATTER,    'codegen'),
    ('#182012', 'scatter-only',         SF_SCATTER,    'codegen'),
    ('#182200', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#183017', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#183987', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#184088', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#184100', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#184101', 'pointwise-only',       SF_POINTWISE,  'codegen'),

    # Split-scan boundary (3)
    ('#180221', 'scan-only',            SF_SPLIT_SCAN, 'codegen'),
    ('#181584', 'scan-only',            SF_SPLIT_SCAN, 'codegen'),
    ('#183250', 'scan-only',            SF_SPLIT_SCAN, 'codegen'),

    # Scatter/indexing (2)
    ('#183986', 'scatter-only',         SF_SCATTER,    'codegen'),
    ('#183988', None,                   SF_CONTEXT,    'pre-codegen'),

    # Context/higher-order (5)
    ('#181581', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#181512', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#182217', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#183015', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#184222', None,                   SF_CONTEXT,    'pre-codegen'),

    # Codegen/lowering (8)
    ('#183249', 'reduction-only',       SF_RED,        'codegen'),
    ('#184408', 'pointwise-only',       SF_POINTWISE,  'codegen'),
    ('#184410', 'pointwise-only',       SF_POINTWISE,  'codegen'),
    ('#184415', 'pointwise-only',       SF_FOREACH,    'codegen'),
    ('#184416', 'extern-only',          SF_EXTERN,     'codegen'),
    ('#184081', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#184089', None,                   SF_CONTEXT,    'pre-codegen'),
    ('#184992', None,                   SF_CONTEXT,    'pre-codegen'),
]

assert len(BUGS) == 30, f"Expected 30, got {len(BUGS)}"


def analyze_random():
    """Analyze random baseline for schedule feature distribution."""
    PREFIX_TO_SCHED = {
        "poi": "pointwise tile/mask",
        "per": "persistent reduction",
        "red": "standard reduction",
        "hel": "cooperative reduction",
    }

    path = '${ARTIFACT_ROOT}/experiments/results/fusion_baseline/traces_random_1k.json'
    with open(path) as f:
        traces = json.load(f)

    sched_counts = Counter()
    n_kernels = 0
    n_programs = 0

    for entry in traces:
        trace = entry.get('trace', {})
        if trace.get('error'):
            continue
        n_programs += 1

        for kname in trace.get('kernel_names', []):
            m = re.match(r'triton_(\w{3})_fused_', kname)
            if m:
                sched = PREFIX_TO_SCHED.get(m.group(1), m.group(1))
                sched_counts[sched] += 1
                n_kernels += 1

        for ext in trace.get('extern_calls', []):
            sched_counts['extern dispatch'] += 1
            n_kernels += 1

    return sched_counts, n_programs, n_kernels


def main():
    sched_counts, n_programs, n_kernels = analyze_random()

    print(f"Random baseline: {n_programs} programs, {n_kernels} kernels")
    print()

    # ── Random schedule feature distribution ─────────────────────
    print("Random schedule feature distribution:")
    for s, c in sched_counts.most_common():
        print(f"  {s:<30s} {c:>5d} ({100*c/n_kernels:.1f}%)")

    has_scan = any('scan' in s for s in sched_counts)
    print(f"\n  Scan/split-scan kernels: {'YES' if has_scan else 'ZERO (0 out of ' + str(n_kernels) + ')'}")

    # ── Bug classification ───────────────────────────────────────
    codegen = [(i, inter, sf, cat) for i, inter, sf, cat in BUGS if cat == 'codegen']
    precodegen = [(i, inter, sf, cat) for i, inter, sf, cat in BUGS if cat == 'pre-codegen']

    # Interaction frequencies from boundary analysis
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boundary_analysis_results.json')) as f:
        bd = json.load(f)
    boundary_freq = bd['boundary_frequency']
    total_fusions = bd['total_fusions']

    # ── Table rows ───────────────────────────────────────────────

    # TABLE DESIGN:
    # Row = (fused_interaction, schedule_feature)
    # "Fusions" column = % of random fusions in this interaction region
    # "Bugs" column = count of bugs requiring this (interaction, schedule)
    #
    # For scan-only: random reaches the interaction (0.6%) but uses
    # reduction scheduling. The bugs need scan scheduling. We show this
    # with a footnote marker.

    # Explored regions with their dominant schedule feature
    EXPLORED_ROWS = OrderedDict([
        ('pointwise × reduction',          SF_RED),
        ('pointwise-only',                 SF_POINTWISE),
        ('layout × pointwise',             SF_POINTWISE),
        ('extern-only',                    SF_EXTERN),
        ('reduction-only',                 SF_RED),
        ('pointwise × scan',              SF_RED),
        ('scan-only',                      SF_SPLIT_SCAN),  # bugs need scan, not the reduction that random produces
    ])

    explored_interactions = set(boundary_freq.keys())

    # Count bugs per table row
    codegen_by_inter = Counter()
    for issue, inter, sf, cat in codegen:
        codegen_by_inter[inter] += 1

    # Print table
    print()
    print("=" * 95)
    print("PROPOSED TABLE: Bug density by decision-state region")
    print("=" * 95)
    fmt = "{:<30s} {:<32s} {:>8s} {:>5s}"
    print(fmt.format("Fused interaction", "Schedule feature", "Fusions", "Bugs"))
    print("-" * 77)

    explored_bug_total = 0
    for inter_label, sched_label in EXPLORED_ROWS.items():
        # Get fusions %
        freq = boundary_freq.get(inter_label, 0)
        pct = f"{100.0 * freq / total_fusions:.1f}%"

        # Count bugs in this interaction
        bugs = codegen_by_inter.get(inter_label, 0)
        explored_bug_total += bugs

        # Mark scan-only specially
        marker = ""
        if inter_label == 'scan-only':
            marker = "†"
        elif inter_label == 'pointwise × scan':
            marker = "†"

        print(fmt.format(
            inter_label,
            sched_label + marker,
            pct,
            str(bugs)
        ))

    # Merge remaining explored
    other_freq = sum(f for label, f in boundary_freq.items()
                     if label not in EXPLORED_ROWS)
    if other_freq > 0:
        pct = f"{100.0 * other_freq / total_fusions:.1f}%"
        print(fmt.format("other explored", "mixed", pct, "0"))

    print("-" * 77)
    print(fmt.format("Explored regions", "", "100.0%", str(explored_bug_total)))

    # Unexplored codegen regions
    unexplored_bugs = []
    for issue, inter, sf, cat in codegen:
        if inter and inter not in explored_interactions and inter not in EXPLORED_ROWS:
            unexplored_bugs.append((inter, sf))
    unexplored_count = Counter()
    for inter, sf in unexplored_bugs:
        unexplored_count[(inter, sf)] += 1

    unexplored_total = sum(unexplored_count.values())
    unexplored_scheds = sorted(set(sf for _, sf in unexplored_count.keys()))
    print(fmt.format(
        "Unexplored codegen regions",
        ', '.join(unexplored_scheds),
        "0%",
        str(unexplored_total)
    ))
    for (inter, sf), cnt in unexplored_count.most_common():
        print(f"  └ {inter}: {sf} ({cnt})")

    print(fmt.format(
        "Pre-codegen/context",
        SF_CONTEXT,
        "—",
        str(len(precodegen))
    ))
    print("-" * 77)
    total_bugs = explored_bug_total + unexplored_total + len(precodegen)
    print(f"Total: {total_bugs} bugs")

    print()
    print("† Random compiles scan ops to reduction kernels; 0 scan/split-scan")
    print("  kernels in 1,732 random Triton kernels.")

    # ── Quantified schedule gap ──────────────────────────────────
    print()
    print("=" * 95)
    print("SCHEDULE GAP ANALYSIS")
    print("=" * 95)

    bug_sf_counts = Counter(sf for _, _, sf, _ in BUGS if sf != SF_CONTEXT)
    for sf, cnt in bug_sf_counts.most_common():
        rc = sched_counts.get(sf, 0)
        rc_pct = f"{100*rc/n_kernels:.1f}%" if rc > 0 else "0%"
        print(f"  {sf:<32s}  {cnt:>2d} bugs   random: {rc_pct}")

    # ── LaTeX output ─────────────────────────────────────────────
    print()
    print("=" * 95)
    print("LATEX")
    print("=" * 95)
    print()

    tex = r"""\begin{table}[tbp]
\centering
\caption{Bug density by decision-state region.
  Each region is defined by fused-operator interaction and
  schedule feature, an observable proxy for the scheduling
  choice in the generated kernel.}
\label{tab:region-density}
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabularx}{\columnwidth}{@{}
>{\raggedright\arraybackslash}p{0.26\columnwidth}
>{\raggedright\arraybackslash}p{0.30\columnwidth}
rr@{}}
\toprule
Fused interaction & Schedule feature & Fusions & Bugs \\
\midrule"""

    for inter_label, sched_label in EXPLORED_ROWS.items():
        freq = boundary_freq.get(inter_label, 0)
        pct = f"{100.0 * freq / total_fusions:.1f}\\%"
        bugs = codegen_by_inter.get(inter_label, 0)

        inter_tex = inter_label.replace(' × ', ' $\\times$ ')

        sched_tex = sched_label
        if inter_label in ('scan-only', 'pointwise × scan'):
            sched_tex += '$^\\dagger$'

        tex += f"\n{inter_tex} & {sched_tex} & {pct} & {bugs} \\\\"

    if other_freq > 0:
        pct = f"{100.0 * other_freq / total_fusions:.1f}\\%"
        tex += f"\nother explored & mixed & {pct} & 0 \\\\"

    tex += f"""
\\midrule
Explored regions & & 100.0\\% & {explored_bug_total} \\\\
Unexplored codegen regions & {', '.join(unexplored_scheds)} & 0\\% & {unexplored_total} \\\\
Pre-codegen/context & {SF_CONTEXT} & -- & {len(precodegen)} \\\\
\\bottomrule
\\end{{tabularx}}
\\vspace{{2pt}}
\\raggedright\\scriptsize
$^\\dagger$Random compiles scan ops to reduction kernels;
zero scan/split-scan kernels in 1,732 random Triton kernels.
\\end{{table}}"""

    print(tex)


if __name__ == '__main__':
    main()
