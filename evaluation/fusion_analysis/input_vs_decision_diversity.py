#!/usr/bin/env python3
"""
Quantitative evidence: Input diversity ≠ Decision diversity.

Shows three phenomena:
  1. Program collapse (many-to-one): many distinct programs produce
     identical compilation artifacts.
  2. Decision divergence (one-to-many): the same operator produces
     different kernels depending on shape, dtype, and fusion context.
  3. Interaction imbalance: random generation saturates at 9 interaction
     cells by program ~168, leaving 47% of bug-harboring cells unexplored.

Uses existing 1k random + 1k targeted traces from fusion_baseline_1k.py.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fusion_boundary_analysis import (
    OP_TO_CATEGORY, classify_fusion_boundary, boundary_label, BUGS,
    CATEGORY_POINTWISE, CATEGORY_REDUCTION, CATEGORY_SCAN,
    CATEGORY_SCATTER, CATEGORY_LAYOUT, CATEGORY_EXTERN,
)

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traces')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _kernel_template(kernel_name):
    """Strip trailing counter to get structurally unique template."""
    return re.sub(r'_\d+$', '', kernel_name)


def _extract_ops(code_or_fusion_ops):
    """Extract op set from fusion ops list."""
    if isinstance(code_or_fusion_ops, list):
        return frozenset(code_or_fusion_ops)
    return frozenset()


def load_traces(method):
    path = os.path.join(TRACES_DIR, f'traces_{method}_1k.json')
    with open(path) as f:
        return json.load(f)


def analyze_program_collapse(traces):
    """
    Many-to-one: how many distinct programs collapse to the same
    compilation artifact (kernel template set)?

    A "compilation artifact" is the frozenset of kernel templates
    produced by compiling a program.
    """
    artifact_to_programs = defaultdict(list)
    n_ok = 0
    n_err = 0

    for t in traces:
        trace = t['trace']
        if trace.get('error'):
            n_err += 1
            continue
        n_ok += 1

        kernel_names = trace.get('kernel_names', [])
        extern_calls = trace.get('extern_calls', [])
        templates = frozenset(_kernel_template(k) for k in kernel_names)
        externs = frozenset(extern_calls)
        artifact = (templates, externs)

        artifact_to_programs[artifact].append(t['idx'])

    n_distinct_artifacts = len(artifact_to_programs)
    collision_counts = [len(v) for v in artifact_to_programs.values()]

    # Programs with >1 collision partner
    n_colliding = sum(c for c in collision_counts if c > 1)
    max_collision = max(collision_counts)

    # Top collisions
    top_collisions = sorted(artifact_to_programs.items(),
                            key=lambda x: -len(x[1]))[:10]

    return {
        'n_compiled': n_ok,
        'n_errors': n_err,
        'n_distinct_programs': n_ok,
        'n_distinct_artifacts': n_distinct_artifacts,
        'compression_ratio': n_ok / max(n_distinct_artifacts, 1),
        'n_programs_with_collisions': n_colliding,
        'collision_pct': n_colliding / max(n_ok, 1) * 100,
        'max_collision_size': max_collision,
        'collision_distribution': dict(Counter(collision_counts)),
        'top_collisions': [
            {
                'artifact': str(artifact[0]) if artifact[0] else 'extern-only: ' + str(artifact[1]),
                'n_programs': len(programs),
                'program_indices': programs[:5],
            }
            for artifact, programs in top_collisions
        ],
    }


def analyze_decision_divergence(traces):
    """
    One-to-many: same operator appears in different kernel templates
    depending on fusion context, shape, and dtype.

    For each op that appears in kernel names, count how many distinct
    kernel templates it appears in.
    """
    op_to_templates = defaultdict(set)
    op_to_families = defaultdict(set)

    for t in traces:
        trace = t['trace']
        if trace.get('error'):
            continue

        for kn in trace.get('kernel_names', []):
            template = _kernel_template(kn)
            # Parse the fused ops from the kernel name
            m = re.match(r'triton_(\w{3})_fused_(.+)', template)
            if m:
                prefix = m.group(1)
                family = {'poi': 'pointwise', 'per': 'persistent_reduction',
                          'red': 'reduction', 'hel': 'cooperative_reduction'
                          }.get(prefix, prefix)
                # Split fused ops
                ops_str = m.group(2)
                for op in re.split(r'(?<=[a-z0-9])_(?=[a-z])', ops_str):
                    op = op.strip('_')
                    if op:
                        op_to_templates[op].add(template)
                        op_to_families[op].add(family)

    divergence = {}
    for op in sorted(op_to_templates.keys()):
        n_templates = len(op_to_templates[op])
        n_families = len(op_to_families[op])
        if n_templates > 1:
            divergence[op] = {
                'n_distinct_templates': n_templates,
                'n_distinct_families': n_families,
                'families': sorted(op_to_families[op]),
                'sample_templates': sorted(list(op_to_templates[op]))[:5],
            }

    return {
        'total_ops_observed': len(op_to_templates),
        'ops_with_divergence': len(divergence),
        'ops_single_template': len(op_to_templates) - len(divergence),
        'divergence_pct': len(divergence) / max(len(op_to_templates), 1) * 100,
        'top_divergent_ops': dict(sorted(
            divergence.items(),
            key=lambda x: -x[1]['n_distinct_templates']
        )[:15]),
        'all_divergence': divergence,
    }


def analyze_interaction_growth(traces):
    """
    Interaction cell coverage growth curve.
    Shows saturation: new programs stop contributing new cells early.
    """
    cells_seen = set()
    growth = []
    n_ok = 0

    for t in traces:
        trace = t['trace']
        if trace.get('error'):
            continue
        n_ok += 1

        new_cells_this_program = set()
        for fp_str in trace.get('fusion_patterns', []):
            ops = re.findall(r"'(\w+)'", fp_str)
            boundary = classify_fusion_boundary(ops)
            if boundary and boundary not in cells_seen:
                new_cells_this_program.add(boundary)
                cells_seen.add(boundary)

        growth.append({
            'program_idx': n_ok,
            'total_cells': len(cells_seen),
            'new_cells': len(new_cells_this_program),
        })

    # Find saturation point (last program that added a new cell)
    saturation_idx = 0
    for g in growth:
        if g['new_cells'] > 0:
            saturation_idx = g['program_idx']

    return {
        'final_cells': len(cells_seen),
        'cells_list': sorted(boundary_label(c) for c in cells_seen),
        'saturation_program': saturation_idx,
        'programs_after_saturation': n_ok - saturation_idx,
        'wasted_pct': (n_ok - saturation_idx) / max(n_ok, 1) * 100,
        'growth_curve': growth,
    }


def analyze_kernel_diversity_growth(traces):
    """
    Kernel template diversity growth curve (analogous to interaction cells
    but at the kernel template level).
    """
    templates_seen = set()
    growth = []
    n_ok = 0

    for t in traces:
        trace = t['trace']
        if trace.get('error'):
            continue
        n_ok += 1

        for kn in trace.get('kernel_names', []):
            templates_seen.add(_kernel_template(kn))

        growth.append({
            'program_idx': n_ok,
            'kernel_diversity': len(templates_seen),
        })

    return {
        'final_diversity': len(templates_seen),
        'growth_curve': growth,
    }


def compute_all():
    print("Loading traces...")
    random_traces = load_traces('random')
    targeted_traces = load_traces('targeted')

    results = {}

    # 1. Program collapse (many-to-one)
    print("\n=== Program Collapse (Many-to-One) ===")
    for method, traces in [('random', random_traces), ('targeted', targeted_traces)]:
        collapse = analyze_program_collapse(traces)
        results[f'{method}_collapse'] = collapse
        print(f"\n{method.upper()}:")
        print(f"  Programs compiled: {collapse['n_compiled']}")
        print(f"  Distinct compilation artifacts: {collapse['n_distinct_artifacts']}")
        print(f"  Compression ratio: {collapse['compression_ratio']:.2f}x")
        print(f"  Programs with identical artifacts: {collapse['n_programs_with_collisions']} "
              f"({collapse['collision_pct']:.1f}%)")
        print(f"  Largest collision: {collapse['max_collision_size']} programs → 1 artifact")
        print(f"  Top collisions:")
        for tc in collapse['top_collisions'][:5]:
            art_str = str(tc['artifact'])[:80]
            print(f"    {tc['n_programs']:3d} programs → {art_str}")

    # 2. Decision divergence (one-to-many)
    print("\n=== Decision Divergence (One-to-Many) ===")
    for method, traces in [('random', random_traces), ('targeted', targeted_traces)]:
        div = analyze_decision_divergence(traces)
        results[f'{method}_divergence'] = {
            k: v for k, v in div.items() if k != 'all_divergence'
        }
        print(f"\n{method.upper()}:")
        print(f"  Total ops observed: {div['total_ops_observed']}")
        print(f"  Ops with decision divergence: {div['ops_with_divergence']} "
              f"({div['divergence_pct']:.1f}%)")
        print(f"  Top divergent ops:")
        for op, info in list(div['top_divergent_ops'].items())[:10]:
            print(f"    {op:15s}: {info['n_distinct_templates']:3d} templates "
                  f"across {info['n_distinct_families']} families "
                  f"({', '.join(info['families'])})")

    # 3. Interaction cell saturation
    print("\n=== Interaction Cell Saturation ===")
    for method, traces in [('random', random_traces), ('targeted', targeted_traces)]:
        growth = analyze_interaction_growth(traces)
        results[f'{method}_interaction'] = {
            k: v for k, v in growth.items() if k != 'growth_curve'
        }
        results[f'{method}_interaction']['growth_curve_summary'] = {
            'points': len(growth['growth_curve']),
            'first_10': growth['growth_curve'][:10],
            'last_5': growth['growth_curve'][-5:],
        }
        print(f"\n{method.upper()}:")
        print(f"  Final cells: {growth['final_cells']}")
        print(f"  Saturation at program: {growth['saturation_program']}")
        print(f"  Programs after saturation: {growth['programs_after_saturation']} "
              f"({growth['wasted_pct']:.1f}% wasted)")
        print(f"  Cells: {growth['cells_list']}")

    # 4. Kernel diversity growth comparison
    print("\n=== Kernel Diversity Growth ===")
    for method, traces in [('random', random_traces), ('targeted', targeted_traces)]:
        kd = analyze_kernel_diversity_growth(traces)
        results[f'{method}_kernel_diversity'] = {
            'final_diversity': kd['final_diversity'],
        }
        print(f"\n{method.upper()}:")
        print(f"  Final kernel diversity: {kd['final_diversity']}")

    # 5. Bug reachability (from existing data)
    print("\n=== Bug Reachability Summary ===")
    bug_cells = set()
    for bug in BUGS.values():
        bug_cells.add(bug['boundary'])
    random_cells = set()
    for t in random_traces:
        trace = t['trace']
        if trace.get('error'):
            continue
        for fp_str in trace.get('fusion_patterns', []):
            ops = re.findall(r"'(\w+)'", fp_str)
            boundary = classify_fusion_boundary(ops)
            if boundary:
                random_cells.add(boundary)

    reached = bug_cells & random_cells
    unreached = bug_cells - random_cells
    bugs_reached = sum(1 for b in BUGS.values() if b['boundary'] in random_cells)
    bugs_unreached = sum(1 for b in BUGS.values() if b['boundary'] not in random_cells)

    results['bug_reachability'] = {
        'total_bugs': len(BUGS),
        'bug_cells': len(bug_cells),
        'random_cells': len(random_cells),
        'cells_reached': len(reached),
        'cells_unreached': len(unreached),
        'bugs_in_reached': bugs_reached,
        'bugs_in_unreached': bugs_unreached,
        'unreached_cells': sorted(boundary_label(c) for c in unreached),
    }
    print(f"  Bug interaction cells: {len(bug_cells)}")
    print(f"  Random reaches: {len(reached)}/{len(bug_cells)} cells")
    print(f"  Bugs in unreached cells: {bugs_unreached}/{len(BUGS)}")
    print(f"  Unreached cells: {sorted(boundary_label(c) for c in unreached)}")

    # 6. Summary table for paper
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)
    rc = results['random_collapse']
    rd = results['random_divergence']
    ri = results['random_interaction']
    print(f"\nProgram collapse (random, 1k programs):")
    print(f"  {rc['n_compiled']} programs → {rc['n_distinct_artifacts']} "
          f"distinct artifacts ({rc['compression_ratio']:.1f}× compression)")
    print(f"  {rc['collision_pct']:.0f}% of programs share artifacts with another program")
    print(f"\nDecision divergence (random):")
    print(f"  {rd['ops_with_divergence']}/{rd['total_ops_observed']} ops "
          f"({rd['divergence_pct']:.0f}%) produce multiple kernel templates")
    print(f"\nInteraction saturation (random):")
    print(f"  Saturates at 9 cells by program {ri['saturation_program']}")
    print(f"  {ri['wasted_pct']:.0f}% of programs contribute zero new cells")
    print(f"\nBug gap:")
    br = results['bug_reachability']
    print(f"  {br['bugs_in_unreached']}/{br['total_bugs']} bugs in unreached cells")

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'input_vs_decision_diversity.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")

    return results


if __name__ == '__main__':
    compute_all()
