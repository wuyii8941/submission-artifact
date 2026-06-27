#!/usr/bin/env python3
"""
30 PyTorch/Inductor campaign bugs classified by decision region.

Maps each bug to:
  - decision_region: one of the 6 Inductor pipeline stages
  - required_hit: what the baseline program must exercise to reach this bug
  - structural_requirements: concrete program features needed

NOTE: Upstream issue identifiers are intentionally omitted for
double-blind review; bug IDs and structural requirements are sufficient
for the attribution/matching logic in attribute_table2.py.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BugSpec:
    bug_id: str
    directory: str
    decision_region: str
    required_hit: str
    silent: bool
    structural_reqs: List[str] = field(default_factory=list)


DECISION_REGIONS = [
    "Frontend/context",
    "AOT/backward",
    "Lowering/IR",
    "Fusion",
    "Scheduling/template",
    "Triton codegen",
]

BUGS: List[BugSpec] = [
    # ── Frontend/context (3) ──
    BugSpec("bug_020", "bug_020_inference_mode_batched_matmul",
            "Frontend/context", "compiled frontend context",
            silent=False,
            structural_reqs=["inference_mode", "matmul"]),
    BugSpec("bug_023", "bug_023_layer_norm_nestedtensor_guard_crash",
            "Frontend/context", "compiled frontend context",
            silent=False,
            structural_reqs=["nested_tensor", "layer_norm"]),
    BugSpec("bug_051", "bug_051_scheduler_prologue_fusion_typo",
            "Frontend/context", "compiled frontend context",
            silent=True,
            structural_reqs=["prologue_fusion", "scheduler"]),

    # ── AOT/backward (5) ──
    BugSpec("bug_019", "bug_019_create_graph_requires_grad_dropped",
            "AOT/backward", "compiled backward artifact",
            silent=True,
            structural_reqs=["backward", "create_graph"]),
    BugSpec("bug_029", "bug_029_while_loop_backward_stride",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "while_loop"]),
    BugSpec("bug_042", "bug_042_expand_backward_aot_crash",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "expand"]),
    BugSpec("bug_044", "bug_044_complex_scalar_backward_crash",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "complex"]),
    BugSpec("bug_045", "bug_045_complex_abs_transpose_backward",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "complex", "abs"]),

    # ── Lowering/IR (6) ──
    BugSpec("bug_032", "bug_032_frac_maxpool_random_compile",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["fractional_max_pool"]),
    BugSpec("bug_053", "bug_053_div_trunc_approximate_division",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["div", "rounding_mode"]),
    BugSpec("bug_054", "",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["lowering"]),
    BugSpec("bug_055", "bug_055_foreach_sub_alpha_ignored",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["foreach", "alpha"]),
    BugSpec("bug_059", "bug_059_deterministic_extern_kernel",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["deterministic", "extern_kernel"]),
    BugSpec("bug_061", "bug_061_float8_cast_arch_crash",
            "Lowering/IR", "specific lowering path",
            silent=False,
            structural_reqs=["float8"]),

    # ── Fusion (3) ──
    BugSpec("bug_004", "bug_004_bf16_cast_elided",
            "Fusion", "composed fusion group",
            silent=True,
            structural_reqs=["bf16", "cast", "fusion"]),
    BugSpec("bug_006", "bug_006_adaptive_avgpool_flatten_sum",
            "Fusion", "composed fusion group",
            silent=True,
            structural_reqs=["adaptive_avg_pool", "flatten", "sum"]),
    BugSpec("bug_034", "bug_034_addcmul_addcdiv_inplace_bwd_crash",
            "Fusion", "composed fusion group",
            silent=False,
            structural_reqs=["backward", "addcmul", "inplace"]),

    # ── Scheduling/template (4) ──
    BugSpec("bug_011", "bug_011_cumsum_splitscan_crash",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["cumsum", "split_scan", "large_dim"]),
    BugSpec("bug_015", "bug_015_associative_scan_dynamic",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["associative_scan", "dynamic"]),
    BugSpec("bug_031", "bug_031_cumprod_splitdim_bwd_crash",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["cumprod", "backward", "split_scan"]),
    BugSpec("bug_036", "bug_036_randperm_index_pattern_crash",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["randperm", "index"]),

    # ── Triton codegen (9) ──
    BugSpec("bug_007", "bug_007_slice_scatter_backward",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "slice_scatter"]),
    BugSpec("bug_013", "bug_013_slice_scatter_backward",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "slice_scatter"]),
    BugSpec("bug_013v", "bug_013v_slice_scatter_backward_variant",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "slice_scatter"]),
    BugSpec("bug_022", "bug_022_diagonal_scatter_backward",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "diagonal_scatter"]),
    BugSpec("bug_033", "bug_033_index_family_expanded_wrong_result",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["index_put", "expand"]),
    BugSpec("bug_043", "bug_043_flip_0d_inductor_crash",
            "Triton codegen", "generated-kernel feature",
            silent=False,
            structural_reqs=["flip", "0d_tensor"]),
    BugSpec("bug_063", "bug_063_cat_axis_sink_crash",
            "Triton codegen", "generated-kernel feature",
            silent=False,
            structural_reqs=["cat", "axis_sink"]),
    BugSpec("bug_065", "",
            "Triton codegen", "generated-kernel feature",
            silent=False,
            structural_reqs=["backward"]),
    BugSpec("bug_066", "",
            "Triton codegen", "generated-kernel feature",
            silent=False,
            structural_reqs=["context"]),
]

assert len(BUGS) == 30, f"Expected 30 bugs, got {len(BUGS)}"

for region in DECISION_REGIONS:
    count = sum(1 for b in BUGS if b.decision_region == region)
    assert count > 0, f"No bugs in region {region}"


def region_counts():
    from collections import Counter
    return Counter(b.decision_region for b in BUGS)


if __name__ == "__main__":
    counts = region_counts()
    print("Decision region distribution:")
    for r in DECISION_REGIONS:
        bugs = [b for b in BUGS if b.decision_region == r]
        print(f"  {r}: {counts[r]} bugs")
        for b in bugs:
            print(f"    {b.bug_id} [{', '.join(b.structural_reqs)}]")
