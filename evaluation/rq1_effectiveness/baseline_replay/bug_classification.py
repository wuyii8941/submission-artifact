#!/usr/bin/env python3
"""
30 PyTorch/Inductor campaign bugs classified by decision region.

Maps each bug to:
  - decision_region: one of the 6 Inductor pipeline stages
  - required_hit: what the baseline program must exercise to reach this bug
  - structural_requirements: concrete program features needed
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BugSpec:
    bug_id: str
    issue: int
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
    BugSpec("bug_020", 181512, "bug_020_inference_mode_batched_matmul",
            "Frontend/context", "compiled frontend context",
            silent=False,
            structural_reqs=["inference_mode", "matmul"]),
    BugSpec("bug_023", 182217, "bug_023_layer_norm_nestedtensor_guard_crash",
            "Frontend/context", "compiled frontend context",
            silent=False,
            structural_reqs=["nested_tensor", "layer_norm"]),
    BugSpec("bug_051", 184222, "bug_051_scheduler_prologue_fusion_typo",
            "Frontend/context", "compiled frontend context",
            silent=True,
            structural_reqs=["prologue_fusion", "scheduler"]),

    # ── AOT/backward (5) ──
    BugSpec("bug_019", 181581, "bug_019_create_graph_requires_grad_dropped",
            "AOT/backward", "compiled backward artifact",
            silent=True,
            structural_reqs=["backward", "create_graph"]),
    BugSpec("bug_029", 183017, "bug_029_while_loop_backward_stride",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "while_loop"]),
    BugSpec("bug_042", 184088, "bug_042_expand_backward_aot_crash",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "expand"]),
    BugSpec("bug_044", 184100, "bug_044_complex_scalar_backward_crash",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "complex"]),
    BugSpec("bug_045", 184101, "bug_045_complex_abs_transpose_backward",
            "AOT/backward", "compiled backward artifact",
            silent=False,
            structural_reqs=["backward", "complex", "abs"]),

    # ── Lowering/IR (6) ──
    BugSpec("bug_032", 183249, "bug_032_frac_maxpool_random_compile",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["fractional_max_pool"]),
    BugSpec("bug_053", 184408, "bug_053_div_trunc_approximate_division",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["div", "rounding_mode"]),
    BugSpec("bug_054", 184410, "",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["lowering"]),
    BugSpec("bug_055", 184415, "bug_055_foreach_sub_alpha_ignored",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["foreach", "alpha"]),
    BugSpec("bug_059", 184416, "bug_059_deterministic_extern_kernel",
            "Lowering/IR", "specific lowering path",
            silent=True,
            structural_reqs=["deterministic", "extern_kernel"]),
    BugSpec("bug_061", 184081, "bug_061_float8_cast_arch_crash",
            "Lowering/IR", "specific lowering path",
            silent=False,
            structural_reqs=["float8"]),

    # ── Fusion (3) ──
    BugSpec("bug_004", 179561, "bug_004_bf16_cast_elided",
            "Fusion", "composed fusion group",
            silent=True,
            structural_reqs=["bf16", "cast", "fusion"]),
    BugSpec("bug_006", 179931, "bug_006_adaptive_avgpool_flatten_sum",
            "Fusion", "composed fusion group",
            silent=True,
            structural_reqs=["adaptive_avg_pool", "flatten", "sum"]),
    BugSpec("bug_034", 183987, "bug_034_addcmul_addcdiv_inplace_bwd_crash",
            "Fusion", "composed fusion group",
            silent=False,
            structural_reqs=["backward", "addcmul", "inplace"]),

    # ── Scheduling/template (4) ──
    BugSpec("bug_011", 180221, "bug_011_cumsum_splitscan_crash",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["cumsum", "split_scan", "large_dim"]),
    BugSpec("bug_015", 181584, "bug_015_associative_scan_dynamic",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["associative_scan", "dynamic"]),
    BugSpec("bug_031", 183250, "bug_031_cumprod_splitdim_bwd_crash",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["cumprod", "backward", "split_scan"]),
    BugSpec("bug_036", 183988, "bug_036_randperm_index_pattern_crash",
            "Scheduling/template", "split/template branch",
            silent=False,
            structural_reqs=["randperm", "index"]),

    # ── Triton codegen (9) ──
    BugSpec("bug_007", 180164, "bug_007_slice_scatter_backward",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "slice_scatter"]),
    BugSpec("bug_013", 180771, "bug_013_slice_scatter_backward",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "slice_scatter"]),
    BugSpec("bug_013v", 180772, "bug_013v_slice_scatter_backward_variant",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "slice_scatter"]),
    BugSpec("bug_022", 182012, "bug_022_diagonal_scatter_backward",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["backward", "diagonal_scatter"]),
    BugSpec("bug_033", 183986, "bug_033_index_family_expanded_wrong_result",
            "Triton codegen", "generated-kernel feature",
            silent=True,
            structural_reqs=["index_put", "expand"]),
    BugSpec("bug_043", 184089, "bug_043_flip_0d_inductor_crash",
            "Triton codegen", "generated-kernel feature",
            silent=False,
            structural_reqs=["flip", "0d_tensor"]),
    BugSpec("bug_063", 184992, "bug_063_cat_axis_sink_crash",
            "Triton codegen", "generated-kernel feature",
            silent=False,
            structural_reqs=["cat", "axis_sink"]),
    BugSpec("bug_182200", 182200, "",
            "Triton codegen", "generated-kernel feature",
            silent=False,
            structural_reqs=["backward"]),
    BugSpec("bug_183015", 183015, "",
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
            print(f"    {b.bug_id} #{b.issue} [{', '.join(b.structural_reqs)}]")
