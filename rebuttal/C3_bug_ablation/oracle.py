#!/usr/bin/env python3
import json, re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
M = re.M
ORACLE = {
 'bug_014_frac_maxpool_random_compile': ('nightly', 'stdout', r'Without _random_samples:\s*\n\s*max_diff = [\d.]+\s*\n\s*BUG: YES'),
 'bug_015_index_family_expanded_wrong_result': ('nightly', 'stdout', r'--- Minimal Repro ---[\s\S]*?Max diff: (?!0\.000000)\d'),
 'bug_017_randperm_index_pattern_crash': ('nightly', 'stderr', r'RuntimeError: Not all inputs to pattern found in match\.kwargs'),
 'bug_021_complex_abs_transpose_backward': ('nightly', 'stdout', r'inductor: CRASH - AssertionError: expected size'),
 'bug_023_div_trunc_approximate_division': ('nightly', 'stdout', r'BUG CONFIRMED: div_trunc'),
 'bug_029_fft_backward_stride_crash': ('nightly', 'stdout', r'CRASH: AssertionError: expected size'),
 'bug_031_vmap_hessian_crash': ('nightly', 'stdout', r'\[hessian path\] CRASH: TorchRuntimeError'),
 'bug_002_adaptive_avgpool_flatten_sum': ('pt2121', 'stdout', r'^  max_diff: \d\.\d+e\+'),
 'bug_004_cumsum_splitscan_crash': ('pt2121', 'stderr', r'InductorError: TypeError: list indices must be integers or slices, not NoneType'),
 'bug_007_associative_scan_dynamic': ('pt2121', 'stdout', r'2D dynamic=True: CRASH .*lifted a'),
 'bug_008_create_graph_requires_grad_dropped': ('pt2121', 'stdout', r'Compile: requires_grad=False, grad_fn=None'),
 'bug_009_inference_mode_batched_matmul': ('pt2121', 'stdout', r'3D matmul \+ inference_mode \(BUG\) ---\s*\n\s*CRASH'),
 'bug_010_diagonal_scatter_backward': ('pt2121', 'stdout', r'^  BUG \d+x\d+'),
 'bug_012_while_loop_backward_stride': ('pt2121', 'stdout', r"differ in 'stride: \(0, 0\) vs \(64, 1\)'"),
 'bug_016_addcmul_addcdiv_inplace_bwd_crash': ('pt2121', 'stderr', r'RuntimeError: Trying to backward through prims\.fma'),
 'bug_018_expand_backward_aot_crash': ('pt2121', 'stdout', r'inductor: CRASH - unsupported operation: more than one element of the written-to tensor'),
 'bug_019_flip_0d_inductor_crash': ('pt2121', 'stdout', r'inductor: CRASH - InductorError: IndexError: list index out of range'),
 'bug_013_cumprod_splitdim_bwd_crash': ('pt2121', 'stdout', r'Compiled CRASH \(unexpected\): InductorError: NotImplementedError: NYI TritonSplitDimKernel reductions'),
 'bug_020_complex_scalar_backward_crash': ('pt2121', 'stdout', r"inductor: CRASH - InductorError: AttributeError: 'complex' object has no attribute 'get_name'"),
 'bug_024_foreach_sub_alpha_ignored': ('pt2121', 'stdout', r'BUG CONFIRMED: _foreach_sub'),
 'bug_026_float8_cast_arch_crash': ('pt2121', 'stderr', r'fp8e4nv not supported'),
 'bug_027_cat_axis_sink_crash': ('pt2121', 'stdout', r'^Compiled CRASH'),
 'bug_003_slice_scatter_backward': ('pt2110', 'stdout', r'x\.grad max_diff:\s+(?!0\.0000e\+00)\d'),
 'bug_005_slice_scatter_backward': ('pt2110', 'stdout', r'=== CUDA ===\s*\n❌'),
}
EXCLUDED = {'bug_001_bf16_cast_elided': 'no canonical reproducer (two scripts, neither named as in tools/run_repro.sh)',
            'bug_006_slice_scatter_backward_variant': 'the reproducer stops before its check on the builds used here (x1.grad is None)',
            'bug_011_layer_norm_nestedtensor_guard_crash': 'does not manifest on the builds used here',
            'bug_028_hardtanh_bf16_boundary_grad': 'needs BF16 kernels; the T4 has none',
            'bug_022_scheduler_prologue_fusion_typo': 'static source check (not a program to compile)',
            'bug_025_deterministic_extern_kernel': 'static source check (not a program to compile)',
            'bug_030_pad_mm_getitem_dead_code': 'static source check (not a program to compile)'}
PY = {'nightly': os.environ.get('PT_NIGHTLY', 'python3'), 'pt2130': os.environ.get('PT_2130', 'python3'),
      'pt2121': os.environ.get('PT_2121', 'python3'), 'pt2110': os.environ.get('PT_2110', 'python3')}


def run_dir(build, mode):
    return os.path.join(HERE, 'runs', mode) if build == 'nightly' else os.path.join(HERE, 'runs', build, mode)


def manifests(bug, mode):
    build, stream, pat = ORACLE[bug]
    f = os.path.join(run_dir(build, mode), f'{bug}.{stream}.txt')
    return bool(os.path.exists(f) and re.search(pat, open(f).read(), M)) if os.path.exists(f) else None


if __name__ == '__main__':
    mode = sys.argv[1]
    for bug in ORACLE:
        print(f'{bug:48s} {ORACLE[bug][0]:8s} manifests: {manifests(bug, mode)}')
