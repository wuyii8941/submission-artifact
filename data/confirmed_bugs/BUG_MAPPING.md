# Bug Mapping: Issue → Reproducer Directory

All 71 counted bugs mapped to their reproducer directories and paper tables.

## PyTorch/Inductor (30 bugs, plus 1 TBD unfiled)

| # | Issue | Directory | Category | Silent/Crash | Paper Table |
|---|-------|-----------|----------|-------------|-------------|
| 1 | [#179561](<upstream issue, anonymized>) | `pytorch/bug_004_bf16_cast_elided` | Fusion composition | Silent | Tab 4 row 1 |
| 2 | [#179931](<upstream issue, anonymized>) | `pytorch/bug_006_adaptive_avgpool_flatten_sum` | Fusion composition | Silent | Tab 4 row 2 |
| 3 | [#180164](<upstream issue, anonymized>) | `pytorch/bug_007_slice_scatter_backward` | Backward codegen | Silent | Tab 4 row 3 |
| 4 | [#180221](<upstream issue, anonymized>) | `pytorch/bug_011_cumsum_splitscan_crash` | Split-scan boundary | Crash | Tab 4 row 4 |
| 5 | [#180771](<upstream issue, anonymized>) | `pytorch/bug_013_slice_scatter_backward` | Backward codegen | Silent | Tab 4 row 5 |
| 6 | [#180772](<upstream issue, anonymized>) | `pytorch/bug_013v_slice_scatter_backward_variant` | Backward codegen | Silent | Tab 4 row 6 |
| 7 | [#181512](<upstream issue, anonymized>) | `pytorch/bug_020_inference_mode_batched_matmul` | Context/higher-order | Crash | Tab 4 row 7 |
| 8 | [#181581](<upstream issue, anonymized>) | `pytorch/bug_019_create_graph_requires_grad_dropped` | Context/higher-order | Silent | Tab 4 row 8 |
| 9 | [#181584](<upstream issue, anonymized>) | `pytorch/bug_015_associative_scan_dynamic` | Split-scan boundary | Crash | Tab 4 row 9 |
| 10 | [#182012](<upstream issue, anonymized>) | `pytorch/bug_022_diagonal_scatter_backward` | Backward codegen | Silent | Tab 4 row 10 |
| 11 | [#182217](<upstream issue, anonymized>) | `pytorch/bug_023_layer_norm_nestedtensor_guard_crash` | Context/higher-order | Crash | Tab 4 row 11 |
| 12 | [#183017](<upstream issue, anonymized>) | `pytorch/bug_029_while_loop_backward_stride` | Backward codegen | Crash | Tab 4 row 12 |
| 13 | [#183249](<upstream issue, anonymized>) | `pytorch/bug_032_frac_maxpool_random_compile` | Codegen/lowering | Silent | Tab 4 row 13 |
| 14 | [#183250](<upstream issue, anonymized>) | `pytorch/bug_031_cumprod_splitdim_bwd_crash` | Split-scan boundary | Crash | Tab 4 row 14 |
| 15 | [#183986](<upstream issue, anonymized>) | `pytorch/bug_033_index_family_expanded_wrong_result` | Scatter/indexing | Silent | Tab 4 row 15 |
| 16 | [#183987](<upstream issue, anonymized>) | `pytorch/bug_034_addcmul_addcdiv_inplace_bwd_crash` | Backward codegen | Crash | Tab 4 row 16 |
| 17 | [#183988](<upstream issue, anonymized>) | `pytorch/bug_036_randperm_index_pattern_crash` | Scatter/indexing | Crash | Tab 4 row 17 |
| 18 | [#184081](<upstream issue, anonymized>) | `pytorch/bug_061_float8_cast_arch_crash` | Codegen/lowering | Crash | Tab 4 row 18 |
| 19 | [#184089](<upstream issue, anonymized>) | `pytorch/bug_043_flip_0d_inductor_crash` | Codegen/lowering | Crash | Tab 4 row 19 |
| 20 | [#184100](<upstream issue, anonymized>) | `pytorch/bug_044_complex_scalar_backward_crash` | Backward codegen | Crash | Tab 4 row 20 |
| 21 | [#184101](<upstream issue, anonymized>) | `pytorch/bug_045_complex_abs_transpose_backward` | Backward codegen | Crash | Tab 4 row 21 |
| 22 | [#182200](<upstream issue, anonymized>) | — | Backward codegen | Crash | Appendix (Backward) |
| 23 | [#183015](<upstream issue, anonymized>) | — | Context/higher-order | Crash | Appendix (Context) |
| 24 | [#184222](<upstream issue, anonymized>) | `pytorch/bug_051_scheduler_prologue_fusion_typo` | Context/higher-order | Silent | Tab 4 row 22 |
| 25 | [#184088](<upstream issue, anonymized>) | `pytorch/bug_042_expand_backward_aot_crash` | Backward codegen | Crash | Tab 4 row 24 |
| 26 | [#184408](<upstream issue, anonymized>) | `pytorch/bug_053_div_trunc_approximate_division` (graph_exp) | Codegen/lowering | Silent | Appendix (Codegen) |
| 27 | [#184410](<upstream issue, anonymized>) | — | Codegen/lowering | Silent | Appendix (Codegen) |
| 28 | [#184415](<upstream issue, anonymized>) | `pytorch/bug_055_foreach_sub_alpha_ignored` | Codegen/lowering | Silent | Tab 4 row 23 |
| 29 | [#184416](<upstream issue, anonymized>) | `pytorch/bug_059_deterministic_extern_kernel` | Codegen/lowering | Silent | Tab 4 row 25 |
| 30 | [#184992](<upstream issue, anonymized>) | `pytorch/bug_063_cat_axis_sink_crash` | Codegen/lowering | Crash | Tab 4 row 27 |
| — | TBD | `pytorch/bug_062_var_norm_donated_buffer_create_graph` | Context/higher-order | Crash | — |

**Silent: 14, Crash: 16, Total: 30** (excluding TBD unfiled)

## TensorFlow (24 bugs)

| # | Issue | Directory | Status |
|---|-------|-----------|--------|
| 1 | [#116436](<upstream issue, anonymized>) | `tensorflow/tf_grappler_argmin_monotonic` | Awaiting PR merge |
| 2 | [#116943](<upstream issue, anonymized>) | `tensorflow/tf_selu_elu_expm1` | Awaiting PR merge |
| 3 | [#116945](<upstream issue, anonymized>) | `tensorflow/tf_complex_sign_eager` | Assigned |
| 4 | [#116992](<upstream issue, anonymized>) | `tensorflow/tf_xla_cpu_nan_minmax` | Awaiting PR merge |
| 5 | [#117319](<upstream issue, anonymized>) | `tensorflow/tf_check_numerics_xla_silent` | Assigned |
| 6 | [#117546](<upstream issue, anonymized>) | `tensorflow/tf_xla_cumsum_cumprod_complex_crash` | Assigned |
| 7 | [#117547](<upstream issue, anonymized>) | `tensorflow/tf_xla_argmin_dead_code_crash` | Assigned |
| 8 | [#117802](<upstream issue, anonymized>) | `tensorflow/tf_complex_pow_zero` | Confirmed |
| 9 | [#117803](<upstream issue, anonymized>) | `tensorflow/tf_resize_bilinear_nan_xla` | Confirmed |
| 10 | [#117804](<upstream issue, anonymized>) | `tensorflow/tf_searchsorted_nan_xla` | Fix PR in progress |
| 11 | [#117807](<upstream issue, anonymized>) | `tensorflow/tf_eigvalsh_large_xla` | Confirmed |
| 12 | [#117862](<upstream issue, anonymized>) | `tensorflow/tf_segment_min_max_nan_position` | Assigned |
| 13 | [#117864](<upstream issue, anonymized>) | `tensorflow/tf_eigvalsh_complex_xla` | Community reproduced |
| 14 | [#117962](<upstream issue, anonymized>) | `tensorflow/tf_fake_quant_nudge_rounding` | Assigned |
| 15 | [#118642](<upstream issue, anonymized>) | `tensorflow/tf_xla_floordiv_floormod_identity` | Assigned |
| 16 | [#118643](<upstream issue, anonymized>) | `tensorflow/tf_atan2_subnormal_nan` | Assigned |
| 17 | [#118644](<upstream issue, anonymized>) | `tensorflow/tf_rsqrt_subnormal_inf` | Assigned |
| 18 | [#118645](<upstream issue, anonymized>) | `tensorflow/tf_gpu_reciprocal_ftz` | Assigned |
| 19 | [#118647](<upstream issue, anonymized>) | `tensorflow/tf_xla_argmax_nan_nonpow2` | Assigned |
| 20 | [#118673](<upstream issue, anonymized>) | `tensorflow/tf_xla_reduce_max_grad_nan` | Assigned |
| 21 | [#118674](<upstream issue, anonymized>) | `tensorflow/tf_xla_where_gradient_dead_branch` | Assigned |
| 22 | [#118675](<upstream issue, anonymized>) | `tensorflow/tf_grappler_monotonic_saturation` | Assigned |
| 23 | [#118701](<upstream issue, anonymized>) | `tensorflow/tf_xla_fused_batchnorm_negative_variance` | Assigned |
| 24 | [#119476](<upstream issue, anonymized>) | `tensorflow/tf_xlogy_xlog1py_gradient_zero` (tf_cross) | `type:bug`, `comp:ops`, assigned @Venkat6871 |

**6 TF issues need local reproducers**: #117802, #117804, #117807, #117862, #117864, #117962.
All 6 have reproducers in the GitHub issues themselves; local scripts need to be extracted.

## XLA (1 bug)

| Issue | Directory | Status |
|-------|-----------|--------|
| [openxla/xla#41167](<upstream PR, anonymized>) | `xla/tf_fuzz_real_seed71684` | Merged |

## IREE (6 bugs)

| # | Issue | Directory |
|---|-------|-----------|
| 1 | [#24180](<upstream issue, anonymized>) | `iree/bug_006_f64_demote_buffer_mismatch` |
| 2 | [#24181](<upstream issue, anonymized>) | `iree/bug_002_reduceprod_keepdims0` |
| 3 | [#24228](<upstream issue, anonymized>) | `iree/iree_bug_004_sceloss_weight_mean` |
| 4 | [#24229](<upstream issue, anonymized>) | `iree/bug_005_reverse_sequence` |
| 5 | [#24230](<upstream issue, anonymized>) | `iree/iree_bug_006_unique_segfault` |
| 6 | [#24265](<upstream issue, anonymized>) | `iree/iree_bug_009_reduce_segfault` |

## TVM (10 bugs)

| # | Issue | Directory |
|---|-------|-----------|
| 1 | [#19419](<upstream issue, anonymized>) | `tvm/bug_001_gemv_keyerror` |
| 2 | [#19420](<upstream issue, anonymized>) | `tvm/bug_004_softmax_opset11` |
| 3 | [#19435](<upstream issue, anonymized>) | `tvm/bug_005_scatter_reduction` |
| 4 | [#19436](<upstream issue, anonymized>) | `tvm/bug_010_gather_negative_idx` |
| 5 | [#19437](<upstream issue, anonymized>) | `tvm/bug_011_cumsum_axis` |
| 6 | [#19451](<upstream issue, anonymized>) | `tvm/bug_018_scatter_elements_cuda` |
| 7 | [#19452](<upstream issue, anonymized>) | `tvm/bug_020_isnan_isinf_not_implemented` |
| 8 | [#19571](<upstream issue, anonymized>) | `tvm/bug_029_topk_crash` |
| 9 | [#19575](<upstream issue, anonymized>) | `tvm/bug_032_reorder_permute_concat_neg_axis` |
| 10 | [#19579](<upstream issue, anonymized>) | `tvm/bug_036_nan_propagation` |
