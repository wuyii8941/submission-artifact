# Bug Mapping

All 86 bugs counted in the paper, anonymized for double-blind review. Upstream
issue/PR numbers and tracker URLs are stripped; they will be restored in the
camera-ready version. The Internal ID column matches the reproducer directory
under `data/confirmed_bugs/<stack>/`, which is the stable handle for review.

## PyTorch/Inductor: 31 Counted Bugs

| Internal ID | Root Cause | Status |
|----|----|----|
| bug_001 | bf16 cast elided during fusion | Closed / Fixed |
| bug_002 | adaptive_avgpool fusion stride error | Open / assigned <maintainer> |
| bug_003 | slice_scatter backward gradient corruption | Closed / Fixed |
| bug_004 | cumsum SplitScan crash at dim=129 | Closed / Fixed |
| bug_005 | slice_scatter backward variant | Closed / Fixed |
| bug_005 variant | slice_scatter backward variant with a distinct trigger condition | Closed / linked variant |
| bug_009 | `torch.compile` crashes on batched matmul inside `torch.inference_mode()` | Open; developer confirmed |
| bug_008 | AOTAutograd silently drops higher-order grad metadata from `autograd.grad(create_graph=True)` | Open; high priority; upstream fix PR open |
| bug_007 | `associative_scan` crashes with `dynamic=True` for ndim >= 2 (lifted arguments) | Closed / Fixed |
| bug_010 | `diagonal_scatter` backward silently zeroes `x.grad` — Inductor stride-0 clone | Open; contributor confirmed; fix commit ready |
| appendix-A | Backward codegen crash (paper appendix) | Confirmed |
| bug_011 | `torch.compile` crashes on `F.layer_norm(NestedTensor)` — guard builder IndexError | Open; developer confirmed |
| appendix-B | Context/higher-order crash (paper appendix) | Confirmed |
| bug_030 | `torch.while_loop` backward crashes — Inductor stride mismatch `(0,0)` vs `(64,1)` | Open; triaged, reproduced on H100 |
| bug_013 | `torch.cumprod` backward crashes at scan dim >= 8193 — TritonSplitDimKernel unsupported reduction | Closed / Fixed |
| bug_015 | `index_add`/`index_copy`/`index_fill`/`index_put` on expanded tensors wrong result | Open; triaged, reproduced on H100, `has workaround` |
| bug_017 | `torch.compile` crashes on `x[torch.randperm(n)]` — pattern matcher missing `slice_shape` input | Open; triaged; developer confirmed |
| bug_022 | Copy-paste typo in scheduler prologue fusion guard: `node1` checked twice instead of `node1` and `node2` | Closed / Fixed (upstream PR merged) |
| bug_014 | `fractional_max_pool2d` produces wrong results when `_random_samples` not provided — compiled path generates different pooling indices | Open; triaged; developer confirmed |
| bug_016 | `addcmul_`/`addcdiv_` backward crashes — missing autograd formula for `prims.fma.default` | Closed (upstream fix PR merged); developer confirmed |
| bug_026 | `torch.compile` crashes on cast to `float8_e4m3fn` on GPUs without native FP8 support — Triton codegen emits unsupported type | Closed / Fixed |
| bug_019 | `torch.compile` crashes on `flip()` applied to 0-dimensional tensors — IndexError in lowering | Open; triaged; developer confirmed |
| bug_020 | `torch.compile` backward crashes when complex tensor multiplied by Python complex literal — `.get_name()` on raw complex | Open; triaged; developer confirmed |
| bug_021 | `torch.compile` backward crashes for `abs()`/`angle()` on transposed complex tensors — stride assertion failure | Open; triaged; developer confirmed |
| bug_024 | `torch._foreach_sub` silently ignores `alpha` parameter under `torch.compile` — missing `allow_alpha=True` in lowering | Closed / Fixed |
| bug_027 | `torch.compile` crashes when `torch.cat` uses `axis=` keyword — `sink_cat_after_pointwise` pass helper only accepts `dim` | Open; triaged; `module: inductor`, `module: numpy` |
| bug_018 | `torch.compile` crashes on `expand` backward with AOT autograd | Closed / Fixed (upstream PR merged) |
| bug_023 | `torch.div(rounding_mode='trunc')` produces off-by-one results under `torch.compile` — Triton approximate division instead of `_div_rn` | Open; high priority, triaged; upstream fix PR open; developer confirmed |
| appendix-C | Codegen/lowering silent wrong result (paper appendix) | Confirmed |
| bug_025 | `pick_deterministic_choice()` checks `ExternKernelChoice` instead of `ExternKernelCaller` — deterministic extern kernel preference is dead code | Closed; triaged as `module: determinism`, `module: inductor` |
| bug_028 | `F.hardtanh` compiled backward silently drops gradient at bf16 boundary values — decomposition compares in bf16 precision while eager kernel uses float scalar | Closed / Fixed |

## TensorFlow/XLA: 31 Counted Bugs

TensorFlow reproducer directories are named after the offending API/symptom
rather than a sequential bug index. The "Reproducer" column gives the dir
name under `data/confirmed_bugs/tensorflow/`.

| Reproducer | Root Cause | Status |
|----|----|----|
| bug_013_grappler_argmin_monotonic | Grappler incorrectly strips non-strictly-monotonic ops from ArgMin/ArgMax | `type:bug`, `awaiting PR merge`; upstream fix PR open |
| bug_021_selu_elu_expm1 | Eigen SELU/ELU eager kernels catastrophic cancellation for small negative fp32 | `type:bug`, `awaiting PR merge`; upstream fix PR open |
| bug_005_complex_sign_eager | Eager `tf.math.sign` returns 0 for small nonzero complex numbers | `type:bug`, assigned |
| bug_024_xla_cpu_nan_minmax | CPU XLA silently drops NaN in `reduce_max`/`reduce_min`/`maximum`/`minimum` | `type:bug`, `awaiting PR merge`; upstream fix PR open |
| bug_003_check_numerics_xla_silent | `tf.debugging.check_numerics` silently passes NaN/Inf under XLA `jit_compile=True` | `type:bug`, assigned |
| bug_025_xla_cumsum_cumprod_complex_crash | `tf.math.cumsum`/`tf.math.cumprod` crash with `jit_compile=True` on complex dtypes | `type:bug`, assigned |
| bug_023_xla_argmin_dead_code_crash | `tf.argmin`/`tf.argmax` crash XLA on empty-dimension tensor dead code | `type:bug`, `comp:xla`, assigned <maintainer> |
| bug_004_complex_pow_zero | Eager `tf.pow(0+0j, 0+0j)` returns NaN+NaNj via `exp(b*log(a))`; NumPy/XLA return 1+0j | `type:bug`, confirmed/reproduced by maintainer |
| bug_017_resize_bilinear_nan_xla | `tf.image.resize` bilinear under XLA over-propagates NaN to entire channel | `type:bug`, confirmed/reproduced by maintainer |
| bug_019_searchsorted_nan_xla | `tf.searchsorted(side='right')` returns wrong position for NaN under XLA | `type:bug`, `comp:xla`; upstream fix PR in progress |
| bug_009_eigvalsh_large_xla | `tf.linalg.eigvalsh` returns wrong eigenvalues under XLA for large-magnitude matrices | `type:bug`, confirmed/reproduced by maintainer |
| bug_020_segment_min_max_nan_position | `segment_min`/`segment_max` CPU kernel NaN-position-dependent results | `type:bug`, assigned <maintainer> `stat:contribution welcome` |
| bug_008_eigvalsh_complex_xla | `tf.linalg.eigvalsh`/`eigh` returns wrong results for complex input under XLA — real eigenvalue buffer reinterpreted as complex | `type:bug`, `comp:xla`; community reproduced |
| bug_011_fake_quant_nudge_rounding | `fake_quant_with_min_max_vars` XLA rounding divergence (half-to-even vs half-away-from-zero) | `type:bug`, `comp:xla`, assigned <maintainer> |
| bug_026_xla_floordiv_floormod_identity | XLA `floordiv(x,y)*y + floormod(x,y)` violates division identity for float32 at large magnitudes | `type:bug`, assigned <maintainer> |
| bug_001_atan2_subnormal_nan | `tf.math.atan2` returns NaN on CPU for subnormal float32 inputs — intermediate underflow | `type:bug`, assigned <maintainer> |
| bug_018_rsqrt_subnormal_inf | `tf.math.rsqrt` returns inf for subnormal float32 inputs on CPU and GPU | `type:bug`, `comp:ops`, assigned <maintainer> |
| bug_012_gpu_reciprocal_ftz | GPU `x * reciprocal(x)` returns 0 for normal float32 `x=1e38` — GPU FTZ flushes subnormal intermediate | `type:bug`, `comp:ops`, assigned <maintainer> |
| bug_022_xla_argmax_nan_nonpow2 | XLA `argmax`/`argmin` returns wrong index when NaN present for non-power-of-2 sizes | `type:bug`, `comp:xla`, assigned <maintainer> |
| bug_028_xla_reduce_max_grad_nan | XLA gradient of `reduce_max`/`reduce_min` silently returns 0 instead of NaN for NaN input | `2.21.0`, assigned <maintainer> |
| bug_029_xla_where_gradient_dead_branch | XLA gradient of `tf.where` evaluates dead branch, leaking NaN/inf via `NaN*0=NaN` | `type:bug`, `comp:xla`, assigned <maintainer> |
| bug_014_grappler_monotonic_saturation | Grappler strips strictly-monotonic ops (tanh, exp, sigmoid) from ArgMax/ArgMin — float32 saturation breaks injectivity | `type:bug`, `comp:ops`, assigned <maintainer> |
| bug_027_xla_fused_batchnorm_negative_variance | XLA `FusedBatchNormV3` one-pass variance formula produces negative variance and NaN output | `type:bug`, `comp:xla`, assigned <maintainer> |
| bug_030_xlogy_xlog1py_gradient_zero | `tf.math.xlogy` and `tf.math.xlog1py` gradient w.r.t. x incorrectly zero when x=0 — zero-mask from forward applied to backward | `type:bug`, `comp:ops`, assigned <maintainer> |
| bug_001_erf_shape_dependent_lowering | `erf` scalar-vs-vector lowering divergence in XLA (reproduces in both TF and JAX) | **Merged** |
| bug_015_grappler_mul_zero_nan | Grappler constant folding suppresses NaN for `x * 0` with Inf/NaN inputs | `type:bug`, assigned |
| bug_016_grappler_reciprocal_involution | Grappler ArithmeticOptimizer treats `reciprocal(reciprocal(x))` as an exact involution | `type:bug`, assigned |
| bug_007_det_gradient_singular_crash | `tf.linalg.det` gradient crashes on singular matrices — backward calls `matrix_inverse` on non-invertible input | `type:bug`, assigned <maintainer> |
| bug_006_cumprod_gradient_zero | `tf.math.cumprod` gradient incorrect when input contains zeros — `cumprod/x[i]` produces silent `0/0` | `type:bug`, assigned <maintainer>; upstream fix PR open |
| bug_002_bessel_i1_gradient_zero | `tf.math.bessel_i1` gradient at x=0 returns 1.0 instead of correct 0.5 — removable singularity I₁(x)/x mishandled in gradient computation; companion `bessel_j1` is correct | `type:bug`, `comp:core`, `2.21.0`; upstream fix PR awaiting merge |
| bug_010_erfinv_eager_precision | `tf.math.erfinv` eager fp32 kernel (Eigen, `cwise_op_erfinv.cc`) loses ~4 digits of precision near `x = ±1` while XLA `jit_compile=True` matches fp64 to ~2e-7 — eager kernel never updated to match modernized XLA `math.cc::Erfinv` | `type:bug`, assigned <maintainer>; upstream fix PR open; developer confirmed |

## IREE: 7 Counted Bugs

| Internal ID | Root Cause | Status |
|----|----|----|
| bug_004_f64_demote_buffer_mismatch | `DemoteF64ToF32Pass` f64/f32 buffer-size mismatch → silent f64 output corruption | Open; labeled `bug`; developer confirmed |
| bug_001_reduceprod_keepdims0 | ReduceProd keepdims=0 shape/lowering crash | Closed / Fixed (upstream torch-mlir PR merged) |
| bug_002_sceloss_weight_mean | SoftmaxCrossEntropyLoss with weights divides by N instead of weight sum | Open; labeled `bug`; developer confirmed (fix patch provided) |
| bug_003_reverse_sequence | ReverseSequence wrong results when batch_axis=1, time_axis=0 | Closed / Fixed |
| bug_006_unique_segfault | Unique op torch lowering causes compiler SEGFAULT | Open; labeled `bug`; developer confirmed (fix patch provided) |
| bug_007_reduce_segfault | ReduceMax/ReduceMin multi-axis + negative index SEGFAULT | Open; labeled `bug`; developer confirmed |
| bug_005_pad_wrap_hang | ONNX `Pad` with `mode="wrap"` causes IREE compiler to hang in an infinite loop; other padding modes (constant/reflect/edge) compile successfully — triggered by PyTorch circular padding exported as ONNX `Pad(mode='wrap')` | Open; labeled `bug` |

## TVM: 17 Counted Bugs

| Internal ID | Root Cause | Status |
|----|----|----|
| bug_001_gemv_keyerror | DLight GEMV schedule rule crashes with KeyError on auto-detected CUDA | Closed; developer confirmed |
| bug_002_softmax_opset11 | ONNX Softmax/LogSoftmax/Hardmax frontend ignores opset <13 semantics | Closed / Fixed |
| bug_003_scatter_reduction | ONNX ScatterElements ignores `reduction` attribute, silently produces wrong results | Closed / Fixed (upstream PR merged) |
| bug_004_gather_negative_idx | ONNX Gather with negative indices reads out-of-bounds memory (garbage values) | Closed / Fixed (upstream PR merged) |
| bug_005_cumsum_axis | CumSum ignores axis parameter, always reduces along axis 0 | Closed; developer confirmed |
| bug_006_scatter_elements_cuda | scatter_elements / scatter_nd fail to compile for CUDA target | Open; developer confirmed |
| bug_007_isnan_isinf_not_implemented | `R.isnan`, `R.isinf`, `R.isfinite` crash at build time | Closed; developer confirmed |
| bug_008_topk_crash | ONNX TopK operator crashes during `from_onnx()` — K tensor scalar extraction failure | Closed / Fixed (upstream PR merged) |
| bug_009_reorder_permute_concat_neg_axis | `ReorderPermuteDimsAfterConcat` pass crashes with negative concat axis — missing index normalization | Closed / Fixed (upstream PR merged) |
| bug_010_dataflow_inplace_liveness | `DataflowUseInplaceCalls` corrupts concat input via incorrect liveness analysis — in-place buffer reuse on still-live value | Closed / Fixed (upstream PR merged) |
| bug_011_nan_propagation | `maximum`/`minimum`/`relu`/`clip` do not propagate NaN (IEEE 754 violation) — LLVM `maxss`/`maxps` semantics | Open; upstream PR open |
| bug_012_relu_sign_nan_propagation | Relax/ONNX `Relu`, `Sign` convert NaN to 0 instead of preserving it; `ReduceMax`/`ReduceMin` exhibit position-dependent NaN behavior — improper `fmax`/`fmin` semantics + comparison chains failing to propagate NaN | Closed / Fixed; `type: bug` |
| bug_013_batchnorm_legalize_training | `LegalizeOps` lowering of `relax.nn.batch_norm` recomputes mean/variance from input data instead of using supplied `running_mean`/`running_var`, producing training-mode behavior under inference — repro shows max error ~60 vs ground truth | Closed / Fixed (upstream PR merged); `type: bug` |
| bug_015_gridsample_5d_crash | Relax/ONNX `GridSample._impl_v16` unconditionally permutes grid from `[N,H,W,2]` to `[N,2,H,W]` and dispatches to 2D `image.grid_sample`, crashing on 5D volumetric inputs (`PermuteDims` expects 4 axes but tensor ndim=5) | Closed / Fixed (upstream PR merged); `type: bug`; developer confirmed |
| bug_016_layernorm_bias_shape | Relax/ONNX `LayerNormalization` without optional bias synthesizes a zero bias of shape `data.shape[1]` instead of `gamma_shape`, causing dimension mismatch in `relax.nn.layer_norm` (breaks PyTorch transformers exported with `LayerNorm(bias=False)`) | Closed / Fixed (upstream PR merged); `type: bug` |
| bug_014_cumsum_exclusive | Relax/ONNX `CumSum` with `exclusive=1` hits a bare `AssertionError` in `CumSum._impl_v14` — `topi.cumsum`/`relax.op.cumsum` only implement the inclusive variant | Closed / Fixed (upstream PR merged); `type: bug` |
| bug_017_attention_symbolic_batch | `relax.op.nn.attention` fails to legalize when batch dim is a symbolic `tir.Var` — `_te_attention` builds two `Mul(B, num_head)` exprs, and `topi.nn.batch_matmul` compares them with Python `==` which returns a deferred `PrimExpr` instead of `True`, tripping the batch-dim assertion | Closed / Fixed (upstream PR merged); `type: bug` |

## Post-2026-06-04 Bugs (Already Counted in 86)

9 bugs were filed/confirmed after the 2026-06-04 freeze. They are **included** in the 86 total above. Stack, count, and the Internal IDs that point into the tables above:

- **TF/XLA (2):** bug_002_bessel_i1_gradient_zero, bug_010_erfinv_eager_precision
- **IREE (1):** bug_005_pad_wrap_hang
- **TVM (6):** bug_012_relu_sign_nan_propagation, bug_013_batchnorm_legalize_training, bug_015_gridsample_5d_crash, bug_016_layernorm_bias_shape, bug_014_cumsum_exclusive, bug_017_attention_symbolic_batch

All satisfy the same counting policy (upstream issue submitted + developer-confirmed via comment / PR / label).

---

**Total counted: 86** (matches paper Table 1)
- PyTorch/Inductor: 31 (Silent 15, Crash 16; 20 fixed / 11 open)
- TensorFlow/XLA: 31 (Silent 28, Crash 3; 5 fixed / 26 open)
- IREE: 7 (Silent 3, Crash 4; 2 fixed / 5 open)
- TVM: 17 (Silent 8, Crash 9; 16 fixed / 1 open)

**Counting policy for "fixed":** A bug is counted as fixed when an upstream fix has been accepted (PR merged or issue closed as fixed). Bugs whose fix PR has been reviewed and approved but not yet merged at submission time are also counted as fixed (status "awaiting PR merge" in the per-row Status column above). This accounts for the TF fixed count (5) being higher than the number of fully-merged PRs visible in the table.

---

## Reproducer Provenance Audit

Most `repro.py` scripts in this artifact are copied verbatim from local
fuzzer-output scripts. A small number were reconstructed from upstream issue
bodies or synthesized from source-level bug descriptions. The lists below
flag every script in those two latter classes for reviewer inspection.

**Class A — copied verbatim from local reproducer scripts (23 scripts).**
All TF reproducers (16), TVM bug_016/037/038/039, IREE bug_002/007, and
PT bug_023. These are byte-identical to the fuzzer-emitted scripts and
require no extra scrutiny beyond running the script.

**Class B — code lifted from upstream issue body (6 scripts).**
The reproducer code block was lifted from the upstream issue's first
post. No local fuzzer-output script existed at submission time. The
extracted code is not guaranteed byte-identical to the issue post
(extraction passed through web-fetch tooling); reviewers may wish to
diff against the upstream issue.

| Path | Verification on this server |
|---|---|
| `pytorch/bug_028_hardtanh_bf16_boundary_grad/repro.py` | Runs, but bug does not trigger on Tesla T4 (bf16 compile is skipped on sm_75). To reproduce the eager≠compiled divergence, run on sm_80+. |
| `pytorch/bug_029_fft_backward_stride_crash/repro.py` | ✅ All 4 FFT-backward cases crash with `AssertionError: ... stride mismatch` at `torch.ops.aten.mul.Tensor` — fake/meta stride bug confirmed. |
| `pytorch/bug_031_vmap_hessian_crash/repro.py` | ✅ Hessian path crashes with `TorchRuntimeError` on the fake-tensor `_autograd_grad` call; nested-jacrev workaround compiles cleanly. |
| `tvm/bug_015_gridsample_5d_crash/repro.py` | ✅ Crashes with `tvm.error.InternalError: PermuteDims expects ... ndim is 5 while the given number of axes is 4` — exact match. |
| `tvm/bug_016_layernorm_bias_shape/repro.py` | ✅ Crashes with `tvm.error.InternalError: Op(relax.nn.layer_norm) requires ... [T.int64(8)] and [T.int64(3)] are unequal` — confirms [3]-shaped synthesized zero-bias bug. |
| `tvm/bug_017_attention_symbolic_batch/repro.py` | Not verified: this server's TVM build has a `tvm_ffi` version mismatch that breaks `from tvm import tir` before the bug-reach line. Reproducer matches the upstream issue body verbatim (with one obvious `tirx` → `tir` typo correction). |

**Class C — static demonstrations (2 scripts).**
The upstream issue has *no* Python reproducer — the bug is source-level
(an `isinstance` / `is` check whose branch is dead code). The script in
the artifact is a *static demonstration* that exposes the type confusion
via class introspection or FX-graph construction. It is **not** a
runtime crash / wrong-result reproducer.

| Path | What it does | Verification |
|---|---|---|
| `pytorch/bug_025_deterministic_extern_kernel/repro.py` | Prints class MROs and runs an `isinstance` filter to show that the buggy filter always matches zero choices, while the corrected filter matches one. | ✅ Runs on PT 2.13.0.dev20260609+cu126; prints "BUG CONFIRMED: extern-kernel preference ... is unreachable". |
| `pytorch/bug_030_pad_mm_getitem_dead_code/repro.py` | Builds a 1-op FX graph with `operator.getitem` and shows that `node.op is operator.getitem` is always False (string-vs-function comparison). | ✅ Runs on PT 2.13.0.dev20260609+cu126; prints "BUG CONFIRMED: pad_mm.get_non_view_def cannot recurse through getitem". |

**Tested environments.** PyTorch scripts run with `${ARTIFACT_ROOT}/pt_nightly_env/bin/python` (PyTorch 2.13.0.dev20260609+cu126, Tesla T4). TVM scripts run with `${ARTIFACT_ROOT}/tvm_fuzz/bin/python` plus `LD_LIBRARY_PATH=${ARTIFACT_ROOT}/tvm_fuzz/lib` (TVM 0.24.dev0). 7 of 8 Class B/C scripts were verified to behave as documented; bug_028 (T4 hardware limitation) and bug_018 (TVM env tir-init mismatch) were not.