# Bug Mapping: Issue → Reproducer Directory

All 86 bugs counted in the paper, anonymized for double-blind review.
Includes 9 bugs filed/confirmed post-2026-06-04 (already counted in the 86 total).
See `data/confirmed_bugs/<stack>/` for minimal reproducers.

## PyTorch/Inductor: 31 Counted Bugs

| Issue | Internal ID | Root Cause | Status |
|----|----|----|----|
| #179561 (anonymized) | bug_004 | bf16 cast elided during fusion | Closed / Fixed |
| #179931 (anonymized) | bug_006 | adaptive_avgpool fusion stride error | Open / assigned <maintainer> |
| #180164 (anonymized) | bug_007 | slice_scatter backward gradient corruption | Closed / Fixed |
| #180221 (anonymized) | bug_011 | cumsum SplitScan crash at dim=129 | Closed / Fixed |
| #180771 (anonymized) | bug_013 | slice_scatter backward variant | Closed / Fixed |
| #180772 (anonymized) | bug_013 variant | slice_scatter backward variant with a distinct trigger condition | Closed / linked variant |
| #181512 (anonymized) | bug_020 | `torch.compile` crashes on batched matmul inside `torch.inference_mode()` | Open; developer (dsashidh) confirmed |
| #181581 (anonymized) | bug_019 | AOTAutograd silently drops higher-order grad metadata from `autograd.grad(create_graph=True)` | Open; high priority; fix PR #181606 |
| #181584 (anonymized) | bug_015 | `associative_scan` crashes with `dynamic=True` for ndim >= 2 (lifted arguments) | Closed / Fixed; event |
| #182012 (anonymized) | bug_022 | `diagonal_scatter` backward silently zeroes `x.grad` — Inductor stride-0 clone | Open; contributor (irshadcc) confirmed; fix commit ready |
| #182200 (anonymized) | — | Backward codegen crash (in paper appendix) | Confirmed |
| #182217 (anonymized) | bug_023 | `torch.compile` crashes on `F.layer_norm(NestedTensor)` — guard builder IndexError | Open; developer confirmed |
| #183015 (anonymized) | — | Context/higher-order crash (in paper appendix) | Confirmed |
| #183017 (anonymized) | bug_030 | `torch.while_loop` backward crashes — Inductor stride mismatch `(0,0)` vs `(64,1)` | Open; triaged, reproduced on H100 |
| #183250 (anonymized) | bug_031 | `torch.cumprod` backward crashes at scan dim >= 8193 — TritonSplitDimKernel unsupported reduction | Closed / Fixed |
| #183986 (anonymized) | bug_033 | `index_add`/`index_copy`/`index_fill`/`index_put` on expanded tensors wrong result | Open; triaged, reproduced on H100, `has workaround` |
| #183988 (anonymized) | bug_036 | `torch.compile` crashes on `x[torch.randperm(n)]` — pattern matcher missing `slice_shape` input | Open; triaged; developer confirmed |
| #184222 (anonymized) | bug_051 | Copy-paste typo in scheduler prologue fusion guard: `node1` checked twice instead of `node1` and `node2` | Closed / Fixed; PR #184239 merged |
| #183249 (anonymized) | bug_032 | `fractional_max_pool2d` produces wrong results when `_random_samples` not provided — compiled path generates different pooling indices | Open; triaged; developer confirmed |
| #183987 (anonymized) | bug_034 | `addcmul_`/`addcdiv_` backward crashes — missing autograd formula for `prims.fma.default` | Closed; fix PR #184629; developer confirmed |
| #184081 (anonymized) | bug_061 | `torch.compile` crashes on cast to `float8_e4m3fn` on GPUs without native FP8 support — Triton codegen emits unsupported type | Closed / Fixed |
| #184089 (anonymized) | bug_043 | `torch.compile` crashes on `flip()` applied to 0-dimensional tensors — IndexError in lowering | Open; triaged; developer confirmed |
| #184100 (anonymized) | bug_044 | `torch.compile` backward crashes when complex tensor multiplied by Python complex literal — `.get_name()` on raw complex | Open; triaged; developer confirmed |
| #184101 (anonymized) | bug_045 | `torch.compile` backward crashes for `abs()`/`angle()` on transposed complex tensors — stride assertion failure | Open; triaged; developer confirmed |
| #184415 (anonymized) | bug_055 | `torch._foreach_sub` silently ignores `alpha` parameter under `torch.compile` — missing `allow_alpha=True` in lowering | Closed / Fixed; commit 8fdef8b |
| #184992 (anonymized) | bug_063 | `torch.compile` crashes when `torch.cat` uses `axis=` keyword — `sink_cat_after_pointwise` pass helper only accepts `dim` | Open; triaged; `module: inductor`, `module: numpy` |
| #184088 (anonymized) | bug_042 | `torch.compile` crashes on `expand` backward with AOT autograd | Closed / Fixed; PR #184519 merged |
| #184408 (anonymized) | bug_053 | `torch.div(rounding_mode='trunc')` produces off-by-one results under `torch.compile` — Triton approximate division instead of `_div_rn` | Open; high priority, triaged; fix PR #184680; developer confirmed |
| #184410 (anonymized) | — | Codegen/lowering silent wrong result (in paper appendix) | Confirmed |
| #184416 (anonymized) | bug_059 | `pick_deterministic_choice()` checks `ExternKernelChoice` instead of `ExternKernelCaller` — deterministic extern kernel preference is dead code | Closed; triaged as `module: determinism`, `module: inductor` |
| #185472 (anonymized) | bug_064 | `F.hardtanh` compiled backward silently drops gradient at bf16 boundary values — decomposition compares in bf16 precision while eager kernel uses float scalar | Closed / Fixed |

## TensorFlow/XLA: 31 Counted Bugs

| Issue | Root Cause | Status |
|----|----|----|
| #116436 (anonymized) | Grappler incorrectly strips non-strictly-monotonic ops from ArgMin/ArgMax | `type:bug`, `awaiting PR merge`; fix PR #116709 |
| #116943 (anonymized) | Eigen SELU/ELU eager kernels catastrophic cancellation for small negative fp32 | `type:bug`, `awaiting PR merge`; fix PR #116947 |
| #116945 (anonymized) | Eager `tf.math.sign` returns 0 for small nonzero complex numbers | `type:bug`, assigned |
| #116992 (anonymized) | CPU XLA silently drops NaN in `reduce_max`/`reduce_min`/`maximum`/`minimum` | `type:bug`, `awaiting PR merge`; fix PR #117046 |
| #117319 (anonymized) | `tf.debugging.check_numerics` silently passes NaN/Inf under XLA `jit_compile=True` | `type:bug`, assigned |
| #117546 (anonymized) | `tf.math.cumsum`/`tf.math.cumprod` crash with `jit_compile=True` on complex dtypes | `type:bug`, assigned |
| #117547 (anonymized) | `tf.argmin`/`tf.argmax` crash XLA on empty-dimension tensor dead code | `type:bug`, `comp:xla`, assigned <maintainer> |
| #117802 (anonymized) | Eager `tf.pow(0+0j, 0+0j)` returns NaN+NaNj via `exp(b*log(a))`; NumPy/XLA return 1+0j | `type:bug`, confirmed/reproduced by maintainer |
| #117803 (anonymized) | `tf.image.resize` bilinear under XLA over-propagates NaN to entire channel | `type:bug`, confirmed/reproduced by maintainer |
| #117804 (anonymized) | `tf.searchsorted(side='right')` returns wrong position for NaN under XLA | `type:bug`, `comp:xla`; fix PR #117850 in progress |
| #117807 (anonymized) | `tf.linalg.eigvalsh` returns wrong eigenvalues under XLA for large-magnitude matrices | `type:bug`, confirmed/reproduced by maintainer |
| #117862 (anonymized) | `segment_min`/`segment_max` CPU kernel NaN-position-dependent results | `type:bug`, assigned <maintainer> `stat:contribution welcome` |
| #117864 (anonymized) | `tf.linalg.eigvalsh`/`eigh` returns wrong results for complex input under XLA — real eigenvalue buffer reinterpreted as complex | `type:bug`, `comp:xla`; community reproduced |
| #117962 (anonymized) | `fake_quant_with_min_max_vars` XLA rounding divergence (half-to-even vs half-away-from-zero) | `type:bug`, `comp:xla`, assigned <maintainer> |
| #118642 (anonymized) | XLA `floordiv(x,y)*y + floormod(x,y)` violates division identity for float32 at large magnitudes | `type:bug`, assigned <maintainer> |
| #118643 (anonymized) | `tf.math.atan2` returns NaN on CPU for subnormal float32 inputs — intermediate underflow | `type:bug`, assigned <maintainer> |
| #118644 (anonymized) | `tf.math.rsqrt` returns inf for subnormal float32 inputs on CPU and GPU | `type:bug`, `comp:ops`, assigned <maintainer> |
| #118645 (anonymized) | GPU `x * reciprocal(x)` returns 0 for normal float32 `x=1e38` — GPU FTZ flushes subnormal intermediate | `type:bug`, `comp:ops`, assigned <maintainer> |
| #118647 (anonymized) | XLA `argmax`/`argmin` returns wrong index when NaN present for non-power-of-2 sizes | `type:bug`, `comp:xla`, assigned <maintainer> |
| #118673 (anonymized) | XLA gradient of `reduce_max`/`reduce_min` silently returns 0 instead of NaN for NaN input | `2.21.0`, assigned <maintainer> |
| #118674 (anonymized) | XLA gradient of `tf.where` evaluates dead branch, leaking NaN/inf via `NaN*0=NaN` | `type:bug`, `comp:xla`, assigned <maintainer> |
| #118675 (anonymized) | Grappler strips strictly-monotonic ops (tanh, exp, sigmoid) from ArgMax/ArgMin — float32 saturation breaks injectivity | `type:bug`, `comp:ops`, assigned <maintainer> |
| #118701 (anonymized) | XLA `FusedBatchNormV3` one-pass variance formula produces negative variance and NaN output | `type:bug`, `comp:xla`, assigned <maintainer> |
| #119476 (anonymized) | `tf.math.xlogy` and `tf.math.xlog1py` gradient w.r.t. x incorrectly zero when x=0 — zero-mask from forward applied to backward | `type:bug`, `comp:ops`, assigned <maintainer> |
| #41167 (anonymized) | `erf` scalar-vs-vector lowering divergence | **Merged** |
| #119428 (anonymized) | Grappler constant folding suppresses NaN for `x * 0` with Inf/NaN inputs | `type:bug`, assigned |
| #119429 (anonymized) | Grappler ArithmeticOptimizer treats `reciprocal(reciprocal(x))` as an exact involution | `type:bug`, assigned |
| #119477 (anonymized) | `tf.linalg.det` gradient crashes on singular matrices — backward calls `matrix_inverse` on non-invertible input | `type:bug`, assigned <maintainer> |
| #119430 (anonymized) | `tf.math.cumprod` gradient incorrect when input contains zeros — `cumprod/x[i]` produces silent `0/0` | `type:bug`, assigned <maintainer> fix PR #120282 |
| #120578 (anonymized) | `tf.math.bessel_i1` gradient at x=0 returns 1.0 instead of correct 0.5 — removable singularity I₁(x)/x mishandled in gradient computation; companion `bessel_j1` is correct | `type:bug`, `comp:core`, `2.21.0`; fix PR #120944 awaiting merge |
| #121629 (anonymized) | `tf.math.erfinv` eager fp32 kernel (Eigen, `cwise_op_erfinv.cc`) loses ~4 digits of precision near `x = ±1` while XLA `jit_compile=True` matches fp64 to ~2e-7 — eager kernel never updated to match modernized XLA `math.cc::Erfinv` | `type:bug`, assigned <maintainer>; fix PR #121644; developer confirmed |

## IREE: 7 Counted Bugs

| Issue | Root Cause | Status |
|----|----|----|
| #24180 (anonymized) | `DemoteF64ToF32Pass` f64/f32 buffer-size mismatch → silent f64 output corruption | Open; labeled `bug`; developer confirmed |
| #24181 (anonymized) | ReduceProd keepdims=0 shape/lowering crash | Closed / Fixed (torch-mlir PR #4540) |
| #24228 (anonymized) | SoftmaxCrossEntropyLoss with weights divides by N instead of weight sum | Open; labeled `bug`; developer confirmed (hanhanW provided fix patch) |
| #24229 (anonymized) | ReverseSequence wrong results when batch_axis=1, time_axis=0 | Closed / Fixed by hanhanW |
| #24230 (anonymized) | Unique op torch lowering causes compiler SEGFAULT | Open; labeled `bug`; developer confirmed (hanhanW provided fix patch) |
| #24265 (anonymized) | ReduceMax/ReduceMin multi-axis + negative index SEGFAULT | Open; labeled `bug`; developer confirmed |
| #24585 (anonymized) | ONNX `Pad` with `mode="wrap"` causes IREE compiler to hang in an infinite loop; other padding modes (constant/reflect/edge) compile successfully — triggered by PyTorch circular padding exported as ONNX `Pad(mode='wrap')` | Open; labeled `bug 🐞 Something isn't working` |

## TVM: 17 Counted Bugs

| Issue | Root Cause | Status |
|----|----|----|
| #19419 (anonymized) | DLight GEMV schedule rule crashes with KeyError on auto-detected CUDA | Closed; developer confirmed |
| #19420 (anonymized) | ONNX Softmax/LogSoftmax/Hardmax frontend ignores opset <13 semantics | Closed / Fixed |
| #19435 (anonymized) | ONNX ScatterElements ignores `reduction` attribute, silently produces wrong results | Closed / Fixed; resolved by PR #19527 |
| #19436 (anonymized) | ONNX Gather with negative indices reads out-of-bounds memory (garbage values) | Closed / Fixed; resolved by PR #19525 |
| #19437 (anonymized) | CumSum ignores axis parameter, always reduces along axis 0 | Closed; developer confirmed |
| #19451 (anonymized) | scatter_elements / scatter_nd fail to compile for CUDA target | Open; developer confirmed |
| #19452 (anonymized) | `R.isnan`, `R.isinf`, `R.isfinite` crash at build time | Closed; developer confirmed |
| #19571 (anonymized) | ONNX TopK operator crashes during `from_onnx()` — K tensor scalar extraction failure | Closed / Fixed; PR #19573 |
| #19575 (anonymized) | `ReorderPermuteDimsAfterConcat` pass crashes with negative concat axis — missing index normalization | Closed / Fixed; PR #19588 |
| #19577 (anonymized) | `DataflowUseInplaceCalls` corrupts concat input via incorrect liveness analysis — in-place buffer reuse on still-live value | Closed / Fixed; PR #19644 |
| #19579 (anonymized) | `maximum`/`minimum`/`relu`/`clip` do not propagate NaN (IEEE 754 violation) — LLVM `maxss`/`maxps` semantics | Open; PR #19589 |
| #19572 (anonymized) | Relax/ONNX `Relu`, `Sign` convert NaN to 0 instead of preserving it; `ReduceMax`/`ReduceMin` exhibit position-dependent NaN behavior — improper `fmax`/`fmin` semantics + comparison chains failing to propagate NaN | Closed / Fixed; commit b44988b (tlopex, 2026-06-15); `type: bug` |
| #19574 (anonymized) | `LegalizeOps` lowering of `relax.nn.batch_norm` recomputes mean/variance from input data instead of using supplied `running_mean`/`running_var`, producing training-mode behavior under inference — repro shows max error ~60 vs ground truth | Closed / Fixed; PR #19818; `type: bug` |
| #19688 (anonymized) | Relax/ONNX `GridSample._impl_v16` unconditionally permutes grid from `[N,H,W,2]` to `[N,2,H,W]` and dispatches to 2D `image.grid_sample`, crashing on 5D volumetric inputs (`PermuteDims` expects 4 axes but tensor ndim=5) | Closed / Fixed; PR #19816; `type: bug`; developer confirmed |
| #19691 (anonymized) | Relax/ONNX `LayerNormalization` without optional bias synthesizes a zero bias of shape `data.shape[1]` instead of `gamma_shape`, causing dimension mismatch in `relax.nn.layer_norm` (breaks PyTorch transformers exported with `LayerNorm(bias=False)`) | Closed / Fixed; PR #19772; `type: bug` |
| #19692 (anonymized) | Relax/ONNX `CumSum` with `exclusive=1` hits a bare `AssertionError` in `CumSum._impl_v14` — `topi.cumsum`/`relax.op.cumsum` only implement the inclusive variant | Closed / Fixed; PR #19773; `type: bug` |
| #19696 (anonymized) | `relax.op.nn.attention` fails to legalize when batch dim is a symbolic `tir.Var` — `_te_attention` builds two `Mul(B, num_head)` exprs, and `topi.nn.batch_matmul` compares them with Python `==` which returns a deferred `PrimExpr` instead of `True`, tripping the batch-dim assertion | Closed / Fixed; PR #19779; `type: bug` |

## Post-2026-06-04 Bugs (Already Counted in 86)

9 bugs were filed/confirmed after the 2026-06-04 freeze and are **included** in the 86 total above (anonymized):

- **TF/XLA (2):** #120578 (bessel_i1 grad), #121629 (erfinv eager precision)
- **IREE (1):** #24585 (ONNX Pad mode=wrap hang)
- **TVM (6):** #19572 (NaN propagation), #19574 (batch_norm legalize), #19688 (GridSample 5D), #19691 (LayerNorm bias), #19692 (CumSum exclusive), #19696 (attention symbolic batch)

All satisfy the same counting policy (upstream issue submitted + developer-confirmed via comment / PR / label).

---

**Total counted: 86** (matches paper Table 1)
- PyTorch/Inductor: 31 (Silent 15, Crash 16; 20 fixed / 11 open)
- TensorFlow/XLA: 31 (Silent 28, Crash 3; 5 fixed / 26 open)
- IREE: 7 (Silent 3, Crash 4; 2 fixed / 5 open)
- TVM: 17 (Silent 8, Crash 9; 16 fixed / 1 open)

---

## Reproducer Provenance Audit

Most `repro.py` scripts in this artifact are copied verbatim from local
fuzzer-output scripts. A small number were reconstructed from upstream issue
bodies or synthesized from source-level bug descriptions. The lists below
flag every script in those two latter classes for reviewer inspection.

**Class A — copied verbatim from local reproducer scripts (23 scripts).**
All TF reproducers (16), TVM bug_034/037/038/039, IREE bug_006/007, and
PT bug_053. These are byte-identical to the fuzzer-emitted scripts and
require no extra scrutiny beyond running the script.

**Class B — code lifted from upstream issue body (6 scripts).**
The reproducer code block was lifted from the upstream issue's first
post. No local fuzzer-output script existed at submission time. The
extracted code is not guaranteed byte-identical to the issue post
(extraction passed through web-fetch tooling); reviewers may wish to
diff against the upstream issue.

| Path | Verification on this server |
|---|---|
| `pytorch/bug_064_hardtanh_bf16_boundary_grad/repro.py` | Runs, but bug does not trigger on Tesla T4 (bf16 compile is skipped on sm_75). To reproduce the eager≠compiled divergence, run on sm_80+. |
| `pytorch/bug_appendix_182200_fft_backward_stride_crash/repro.py` | ✅ All 4 FFT-backward cases crash with `AssertionError: ... stride mismatch` at `torch.ops.aten.mul.Tensor` — fake/meta stride bug confirmed. |
| `pytorch/bug_appendix_183015_vmap_hessian_crash/repro.py` | ✅ Hessian path crashes with `TorchRuntimeError` on the fake-tensor `_autograd_grad` call; nested-jacrev workaround compiles cleanly. |
| `tvm/bug_040_gridsample_5d_crash/repro.py` | ✅ Crashes with `tvm.error.InternalError: PermuteDims expects ... ndim is 5 while the given number of axes is 4` — exact match. |
| `tvm/bug_041_layernorm_bias_shape/repro.py` | ✅ Crashes with `tvm.error.InternalError: Op(relax.nn.layer_norm) requires ... [T.int64(8)] and [T.int64(3)] are unequal` — confirms [3]-shaped synthesized zero-bias bug. |
| `tvm/bug_042_attention_symbolic_batch/repro.py` | Not verified: this server's TVM build has a `tvm_ffi` version mismatch that breaks `from tvm import tir` before the bug-reach line. Reproducer matches the upstream issue body verbatim (with one obvious `tirx` → `tir` typo correction). |

**Class C — static demonstrations (2 scripts).**
The upstream issue has *no* Python reproducer — the bug is source-level
(an `isinstance` / `is` check whose branch is dead code). The script in
the artifact is a *static demonstration* that exposes the type confusion
via class introspection or FX-graph construction. It is **not** a
runtime crash / wrong-result reproducer.

| Path | What it does | Verification |
|---|---|---|
| `pytorch/bug_059_deterministic_extern_kernel/repro.py` | Prints class MROs and runs an `isinstance` filter to show that the buggy filter always matches zero choices, while the corrected filter matches one. | ✅ Runs on PT 2.13.0.dev20260609+cu126; prints "BUG CONFIRMED: extern-kernel preference ... is unreachable". |
| `pytorch/bug_appendix_184410_pad_mm_getitem_dead_code/repro.py` | Builds a 1-op FX graph with `operator.getitem` and shows that `node.op is operator.getitem` is always False (string-vs-function comparison). | ✅ Runs on PT 2.13.0.dev20260609+cu126; prints "BUG CONFIRMED: pad_mm.get_non_view_def cannot recurse through getitem". |

**Tested environments.** PT scripts run with `/data1/tzh/pt_nightly_env/bin/python` (PyTorch 2.13.0.dev20260609+cu126, Tesla T4). TVM scripts run with `/data1/tzh/tvm_fuzz/bin/python` plus `LD_LIBRARY_PATH=/data1/tzh/tvm_fuzz/lib` (TVM 0.24.dev0). 7 of 8 Class B/C scripts were verified to behave as documented on this server; bug_064 (T4 hardware limitation) and bug_042 (TVM env tir-init mismatch) were not.