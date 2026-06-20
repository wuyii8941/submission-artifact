# Bug Mapping: Issue → Reproducer Directory

All 77 counted bugs from the paper, anonymized for double-blind
review. See `data/confirmed_bugs/<stack>/` for minimal reproducers.

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

## TensorFlow/XLA: 29 Counted Bugs

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

## IREE: 6 Counted Bugs

| Issue | Root Cause | Status |
|----|----|----|
| #24180 (anonymized) | `DemoteF64ToF32Pass` f64/f32 buffer-size mismatch → silent f64 output corruption | Open; labeled `bug`; developer confirmed |
| #24181 (anonymized) | ReduceProd keepdims=0 shape/lowering crash | Closed / Fixed (torch-mlir PR #4540) |
| #24228 (anonymized) | SoftmaxCrossEntropyLoss with weights divides by N instead of weight sum | Open; labeled `bug`; developer confirmed (hanhanW provided fix patch) |
| #24229 (anonymized) | ReverseSequence wrong results when batch_axis=1, time_axis=0 | Closed / Fixed by hanhanW |
| #24230 (anonymized) | Unique op torch lowering causes compiler SEGFAULT | Open; labeled `bug`; developer confirmed (hanhanW provided fix patch) |
| #24265 (anonymized) | ReduceMax/ReduceMin multi-axis + negative index SEGFAULT | Open; labeled `bug`; developer confirmed |

## TVM: 11 Counted Bugs

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

---

**Total: 77 counted bugs** (matches paper Table 1).

Note: 6 additional bugs were discovered after the paper freeze
(post-submission). They are tracked separately in upstream trackers
and are **not** included in the paper's 77-bug count.