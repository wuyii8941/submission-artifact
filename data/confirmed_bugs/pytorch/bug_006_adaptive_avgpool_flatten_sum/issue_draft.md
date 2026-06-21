# `torch.compile` silently produces wrong output for `adaptive_avg_pool2d + flatten + sum` — all batches except batch 0 are wrong

## Summary

For the common vision-model pattern
```python
def program(x):
    y = torch.nn.functional.adaptive_avg_pool2d(x, k)
    return y.flatten(1).sum(dim=-1)
```
`torch.compile(backend="inductor")` produces **silently wrong output for every batch except batch 0**. Eager is correct; inductor is wrong. No warning, no error — just wrong numbers.

Root cause: the fused reduction kernel hard-codes a stride of `3200` (= `ceil(3185/32)*32`) while the pool-output buffer it reads from has an actual batch stride of `3185` (= `65*7*7`). Batch 1 loads start **15 elements past** the batch-1 row and continue **15 elements past** the allocated buffer.

## Versions

Reproduced on:

| Version | Triton | GPU | Status |
| --- | --- | --- | --- |
| **2.11.0+cu126** (stable) | 3.6.0 | Tesla T4 (sm_75) | **Reproduces** |
| **2.12.0.dev20260410+cu126** (nightly) | 3.7.0 | Tesla T4 (sm_75) | **Reproduces** — bit-identical output |

The nightly (built today) and 2.11.0 produce byte-identical wrong values, so the bug lives in Inductor's codegen and has not been touched since at least April 2025.

## Minimal reproducer

```python
import torch
torch.manual_seed(0)
# r0_numel = 2049 * 7 * 7 = 100401 — large enough that Inductor picks a
# non-persistent reduction on every GPU architecture, so the reproducer
# does not depend on T4-specific tile heuristics.
x = torch.randn(4, 2049, 8, 8, dtype=torch.float32, device="cuda")

def program(x):
    y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
    return y.flatten(1).sum(dim=-1)

eager    = program(x.clone())
compiled = torch.compile(program, backend="inductor")(x.clone())

print("eager   :", eager.tolist())
print("inductor:", compiled.tolist())
print("diff    :", (eager - compiled).abs().tolist())
```

Output on 2.11.0, Tesla T4:
```
diff: [1.5e-05, 0.57, 2.17, 8.72]
```

Batch 0 is correct (at fp32 noise level). Batches 1, 2, 3 are silently wrong
by increasingly large amounts.

A smaller shape also reproduces (`[4, 65, 8, 8]`, `diff ≈ [5e-6, 6.8, 2.5, 2.5]`),
but a reviewer on a higher-end GPU may find the smaller shape routes to a
persistent reduction kernel and does not reproduce — the `[4, 2049, 8, 8]`
shape above has a reduction size that is too large to fit in any GPU's
shared memory, forcing the non-persistent reduction codegen path on every
architecture.

## Per-batch signature: only batch 0 is ever correct

Running the same program across batch sizes:
```
shape [B, 65, 8, 8]:
  B=2: wrong batches = [1]
  B=3: wrong batches = [1, 2]
  B=4: wrong batches = [1, 2, 3]
  B=8: wrong batches = [1, 2, 3, 4, 5, 6, 7]
```

Across 30 different seeds at shape `[2, 257, 22, 22]`: batch 0 wrong in 0/30 runs, batch 1 wrong in 30/30 runs — fully deterministic.

`fp64` reproduces identically (`max_diff ≈ 6.82`), so it is not a floating-point accumulation artifact.

## Inductor-generated kernel shows the wrong stride directly

Running with `TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+output_code"` dumps the following kernels for shape `[2, 65, 8, 8]`:

The pool buffer is allocated correctly:
```python
buf0 = empty_strided_cuda((2, 65, 7, 7), (3185, 49, 7, 1), torch.float32)
#                              ^^^^ shape                  ^^^^ stride
#                                                          batch stride = 3185 ✓
```

But the fused view+reduction kernel uses **stride 3200** instead:
```python
<upstream maintainer>.jit
def triton_red_fused_sum_view_1(in_ptr0, out_ptr0, xnumel, r0_numel,
                                XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 2
    r0_numel = 3185        # correct: 65 * 7 * 7 = 3185
    ...
    x0 = xindex            # batch index ∈ [0, 1]
    ...
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask  = r0_index < r0_numel
        r0_1     = r0_index
        # ⚠️  WRONG: stride should be 3185 (buf0 batch stride), not 3200
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3200*x0),
                       r0_mask & xmask,
                       eviction_policy="evict_first",
                       other=0.0)
        ...
```

The FX-IR comment embedded in the generated code even labels the expected type correctly:
```
%view : Tensor "f32[2, 3185][3185, 1]cuda:0" = call_function[target=torch.ops.aten.reshape.default]
        (args = (%_adaptive_avg_pool2d, [2, 3185]), kwargs = {})
```
— stride is documented as `(3185, 1)`. But the kernel uses `3200`.

### What `3200` is

`3200 == ceil(3185 / 32) * 32 == 100 * 32`. Inductor has rounded `r0_numel` **up to the next multiple of 32** (the warp size) and is using the rounded value as if it were the batch stride of the source tensor.

### Consequence

For batch index `x0 = 1`, the kernel reads `in_ptr0[3200 .. 3200+3184]` = `in_ptr0[3200..6384]`. But batch 1's valid data lives at `in_ptr0[3185..6369]`. Therefore:

1. The loads **miss** the first 15 elements of batch 1 (offsets 3185..3199).
2. The loads **read 15 elements past the end** of the allocated buffer (offsets 6370..6384), picking up whatever garbage is adjacent in memory.
3. The reduction over the shifted window is close to — but not equal to — the correct sum.

This is why `diff ≈ 2…7` for a randn-based input: most of the 3185-element reduction overlaps the correct range, but 15 elements differ, plus 15 garbage reads, so the final sum drifts by a value proportional to the input variance.

This also explains every observation:

| Observation | Explanation |
| --- | --- |
| Only batch 0 correct | For `x0=0` the offset is `0*3200 = 0`, which happens to match `0*3185 = 0` |
| Both fp32 and fp64 reproduce | Pure indexing bug, precision-independent |
| Run-to-run deterministic | No RNG involved, the shift is fixed |
| Cross-seed wrong-batch pattern identical | Same |
| Bug disappears when fusion is broken (e.g. `pool.sum((1,2,3))`, `(y + 0.0).flatten(1).sum(-1)`) | Different kernel is generated, no view-through-reduction codegen |
| Bug appears whenever `C * out_h * out_w ≠ multiple of 32` **and** the reduction is large enough to use a `triton_red_fused_*` (non-persistent) kernel | Matches the `ceil_to_32` rounding hypothesis exactly |

## Isolating the bug to the single `3200` literal

To rule out Triton, CUDA, the driver, the buffer allocator, and the pool kernel, I ran the **exact** generated Triton reduction kernel verbatim and an identical copy with only the literal `3200` replaced by `3185`, on the **same** `buf0` produced by eager PyTorch.

```python
# buf0 produced by eager — confirmed contiguous (3185, 49, 7, 1)
buf0 = torch.nn.functional.adaptive_avg_pool2d(x, 7).contiguous()
assert buf0.stride() == (3185, 49, 7, 1)
assert buf0.is_contiguous()

expected = buf0.flatten(1).sum(dim=-1)   # ground truth computed in eager

<upstream maintainer>.jit
def inductor_broken(in_ptr0, out_ptr0, xnumel, r0_numel,
                    XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    # ... (verbatim copy of inductor's generated kernel) ...
    tmp0 = tl.load(in_ptr0 + (r0_1 + 3200 * x0), r0_mask & xmask, other=0.0)
    # ...

<upstream maintainer>.jit
def inductor_patched(in_ptr0, out_ptr0, xnumel, r0_numel,
                     XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    # ... (identical kernel) ...
    tmp0 = tl.load(in_ptr0 + (r0_1 + 3185 * x0), r0_mask & xmask, other=0.0)   # only change
    # ...
```

Results on `buf0` with shape `[2, 65, 7, 7]`:

| Kernel | batch-1 output | diff vs eager ground truth |
| --- | ---: | ---: |
| `inductor_broken` (verbatim, `3200`) | 71.60494995 | **3.63** |
| `inductor_patched` (`3200` → `3185`) | 67.97589111 | **1.43e-06** (bit-exact) |
| `torch.compile(prog)(x)` | 71.60494232 | **3.63** (byte-identical to broken) |
| direct `buf0.flatten(1).sum(-1)` in eager | 67.97589111 | ground truth |

`torch.compile`'s output is byte-identical to the `inductor_broken` kernel run against the same `buf0`, confirming that `torch.compile` is in fact executing this exact kernel. The one-character fix `3200 → 3185` makes it correct.

This rules out:

- **Triton**: the same Triton 3.6.0 compiles the patched kernel — correct.
- **CUDA / driver / GPU**: same Tesla T4, same CUDA 12.6, same launch config — correct once the literal is fixed.
- **Buffer allocator**: `buf0.stride() == (3185, 49, 7, 1)` and `buf0.is_contiguous() == True`.
- **Pool kernel (kernel 1)**: `buf0.flatten(1).sum(-1)` in eager equals the patched-kernel output.
- **numpy / eager**: not on the execution path.

The only code path that differs between the broken run and the correct run is **the literal `3200` that Inductor's codegen writes into the generated Triton source file**. The bug is unambiguously in `torch/_inductor/` codegen for fused `view + reduction` kernels over a pool output, in the place that computes the batch-stride literal for `in_ptr0` loads.

I also verified the same byte-for-byte wrong output on PyTorch nightly `2.12.0.dev20260410+cu126` with Triton `3.7.0`, so the codegen path has not moved in the last year and the bug is present on current `main`.

## When the bug does **not** fire

- `pool(x).sum(dim=(1, 2, 3))` (direct multi-dim reduction, no `view`): clean
- `(pool(x) + 0.0).flatten(1).sum(-1)` (no-op pointwise breaks fusion): clean
- `pool(x).contiguous().flatten(1).sum(-1)`: **still buggy** (`contiguous()` is a no-op on already-contiguous tensor, doesn't create a new buffer, fusion still happens)
- `C × out_h × out_w` is a multiple of 32 (e.g. C=64/128/192/256/384 with out=7): clean
- `r0_numel` small enough that Inductor picks a persistent reduction (`triton_per_fused_*` template) instead of a non-persistent reduction (`triton_red_fused_*`): clean

## Channel-sweep evidence for the `ceil_to_32` rule

Shape `[3, C, 22, 22]`, `adaptive_avg_pool2d(x, 7)`, `flatten(1).sum(-1)`:

| C | `C*49 % 32` | `ceil_to_32 - C*49` | observed `max_diff` |
| ---: | ---: | ---: | ---: |
| 16  | 16 | 16 | 9.5e-07 *(too small, persistent reduction)* |
| 32  | 0  | 0  | 1.9e-06 |
| 64  | 0  | 0  | 3.8e-06 |
| 128 | 0  | 0  | 3.8e-06 |
| 192 | 0  | 0  | 3.8e-06 |
| 255 | 15 | 17 | **1.07e+02** |
| 256 | 0  | 0  | 7.6e-06 |
| 257 | 17 | 15 | **4.40e+00** |
| 300 | 12 | 20 | **9.88e+00** |
| 384 | 0  | 0  | 7.6e-06 |
| 512 | 0  | 0  | 9.5e-06 |

Pattern: clean iff `(C * 49) mod 32 == 0`.

## Workarounds

Until this is fixed, any of the following restore correct output:

1. `pool(x).sum(dim=(1, 2, 3))` — use an explicit multi-dim reduction instead of `flatten + sum`.
2. `(pool(x) + 0.0).flatten(1).sum(-1)` — insert a trivial pointwise op to break fusion.
3. `torch._inductor.config.disable_fused_view = True` or similar config that prevents the view+reduction fusion (if present).
4. Move the pool output out-of-graph, e.g. `pool_out = pool(x).detach().requires_grad_(x.requires_grad); return pool_out.flatten(1).sum(-1)` — forces a buffer boundary.

## Why this matters

`adaptive_avg_pool2d + flatten + linear / sum / mean` is **the standard global-average-pooling head** used in nearly every CNN vision model (ResNet, ResNeXt, EfficientNet, ConvNeXt, Vision Transformer hybrids, many others). Any model with a channel count that is not a multiple of 32 in its final feature map will silently return wrong classification logits when compiled with Inductor.

## Environment

```
PyTorch: 2.11.0+cu126   (also reproduces on 2.12.0.dev20260410+cu126)
Triton:  3.6.0          (also 3.7.0)
GPU:     Tesla T4, sm_75
CUDA:    12.6
OS:      Linux 5.4.0-42-generic
```

## Related — and why this is NOT a duplicate of #<anon> / PR <anon>

At first glance this looks related to the ReflectionPad + view bug (#<anon>,
fixed by PR <anon>), since both produce silent wrong output from a
fused-view-then-reduction pattern. **They are not the same bug and #<anon>
does not fix this one.** Concretely:

| | #<anon> (fixed) | This issue |
| --- | --- | --- |
| Triggering op | `reflection_pad + view + chunk + sum` | `adaptive_avg_pool2d + flatten + sum` |
| Layer the bug lives in | SymPy IR (`torch/utils/_sympy/functions.py`) | Triton kernel codegen (`torch/_inductor/codegen/triton.py` or called helpers) |
| Root cause | `ModularIndexing._eval_is_positive` returned `True` when the operands were positive, but the result can be zero, which let the simplifier drop an `abs()` it should have kept | An integer literal emitted by codegen as the outer/batch stride of `in_ptr0` inside a fused `triton_red_fused_view_sum` kernel; the literal is `ceil(r0_numel / 32) * 32` instead of the actual buffer stride `r0_numel` |
| Fix shape | 4-line delete in `functions.py` | Will need a fix in the reduction-kernel codegen path that computes the outer-dim stride of a view over a pool output |
| Verifiable by `grep`ing PR <anon> diff | Yes | No — PR <anon> does not touch `_inductor/codegen/*` or `_inductor/scheduler.py` |

I've verified PR <anon> merged commit `e5f8699` is present in 2.11.0 and in
today's nightly (`2.12.0.dev20260410`). The present bug reproduces on both
versions with byte-identical output, so the fix for the ReflectionPad case
does not reach this path.

The two bugs share a *symptom family* ("fused view + reduction produces
silent wrong output") but live in unrelated code, so they should be treated
independently.

cc <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer>
