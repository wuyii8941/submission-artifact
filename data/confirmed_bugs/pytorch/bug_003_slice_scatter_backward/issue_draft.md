# `torch.compile` silently zeroes `x.grad` for the pattern `slice_scatter(x, y, dim=K).sum(dim=K)` — backward is wrong, forward is correct

## Summary

For the pattern
```python
def program(x, y):
    out = torch.slice_scatter(x, y, dim=1, start=0, end=6)
    return out.sum(dim=1)
```
`torch.compile(program, backend="inductor")` produces **correct forward** but **silently wrong `x.grad`** — specifically, the gradient w.r.t. the `self` argument of `slice_scatter` is either fully zeroed or scaled by the wrong factor, depending on whether the slice region contains the collapsed dim-1 origin.

- `y.grad` is correct.
- `aot_eager` produces the correct `x.grad` — the bug is **inductor-specific**, not in the decomposition table.
- Repro is deterministic across seeds, shapes, and dtypes.
- `fp64` reproduces identically.
- **Present in 2.10.0, 2.11.0, and `2.12.0.dev20260410+cu126` (today's nightly) with byte-identical generated code** — so this is not a known regression in flight.

The generated backward Triton kernel folds the slice-region mask check into a compile-time constant because the output stride inherited from a broadcast `self` collapses the dim-1 coordinate. Details under "Root cause" below.

## Reproducer

```python
import torch

torch.manual_seed(0)

def program(x, y):
    out = torch.slice_scatter(x, y, dim=1, start=0, end=6)
    return out.sum(dim=1)

x = torch.randn(4, 13, 33, dtype=torch.float32, device="cuda", requires_grad=True)
y = torch.randn(4, 6, 33,  dtype=torch.float32, device="cuda", requires_grad=True)
g = torch.randn(4, 33,     dtype=torch.float32, device="cuda")

torch._dynamo.reset()
compiled = torch.compile(program, backend="inductor")
out = compiled(x, y)
out.backward(g)

# Compare against eager
xE, yE = x.detach().requires_grad_(True), y.detach().requires_grad_(True)
outE = program(xE, yE); outE.backward(g)

print("forward max_diff :", (out - outE).abs().max().item())  # ~1e-7
print("x.grad eager sum :", xE.grad.sum().item())              # 140.6300
print("x.grad ind   sum :", x.grad.sum().item())               # 0.0      <-- WRONG
print("y.grad max_diff  :", (y.grad - yE.grad).abs().max().item())  # 0.0
```

Output:
```
forward max_diff : 4.768e-07
x.grad eager sum : 140.6300
x.grad ind   sum : 0.0000     ← WRONG
y.grad max_diff  : 0.0
```

## Expected vs. Actual

Mathematical expectation:
```
out[b, i, h] = slice_scatter(x, y, dim=1, 0, 6)
             = y[b, i, h]          for i ∈ [0, 6)
             = x[b, i, h]          for i ∈ [6, 13)
L[b, h]      = sum_i out[b, i, h]
             = sum_{i<6} y[b, i, h] + sum_{i>=6} x[b, i, h]
dL/dx[b, i, h] = g[b, h]           for i ∈ [6, 13)
               = 0                 for i ∈ [0, 6)
```

Eager's `x.grad` matches this formula exactly (verified with a hand-built analytic tensor — max diff 0.0e+00 in fp64).

Inductor's `x.grad` is **entirely zero**. The positions `[6, 13)` that should contain `g[b, h]` are all zero.

## Ruling out alternative explanations

1. **Not a precision artifact** — fp64 reproduces identically with `max_diff = 2.45e+00`.
2. **Not an eager-side bug** — a hand-built analytic gradient (literally iterating `for j in range(6, 13): expected[b, j, h] = g[b, h]`) matches eager's `x.grad` to 0.0 error, and disagrees with inductor's output by 2.45.
3. **Not in the decomposition / autograd** — `torch.compile(program, backend="aot_eager")` produces a correct `x.grad` (max diff vs eager = 0.0). The bug is introduced when inductor lowers the AOT backward graph, not by the graph itself.
4. **Not seed-sensitive** — `x.grad.sum()` is deterministically 0 across seeds 1/2/42.
5. **Not a dtype artifact** — fp32 and fp64 both reproduce.
6. **Not a `requires_grad` alias issue** — the script uses fresh tensors each iteration.
7. **Not a `reduce-overhead` / cudagraphs / dynamic interaction** — default `torch.compile(fn)` with no extras reproduces.

## Cross-version matrix

| PyTorch | Triton | GPU | `x.grad.sum()` eager | `x.grad.sum()` inductor | forward max_diff |
|---|---|---|---|---|---|
| 2.10.0+cu128 | 3.6.0 | T4 sm_75 | 140.6300 | **0.0000** | 7.15e-07 |
| 2.11.0+cu126 | 3.6.0 | T4 sm_75 | 140.6300 | **0.0000** | 4.77e-07 |
| 2.12.0.dev20260410+cu126 | 3.7.0 | T4 sm_75 | 140.6300 | **0.0000** | 7.15e-07 |

The generated backward Triton kernel is **byte-identical** on 2.10.0, 2.11.0, and today's nightly, so this has been unfixed for at least 6 months.

## Cross-dtype × Cross-shape matrix (auto-found by fuzzer, 2026-04-12)

In a follow-up automated fuzzing run (V4 backward primary oracle with P0.2 AOTI integration, 399 iter / 6h), the fuzzer independently discovered 3 additional variants of this bug — including a **float64 variant** proving the bug is dtype-independent:

| dtype | shape | slice | config | `x.grad` eager | `x.grad` inductor | forward diff | verified |
|---|---|---|---|---|---|---|---|
| **float32** | (4, 13, 33) | [0:6] | default | 140.6300 | **0.0000** | 4.77e-07 | manual |
| **float32** | (1, 64, 16) | [0:32] | freezing | (positive) | **0.0000** | ~1e-6 | manual |
| **float32** | (1, 33, 257) | [0:16] | freezing | 583.3989 | **0.0000** | 1.9e-06 | fresh-process |
| **float64** | (7, 63, 33) | [0:31] | max_fusion_low | 621.618195 | **0.0000** | **0.0** | fresh-process |

The float64 row is especially damning: the forward is **bit-identical** (`max_diff=0.0`) between eager and inductor, yet the backward `x.grad` is entirely zero. This eliminates any remaining theory that the bug might be a precision artifact — the zero-stride collapse in the backward graph is structural, not numerical.

All 4 variants share the same root cause: `slice_scatter(x, y, dim=K, start=0, end=N).sum(dim=K)` → `x.grad` is fully zeroed because the backward graph's slice_scatter output has a stride-0 dim that collapses the inductor iteration space.

## Shape sensitivity (confirms root cause)

```
shape          slice[start:end]  sum(dim=1)    eager.sum    inductor.sum    predicted
(4, 13, 33)    [0:6]             dim=1          140.63       0.00            0       ✓
(2,  8, 16)    [0:4]             dim=1           22.45       0.00            0       ✓
(1,  6, 100)   [0:3]             dim=1           29.55       0.00            0       ✓
(3, 20,  7)    [5:12]            dim=1           50.17       77.18           77.1846 ✓
```

The `(3, 20, 7)` case is particularly illuminating: inductor gives `77.18`, eager gives `50.17`, and the ratio is **exactly** `77.18/50.17 = 1.53846… = 20/13`. This is the ratio `dim_size / (dim_size - slice_size) = 20 / (20 - 7)` — i.e., inductor is summing the gradient over **all 20 positions** along dim 1 (with the upstream-broadcast value) instead of the correct **13 non-slice positions**. The prediction is bit-exact.

This is explained by the root cause: the slice region `[5, 12)` does not include the collapsed dim-1 coordinate 0, so the mask check constant-folds to `False`, the `slice_scatter` degenerates into an identity over the broadcast tensor, and the summed result is 20 copies of the per-element gradient instead of 13.

When the slice region **does** include 0 (as in all the `start=0` cases above), the mask constant-folds to `True`, everything is overwritten with 0, and `x.grad` is entirely zero.

## Root cause — AOT backward graph + generated Triton

Running with `TORCH_LOGS="aot_graphs"` on the main repro prints:

```python
===== Backward graph 0 =====
def forward(self, tangents_1: "f32[4, 33][33, 1]cuda:0"):
    unsqueeze     : "f32[4, 1, 33][33, 33, 1]cuda:0"  = aten.unsqueeze.default(tangents_1, 1)
    expand        : "f32[4, 13, 33][33, 0, 1]cuda:0"  = aten.expand.default(unsqueeze, [4, 13, 33])
    full_default  : "f32[4, 6, 33][198, 33, 1]cuda:0" = aten.full.default([4, 6, 33], 0, ...)
    slice_scatter : "f32[4, 13, 33][33, 0, 1]cuda:0"  = aten.slice_scatter.default(expand, full_default, 1, 0, 6)
    slice_1       : "f32[4, 6, 33][33, 0, 1]cuda:0"   = aten.slice.Tensor(expand, 1, 0, 6)
    return (slice_scatter, slice_1)
```

The key anomaly is the stride annotation on `slice_scatter`:

```
slice_scatter : "f32[4, 13, 33][33, 0, 1]cuda:0"
                              ~~~~~~~~~~~
                              stride[1] == 0
```

`aten.slice_scatter` is a **functional op**: its semantics are clone-and-scatter, so its output **must** be a newly-allocated tensor with a stride that faithfully represents the actual per-element layout (i.e. a contiguous `[429, 33, 1]` for shape `[4, 13, 33]`). But the fake / meta kernel here has **inherited the stride of `self`** (the broadcast `expand`), producing an output stride with `stride[1] == 0`. A tensor with `stride[1] == 0` means **all 13 positions along dim 1 alias the same physical memory** — it is physically only `4 × 1 × 33 = 132` elements wide, not `4 × 13 × 33 = 1716`.

Inductor trusts the stride annotation and generates a kernel that iterates only over those 132 unique elements:

```python
# Wrapper (from TORCH_COMPILE_DEBUG output_code.py)
buf0 = empty_strided_cuda((4, 1, 33), (33, 0, 1), torch.float32)
triton_poi_fused_expand_unsqueeze_zeros_0.run(tangents_1, buf0, 132, stream=stream0)
return (
    reinterpret_tensor(buf0, (4, 13, 33), (33, 0, 1), 0),  # <- x.grad
    reinterpret_tensor(tangents_1, (4, 6, 33), (33, 0, 1), 0),  # y.grad
)

# Kernel
<upstream maintainer>.jit
def triton_poi_fused_expand_unsqueeze_zeros_0(in_ptr0, out_ptr0, xnumel, XBLOCK):
    xnumel = 132                                  # <- only 132, not 1716
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]
    xmask   = xindex < xnumel
    x0      = xindex
    tmp6    = tl.load(in_ptr0 + (x0), xmask)      # load from tangents_1
    tmp0    = tl.full([1], 0, tl.int64)           # <- "current dim-1 coord", ALWAYS 0
    tmp1    = tl.full([1], 6, tl.int64)           # slice end
    tmp2    = tmp0 < tmp1                          # 0 < 6 == True (compile-time constant)
    tmp3    = tl.full([1], 0.0, tl.float32)
    tmp4    = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5    = tl.where(tmp2, tmp3, tmp4)
    tmp7    = tl.where(tmp2, tmp5, tmp6)          # True → tmp5 == 0.0 (discards tmp6)
    tl.store(out_ptr0 + (x0), tmp7, xmask)        # writes 0.0 to every element of buf0
```

Step-by-step:

1. The iteration space is `xnumel = 132 = 4 × 1 × 33`, because the output tensor's stride-0 dim-1 has been collapsed.
2. In that collapsed space **there is no dim-1 coordinate** — the code generator needs to compute the dim-1 index to evaluate `start <= idx < end`, but all it has is a single physical row per `(b, h)`, so the dim-1 index is hard-coded to 0 (`tmp0 = tl.full([1], 0, tl.int64)`).
3. `tmp2 = 0 < 6` is therefore a compile-time True; the Triton compiler folds it, so `tmp7 = tl.where(True, 0.0, tmp6) = 0.0` unconditionally.
4. Every element of `buf0` is written as `0.0`.
5. The wrapper returns `reinterpret_tensor(buf0, (4, 13, 33), (33, 0, 1), 0)`. Autograd receives a `[4, 13, 33]` "tensor" that aliases 132 zeros 13 times. `x.grad` is entirely zero.

For the `(3, 20, 7) slice[5:12]` case, the check becomes `tmp0 = 0; tmp1_lo = 5; tmp1_hi = 12; mask = (0 >= 5) AND (0 < 12) = False`, so `tmp7 = tmp6` unconditionally — `buf0` holds the upstream gradient and the `reinterpret_tensor` broadcast gives 20 copies of it instead of the correct 13. This matches the `20/13` ratio bit-exactly, as shown in the shape sensitivity section.

## Why `aot_eager` is correct

`torch.compile(program, backend="aot_eager")` runs the same AOT backward graph (same `slice_scatter : f32[4, 13, 33][33, 0, 1]` annotation), but dispatches each op to eager aten kernels. Eager's `aten::slice_scatter` implementation **materializes a fresh contiguous output** regardless of self's stride, so the stride-0 annotation is harmless — by the time `aten::slice_scatter` returns, the output is a real `[429, 33, 1]` tensor. The bug only appears when inductor reads the stride annotation and uses it to generate the iteration space.

## Proposed fix

Two possible fix locations, in order of cleanliness:

1. **Fix the meta / fake kernel for `aten.slice_scatter`** to always return a contiguous-strided output, regardless of `self`'s stride. This is semantically correct: `slice_scatter` is functional and must clone, so its output is a new contiguous tensor. This eliminates the bug at the source and is a one-place fix in `torch/_meta_registrations.py` or wherever the meta kernel lives.

2. **Fix inductor's lowering of `slice_scatter`** to materialize `self` via `ir.ExternKernel.require_stride_order` (or equivalent) when `self` has a zero-stride dim. This fixes only the inductor path and leaves the meta kernel misleading for other users.

Option 1 is strongly preferable.

## Why this matters

`slice_scatter` is the functionalized form of any in-place slice assignment. It is heavily used in:

- KV-cache updates in autoregressive decoders (`kv_cache[:, :, t:t+1] = ...`),
- ring buffers and sliding window attention,
- prefix tuning / partial parameter updates where a subset of a tensor is overwritten and the rest must propagate gradients,
- functional-style rewrites by `torch.func.functionalize` / `make_fx`.

Combining `slice_scatter` with a reduction over the scatter dimension is common (cross-entropy over a masked prefix, prefix-sum bookkeeping, etc.). Any such model that is fine-tuned under `torch.compile` currently receives **silently wrong gradients with no warning or error**.

## Environment

```
PyTorch: 2.11.0+cu126  (also reproduces on 2.10.0+cu128 and 2.12.0.dev20260410+cu126)
Triton:  3.6.0          (also 3.7.0 on nightly)
GPU:     Tesla T4, sm_75
CUDA:    12.6
OS:      Linux 5.4.0-42-generic
```

## Relationship to #<anon> (same family, different code path)

While searching for prior reports I found [#<anon>](<upstream issue, anonymized for review>) ("`reinplace_inplaceable_ops` fails with `diagonal_scatter` on expanded (stride-zero) tensor in backward graph"), which is **open** and labeled `high priority` as of April 2026. Both bugs share the same root cause — **a `*_scatter` op receiving an expanded (stride-0) tensor as `self` in the backward graph** — but manifest on different inductor code paths and with different symptoms:

|  | #<anon> | **this issue** |
| --- | --- | --- |
| Trigger path | Two-stage `aot_export → transform → aot_compile` (AutoParallel) | Plain single-stage `torch.compile(fn); fn(x); loss.backward()` |
| Failure location | `torch/_inductor/fx_passes/reinplace.py` — the reinplace pass tries to clone an overlapping layout and raises | Inductor codegen — the stride-0 output annotation collapses the iteration space and the slice-region mask is folded to a constant |
| Symptom | `RuntimeError: more than one element of the written-to tensor refers to a single memory location` (loud crash) | `x.grad` silently returns wrong values (zeros when the slice covers index 0, or `dim_size / (dim_size - slice_size)` times the correct value otherwise) |
| Affected ops listed | `diagonal_scatter`, `select_scatter`, `slice_scatter`, `as_strided_scatter` | Same family — verified on `slice_scatter`, likely applies to the siblings |

Critically, #<anon> explicitly notes:

> "torch.compile's single-shot path avoids this because functionalization ordering prevents the pattern from reaching reinplace."

That statement is correct about the crash but the same stride-0 `self` still flows *past* the reinplace pass into inductor's codegen, and codegen trusts the `[33, 0, 1]` stride annotation on the `slice_scatter` output and generates a kernel with a collapsed iteration space (see "Root cause" above). That is the bug this issue reports.

PR <anon> ("Handle overlapping bases in scatter decomposition") attempted to fix #<anon> by materializing the clone in `_decompose_scatter_mutating` when the scatter base has internal overlap, but was **closed without merging** on 2026-04-10. Because that PR only modified `torch/_inductor/fx_passes/reinplace.py`, it would not have fixed this issue even if it had landed — the single-shot path never reaches the reinplace pass.

A fix at the level of `aten.slice_scatter`'s meta kernel — returning a contiguous-strided output regardless of `self`'s stride — would address **both** #<anon> and this issue with a single one-file change, because both code paths trust the meta kernel's stride annotation to decide whether to materialize or not. The same fix should be mirrored to the other `*_scatter` ops (`diagonal_scatter`, `select_scatter`, `as_strided_scatter`) listed in #<anon>.

## Not a duplicate of #<anon>

[#<anon>](<upstream issue, anonymized for review>) is a forward-pass bug in `slice_scatter` where a mismatch between the slice range and the source tensor size produces garbage output; it is unrelated to the backward path and was reported/assigned in March 2024. The minimal reproducer above uses well-formed shapes that eager accepts without error, and the forward result is numerically identical between eager and inductor (`max_diff = 4.77e-07`). The bug is purely in the backward path.

cc <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer>
