# `erf` produces different fp32 result for the same input based on tensor shape

> **Status: CONFIRMED BY DEVELOPER — FIXED (merged)**
> Fix PR: <upstream PR, anonymized for review>

## Summary

XLA's `erf` lowering is not shape-invariant: the same fp32 input produces a different output depending on the innermost axis size of the enclosing tensor. For `x = 4.5`:

- Shape with innermost dim **= 1** (e.g. `[1]`, `[7, 64, 1]`) → `erf(x) = 0.9999998212`
- Shape with innermost dim **≥ 2** (e.g. `[2]`, `[7, 64, 5]`) → `erf(x) = 1.0` (correctly rounded)

The two outputs differ by ~3 fp32 ULPs. A single downstream `floor` / `ceil` / `>=` call amplifies this into an integer-valued, silent divergence.

The bug reproduces identically in JAX, which confirms the root cause lives in XLA's `erf` HLO lowering (not in TensorFlow's XLA bridge).

## Versions tested

| stack | version | reproduces? |
|---|---|---|
| TensorFlow | 2.21.0 (stable, 2026-03-06) | ✓ |
| TensorFlow | tf-nightly `2.22.0-dev20260417` | ✓ |
| JAX | 0.9.2 | ✓ |

Linux x86_64; reproduced on both CPU-only and CUDA 12 / NVIDIA T4 configurations.

## Minimal reproducer (TensorFlow, self-contained, ~20 s)

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

<upstream maintainer>.function(jit_compile=True)
def xla_erf(x):
    return tf.math.erf(x)

print(f"TF version: {tf.__version__}")
for shape in [(1,), (2,), (7, 1), (7, 64, 1), (7, 64, 5), (1, 1, 1)]:
    x = tf.fill(shape, tf.constant(4.5, dtype=tf.float32))
    e  = tf.math.erf(x).numpy()
    xl = xla_erf(x).numpy()
    print(f"shape {str(shape):>13}  eager={float(e.flat[0]):.10f}  XLA={float(xl.flat[0]):.10f}  match={float(e.flat[0]) == float(xl.flat[0])}")
```

(The same result holds for `tf.raw_ops.Erf`.)

Expected (what `math.erf(4.5)` rounds to in fp32): `1.0` for every shape.
Observed on TF 2.21.0 and tf-nightly `2.22.0-dev20260417`:

```
shape          (1,)  eager=1.0000000000  XLA=0.9999998212  match=False
shape          (2,)  eager=1.0000000000  XLA=1.0000000000  match=True
shape        (7, 1)  eager=1.0000000000  XLA=0.9999998212  match=False
shape    (7, 64, 1)  eager=1.0000000000  XLA=0.9999998212  match=False
shape    (7, 64, 5)  eager=1.0000000000  XLA=1.0000000000  match=True
shape     (1, 1, 1)  eager=1.0000000000  XLA=0.9999998212  match=False
```

Exactly the same pattern appears when we bypass TF entirely and go through JAX:

## Cross-framework reproducer (JAX, confirms the bug is in XLA)

```python
import jax, jax.numpy as jnp
from jax import lax
import numpy as np

<upstream maintainer>.jit
def jit_erf(x):
    return lax.erf(x)

print(f"jax: {jax.__version__}")
for shape in [(1,), (2,), (7, 1), (7, 64, 1), (7, 64, 5), (1, 1, 1)]:
    x = jnp.full(shape, 4.5, dtype=jnp.float32)
    e  = np.asarray(lax.erf(x)).flat[0]
    xl = np.asarray(jit_erf(x)).flat[0]
    print(f"shape {str(shape):>13}  eager={float(e):.10f}  jit={float(xl):.10f}")
```

Output on jax 0.9.2:

```
shape          (1,)  eager=0.9999998212  jit=0.9999998212
shape          (2,)  eager=1.0000000000  jit=1.0000000000
shape        (7, 1)  eager=0.9999998212  jit=0.9999998212
shape    (7, 64, 1)  eager=0.9999998212  jit=0.9999998212
shape    (7, 64, 5)  eager=1.0000000000  jit=1.0000000000
shape     (1, 1, 1)  eager=0.9999998212  jit=0.9999998212
```

Every XLA-backed path (TF+jit, JAX-eager, JAX+jit) agrees on the shape-dependent output. Only TensorFlow's non-XLA Eigen kernel (used in TF eager) returns the correctly-rounded `1.0` for every shape.

| shape | TF eager (Eigen) | TF `jit_compile=True` (XLA) | JAX `lax.erf` (XLA) | JAX `jit(lax.erf)` (XLA) |
|---|---|---|---|---|
| `(1,)`        | `1.0000000000` | `0.9999998212` | `0.9999998212` | `0.9999998212` |
| `(2,)`        | `1.0000000000` | `1.0000000000` | `1.0000000000` | `1.0000000000` |
| `(7, 64, 1)`  | `1.0000000000` | `0.9999998212` | `0.9999998212` | `0.9999998212` |
| `(7, 64, 5)`  | `1.0000000000` | `1.0000000000` | `1.0000000000` | `1.0000000000` |

This rules out TF's XLA bridge as the source: same XLA, different frontend, same bug.

## Why this matters: silent amplification under downstream `floor`

```python
x = tf.fill((7, 64, 1), tf.constant(4.5, dtype=tf.float32))
eager_sum = float(tf.reduce_sum(tf.math.floor(tf.math.erf(x))))
xla_sum   = float(tf.reduce_sum(tf.math.floor(xla_erf(x))))
print(f"sum(floor(eager erf(4.5))) = {eager_sum}  (expected 448)")
print(f"sum(floor(XLA   erf(4.5))) = {xla_sum}    (observed 0)")
```

Output: `eager = 448, XLA = 0`. All 448 elements evaluate to `1` in eager and `0` under XLA after a single `floor` — 100% misclassification with no warning.

Size-1 innermost axes arise naturally in ML code whenever a reduction uses `keepdims=True` (attention score scaling, per-row normalization, mixture-of-experts routing); a subsequent elementwise saturating op such as `erf`, `tanh`, `sigmoid`, `gelu` followed by a thresholding op is a common pipeline shape.

## Why we believe this is a bug (not a precision tradeoff)

1. **Elementwise ops must be shape-invariant.** The same scalar input `4.5` producing `1.0000000000` at shape `(2,)` and `0.9999998212` at shape `(1,)` violates the semantics of `erf`.
2. **The accurate lowering is already available in XLA.** Shapes with innermost dim ≥ 2 produce the correctly-rounded fp32 result, so there is no missing approximation — XLA already has the accurate path; it just is not taken for size-1 innermost.
3. **Silent wrong result with no dtype/precision annotation.** Nothing in the TF or JAX API surface signals that `erf` precision depends on tensor shape.

## Root cause — two structurally different lowerings in LLVM IR

Inspecting the XLA-dumped LLVM IR (`--xla_dump_to=...`, `.ir-with-opt.ll`) for shape `(4, 1)` vs `(4, 2)` on the same `Erf` HLO op shows they take two different code paths, not one shared intrinsic instantiated at different vector widths.

**Shape (4, 2) — vector path, correct:**
```llvm
%11 = call <2 x float> <upstream maintainer>.fabs.v2f32(<2 x float> %10)
%12 = fcmp oge <2 x float> %11, splat (float 3.83251)      ; |x| >= 3.83
%13 = fmul <2 x float> %10, %10                             ; x²
%14..%17 = 4 FMAs over x² for numerator polynomial
%18 = fmul <2 x float> %10, %17                             ; x · P(x²)
%19..%24 = 6 FMAs for denominator polynomial
%25 = call <2 x float> <upstream maintainer>.copysign.v2f32(splat 1.0, %10) ; sign(x) · 1.0
%26 = fdiv <2 x float> %18, %24                             ; P(x)/Q(x²)
%27 = select <2 x i1> %12, <2 x float> %25, <2 x float> %26 ; saturation select
store <2 x float> %27, ptr %8
```

**Shape (4, 1) — scalar path, missing output saturation:**
```llvm
%12 = tail call float <upstream maintainer>.maximum.f32(%11, -3.74392)      ; input clamp
%13 = tail call float <upstream maintainer>.minimum.f32(%12,  3.74392)      ; (different constant!)
%14 = fmul float %13, %13
%15..%18 = 4 FMAs (numerator)
%19 = fmul float %13, %18
%20..%25 = 6 FMAs (denominator)
%26 = fdiv float %19, %25                                    ; polynomial result
store float %26, ptr %6                                       ; NO saturation select
```

Two independent pieces of evidence that the two paths are separate implementations, not a single intrinsic at different widths:

1. **Different saturation constants.** The vector path checks `|x| >= 3.83251` (`0x400EA8F960000000`) and replaces with `sign(x) · 1.0` via `select`. The scalar path clamps the input to `±3.74392` (`0x400DF38D00000000`) and stores the polynomial output directly. The two constants are provably different.

2. **Only the vector path has a `select` / `copysign` saturation tail.** For `x = 4.5` the scalar path evaluates the rational polynomial at the clamped input `3.74392` and produces `0.9999998212`. The vector path's `select` replaces the polynomial output with exactly `1.0`. This is the direct mechanism for the observed discrepancy.

The saturation semantics in the vector path correspond to what XLA's `EmitErfF32` intrinsic in `xla/codegen/intrinsic/erf.cc` is documented to perform (`|x| >= 3.83 → sign(x)·1.0`). The scalar path appears to bypass `EmitErfF32` entirely and emit a separate rational approximation that has only input clamping, no output saturation, and a different clamp constant.

Git history shows this saturation behaviour was deliberately added to `EmitErfF32` — the scalar bypass appears to be a regression relative to that intent.

## Suggested fix

Route the f32 scalar `Erf` lowering through the same `EmitErfF32` intrinsic the vector path uses, or equivalently add the same `fabs + fcmp + copysign + select` saturation tail to the scalar path. The two paths must agree on both the saturation constant and the output replacement step.

## Discovery context

Found during random graph fuzzing of TF 2.21.0. The triggering program contained a `keepdims=True` L2-norm followed by `erf` and `floor`; the innermost axis was size 1 because of the `keepdims`. Minimized to the shape-enumeration repro above, cross-checked in JAX to confirm XLA-side attribution, then root-caused to the scalar-vs-vector lowering divergence by inspecting `--xla_dump_to=...` LLVM IR.
