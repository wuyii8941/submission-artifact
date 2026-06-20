# `tf.math.igamma` and `tf.math.igammac` return values outside [0, 1] for large arguments

## Summary

`tf.math.igamma(a, x)` computes the regularized lower incomplete gamma function P(a, x), which is mathematically bounded to [0, 1] for all valid inputs (a > 0, x ≥ 0). For large arguments (a ≈ x ≥ ~5×10⁴ in float32), TF returns values well outside this range — up to 1900× the correct value.

`tf.math.igammac` (the complement Q = 1 - P) is similarly affected. The identity P(a,x) + Q(a,x) = 1 breaks down: at a = x = 800000, P + Q ≈ 3.03 instead of 1.0.

## Reproduction

```python
import tensorflow as tf

a = tf.constant(800000.0, dtype=tf.float32)
x = tf.constant(800000.0, dtype=tf.float32)

ig = tf.math.igamma(a, x)
igc = tf.math.igammac(a, x)

print(f"igamma  = {ig.numpy():.6f}")   # 1.498142 (should be ~0.5)
print(f"igammac = {igc.numpy():.6f}")  # 1.536196 (should be ~0.5)
print(f"sum     = {(ig + igc).numpy():.6f}")  # 3.034338 (should be 1.0)
```

## Scale of the problem

| a = x | igamma | igammac | sum (should be 1.0) |
|-------|--------|---------|---------------------|
| 100 | 0.5133 | 0.4867 | 1.0000 |
| 1,000 | 0.5044 | 0.4960 | 1.0003 |
| 10,000 | 0.5020 | 0.4993 | 1.0013 |
| 100,000 | 0.5195 | 0.5187 | 1.0382 |
| **800,000** | **1.4981** | **1.5362** | **3.0343** |
| **1,000,000** | **1.3122** | **1.3741** | **2.6863** |
| **8,685,113** | **1900.3** | — | — |

Degradation starts around a ≈ 5×10⁴ and gets worse with increasing values. At a = 8.7×10⁶, the result is off by a factor of ~3800.

## Affected devices

Both CPU and GPU eager kernels return identical wrong results. XLA (`jit_compile=True`) uses a different implementation that stays in range longer but also degrades at very large values.

## Why this matters

1. **Violates mathematical invariant**: igamma ∈ [0, 1] is not a soft tolerance — it's a hard mathematical bound. Returning 1.498 or 1900 is not a precision issue, it's a qualitative failure.
2. **Breaks the complement identity**: Any code using `igamma + igammac == 1` (e.g., computing one from the other) will produce wrong results.
3. **Used in statistical distributions**: `igamma`/`igammac` underlie `tf.math.igamma`, `tfp.distributions.Gamma.cdf`, chi-squared tests, and Poisson distributions. Large shape parameters (a > 10⁴) are common in batch statistics and hypothesis testing.

## Expected behavior

For a = x, the regularized incomplete gamma P(a, a) → 0.5 as a → ∞ (by the central limit theorem applied to the Gamma distribution). The function should always return values in [0, 1], regardless of input magnitude.

## Also affects `tf.math.betainc`

`tf.math.betainc(a, b, x)` (the regularized incomplete beta function) has the same root cause. By symmetry of the Beta distribution, `betainc(a, a, 0.5) = 0.5` exactly for all `a > 0`. TF's f64 implementation confirms this. But in f32:

```python
tf.math.betainc(
    tf.constant(5e6, tf.float32),
    tf.constant(5e6, tf.float32),
    tf.constant(0.5, tf.float32)
).numpy()
# Returns: 12552954.0  (should be 0.5)
```

| a = b | betainc(a, a, 0.5) f32 | betainc(a, a, 0.5) f64 |
|-------|------------------------|------------------------|
| 10,000 | 0.5005 | 0.5000 |
| 100,000 | **0.6697** | 0.5000 |
| 500,000 | **0.1973** | 0.5000 |
| **5,000,000** | **12,552,954** | 0.5000 |

Since `betainc` is implemented via the incomplete beta function which shares convergence machinery with `igamma`, this is the same underlying issue.

## Root cause hypothesis

The Eigen `igamma` / `ibeta` implementation uses a series expansion and continued fraction. For large equal arguments (a ≈ x for igamma, or a = b with x = 0.5 for betainc), the computation sits at the boundary of the two regimes where convergence is slowest. The series requires an extremely large number of terms, leading to accumulated error and overflow of intermediate sums. A numerically stable implementation would use the asymptotic uniform expansion for large a ≈ x (see DLMF §8.12 or Temme's method).

## Environment

- TensorFlow 2.22.0-dev20260421 (nightly)
- Also confirmed on TensorFlow 2.21.0 (stable)
- CPU: x86_64
- GPU: NVIDIA T4 (CUDA 12)
- Both CPU and GPU affected identically
