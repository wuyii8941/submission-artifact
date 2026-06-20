# Grappler ArithmeticOptimizer incorrectly strips non-strictly-monotonic ops from ArgMin/ArgMax

> **Status: CONFIRMED** — Filed as [upstream issue #116436 (anonymized)]. Developer confirmed and a fix was submitted in [upstream issue #116709 (anonymized)].

## Summary

The `OptimizeMaxOrMinOfMonotonic` stage in Grappler's `ArithmeticOptimizer` transforms `ArgMin(f(x))` into `ArgMin(x)` for ops classified as "monotonic non-decreasing" by `IsElementWiseMonotonic()` in `op_types.cc`. This is incorrect for non-strictly-monotonic functions (e.g., `Relu`, `Relu6`, `Floor`, `Ceil`, `Rint`, `Sign`), which have flat regions that create ties and alter the argmin result.

## Minimal Reproducer

```python
import tensorflow as tf

x = tf.constant([[1.0, -1.0, -5.0, 2.0]])
# relu(x) = [1, 0, 0, 2] → argmin = 1 (first zero)
# x       = [1, -1, -5, 2] → argmin = 2 (most negative)

print("eager:", tf.argmin(tf.nn.relu(x), axis=-1).numpy())  # [1] ✓

tf.config.optimizer.set_experimental_options({"arithmetic_optimization": True})
print("arith ON: ", tf.function(lambda x: tf.argmin(tf.nn.relu(x), axis=-1))(x).numpy())  # [2] ✗

tf.config.optimizer.set_experimental_options({"arithmetic_optimization": False})
print("arith OFF:", tf.function(lambda x: tf.argmin(tf.nn.relu(x), axis=-1))(x).numpy())  # [1] ✓
```

**Output (TF 2.21.0, CUDA 12, Linux):**
```
eager:    [1]
arith ON:  [2]
arith OFF: [1]
```

## Affected ops

The following ops from `IsElementWiseMonotonic()` (in `tensorflow/core/grappler/op_types.cc`) are non-strictly-monotonic and confirmed to trigger incorrect results:

| Op | Flat region | Confirmed |
|----|------------|-----------|
| `Relu` | f(x)=0 for x≤0 | ✅ |
| `Relu6` | f(x)=0 for x≤0, f(x)=6 for x≥6 | ✅ |
| `Floor` | f(x) constant on [n, n+1) | ✅ |
| `Ceil` | f(x) constant on (n-1, n] | ✅ |
| `Rint` | f(x) constant on [n-0.5, n+0.5) | ✅ (same root cause as Floor) |
| `Sign` | f(x) constant on (-∞,0), {0}, (0,+∞) | ✅ |

The remaining ops in the list (Exp, Log, Sigmoid, Tanh, Softplus, etc.) are strictly monotonic and the optimization is valid for them.

## Root cause

The transformation `ArgMin(f(x)) → ArgMin(x)` is only valid when f is **strictly** monotonic, i.e., `a < b ⟹ f(a) < f(b)`. For non-strictly-monotonic functions like Relu, there exist distinct values a ≠ b where f(a) = f(b). When this happens, `ArgMin(f(x))` returns the first index among the tied values (per TF's tie-breaking rule), while `ArgMin(x)` returns the index of the true minimum — a different index.

## Suggested fix

In `op_types.cc`, split `IsElementWiseMonotonic` into strict and non-strict variants, or add a `bool* is_strictly_monotonic` out-parameter. Then in the optimizer stage, only apply the `ArgMin`/`ArgMax` transformation for strictly monotonic ops.

Non-strictly-monotonic ops to exclude: `Relu`, `Relu6`, `Floor`, `Ceil`, `Rint`, `Sign`.

## Impact

This silently produces wrong results for any `tf.function` containing `argmin(relu(...))` or similar patterns, without any error or warning. The Grappler `arithmetic_optimization` pass is enabled by default.

## Environment

Confirmed on both stable and nightly:
- TensorFlow 2.21.0 (stable)
- tf-nightly 2.22.0.dev20260420
- Python 3.11
- CUDA 12 / cuDNN 9
- Ubuntu Linux
