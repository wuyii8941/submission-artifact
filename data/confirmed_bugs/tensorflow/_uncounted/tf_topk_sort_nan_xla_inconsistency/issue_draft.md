# `tf.math.top_k` NaN handling is position-dependent in eager, differs from XLA

## Summary

`tf.math.top_k` produces different results depending on where NaN appears in the input array. This is an inconsistency **within eager execution itself** — the same values in a different order yield different top-k results:

```python
import tensorflow as tf

# NaN at front: included in top-3
tf.math.top_k([float('nan'), 1.0, 2.0, 3.0], k=3)
# values=[nan, 3., 2.],  indices=[0, 3, 2]

# NaN at back: excluded from top-3
tf.math.top_k([1.0, 2.0, 3.0, float('nan')], k=3)
# values=[3., 2., 1.],  indices=[2, 1, 0]
```

The same input values `{1, 2, 3, nan}` produce different top-k sets depending on NaN's position. Under `jit_compile=True`, NaN is always treated as the largest value (consistent regardless of position).

## Reproduction

```python
import tensorflow as tf

# --- Eager: position-dependent NaN behavior ---
t1 = tf.constant([float('nan'), 1.0, 2.0, 3.0])
t2 = tf.constant([1.0, 2.0, 3.0, float('nan')])

v1, _ = tf.math.top_k(t1, k=3)
v2, _ = tf.math.top_k(t2, k=3)
print("NaN at front:", v1.numpy())  # [nan  3.  2.]  — NaN selected
print("NaN at back: ", v2.numpy())  # [3. 2. 1.]     — NaN dropped

# --- XLA: consistent (NaN always selected as max) ---
v1x, _ = tf.function(lambda v: tf.math.top_k(v, k=3), jit_compile=True)(t1)
v2x, _ = tf.function(lambda v: tf.math.top_k(v, k=3), jit_compile=True)(t2)
print("XLA NaN at front:", v1x.numpy())  # [nan  3.  2.]
print("XLA NaN at back: ", v2x.numpy())  # [nan  3.  2.]  — same result regardless of position
```

`tf.sort` and `tf.argsort` show the same issue:

```python
t = tf.constant([3.0, float('nan'), 1.0, 2.0])

# Eager: NaN placement is inconsistent (neither first nor last)
print(tf.sort(t, direction='DESCENDING').numpy())  # [3. nan 2. 1.]
print(tf.sort(t, direction='ASCENDING').numpy())   # [1. 3. nan 2.]

# XLA: NaN consistently placed as maximum
xla_sort = tf.function(lambda v: tf.sort(v, direction='DESCENDING'), jit_compile=True)
print(xla_sort(t).numpy())  # [nan 3. 2. 1.]
```

`tf.math.argmax` is also affected:

```python
t = tf.constant([1.0, 2.0, float('nan')])
print(tf.math.argmax(t).numpy())  # eager: 1  (ignores NaN)
print(tf.function(lambda v: tf.math.argmax(v),
                  jit_compile=True)(t).numpy())  # XLA: 2  (NaN is max)
```

## Root cause

The eager sorting/top-k kernel uses hardware float comparisons where both `NaN < x` and `NaN > x` evaluate to `false`. This makes NaN's effective position in the sorted output depend on where it starts in the input — classic undefined comparison behavior in sorting algorithms.

XLA uses a comparator that explicitly places NaN above all finite values (consistent with IEEE 754 `totalOrder`).

## Relationship to existing issues

This is distinct from the `reduce_min`/`reduce_max` NaN propagation issue (#<anon>). That issue is about whether NaN propagates through fold-style reductions. This issue is about NaN's relative ordering in comparison-based sorting, which affects `top_k`, `sort`, `argsort`, and `argmax`/`argmin`.

## Impact

The position-dependent behavior means eager `top_k` is not a pure function of the multiset of input values — reordering the input can change the output even when the set of values is identical. This affects any code path where NaN may appear in scored candidates (e.g., similarity scores, logits after numerical instability).

## Affected ops

| Op | Eager | XLA |
|----|-------|-----|
| `tf.math.top_k` | Position-dependent | NaN > everything |
| `tf.sort` | Inconsistent NaN placement | NaN first (DESC) / last (ASC) |
| `tf.argsort` | Inconsistent | Consistent |
| `tf.math.argmax` | Ignores NaN | NaN is max |
| `tf.math.argmin` | Ignores NaN | NaN is min |

All float dtypes affected (float16, float32, float64). Tested on CPU; same behavior expected on GPU.

## Environment

- TensorFlow 2.22.0-dev20260429 (nightly)
- Tested on CPU
