# XLA `fake_quant_with_min_max_vars` computes different nudged range than eager

## Summary

`tf.quantization.fake_quant_with_min_max_vars` under XLA produces quantized values that differ from eager by one full quantization step. The root cause is a different rounding mode when computing the nudged zero point: eager uses round-half-away-from-zero (`std::round`), XLA uses round-half-to-even. When `zero_point_from_min` is a half-integer (e.g. 2.5), the nudged quantization range shifts by one step.

```python
import tensorflow as tf

inp = tf.constant([2.579, -0.6, 2.6], dtype=tf.float32)
mn, mx = tf.constant(-0.5), tf.constant(2.5)

# Eager: nudges range to [-0.6, 2.4]
print(tf.quantization.fake_quant_with_min_max_vars(inp, mn, mx, num_bits=4).numpy())
# [2.4, -0.6, 2.4]

# XLA: nudges range to [-0.4, 2.6]
print(tf.function(
    lambda v, lo, hi: tf.quantization.fake_quant_with_min_max_vars(v, lo, hi, num_bits=4),
    jit_compile=True)(inp, mn, mx).numpy())
# [2.6, -0.4, 2.6]  — differs by 0.2 (one full quantization step)
```

## Reproduction

```python
import tensorflow as tf
import numpy as np

inp = tf.constant(np.linspace(-1, 3, 21).astype(np.float32))
mn = tf.constant(-0.5, dtype=tf.float32)
mx = tf.constant(2.5, dtype=tf.float32)

eager = tf.quantization.fake_quant_with_min_max_vars(inp, mn, mx, num_bits=4)
xla = tf.function(
    lambda v, lo, hi: tf.quantization.fake_quant_with_min_max_vars(v, lo, hi, num_bits=4),
    jit_compile=True)(inp, mn, mx)

# 6 out of 21 values differ:
#   inp=-1.0: eager=-0.6, xla=-0.4  (boundary clamped differently)
#   inp=-0.8: eager=-0.6, xla=-0.4
#   inp=-0.6: eager=-0.6, xla=-0.4
#   inp= 2.6: eager= 2.4, xla= 2.6
#   inp= 2.8: eager= 2.4, xla= 2.6
#   inp= 3.0: eager= 2.4, xla= 2.6
```

8-bit quantization is also affected:

```python
eager_8 = tf.quantization.fake_quant_with_min_max_vars(inp, mn, mx, num_bits=8)
xla_8 = tf.function(
    lambda v, lo, hi: tf.quantization.fake_quant_with_min_max_vars(v, lo, hi, num_bits=8),
    jit_compile=True)(inp, mn, mx)

# inp=-1.0: eager=-0.5059, xla=-0.4941  (diff=0.0118, one 8-bit step)
# inp= 2.6: eager= 2.4941, xla= 2.5059
```

Tested on TF 2.22.0-dev20260503.

## Root cause

The fake_quant nudging algorithm ensures 0.0 is exactly representable in the quantization grid:

1. `scale = (max - min) / (2^num_bits - 1)`
2. `zero_point_from_min = -min / scale`
3. `nudged_zero_point = round(zero_point_from_min)` ← **rounding differs**
4. `nudged_min = -nudged_zero_point * scale`
5. `nudged_max = (2^num_bits - 1 - nudged_zero_point) * scale`

For `min=-0.5, max=2.5, num_bits=4`:
- `scale = 3.0 / 15 = 0.2`
- `zero_point_from_min = 0.5 / 0.2 = 2.5` (half-integer)

At step 3:
- **Eager**: `std::round(2.5) = 3` (round-half-away-from-zero) → `nudged_min=-0.6`, `nudged_max=2.4`
- **XLA**: `round(2.5) = 2` (round-half-to-even) → `nudged_min=-0.4`, `nudged_max=2.6`

The entire quantization grid shifts by one step. Values near the boundaries are clamped to different levels, producing errors equal to the step size.

## Trigger condition

The bug triggers when `-min / scale` has a 0.5 fractional part. Examples:

| Range | `zero_point_from_min` | 4-bit error | 8-bit error |
|---|---|---|---|
| `[-0.5, 2.5]` | 2.5 | 0.200 | 0.012 |
| `[-0.1, 0.1]` | 7.5 | 0.013 | 0.001 |
| `[-1.5, 1.5]` | 7.5 | 0.200 | 0.012 |
| `[0, 1]` | 0.0 | — (OK) | — (OK) |
| `[-1, 1]` | 7.5 | boundary only | — (OK) |

## Scope

### Affected:
- `tf.quantization.fake_quant_with_min_max_vars`
- `tf.quantization.fake_quant_with_min_max_args`
- `tf.quantization.fake_quant_with_min_max_vars_per_channel`
- All `num_bits` values (4–16)
- CPU and GPU XLA

### Not affected:
- Ranges where `zero_point_from_min` is an integer (grid aligns identically)
- Eager execution (matches expected behavior)
- `narrow_range=True` with symmetric ranges

## Gradient is also affected

The gradient (used in quantization-aware training) is similarly wrong. The gradient is 1 inside the nudged range and 0 outside it — since the nudged range differs, the gradient differs at the boundaries:

```python
inp = tf.constant([-0.6, 2.6], dtype=tf.float32)
mn, mx = tf.constant(-0.5), tf.constant(2.5)

with tf.GradientTape() as tape:
    tape.watch(inp)
    y = tf.quantization.fake_quant_with_min_max_vars(inp, mn, mx, num_bits=4)
tape.gradient(y, inp)  # [1.0, 0.0]  — eager: -0.6 is inside range

<upstream maintainer>.function(jit_compile=True)
def xla_grad():
    with tf.GradientTape() as tape:
        tape.watch(inp)
        y = tf.quantization.fake_quant_with_min_max_vars(inp, mn, mx, num_bits=4)
    return tape.gradient(y, inp)
xla_grad()  # [0.0, 1.0]  — XLA: -0.6 is outside range
```

The gradient w.r.t. `min` and `max` is also swapped: eager gives `grad_min=4.0, grad_max=6.0`, XLA gives `grad_min=6.0, grad_max=4.0` (counts of clamped values are swapped).

## Impact

Fake quantization is the core primitive for quantization-aware training (QAT). This bug causes XLA-compiled QAT training to use a different quantization grid than eager evaluation, leading to:

1. **Silent accuracy regression**: quantized models trained under XLA may have slightly different weights than expected, with the error proportional to the quantization step size
2. **Non-reproducibility**: switching between eager and XLA changes quantized values, breaking reproducibility
3. **Validation mismatch**: post-training validation in eager mode sees different quantization boundaries than those used during XLA training

## Environment

- TensorFlow 2.22.0-dev20260503
- CPU and GPU XLA both affected
- Eager execution is correct
