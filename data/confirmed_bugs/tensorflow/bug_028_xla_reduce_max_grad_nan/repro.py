#!/usr/bin/env python3
"""
Bug: XLA gradient of reduce_max/reduce_min returns 0 instead of NaN
     when input contains NaN

When the input to tf.reduce_max or tf.reduce_min contains NaN, the forward
pass correctly returns NaN in both eager and XLA. However, the backward pass
(gradient) differs:
  - Eager: gradient is NaN (correct — gradient through NaN should propagate NaN)
  - XLA:   gradient is 0  (wrong — silently drops the NaN signal)

This affects ALL array sizes (not size-dependent like the argmax NaN bug #022).

The root cause is in XLA's gradient implementation for reduce_max/reduce_min:
the gradient is computed as an indicator function (1 where input == max, 0 elsewhere).
When the max is NaN, the comparison input == NaN is always False (IEEE 754),
so all gradients become 0.

Oracle: eager vs XLA (jit_compile=True)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

any_bug = False

# ============================================================
print("\n=== Part 1: reduce_max gradient with NaN at various positions ===")
test_cases = [
    ("all_NaN_sz3", [float('nan')] * 3),
    ("all_NaN_sz5", [float('nan')] * 5),
    ("all_NaN_sz8", [float('nan')] * 8),
    ("NaN_at_end", [1.0, 2.0, float('nan')]),
    ("NaN_at_start", [float('nan'), 1.0, 2.0]),
    ("NaN_in_middle", [1.0, float('nan'), 3.0]),
    ("one_NaN_sz5", [1.0, 2.0, float('nan'), 3.0, 4.0]),
]

print(f"{'case':25s}  {'fwd_eager':>10s}  {'fwd_xla':>10s}  {'grad_eager':>30s}  {'grad_xla':>30s}  {'status'}")
for name, vals in test_cases:
    x = tf.constant(vals)

    # Forward
    fwd_eager = tf.reduce_max(x).numpy()
    fwd_xla = tf.function(tf.reduce_max, jit_compile=True)(x).numpy()

    # Gradient - eager
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_max(x)
    g_eager = t.gradient(y, x).numpy()

    # Gradient - XLA
    @tf.function(jit_compile=True)
    def grad_max(x):
        with tf.GradientTape() as t:
            t.watch(x)
            y = tf.reduce_max(x)
        return t.gradient(y, x)
    g_xla = grad_max(x).numpy()

    fwd_match = (np.isnan(fwd_eager) == np.isnan(fwd_xla))
    grad_match = np.allclose(g_eager, g_xla, equal_nan=True)

    if not grad_match:
        any_bug = True

    print(f"{name:25s}  {fwd_eager:10.4g}  {fwd_xla:10.4g}  {str(g_eager):>30s}  {str(g_xla):>30s}  "
          f"fwd={'OK' if fwd_match else 'DIFF'} grad={'OK' if grad_match else '*** BUG ***'}")

# ============================================================
print("\n=== Part 2: reduce_min gradient with NaN ===")
for name, vals in [("all_NaN_sz3", [float('nan')]*3), ("NaN_at_end", [1.0, 2.0, float('nan')])]:
    x = tf.constant(vals)

    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_min(x)
    g_eager = t.gradient(y, x).numpy()

    @tf.function(jit_compile=True)
    def grad_min(x):
        with tf.GradientTape() as t:
            t.watch(x)
            y = tf.reduce_min(x)
        return t.gradient(y, x)
    g_xla = grad_min(x).numpy()

    grad_match = np.allclose(g_eager, g_xla, equal_nan=True)
    if not grad_match:
        any_bug = True
    print(f"  reduce_min {name:20s}: eager_grad={g_eager}, xla_grad={g_xla} {'OK' if grad_match else '*** BUG ***'}")

# ============================================================
print("\n=== Part 3: Practical impact — gradient through max-based loss ===")

x = tf.constant([1.0, float('nan'), 3.0, 2.0])

# Eager
with tf.GradientTape() as t:
    t.watch(x)
    loss = tf.reduce_max(x)  # max of [1, NaN, 3, 2] = NaN
g_eager = t.gradient(loss, x).numpy()

# XLA
@tf.function(jit_compile=True)
def max_loss_grad(x):
    with tf.GradientTape() as t:
        t.watch(x)
        loss = tf.reduce_max(x)
    return t.gradient(loss, x)
g_xla = max_loss_grad(x).numpy()

print(f"  Loss = reduce_max([1, NaN, 3, 2])")
print(f"  Eager gradient: {g_eager}  (NaN propagated — correct)")
print(f"  XLA gradient:   {g_xla}  (0 everywhere — WRONG)")
print(f"  Impact: NaN in input silently produces zero gradient under XLA,")
print(f"          hiding the problem instead of propagating the error signal.")

# ============================================================
print("\n=== Part 4: Verify forward pass agrees (not a forward bug) ===")
for vals in [[float('nan')]*3, [1.0, 2.0, float('nan')]]:
    x = tf.constant(vals)
    fwd_eager = tf.reduce_max(x).numpy()
    fwd_xla = tf.function(tf.reduce_max, jit_compile=True)(x).numpy()
    print(f"  reduce_max({vals}): eager={fwd_eager}, xla={fwd_xla} — "
          f"{'both NaN, AGREE' if np.isnan(fwd_eager) and np.isnan(fwd_xla) else 'DIFFER'}")

# ============================================================
print("\n=== Part 5: Combined gradient (max + min) with NaN ===")
x = tf.constant([1.0, 2.0, float('nan'), 4.0])
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_max(x) + tf.reduce_min(x)
g_eager = t.gradient(y, x).numpy()

@tf.function(jit_compile=True)
def grad_max_plus_min(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_max(x) + tf.reduce_min(x)
    return t.gradient(y, x)
g_xla = grad_max_plus_min(x).numpy()
print(f"  grad(reduce_max + reduce_min):")
print(f"    eager: {g_eager}")
print(f"    xla:   {g_xla}")

assert any_bug
print("\n[CONFIRMED] XLA gradient of reduce_max/reduce_min returns 0 instead of NaN")
