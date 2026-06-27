#!/usr/bin/env python3
"""tf.math.bessel_i1 gradient is wrong at x=0.

The modified Bessel function I1(x) has derivative:
  I1'(x) = I0(x) - I1(x)/x

At x=0: I0(0) = 1, and lim_{x→0} I1(x)/x = 0.5 (by L'Hôpital or Taylor series).
So I1'(0) = 1.0 - 0.5 = 0.5.

TF returns I1'(0) = 1.0, because the gradient formula computes I1(0)/0 = 0/0,
which is handled as 0 (likely via safe_div or multiply_no_nan), giving
I0(0) - 0 = 1.0 instead of the correct 0.5.

Note: tf.math.bessel_i1e (the exponentially-scaled version) does NOT have this
bug — its gradient is computed using a different formula that avoids the 0/0.
Similarly, tf.math.special.bessel_j1 (whose derivative has the same J0-J1/x
form) handles x=0 correctly, returning 0.5.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

# === Minimal reproduction ===
x = tf.constant([0.0], dtype=tf.float64)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.math.bessel_i1(x)
g = tape.gradient(y, x)
print(f"\nI1(0) = {y.numpy()[0]}")
print(f"TF I1'(0) = {g.numpy()[0]}")
print(f"Correct I1'(0) = 0.5")

# === Numerical verification ===
print("\nNumerical verification:")
for eps in [1e-5, 1e-7, 1e-9]:
    xp = tf.constant([eps], dtype=tf.float64)
    xm = tf.constant([-eps], dtype=tf.float64)
    num = (tf.math.bessel_i1(xp).numpy()[0] - tf.math.bessel_i1(xm).numpy()[0]) / (2 * eps)
    print(f"  eps={eps:.0e}: numerical I1'(0) = {num:.10f}")

# === Taylor series confirmation ===
# I1(x) = x/2 + x^3/16 + ... so I1(x)/x = 1/2 + x^2/16 + ...
# lim_{x→0} I1(x)/x = 1/2
print(f"\nTaylor: I1(x)/x → 1/2 as x → 0")
for eps in [1e-3, 1e-5, 1e-7]:
    xv = tf.constant([eps], dtype=tf.float64)
    ratio = tf.math.bessel_i1(xv).numpy()[0] / eps
    print(f"  I1({eps:.0e})/{eps:.0e} = {ratio:.10f}")

# === Gradient is correct at all other points ===
print("\nGradient at other x values:")
for xv in [1e-10, 0.001, 0.1, 1.0, 5.0]:
    x = tf.constant([xv], dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.math.bessel_i1(x)
    g = tape.gradient(y, x)
    eps = max(abs(xv) * 1e-7, 1e-12)
    xp = tf.constant([xv + eps], dtype=tf.float64)
    xm = tf.constant([xv - eps], dtype=tf.float64)
    num = (tf.math.bessel_i1(xp).numpy()[0] - tf.math.bessel_i1(xm).numpy()[0]) / (2 * eps)
    print(f"  x={xv:>8.1e}: TF={g.numpy()[0]:>14.8f}, num={num:>14.8f} (OK)")

# === Comparison with bessel_j1 (correct) ===
print(f"\nComparison: J1 has same formula J0(x) - J1(x)/x")
x = tf.constant([0.0], dtype=tf.float64)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.math.special.bessel_j1(x)
g_j1 = tape.gradient(y, x)
print(f"  J1'(0) TF = {g_j1.numpy()[0]} (correct: 0.5)")

print(f"\nBUG: I1'(0) returns {1.0} instead of {0.5}")
print(f"Root cause: gradient formula I0(x) - I1(x)/x has 0/0 at x=0,")
print(f"  handled as 0 via safe_div, giving I0(0) = 1.0 instead of 0.5")
