#!/usr/bin/env python3
"""tf.math.xlogy gradient w.r.t. x is incorrectly zero when x=0.

xlogy(x, y) = x * log(y)  (with xlogy(0, y) = 0 by convention)
d/dx xlogy(x, y) = log(y)  for all x (including x=0, when y > 0)

TF returns gradient 0 at x=0 instead of log(y).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

x = tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float64)
y = tf.constant([2.0, 5.0, 10.0, 2.0], dtype=tf.float64)

with tf.GradientTape() as tape:
    tape.watch(x)
    out = tf.math.xlogy(x, y)
g = tape.gradient(out, x)

correct = np.log(y.numpy())
print(f"x = {x.numpy()}")
print(f"y = {y.numpy()}")
print(f"xlogy(x, y)       = {out.numpy()}")
print(f"TF d/dx gradient  = {g.numpy()}")
print(f"Correct gradient  = {correct}")
print(f"\nBUG: gradient is 0 at x=0 positions, should be log(y)")

# Verify with numerical gradient
eps = 1e-7
x_np = x.numpy()
grad_num = np.zeros_like(x_np)
for i in range(len(x_np)):
    xp = x_np.copy(); xp[i] += eps
    xm = x_np.copy(); xm[i] -= eps
    fp = tf.math.xlogy(tf.constant(xp), y).numpy()
    fm = tf.math.xlogy(tf.constant(xm), y).numpy()
    grad_num[i] = (fp[i] - fm[i]) / (2 * eps)
print(f"\nNumerical gradient = {grad_num}")
print(f"Max error (TF vs numerical) = {np.max(np.abs(g.numpy() - grad_num)):.6e}")

# Also test gradient w.r.t. y at x=0
print("\n--- Gradient w.r.t. y at x=0 ---")
with tf.GradientTape() as tape:
    tape.watch(y)
    out = tf.math.xlogy(x, y)
gy = tape.gradient(out, y)
print(f"TF d/dy gradient  = {gy.numpy()}")
print(f"Correct d/dy      = {(x / y).numpy()}")
print(f"d/dy is correct (0/y = 0 when x=0)")
