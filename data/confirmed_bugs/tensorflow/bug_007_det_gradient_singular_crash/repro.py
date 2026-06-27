#!/usr/bin/env python3
"""tf.linalg.det gradient crashes on singular matrices.

The forward pass correctly computes det=0 for singular matrices,
but the gradient computation crashes because it internally calls
matrix_inverse, which fails for non-invertible matrices.

The mathematical gradient of det(A) is adj(A)^T (the adjugate/cofactor matrix),
which is always well-defined regardless of invertibility.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

# Singular matrix: row 2 = 2 * row 1
m = tf.constant([[1.0, 2.0], [2.0, 4.0]], dtype=tf.float64)
print(f"Matrix:\n{m.numpy()}")
print(f"det = {tf.linalg.det(m).numpy()}")  # 0.0 (correct)

# Gradient crashes
print("\nComputing gradient of det...")
try:
    with tf.GradientTape() as tape:
        tape.watch(m)
        d = tf.linalg.det(m)
    g = tape.gradient(d, m)
    print(f"Gradient: {g.numpy() if g is not None else None}")
except tf.errors.InvalidArgumentError as e:
    print(f"CRASH: {e}")

# The correct gradient for det([[1,2],[2,4]]) is the cofactor matrix:
# adj([[1,2],[2,4]])^T = [[4,-2],[-2,1]]
print(f"\nCorrect gradient (cofactor matrix): [[4, -2], [-2, 1]]")

# Non-singular matrix works fine
print("\n--- Non-singular control case ---")
m2 = tf.constant([[2.0, 1.0], [1.0, 3.0]], dtype=tf.float64)
with tf.GradientTape() as tape:
    tape.watch(m2)
    d2 = tf.linalg.det(m2)
g2 = tape.gradient(d2, m2)
print(f"det = {d2.numpy()}, gradient = {g2.numpy()}")

# 3x3 singular matrix
print("\n--- 3x3 singular matrix ---")
m3 = tf.constant([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]], dtype=tf.float64)
print(f"det = {tf.linalg.det(m3).numpy()}")
try:
    with tf.GradientTape() as tape:
        tape.watch(m3)
        d3 = tf.linalg.det(m3)
    g3 = tape.gradient(d3, m3)
    print(f"Gradient:\n{g3.numpy() if g3 is not None else None}")
except tf.errors.InvalidArgumentError as e:
    print(f"CRASH: {e}")
