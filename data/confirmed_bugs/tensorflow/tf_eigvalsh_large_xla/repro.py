#!/usr/bin/env python3
"""
Bug: tf.linalg.eigvalsh returns wrong eigenvalues under XLA for
     matrices with large-magnitude entries.

For the rank-1 matrix [[v, v], [v, v]], eigenvalues are exactly [0, 2v].
Both values are representable in float32 for v up to ~1.7e38.

Under XLA (jit_compile=True), for v >= 1e19, eigvalsh returns [v, v]
(the diagonal elements) instead of [0, 2v]. Eager mode and NumPy
return the correct result.

This also affects non-rank-1 symmetric matrices with large entries
and 3x3+ matrices.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

# 2x2 rank-1 matrix
print("\n2x2 rank-1 matrix [[v, v], [v, v]]:")
print(f"{'v':>10s}  {'eager':>30s}  {'XLA':>30s}  {'NumPy':>30s}")
for exp in [10, 15, 18, 19, 20, 25, 30]:
    v = 10.0 ** exp
    m = tf.constant([[v, v], [v, v]], dtype=tf.float32)
    eager = tf.linalg.eigvalsh(m).numpy()
    xla = tf.function(tf.linalg.eigvalsh, jit_compile=True)(m).numpy()
    np_ref = np.linalg.eigvalsh(m.numpy())
    print(f"  {v:>8.0e}  {str(eager):>30s}  {str(xla):>30s}  {str(np_ref):>30s}")

# Non-rank-1 matrix
print("\n2x2 non-rank-1: [[v, v/2], [v/2, v]], eigenvalues ≈ [v/2, 3v/2]:")
for exp in [15, 19, 20, 25]:
    v = 10.0 ** exp
    m = tf.constant([[v, v/2], [v/2, v]], dtype=tf.float32)
    eager = tf.linalg.eigvalsh(m).numpy()
    xla = tf.function(tf.linalg.eigvalsh, jit_compile=True)(m).numpy()
    np_ref = np.linalg.eigvalsh(m.numpy())
    match = np.allclose(eager, xla, rtol=1e-3)
    print(f"  v={v:.0e}: eager={eager}, XLA={xla} [{'OK' if match else 'DIFF'}]")

# Key assertions
v = 1e20
m = tf.constant([[v, v], [v, v]], dtype=tf.float32)
eager = tf.linalg.eigvalsh(m).numpy()
xla = tf.function(tf.linalg.eigvalsh, jit_compile=True)(m).numpy()

assert eager[0] < 1.0, f"Eager smallest eigenvalue should be ~0, got {eager[0]}"
assert abs(eager[1] - 2e20) / 2e20 < 0.01, f"Eager largest eigenvalue should be ~2e20, got {eager[1]}"
assert xla[0] > 1e19, f"XLA incorrectly returns {xla[0]} instead of ~0"

print(f"\n[CONFIRMED] eigvalsh([[1e20, 1e20], [1e20, 1e20]]):")
print(f"  Eager: {eager}  (correct: [0, 2e20])")
print(f"  XLA:   {xla}  (wrong: returns diagonal)")
