#!/usr/bin/env python3
"""
Bug: XLA tf.math.erfinv loses precision for float64 near ±1

For float64 inputs very close to 1 (e.g., 1 - 1e-15), XLA and eager
produce significantly different results. The XLA result has lower precision,
suggesting a different algorithm or precision loss in the XLA implementation.

For x = 1.0 - 1e-15:
  Eager: erfinv ≈ 5.6861  (matches scipy to 4 digits)
  XLA:   erfinv ≈ 5.6759  (off by ~0.01, relative error ~0.18%)

Root cause: XLA's erfinv implementation likely uses a different polynomial
approximation or fails to handle the extreme tail correctly for float64.

Oracle: eager vs XLA (jit_compile=True)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

test_vals = [
    0.99,
    0.999,
    0.9999,
    0.99999,
    0.999999,
    0.9999999,
    0.99999999,
    1.0 - 1e-10,
    1.0 - 1e-12,
    1.0 - 1e-15,
]

print("\ntf.math.erfinv(x) for float64 near 1.0:")
any_bug = False
for xv in test_vals:
    x = tf.constant([xv], dtype=tf.float64)
    eager = tf.math.erfinv(x).numpy()[0]
    xla = tf.function(tf.math.erfinv, jit_compile=True)(x).numpy()[0]
    diff = abs(eager - xla)
    rel = diff / (abs(eager) + 1e-30)

    if diff > 1e-4:
        any_bug = True
        print(f"  [BUG] x=1-{1-xv:.0e}: eager={eager:.10f} xla={xla:.10f} diff={diff:.2e} rel={rel:.2e}")
    else:
        print(f"  OK    x=1-{1-xv:.0e}: eager={eager:.10f} xla={xla:.10f}")

assert any_bug, "Bug not reproduced"
print("\n[CONFIRMED] XLA erfinv loses float64 precision near ±1")
