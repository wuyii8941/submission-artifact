#!/usr/bin/env python3
"""
Bug: CPU eager tf.math.atan2 returns NaN for subnormal float32 inputs

When both y and x are subnormal (< ~1.175e-38), CPU atan2 returns NaN
instead of the correct finite value (e.g., π/4 for equal positive values).

GPU returns 0 (wrong due to flush-to-zero), XLA returns the correct value.

Root cause: CPU atan2 kernel likely computes y²+x² internally, which
underflows to 0 for subnormal inputs, causing a division by zero
in the normalization step.

Oracle: CPU eager vs XLA / NumPy
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

tiny = np.finfo(np.float32).tiny  # ~1.175e-38
test_vals = [
    (1e-40, 1e-40, np.pi/4),
    (1e-45, 1e-45, np.pi/4),
    (tiny/2, tiny/2, np.pi/4),
    (1e-38, -1e-38, 3*np.pi/4),
    (-1e-38, 1e-38, -np.pi/4),
]

print(f"\natan2(y, x) for subnormal inputs (all < float32 tiny = {tiny:.3e}):")
any_bug = False
for yv, xv, expected in test_vals:
    y = tf.constant([yv], dtype=tf.float32)
    x = tf.constant([xv], dtype=tf.float32)

    with tf.device('/CPU:0'):
        cpu = tf.math.atan2(tf.identity(y), tf.identity(x)).numpy()[0]
    with tf.device('/GPU:0'):
        gpu = tf.math.atan2(tf.identity(y), tf.identity(x)).numpy()[0]
    xla = tf.function(lambda y,x: tf.math.atan2(y,x), jit_compile=True)(y, x).numpy()[0]
    np_ref = np.arctan2(np.float32(yv), np.float32(xv))

    print(f"\n  y={yv:.2e}, x={xv:.2e}  (expected ≈ {expected:.4f}):")
    print(f"    CPU:  {cpu:10.6f}  {'*** NaN ***' if np.isnan(cpu) else ''}")
    print(f"    GPU:  {gpu:10.6f}  {'*** WRONG (FTZ)' if abs(gpu - expected) > 0.1 else ''}")
    print(f"    XLA:  {xla:10.6f}  {'correct' if abs(xla - expected) < 0.001 else ''}")
    print(f"    NumPy:{np_ref:10.6f}")

    if np.isnan(cpu):
        any_bug = True

assert any_bug, "Bug not reproduced"
print("\n[CONFIRMED] CPU atan2 returns NaN for subnormal float32 inputs")
