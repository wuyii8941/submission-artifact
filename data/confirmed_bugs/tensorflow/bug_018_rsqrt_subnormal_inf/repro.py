#!/usr/bin/env python3
"""
Bug: Eager tf.math.rsqrt returns inf for subnormal float32 inputs

Both CPU and GPU eager return inf for rsqrt(x) when x is subnormal
(< ~1.175e-38). The correct result is a large but finite number —
rsqrt(1e-40) = 1/sqrt(1e-40) = 1e20, well within float32 range.

XLA returns the correct finite result.

Root cause: Eager rsqrt kernel doesn't handle subnormal inputs correctly.
On GPU this is likely FTZ (subnormal flushed to 0 → rsqrt(0)=inf).
On CPU the intermediate sqrt(subnormal) may underflow to 0 or the rsqrt
hardware instruction doesn't support subnormals.

Oracle: eager vs XLA (jit_compile=True)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

tiny = np.finfo(np.float32).tiny
test_vals = [1e-40, 1e-42, 1e-45, tiny/4, tiny/2]

print(f"\ntf.math.rsqrt(x) for subnormal float32 inputs (tiny={tiny:.3e}):")
any_bug = False
for xv in test_vals:
    x = tf.constant([xv], dtype=tf.float32)

    with tf.device('/CPU:0'):
        cpu = tf.math.rsqrt(tf.identity(x)).numpy()[0]
    with tf.device('/GPU:0'):
        gpu = tf.math.rsqrt(tf.identity(x)).numpy()[0]
    xla = tf.function(tf.math.rsqrt, jit_compile=True)(x).numpy()[0]

    expected_approx = 1.0 / np.sqrt(np.float64(xv))
    cpu_wrong = np.isinf(cpu) and np.isfinite(xla)
    gpu_wrong = np.isinf(gpu) and np.isfinite(xla)

    print(f"\n  x={xv:.2e}  (expected ≈ {expected_approx:.3e}):")
    print(f"    CPU:  {cpu:15.4e}  {'*** inf (WRONG) ***' if cpu_wrong else ''}")
    print(f"    GPU:  {gpu:15.4e}  {'*** inf (FTZ) ***' if gpu_wrong else ''}")
    print(f"    XLA:  {xla:15.4e}  (correct)")

    if cpu_wrong or gpu_wrong:
        any_bug = True

assert any_bug, "Bug not reproduced"
print("\n[CONFIRMED] Eager rsqrt returns inf for subnormal float32 inputs; XLA is correct")
