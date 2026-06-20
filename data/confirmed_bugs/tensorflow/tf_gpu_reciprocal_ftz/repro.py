#!/usr/bin/env python3
"""
Bug: GPU eager x * reciprocal(x) != 1 for values near float32 extremes

When x is large (e.g., 1e38), reciprocal(x) = 1/x ≈ 1e-38 which is
subnormal. GPU FTZ flushes this to 0, giving x * 0 = 0 instead of ≈ 1.

Similarly, when x is subnormal, reciprocal(x) overflows, and the
subsequent multiplication produces NaN (0 * inf = NaN on GPU, since x
was flushed to 0).

XLA correctly computes x * (1/x) ≈ 1 in both cases.

Oracle: eager (GPU) vs XLA
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

def x_times_reciprocal(x):
    return x * tf.math.reciprocal(x)

test_vals = [
    (1e-38, "subnormal input"),
    (1e-40, "deep subnormal input"),
    (1e38, "large input (reciprocal is subnormal)"),
    (5e37, "large input (reciprocal near subnormal)"),
]

print("\nx * reciprocal(x) should ≈ 1.0:")
any_bug = False
for xv, desc in test_vals:
    x = tf.constant([xv], dtype=tf.float32)
    eager = x_times_reciprocal(x).numpy()[0]
    xla = tf.function(x_times_reciprocal, jit_compile=True)(x).numpy()[0]

    bug = (np.isnan(eager) or abs(eager - 1.0) > 0.5) and abs(xla - 1.0) < 0.01

    print(f"\n  x={xv:.0e} ({desc}):")
    print(f"    eager(GPU): {eager}  {'*** WRONG ***' if bug else ''}")
    print(f"    XLA:        {xla}")

    if bug:
        any_bug = True

assert any_bug, "Bug not reproduced"
print("\n[CONFIRMED] GPU FTZ on intermediates causes x*reciprocal(x) != 1")
