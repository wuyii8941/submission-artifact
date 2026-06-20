#!/usr/bin/env python3
"""
Bug: XLA floordiv/floormod violates the division identity for large float32

The mathematical identity: floordiv(x, y) * y + floormod(x, y) == x
This holds under eager execution but breaks under XLA for large float32 values.

For x=1e10, y=3.0:
  Eager: floordiv*y + floormod = 1e10 (correct)
  XLA:   floordiv*y + floormod = 10000001024.0 (1024 too high)

Root cause: XLA's lowering of floordiv/floormod uses a different algorithm
that introduces rounding error for large values. The error grows with |x|.

Oracle: eager vs XLA (jit_compile=True)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

def divmod_reconstruct(x, y):
    return tf.math.floordiv(x, y) * y + tf.math.floormod(x, y)

test_cases = [
    (1e8, 7.0),
    (1e10, 3.0),
    (1e10, 7.0),
    (1e15, 3.0),
    (1e8, 0.03),
    (1e10, 0.007),
]

print("\nfloordiv(x,y)*y + floormod(x,y) should equal x:")
any_bug = False
for xv, yv in test_cases:
    x = tf.constant([xv], dtype=tf.float32)
    y = tf.constant([yv], dtype=tf.float32)

    eager_result = divmod_reconstruct(x, y).numpy()[0]
    xla_result = tf.function(divmod_reconstruct, jit_compile=True)(x, y).numpy()[0]

    eager_err = abs(eager_result - xv)
    xla_err = abs(xla_result - xv)

    if xla_err > eager_err + 1:
        any_bug = True
        print(f"\n  [BUG] x={xv:.0e}, y={yv}:")
        print(f"    eager: {eager_result}  (error={eager_err:.0f})")
        print(f"    XLA:   {xla_result}  (error={xla_err:.0f})")

assert any_bug, "Bug not reproduced"
print("\n[CONFIRMED] XLA floordiv/floormod violates division identity for large float32")
