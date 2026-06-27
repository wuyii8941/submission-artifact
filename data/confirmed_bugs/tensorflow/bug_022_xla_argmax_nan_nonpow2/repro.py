#!/usr/bin/env python3
"""
Bug: XLA argmax/argmin returns wrong index when NaN is present
     for non-power-of-2 array sizes

XLA's tree-reduction implementation of argmax/argmin returns the LAST index
when NaN is present in the input, instead of the correct answer. This only
happens for non-power-of-2 sizes (3, 5, 6, 7, 9, 10, ...). Power-of-2
sizes (2, 4, 8, 16) work correctly.

Root cause: XLA's argmax/argmin reduction tree handles "leftover" elements
(when size is not power of 2) with NaN comparisons that always favor the
last element.

Examples:
  argmax([1.0, 2.0, NaN]) → eager: 1 (correct), XLA: 2 (wrong)
  argmax([NaN]*5)          → eager: 0, XLA: 4 (last index)
  argmax([NaN]*4)          → eager: 0, XLA: 0 (correct! power of 2)

This also affects compositions like argmax(softmax(x)) with NaN input,
since softmax propagates NaN to all outputs, creating an all-NaN array
whose argmax is then wrong.

Oracle: eager vs XLA (jit_compile=True)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

print("\n=== Part 1: argmax(all-NaN) by array size ===")
print(f"{'size':>5s}  {'eager':>6s}  {'XLA':>6s}  {'power_of_2':>12s}  {'status'}")
any_bug = False
for sz in range(2, 20):
    x = tf.constant([float('nan')] * sz)
    eager = tf.math.argmax(x).numpy()
    xla = tf.function(tf.math.argmax, jit_compile=True)(x).numpy()
    is_pow2 = (sz & (sz - 1) == 0)
    match = eager == xla
    if not match:
        any_bug = True
    print(f"{sz:5d}  {eager:6d}  {xla:6d}  {'yes' if is_pow2 else 'no':>12s}  "
          f"{'OK' if match else '*** BUG ***'}")

print("\n=== Part 2: argmax with NaN at end (size 3) ===")
x = tf.constant([1.0, 2.0, float('nan')])
eager = tf.math.argmax(x).numpy()
xla = tf.function(tf.math.argmax, jit_compile=True)(x).numpy()
print(f"  argmax([1, 2, NaN]): eager={eager} (correct: 2.0 at idx 1), XLA={xla}")
assert eager != xla, "Bug not reproduced"

print("\n=== Part 3: argmin with NaN in middle (size 3) ===")
x = tf.constant([1.0, float('nan'), 3.0])
eager = tf.math.argmin(x).numpy()
xla = tf.function(tf.math.argmin, jit_compile=True)(x).numpy()
print(f"  argmin([1, NaN, 3]): eager={eager} (correct: 1.0 at idx 0), XLA={xla}")

print("\n=== Part 4: Practical impact — argmax(softmax(x)) with NaN ===")
def argmax_softmax(x):
    return tf.math.argmax(tf.nn.softmax(x), axis=-1)
x = tf.constant([[1.0, float('nan'), 2.0]])
sm = tf.nn.softmax(x).numpy()
eager = argmax_softmax(x).numpy()
xla = tf.function(argmax_softmax, jit_compile=True)(x).numpy()
print(f"  softmax([1, NaN, 2]) = {sm}  (all NaN)")
print(f"  argmax of that: eager={eager[0]}, XLA={xla[0]}")
print(f"  (XLA returns {xla[0]} because all-NaN + size 3 → last index)")

print("\n=== Part 5: Verify power-of-2 sizes are correct ===")
for sz in [2, 4, 8, 16]:
    x = tf.constant([float('nan')] * sz)
    eager = tf.math.argmax(x).numpy()
    xla = tf.function(tf.math.argmax, jit_compile=True)(x).numpy()
    print(f"  size={sz}: argmax(all-NaN): eager={eager}, XLA={xla}  {'OK' if eager==xla else 'BUG'}")

assert any_bug
print("\n[CONFIRMED] XLA argmax/argmin NaN handling wrong for non-power-of-2 sizes")
