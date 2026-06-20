#!/usr/bin/env python3
"""
Bug: XLA returns different position for NaN in tf.searchsorted (side='right')

When searching for NaN in a sorted sequence, the result depends on how
NaN comparisons are handled in the binary search:
  - Eager (side='right'): returns len(sequence) = 5 (NaN > everything)
  - XLA (side='right'): returns 0 (NaN < everything)

For side='left', both agree on 0. The inconsistency is only in side='right'.

The root cause is likely that XLA's binary search uses NaN-suppressing
comparisons (where NaN < x for all x), while eager's implementation
uses NaN-propagating comparisons (where NaN > x for all x, or NaN is
treated as larger than any finite value).

Oracle: eager vs XLA (jit_compile=True)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

sorted_seq = tf.constant([[1.0, 3.0, 5.0, 7.0, 9.0]], dtype=tf.float32)

# Test NaN value with both sides
nan_val = tf.constant([[float('nan')]], dtype=tf.float32)

def ss_left(s, v):
    return tf.searchsorted(s, v, side='left')
def ss_right(s, v):
    return tf.searchsorted(s, v, side='right')

e_l = ss_left(sorted_seq, nan_val).numpy()
x_l = tf.function(ss_left, jit_compile=True)(sorted_seq, nan_val).numpy()
e_r = ss_right(sorted_seq, nan_val).numpy()
x_r = tf.function(ss_right, jit_compile=True)(sorted_seq, nan_val).numpy()

print(f"sorted sequence: {sorted_seq.numpy()}")
print(f"\nsearchsorted(NaN, side='left'):")
print(f"  Eager: {e_l[0][0]}")
print(f"  XLA:   {x_l[0][0]}")
print(f"\nsearchsorted(NaN, side='right'):")
print(f"  Eager: {e_r[0][0]}")
print(f"  XLA:   {x_r[0][0]}")

# Also test with multiple NaN positions
values = tf.constant([[0.0, float('nan'), 10.0, float('nan')]], dtype=tf.float32)
e_r_multi = ss_right(sorted_seq, values).numpy()
x_r_multi = tf.function(ss_right, jit_compile=True)(sorted_seq, values).numpy()
print(f"\nMultiple values including NaN (side='right'):")
print(f"  Values: {values.numpy()}")
print(f"  Eager:  {e_r_multi}")
print(f"  XLA:    {x_r_multi}")

# Also test with NaN in the sorted sequence itself
sorted_nan = tf.constant([[1.0, float('nan'), 5.0, 7.0, 9.0]], dtype=tf.float32)
search_val = tf.constant([[3.0, 6.0]], dtype=tf.float32)
e_nan_seq = ss_right(sorted_nan, search_val).numpy()
x_nan_seq = tf.function(ss_right, jit_compile=True)(sorted_nan, search_val).numpy()
print(f"\nNaN IN sorted sequence (side='right'):")
print(f"  Sorted: {sorted_nan.numpy()}")
print(f"  Values: {search_val.numpy()}")
print(f"  Eager:  {e_nan_seq}")
print(f"  XLA:    {x_nan_seq}")

# Key assertion
assert e_r[0][0] != x_r[0][0], "Expected different NaN handling"
assert e_r[0][0] == 5, f"Eager should return 5 (NaN > everything), got {e_r[0][0]}"
assert x_r[0][0] == 0, f"XLA should return 0 (NaN < everything), got {x_r[0][0]}"

print("\n[CONFIRMED] searchsorted(side='right') NaN handling differs:")
print("  Eager: NaN is treated as > all values (returns end position)")
print("  XLA: NaN is treated as < all values (returns 0)")
