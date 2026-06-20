"""tf.math.top_k NaN handling is position-dependent in eager.

Same values {1, 2, 3, nan} produce different top-k results depending
on where NaN appears in the input array.

Also affects tf.sort, tf.argsort, tf.math.argmax, tf.math.argmin.
TF 2.22.0-dev20260429
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print("=== Eager: position-dependent NaN in top_k ===")
t1 = tf.constant([float('nan'), 1.0, 2.0, 3.0])
t2 = tf.constant([1.0, 2.0, 3.0, float('nan')])

v1, i1 = tf.math.top_k(t1, k=3)
v2, i2 = tf.math.top_k(t2, k=3)
print(f"NaN at front: vals={v1.numpy()}, idx={i1.numpy()}")  # NaN selected
print(f"NaN at back:  vals={v2.numpy()}, idx={i2.numpy()}")  # NaN dropped

print("\n=== XLA: consistent (NaN always max) ===")
v1x, i1x = tf.function(lambda v: tf.math.top_k(v, k=3), jit_compile=True)(t1)
v2x, i2x = tf.function(lambda v: tf.math.top_k(v, k=3), jit_compile=True)(t2)
print(f"NaN at front: vals={v1x.numpy()}, idx={i1x.numpy()}")
print(f"NaN at back:  vals={v2x.numpy()}, idx={i2x.numpy()}")

print("\n=== tf.sort: inconsistent NaN placement in eager ===")
t = tf.constant([3.0, float('nan'), 1.0, 2.0])
print(f"eager DESC: {tf.sort(t, direction='DESCENDING').numpy()}")
print(f"XLA   DESC: {tf.function(lambda v: tf.sort(v, direction='DESCENDING'), jit_compile=True)(t).numpy()}")
print(f"eager ASC:  {tf.sort(t, direction='ASCENDING').numpy()}")
print(f"XLA   ASC:  {tf.function(lambda v: tf.sort(v, direction='ASCENDING'), jit_compile=True)(t).numpy()}")

print("\n=== tf.math.argmax: eager ignores NaN, XLA treats NaN as max ===")
t = tf.constant([1.0, 2.0, float('nan')])
print(f"eager argmax: {tf.math.argmax(t).numpy()}")  # 1
print(f"XLA   argmax: {tf.function(lambda v: tf.math.argmax(v), jit_compile=True)(t).numpy()}")  # 2
