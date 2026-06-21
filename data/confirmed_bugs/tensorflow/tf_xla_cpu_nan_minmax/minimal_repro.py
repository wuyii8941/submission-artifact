"""Minimal repro: CPU XLA silently drops NaN in reduce_max / reduce_min.

jit_compile=True on CPU changes tf.reduce_max and tf.reduce_min to ignore NaN,
returning the max/min of the finite values instead. Eager mode correctly
propagates NaN, matching NumPy.

This was previously fixed for eager in tensorflow/tensorflow#<anon> and #<anon>,
but the XLA CPU lowering was never updated to match.

JAX 0.9.2 (same XLA backend) correctly propagates NaN on CPU, confirming this
is a TF-specific XLA integration issue.

Reproduces on TF 2.21.0 (stable) and TF 2.22.0-dev20260424 (nightly).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

vals = tf.constant([float('nan'), 1.0, 3.0, 2.0], dtype=tf.float32)

print("\n=== reduce_max ===")
eager = tf.reduce_max(vals).numpy()
xla = tf.function(lambda x: tf.reduce_max(x), jit_compile=True)(vals).numpy()
print(f"  eager:  {eager}")   # nan  (correct, matches np.max)
print(f"  XLA:    {xla}")     # 3.0  (wrong, NaN silently dropped)
print(f"  numpy:  {np.max(vals.numpy())}")

print("\n=== reduce_min ===")
eager = tf.reduce_min(vals).numpy()
xla = tf.function(lambda x: tf.reduce_min(x), jit_compile=True)(vals).numpy()
print(f"  eager:  {eager}")   # nan
print(f"  XLA:    {xla}")     # 1.0

print("\n=== maximum (scalar tensors) ===")
a = tf.constant(5.0, dtype=tf.float32)
b = tf.constant(float('nan'), dtype=tf.float32)
eager = tf.math.maximum(a, b).numpy()
xla = tf.function(lambda x, y: tf.math.maximum(x, y), jit_compile=True)(a, b).numpy()
print(f"  eager:  {eager}")   # nan
print(f"  XLA:    {xla}")     # 5.0

print("\n=== minimum (scalar tensors) ===")
eager = tf.math.minimum(a, b).numpy()
xla = tf.function(lambda x, y: tf.math.minimum(x, y), jit_compile=True)(a, b).numpy()
print(f"  eager:  {eager}")   # nan
print(f"  XLA:    {xla}")     # 5.0

print("\n=== Practical impact: argmax gives wrong index ===")
data = tf.constant([1.0, float('nan'), 3.0, 2.0])
def find_max_idx(x):
    return tf.argmax(x)
eager_idx = find_max_idx(data).numpy()
xla_idx = tf.function(find_max_idx, jit_compile=True)(data).numpy()
print(f"  eager argmax: {eager_idx}")
print(f"  XLA argmax:   {xla_idx}")

print("\n=== reduce_sum/mean: NaN correctly propagated (not affected) ===")
for name, fn in [('reduce_sum', tf.reduce_sum), ('reduce_mean', tf.reduce_mean)]:
    e = fn(vals).numpy()
    x = tf.function(lambda v: fn(v), jit_compile=True)(vals).numpy()
    print(f"  {name}: eager={e}, XLA={x}, match={str(e)==str(x)}")
