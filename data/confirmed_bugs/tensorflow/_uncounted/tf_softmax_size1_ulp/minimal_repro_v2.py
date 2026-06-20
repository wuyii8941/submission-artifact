"""Minimal repro: tf.nn.softmax over size-1 axis on CPU returns 1.0 - 1 ULP
instead of exactly 1.0, when batch dim >= 8.

Mathematical expectation: softmax(x, axis=-1) where last axis has size 1:
  numerator = exp(x - max(x)) = exp(0) = 1.0
  denominator = sum(numerator) = 1.0
  result = 1.0 / 1.0 = 1.0  (exactly)

Observed on CPU (TF 2.21.0):
  For batches < 8: result is exactly 1.0
  For batches >= 8: the first (N // 8) * 8 positions return 0.99999994
                    (which is 1.0 - 2^-24 = 1 ULP below 1.0)
                    the remaining N % 8 positions return 1.0 correctly

This is exposed by floor(softmax(...)), which should return ones but returns zeros
for the vectorized positions.

GPU path is correct.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

print(f"TF version: {tf.__version__}")

# Minimal failure
x = tf.ones((8, 1), dtype=tf.float32)

with tf.device('/CPU:0'):
    sm = tf.nn.softmax(x, axis=-1)
    print(f"CPU softmax(ones(8,1)): {sm.numpy().flatten()}")
    # Expected: [1. 1. 1. 1. 1. 1. 1. 1.]
    # Actual:   [0.99999994 ... 0.99999994]
    print(f"  all exactly 1.0: {bool((sm.numpy() == 1.0).all())}")
    print(f"  floor sum: {float(tf.reduce_sum(tf.math.floor(sm)).numpy())}  (expected 8)")

with tf.device('/GPU:0'):
    x_g = tf.identity(x)
    sm = tf.nn.softmax(x_g, axis=-1)
    print(f"GPU softmax(ones(8,1)): {sm.numpy().flatten()}")
    print(f"  all exactly 1.0: {bool((sm.numpy() == 1.0).all())}")
    print(f"  floor sum: {float(tf.reduce_sum(tf.math.floor(sm)).numpy())}  (expected 8)")

# Boundary pattern
print("\nBoundary pattern (CPU only, expected=N, observed=N - (N//8)*8):")
for N in [1, 4, 7, 8, 15, 16, 127, 128, 129]:
    ones_N = tf.ones((N, 1), dtype=tf.float32)
    with tf.device('/CPU:0'):
        s = tf.nn.softmax(tf.identity(ones_N), axis=-1)
        got = float(tf.reduce_sum(tf.math.floor(s)).numpy())
    print(f"  N={N:4d}:  expected={N:4d}  got={int(got):4d}  diff={N - int(got):4d}")
