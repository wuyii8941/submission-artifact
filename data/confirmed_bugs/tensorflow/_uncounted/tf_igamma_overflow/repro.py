"""Reproducer: tf.math.igamma returns values outside [0, 1] for large arguments."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

print("\n=== igamma returns values > 1.0 (out of mathematical range) ===")
for v in [800000, 1000000, 1842069, 8685113]:
    a = tf.constant(float(v), dtype=tf.float32)
    result = tf.math.igamma(a, a).numpy()
    print(f"  igamma({v}, {v}) = {result:.6f}  {'** OUT OF RANGE **' if result > 1.0 else ''}")

print("\n=== igamma + igammac should always equal 1.0 ===")
for v in [100, 1000, 10000, 100000, 800000, 1000000]:
    a = tf.constant(float(v), dtype=tf.float32)
    ig = tf.math.igamma(a, a).numpy()
    igc = tf.math.igammac(a, a).numpy()
    print(f"  a=x={v:>10d}: igamma={ig:.6f}  igammac={igc:.6f}  sum={ig+igc:.6f}")

print("\n=== Small values are correct ===")
for v in [1, 10, 100, 1000]:
    a = tf.constant(float(v), dtype=tf.float32)
    ig = tf.math.igamma(a, a).numpy()
    print(f"  igamma({v}, {v}) = {ig:.8f}  (in [0,1]: {0 <= ig <= 1})")
