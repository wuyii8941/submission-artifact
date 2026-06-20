"""Reproducer: XLA complex sin/cos overflow for large imaginary part."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

print("\n=== tf.math.sin ===")
for re, im in [(0, -88), (1, -88), (0, -89), (1, 89)]:
    z = tf.constant(np.complex64(complex(re, im)).reshape(1, 1))
    eager = tf.math.sin(z).numpy()[0, 0]
    xla = tf.function(tf.math.sin, jit_compile=True)(z).numpy()[0, 0]
    print(f"  sin({re}{im:+d}j): eager={eager}  xla={xla}")

print("\n=== tf.math.cos ===")
for re, im in [(0, -88), (0, -89), (0, 89), (1, 89)]:
    z = tf.constant(np.complex64(complex(re, im)).reshape(1, 1))
    eager = tf.math.cos(z).numpy()[0, 0]
    xla = tf.function(tf.math.cos, jit_compile=True)(z).numpy()[0, 0]
    print(f"  cos({re}{im:+d}j): eager={eager}  xla={xla}")

print("\nAll results are representable in float32 (< 3.4e38).")
print("Eager is correct; XLA overflows to inf/nan.")
