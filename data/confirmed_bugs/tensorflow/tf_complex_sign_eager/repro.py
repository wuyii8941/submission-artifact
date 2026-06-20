"""Reproducer: tf.math.sign returns 0 for small nonzero complex numbers."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

print("\n=== tf.math.sign for small complex values ===")
print(f"{'z':>20s} {'eager':>15s} {'XLA':>15s} {'numpy':>15s}")
for re, im in [(1e-18, 0), (1e-19, 0), (1e-20, 0), (1e-30, 0),
               (0, 1e-19), (0, 1e-20), (-1e-20, 0), (0, -1e-20),
               (1e-20, 1e-20)]:
    z = np.complex64(complex(re, im))
    zt = tf.constant(z.reshape(1, 1))
    eager = tf.math.sign(zt).numpy()[0, 0]
    xla = tf.function(tf.math.sign, jit_compile=True)(zt).numpy()[0, 0]
    np_ref = np.sign(z)
    print(f"  {str(z):>20s}  {str(eager):>15s}  {str(xla):>15s}  {str(np_ref):>15s}")

print("\nEager returns 0 for |z| < ~1e-19. XLA and numpy are correct.")
print("All these are normal (non-subnormal) float32 values.")
