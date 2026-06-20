#!/usr/bin/env python3
"""
Bug: XLA returns 1+0j for pow(0+0j, 0+0j), eager returns NaN+NaNj

The mathematical value of 0^0 is conventionally 1 for real numbers
(per IEEE 754: pow(±0, ±0) = 1). However, for complex numbers the
situation is more nuanced — the complex power function has a branch
cut and 0^0 is considered indeterminate in complex analysis.

TF eager mode returns NaN+NaNj (indicating indeterminate).
XLA returns 1+0j (applying the real convention).

This is a semantic inconsistency between execution paths.

Oracle: eager vs XLA (jit_compile=True)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

z_base = tf.constant([0+0j], dtype=tf.complex64)
z_exp = tf.constant([0+0j], dtype=tf.complex64)

def cpow(a, b):
    return tf.pow(a, b)

eager = cpow(z_base, z_exp).numpy()
xla = tf.function(cpow, jit_compile=True)(z_base, z_exp).numpy()

print(f"\npow(0+0j, 0+0j):")
print(f"  Eager: {eager}")    # [nan+nanj]
print(f"  XLA:   {xla}")      # [1+0j]

assert np.isnan(eager[0].real) and np.isnan(eager[0].imag), \
    "Eager should return NaN+NaNj"
assert not np.isnan(xla[0].real), \
    "XLA returns 1+0j instead of NaN+NaNj"

# Also test with additional complex zero variants
bases = tf.constant([0+0j, complex(-0.0, 0), complex(0, -0.0), complex(-0.0, -0.0)], dtype=tf.complex64)
exps = tf.constant([0+0j, 0+0j, 0+0j, 0+0j], dtype=tf.complex64)

eager_all = cpow(bases, exps).numpy()
xla_all = tf.function(cpow, jit_compile=True)(bases, exps).numpy()

print(f"\nAll zero variants:")
print(f"  Eager: {eager_all}")
print(f"  XLA:   {xla_all}")

print("\n[CONFIRMED] XLA returns 1+0j for complex pow(0,0), eager returns NaN+NaNj")
