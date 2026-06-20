#!/usr/bin/env python3
"""Grappler constant folding optimizes x*0 → 0, suppressing NaN for Inf and NaN inputs.

IEEE 754 requires: Inf * 0 = NaN, NaN * 0 = NaN.
But Grappler's constant_folding pass replaces Mul(x, Const(0)) with Const(0),
unconditionally returning 0 regardless of x. This silently suppresses NaN
propagation and violates IEEE 754.

Different from the existing tf_grappler_add_zero_signed_zero bug:
  - That bug: arithmetic_optimization eliminates x+0 → x (AddZero rule)
  - This bug: constant_folding folds x*0 → 0 (MulZero fold)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

x = tf.constant([1.0, float('inf'), float('-inf'), float('nan')], dtype=tf.float32)

# Eager: correct IEEE 754 behavior
eager_r = (x * 0.0).numpy()
print("=== x * 0.0 ===")
print(f"  Input:    {x.numpy()}")
print(f"  Eager:    {eager_r}")  # [0, nan, nan, nan]

# Grappler: constant_folding replaces Mul(x, 0) with Const(0)
@tf.function(jit_compile=False)
def mul_zero(x):
    return x * 0.0

grappler_r = mul_zero(x).numpy()
print(f"  Grappler: {grappler_r}")  # [0, 0, 0, 0] - WRONG

# XLA: correct
@tf.function(jit_compile=True)
def mul_zero_xla(x):
    return x * 0.0

xla_r = mul_zero_xla(x).numpy()
print(f"  XLA:      {xla_r}")  # [0, nan, nan, nan] - correct

print(f"\n  BUG: Grappler says inf*0 = {grappler_r[1]}, should be NaN")
print(f"  BUG: Grappler says nan*0 = {grappler_r[3]}, should be NaN")

# Downstream impact: NaN check is defeated
print("\n=== Downstream: tf.math.is_nan(x * 0) ===")
eager_nan = tf.math.is_nan(x * 0.0).numpy()
grappler_nan = tf.function(lambda x: tf.math.is_nan(x * 0.0), jit_compile=False)(x).numpy()
print(f"  Eager:    {eager_nan}")  # [F, T, T, T]
print(f"  Grappler: {grappler_nan}")  # [F, F, F, F] - NaN check silently defeated

# Confirm root cause: constant_folding pass
print("\n=== Disabling constant_folding fixes it ===")
tf.config.optimizer.set_experimental_options({'constant_folding': False})
no_cf = tf.function(lambda x: x * 0.0, jit_compile=False)(x).numpy()
print(f"  No constant_folding: {no_cf}")  # [0, nan, nan, nan] - correct

tf.config.optimizer.set_experimental_options({'constant_folding': True})
with_cf = tf.function(lambda x: x * 0.0, jit_compile=False)(x).numpy()
print(f"  With constant_folding: {with_cf}")  # [0, 0, 0, 0] - wrong

# Also affects 0*x (commutative)
print("\n=== 0.0 * x (commutative) ===")
eager_r2 = (0.0 * x).numpy()
grappler_r2 = tf.function(lambda x: 0.0 * x, jit_compile=False)(x).numpy()
print(f"  Eager:    {eager_r2}")
print(f"  Grappler: {grappler_r2}")
