#!/usr/bin/env python3
"""Grappler ArithmeticOptimizer eliminates reciprocal(reciprocal(x)) → x,
but reciprocal is not a perfect involution in floating point.

reciprocal(reciprocal(7.0)) = reciprocal(0.14285714...) = 6.9999995,
but Grappler's InvertInvolution pass treats reciprocal as a perfect
involution and eliminates the double-reciprocal, returning exactly 7.0.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

x = tf.constant([3.0, 7.0, 11.0, 13.0, 17.0, 0.1, 0.3], dtype=tf.float32)

# Eager: two actual reciprocal operations, accumulates rounding error
eager_r = tf.math.reciprocal(tf.math.reciprocal(x)).numpy()

# Grappler: InvertInvolution eliminates the pair, returns x unchanged
@tf.function(jit_compile=False)
def recip_recip(x):
    return tf.math.reciprocal(tf.math.reciprocal(x))

grappler_r = recip_recip(x).numpy()

print("=== reciprocal(reciprocal(x)) ===")
print(f"  Input:    {x.numpy()}")
print(f"  Eager:    {eager_r}")
print(f"  Grappler: {grappler_r}")
print(f"  Differ:   {np.where(eager_r != grappler_r)[0].tolist()}")

for i in range(len(x.numpy())):
    if eager_r[i] != grappler_r[i]:
        v = x.numpy()[i]
        r1 = 1.0 / np.float32(v)
        r2 = 1.0 / r1
        print(f"\n  x={v}: 1/x={r1}, 1/(1/x)={r2}")
        print(f"    Eager returns {eager_r[i]} (correct: two operations applied)")
        print(f"    Grappler returns {grappler_r[i]} (wrong: operations skipped)")

# Confirm root cause: arithmetic_optimization
print("\n=== Disabling arithmetic_optimization fixes it ===")
tf.config.optimizer.set_experimental_options({'arithmetic_optimization': False})
no_arith = tf.function(lambda x: tf.math.reciprocal(tf.math.reciprocal(x)),
                        jit_compile=False)(x).numpy()
print(f"  No arith opt: {no_arith}")
print(f"  Match eager:  {np.array_equal(no_arith, eager_r)}")

tf.config.optimizer.set_experimental_options({'arithmetic_optimization': True})
with_arith = tf.function(lambda x: tf.math.reciprocal(tf.math.reciprocal(x)),
                          jit_compile=False)(x).numpy()
print(f"  With arith opt: {with_arith}")
print(f"  Match eager:    {np.array_equal(with_arith, eager_r)}")

# neg(neg(x)) is a true involution (bit-exact), so that optimization is fine
print("\n=== neg(neg(x)) for comparison (true involution, no bug) ===")
eager_neg = tf.math.negative(tf.math.negative(x)).numpy()
grappler_neg = tf.function(lambda x: tf.math.negative(tf.math.negative(x)),
                            jit_compile=False)(x).numpy()
print(f"  Eager:    {eager_neg}")
print(f"  Grappler: {grappler_neg}")
print(f"  Match:    {np.array_equal(eager_neg, grappler_neg)}")
