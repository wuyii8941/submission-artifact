"""tf.debugging.check_numerics silently passes under jit_compile=True.

Eager and Grappler correctly raise InvalidArgumentError on NaN/Inf.
XLA silently passes the tensor through without checking.
TF 2.22.0-dev20260429
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

t = tf.constant([1.0, 2.0, float('nan')])

# Eager: correctly raises
try:
    tf.debugging.check_numerics(t, "test")
    print("eager: passed (WRONG)")
except tf.errors.InvalidArgumentError:
    print("eager: raised InvalidArgumentError (correct)")

# XLA: silently passes
try:
    r = tf.function(lambda v: tf.debugging.check_numerics(v, "test"),
                    jit_compile=True)(t)
    print(f"XLA:   passed silently, result={r.numpy()} (WRONG)")
except tf.errors.InvalidArgumentError:
    print("XLA:   raised InvalidArgumentError (correct)")

# Also passes Inf without raising
t_inf = tf.constant([1.0, float('inf')])
try:
    r = tf.function(lambda v: tf.debugging.check_numerics(v, "test"),
                    jit_compile=True)(t_inf)
    print(f"XLA inf: passed silently, result={r.numpy()} (WRONG)")
except tf.errors.InvalidArgumentError:
    print("XLA inf: raised InvalidArgumentError (correct)")
