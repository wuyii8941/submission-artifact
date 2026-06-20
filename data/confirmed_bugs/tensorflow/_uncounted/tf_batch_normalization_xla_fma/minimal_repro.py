"""tf.nn.batch_normalization returns wrong result under jit_compile=True.

Even with exact pre-computed mean and variance, XLA produces wildly
wrong normalized output for large-magnitude inputs.

Root cause: BN formula uses x*inv - mean*inv instead of (x-mean)*inv.
XLA's FMA instruction computes x*inv at higher intermediate precision,
then subtracts the rounded mean*inv, producing a non-zero residual.
TF 2.22.0-dev20260429
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

val = np.float32(1e15)
t = tf.constant([[val, val, val]], dtype=tf.float32)
mean = tf.constant([val], dtype=tf.float32)
var = tf.constant([0.0], dtype=tf.float32)

# Eager: correct (0.0)
e = tf.nn.batch_normalization(t, mean, var, None, None, 1e-5)
print(f"eager:  {e.numpy()}")

# XLA: wrong (-1.14e10)
x = tf.function(lambda x: tf.nn.batch_normalization(x, mean, var, None, None, 1e-5),
                jit_compile=True)(t)
print(f"XLA:    {x.numpy()}")

# Root cause: formula decomposition
print("\nRoot cause demonstration:")
inv = tf.math.rsqrt(var + 1e-5)
# Correct: (x - mean) * inv
r1 = tf.function(lambda x, m: (x - m) * inv, jit_compile=True)(t, mean)
print(f"(x-mean)*inv XLA = {r1.numpy()}")  # [0, 0, 0]
# What BN actually does: x*inv - mean*inv
r2 = tf.function(lambda x, m: x * inv + (-m * inv), jit_compile=True)(t, mean)
print(f"x*inv-mean*inv XLA = {r2.numpy()}")  # [-1.14e10, ...]

# Scales with magnitude
print("\nScaling:")
for val in [1e5, 1e10, 1e15, 1e20]:
    v = np.float32(val)
    t = tf.constant([[v, v, v]], dtype=tf.float32)
    m = tf.constant([v], dtype=tf.float32)
    vr = tf.constant([0.0], dtype=tf.float32)
    e = tf.nn.batch_normalization(t, m, vr, None, None, 1e-5).numpy()[0][0]
    x = tf.function(lambda x: tf.nn.batch_normalization(x, m, vr, None, None, 1e-5),
                    jit_compile=True)(t).numpy()[0][0]
    print(f"  val={val:.0e}: eager={e:12.2f}, xla={x:15.2f}")
