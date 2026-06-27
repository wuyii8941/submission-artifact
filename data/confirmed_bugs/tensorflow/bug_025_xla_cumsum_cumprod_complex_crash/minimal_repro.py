import tensorflow as tf
import numpy as np

x = tf.constant([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], dtype=tf.complex64)

# Eager: works fine
print("eager cumsum:", tf.math.cumsum(x).numpy())
print("eager cumprod:", tf.math.cumprod(x).numpy())

# XLA (jit_compile=True): crashes
@tf.function(jit_compile=True)
def cumsum_xla(x):
    return tf.math.cumsum(x)

@tf.function(jit_compile=True)
def cumprod_xla(x):
    return tf.math.cumprod(x)

try:
    print("XLA cumsum:", cumsum_xla(x).numpy())
except Exception as e:
    print(f"XLA cumsum CRASHED: {type(e).__name__}: {e}")

try:
    print("XLA cumprod:", cumprod_xla(x).numpy())
except Exception as e:
    print(f"XLA cumprod CRASHED: {type(e).__name__}: {e}")
