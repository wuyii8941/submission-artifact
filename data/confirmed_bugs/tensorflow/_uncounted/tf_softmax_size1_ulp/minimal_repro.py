"""TF fuzzer v1 finding.
seed=58944
severity=CATASTROPHIC
oracles: ['eager_vs_tfn:OK', 'tfn_vs_xla:OK', 'grad_eager_vs_tfn:ALL_NONFINITE', 'cpu_vs_gpu:DIFF']
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

import tensorflow as tf
import numpy as np
rng = np.random.default_rng(58944)
x0 = tf.constant(rng.standard_normal((129, 64)), dtype=tf.float32)

def fn(x0):
    v0 = tf.keras.ops.log1p(tf.abs(x0) + 1e-7)
    v1 = tf.keras.ops.ceil(v0)
    v2 = (lambda _t: (_t - tf.keras.ops.mean(_t, axis=-1, keepdims=True)) / (tf.keras.ops.std(_t, axis=-1, keepdims=True) + 1e-6))(v1)
    v3 = tf.sort(x0, axis=-1)
    v4 = tf.keras.ops.softmax(v3, axis=-1)
    v5 = tf.keras.ops.minimum(v4, v2)
    v6 = tf.keras.ops.prod(tf.clip_by_value(v3, 0.5, 1.5), axis=-1, keepdims=True)
    v7 = tf.math.sin(v1)
    v8 = tf.math.sin(v7)
    v9 = tf.keras.ops.elu(v4)
    v10 = tf.keras.ops.sigmoid(v8)
    v11 = v4[..., :(v4.shape[-1]//2)]
    v12 = tf.keras.ops.power(tf.abs(v11)+1e-7, tf.clip_by_value(tf.abs(v11), 0.1, 3.0))
    v13 = tf.roll(v6, shift=3, axis=-1)
    v14 = (lambda _t: (_t - tf.keras.ops.mean(_t, axis=-1, keepdims=True)) / (tf.keras.ops.std(_t, axis=-1, keepdims=True) + 1e-6))(v10)
    v15 = tf.keras.ops.log_softmax(v12, axis=-1)
    v16 = tf.keras.ops.max(v8, axis=-1, keepdims=True)
    v17 = (lambda _t: _t / (tf.keras.ops.sqrt(tf.keras.ops.mean(tf.keras.ops.square(_t), axis=-1, keepdims=True)) + 1e-6))(v14)
    v18 = tf.keras.ops.log1p(tf.abs(v16) + 1e-7)
    v19 = tf.nn.softplus(v15)
    v20 = tf.keras.ops.softmax(v16, axis=-1)
    v21 = tf.keras.ops.floor(v20)
    return tf.reduce_sum(v21)

args = [x0]

# eager vs tf.function
out_e = fn(*args)
out_t = tf.function(fn, jit_compile=False)(*args)
out_x = tf.function(fn, jit_compile=True)(*args)
e = np.asarray(out_e.numpy(), dtype=np.float64)
t = np.asarray(out_t.numpy(), dtype=np.float64)
x = np.asarray(out_x.numpy(), dtype=np.float64)
print(f"eager_vs_tfn   max_abs={np.abs(e-t).max():.6e}")
print(f"tfn_vs_xla     max_abs={np.abs(t-x).max():.6e}")
print(f"eager_vs_xla   max_abs={np.abs(e-x).max():.6e}")
