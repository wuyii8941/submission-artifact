"""
TensorFlow #117547: jit_compile=True crashes on dead code involving
tf.argmin/tf.argmax on empty-dimension tensors.

Root cause: XLA's tf2xla conversion attempts to lower argmin/argmax on
shape [4,0] (empty reduction axis) even when the result is unused.
Grappler (non-XLA path) correctly eliminates the dead code before execution.

Category: XLA dead-code lowering (T7: backend divergence)
Bug type: Crash
Issue: <upstream issue, anonymized>
Status: type:bug, comp:xla, assigned @Venkat6871
Tested: TF 2.22.0-dev20260424, TF 2.21.0
"""
import tensorflow as tf

x = tf.zeros((4, 0), dtype=tf.float32)


def fn(x):
    dead = tf.cast(tf.argmin(x, axis=-1), tf.float32)
    return tf.constant(42.0)


# Non-XLA path: succeeds (Grappler eliminates dead code)
assert tf.function(fn, jit_compile=False)(x).numpy() == 42.0

# XLA path: crashes with "Reduction axis -1 is empty in shape [4,0]"
try:
    tf.function(fn, jit_compile=True)(x)
    assert False, "Expected crash did not occur"
except Exception as e:
    assert "empty" in str(e).lower() or "tf2xla" in str(e).lower(), f"Unexpected error: {e}"
    print(f"CONFIRMED: XLA crash on dead argmin with empty dim: {type(e).__name__}")
