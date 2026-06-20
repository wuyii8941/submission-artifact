"""Reproducer: tf.nn.selu and tf.nn.elu return 0 for small negative inputs due to
catastrophic cancellation in exp(x)-1 computation."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

alpha_selu = 1.6732632423543772
scale_selu = 1.0507009873554805

print("\n=== tf.nn.selu returns 0 for small negative inputs ===")
print(f"{'x':>12s} {'eager':>15s} {'expected':>15s} {'rel_error':>10s}")
for v in [-1e-1, -1e-3, -1e-5, -1e-7, -1e-8, -1e-10, -1e-20, -1e-30]:
    x = tf.constant([[np.float32(v)]])
    eager = tf.nn.selu(x).numpy()[0, 0]
    expected = scale_selu * alpha_selu * v
    rel_err = abs(eager - expected) / abs(expected) if expected != 0 else 0
    print(f"  {v:>10.0e}  {eager:>15.6e}  {expected:>15.6e}  {rel_err:>9.1%}")

print("\n=== tf.nn.elu has the same bug ===")
print(f"{'x':>12s} {'eager':>15s} {'expected':>15s}")
for v in [-1e-7, -1e-8, -1e-10, -1e-20]:
    x = tf.constant([[np.float32(v)]])
    eager = tf.nn.elu(x).numpy()[0, 0]
    print(f"  {v:>10.0e}  {eager:>15.6e}  {v:>15.6e}")

print("\n=== XLA is correct (uses expm1 internally) ===")
for v in [-1e-7, -1e-10, -1e-20, -1e-30]:
    x = tf.constant([[np.float32(v)]])
    eager = tf.nn.selu(x).numpy()[0, 0]
    xla = tf.function(tf.nn.selu, jit_compile=True)(x).numpy()[0, 0]
    expected = scale_selu * alpha_selu * v
    print(f"  selu({v:>10.0e}): eager={eager:>15.6e}  xla={xla:>15.6e}  expected={expected:>15.6e}")

print("\n=== Root cause: exp(x)-1 vs expm1(x) ===")
for v in [-1e-7, -1e-10]:
    x = np.float32(v)
    print(f"  x={v:.0e}: exp(x)-1 = {np.exp(x) - np.float32(1):.6e}  "
          f"expm1(x) = {np.expm1(np.float64(v)):.6e}")
