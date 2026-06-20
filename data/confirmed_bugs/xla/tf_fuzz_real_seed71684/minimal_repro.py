"""Minimal repro: tf.raw_ops.Erf under XLA jit has ~2 ULP lower precision when
the innermost tensor axis has size 1, compared to eager execution and to
XLA on non-size-1 innermost shapes. The same Erf input returns exactly 1.0
in eager and in jit for shape [N>=2], but returns 0.9999998 in jit for
shapes like [1], [7,1], [7,64,1], etc.

Downstream effect: any discrete op applied after Erf (floor, ceil, argmax,
equality, ...) amplifies the ~1-2 ULP discrepancy to an integer-valued
silent divergence between eager and tf.function(jit_compile=True).

Tested on TF 2.21.0.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

print(f"TF version: {tf.__version__}")


@tf.function(jit_compile=True)
def xla_erf(x):
    return tf.raw_ops.Erf(x=x)


print()
print(f"{'shape':>15}  {'eager Erf[0]':>18}  {'XLA jit Erf[0]':>18}  {'match?':>6}")
for shape in [(1,), (2,), (7, 1), (7, 64, 1), (7, 64, 5), (1, 1, 1)]:
    x = tf.fill(shape, tf.constant(4.5, dtype=tf.float32))
    e = tf.raw_ops.Erf(x=x).numpy()
    xl = xla_erf(x).numpy()
    e0 = float(e.flat[0]); x0 = float(xl.flat[0])
    print(f"{str(shape):>15}  {e0:18.10f}  {x0:18.10f}  {str(e0 == x0):>6}")

print()
print("Discrete-op amplification (shape [7,64,1], value = 4.5):")
x = tf.fill((7, 64, 1), tf.constant(4.5, dtype=tf.float32))
e_floor = tf.math.floor(tf.raw_ops.Erf(x=x))
x_floor = tf.math.floor(xla_erf(x))
print(f"  sum(floor(eager Erf)) = {float(tf.reduce_sum(e_floor).numpy())}  (expected 448)")
print(f"  sum(floor(XLA   Erf)) = {float(tf.reduce_sum(x_floor).numpy())}  (observed 0)")
print()
print("Root cause: XLA compiles `Erf` on rank-N inputs with innermost dim = 1")
print("using a different polynomial than it uses for innermost dim >= 2; the")
print("former is less accurate (~1-2 ULPs low at saturation).")
