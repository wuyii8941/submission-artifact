#!/usr/bin/env python3
"""tf.math.cumprod gradient is wrong when the input contains zeros.

The gradient computation uses cumprod(x)/x[i] internally, which produces
0/0 = NaN when x[i]=0 (because cumprod is also 0 from that point on).
TF silently returns 0 instead of the correct gradient value.

Example: cumprod([1.5, 0, 2, 3]) = [1.5, 0, 0, 0]
  sum = 1.5
  d/dx[0] = 0+0+0 = 0?  No! d/dx[0] = 1 (only first term survives)
  d/dx[1] = 1.5*2*3 + 1.5*3 + 1.5 = 9+4.5+... wait, let me be precise.

  cumprod = [x0, x0*x1, x0*x1*x2, x0*x1*x2*x3]
  d(sum(cumprod))/dx1 = x0 + x0*x2 + x0*x2*x3 = 1.5 + 3.0 + 9.0 = 13.5

TF returns 0 for d/dx[1]. Numerical gradient confirms 13.5 is correct.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np

print(f"TF version: {tf.__version__}")

def cumprod_grad(x_np):
    x = tf.constant(x_np, dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.reduce_sum(tf.math.cumprod(x))
    return tape.gradient(y, x).numpy()

def numerical_grad(x_np, eps=1e-7):
    g = np.zeros_like(x_np)
    for i in range(len(x_np)):
        xp = x_np.copy(); xp[i] += eps
        xm = x_np.copy(); xm[i] -= eps
        fp = float(np.sum(np.cumprod(xp)))
        fm = float(np.sum(np.cumprod(xm)))
        g[i] = (fp - fm) / (2 * eps)
    return g

# Case 1: zero in middle
print("=== Case 1: zero in middle ===")
x1 = np.array([1.5, 0.0, 2.0, 3.0])
tf_g = cumprod_grad(x1)
num_g = numerical_grad(x1)
print(f"  input:    {x1}")
print(f"  cumprod:  {np.cumprod(x1)}")
print(f"  TF grad:  {tf_g}")
print(f"  Num grad: {num_g}")
print(f"  BUG: TF says d/dx[1]={tf_g[1]}, correct is {num_g[1]}")

# Case 2: zero at start
print("\n=== Case 2: zero at start ===")
x2 = np.array([0.0, 2.0, 3.0])
tf_g = cumprod_grad(x2)
num_g = numerical_grad(x2)
print(f"  input:    {x2}")
print(f"  TF grad:  {tf_g}")
print(f"  Num grad: {num_g}")
print(f"  BUG: TF says d/dx[0]={tf_g[0]}, correct is {num_g[0]}")

# Case 3: zero at end
print("\n=== Case 3: zero at end ===")
x3 = np.array([2.0, 3.0, 0.0])
tf_g = cumprod_grad(x3)
num_g = numerical_grad(x3)
print(f"  input:    {x3}")
print(f"  TF grad:  {tf_g}")
print(f"  Num grad: {num_g}")
print(f"  BUG: TF says d/dx[2]={tf_g[2]}, correct is {num_g[2]}")

# Case 4: two zeros
print("\n=== Case 4: two zeros ===")
x4 = np.array([2.0, 0.0, 0.0, 3.0])
tf_g = cumprod_grad(x4)
num_g = numerical_grad(x4)
print(f"  input:    {x4}")
print(f"  TF grad:  {tf_g}")
print(f"  Num grad: {num_g}")

# Case 5: no zeros (should be correct)
print("\n=== Case 5: no zeros (control) ===")
x5 = np.array([1.5, 2.0, 0.5, 3.0])
tf_g = cumprod_grad(x5)
num_g = numerical_grad(x5)
print(f"  input:    {x5}")
print(f"  TF grad:  {tf_g}")
print(f"  Num grad: {num_g}")
print(f"  Match: {np.allclose(tf_g, num_g, rtol=1e-5)}")

# Also affects exclusive and reverse modes
print("\n=== Case 6: exclusive=True ===")
x6 = np.array([2.0, 0.0, 3.0])
x6t = tf.constant(x6, dtype=tf.float64)
with tf.GradientTape() as tape:
    tape.watch(x6t)
    y = tf.reduce_sum(tf.math.cumprod(x6t, exclusive=True))
tf_g6 = tape.gradient(y, x6t).numpy()
def f_excl(xv):
    cp = np.ones(len(xv))
    for i in range(1, len(xv)):
        cp[i] = cp[i-1] * xv[i-1]
    return float(np.sum(cp))
num_g6 = numerical_grad(x6, eps=1e-7)  # reuse for shape, compute manually
num_g6_manual = np.zeros(3)
eps = 1e-7
for i in range(3):
    xp = x6.copy(); xp[i] += eps
    xm = x6.copy(); xm[i] -= eps
    num_g6_manual[i] = (f_excl(xp) - f_excl(xm)) / (2 * eps)
print(f"  TF grad:  {tf_g6}")
print(f"  Num grad: {num_g6_manual}")
if not np.allclose(tf_g6, num_g6_manual, rtol=1e-3):
    print(f"  BUG: also wrong in exclusive mode")

print("\n=== Case 7: reverse=True ===")
x7 = np.array([2.0, 0.0, 3.0])
x7t = tf.constant(x7, dtype=tf.float64)
with tf.GradientTape() as tape:
    tape.watch(x7t)
    y = tf.reduce_sum(tf.math.cumprod(x7t, reverse=True))
tf_g7 = tape.gradient(y, x7t).numpy()
def f_rev(xv):
    return float(np.sum(np.cumprod(xv[::-1])[::-1]))
num_g7 = np.zeros(3)
for i in range(3):
    xp = x7.copy(); xp[i] += eps
    xm = x7.copy(); xm[i] -= eps
    num_g7[i] = (f_rev(xp) - f_rev(xm)) / (2 * eps)
print(f"  TF grad:  {tf_g7}")
print(f"  Num grad: {num_g7}")
if not np.allclose(tf_g7, num_g7, rtol=1e-3):
    print(f"  BUG: also wrong in reverse mode")

# Also affects XLA (CPU)
print("\n=== XLA also affected ===")
with tf.device('/CPU:0'):
    x = tf.constant([1.5, 0.0, 2.0, 3.0], dtype=tf.float64)

    @tf.function(jit_compile=True)
    def xla_cumprod_grad(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.reduce_sum(tf.math.cumprod(x))
        return tape.gradient(y, x)

    xla_g = xla_cumprod_grad(x).numpy()
    print(f"  XLA grad: {xla_g}")
    print(f"  Num grad: {numerical_grad(np.array([1.5, 0.0, 2.0, 3.0]))}")
