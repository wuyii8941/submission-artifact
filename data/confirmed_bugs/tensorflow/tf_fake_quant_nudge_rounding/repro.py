#!/usr/bin/env python3
"""
Bug: tf.quantization.fake_quant_with_min_max_vars produces different results
     between eager and XLA due to different rounding of the nudged zero_point.

     Eager uses round-half-away-from-zero (std::round).
     XLA uses round-half-to-even (banker's rounding).

     When zero_point_from_min has a .5 fractional part, the two implementations
     disagree, causing off-by-one-step quantization error.

     Example: min=-0.5, max=2.5, num_bits=4
       scale = 3.0/15 = 0.2
       zero_point_from_min = 0.5/0.2 = 2.5
       Eager: round(2.5) = 3 → nudged range [-0.6, 2.4]
       XLA:   round(2.5) = 2 → nudged range [-0.4, 2.6]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

# === Minimal reproduction ===
print("\n--- Minimal repro: 4-bit quantization ---")
inp = tf.constant([2.579, -0.6, 2.6, -0.55, 2.55], dtype=tf.float32)
mn = tf.constant(-0.5, dtype=tf.float32)
mx = tf.constant(2.5, dtype=tf.float32)

eager = tf.quantization.fake_quant_with_min_max_vars(inp, mn, mx, num_bits=4)
xla = tf.function(
    lambda v, lo, hi: tf.quantization.fake_quant_with_min_max_vars(v, lo, hi, num_bits=4),
    jit_compile=True)(inp, mn, mx)

print(f"  input:  {inp.numpy()}")
print(f"  eager:  {eager.numpy()}")
print(f"  xla:    {xla.numpy()}")
print(f"  diff:   {(eager - xla).numpy()}")

# === Systematic test ===
print("\n--- Full range test ---")
full_inp = tf.constant(np.linspace(-1, 3, 21).astype(np.float32))
for num_bits in [4, 8]:
    e = tf.quantization.fake_quant_with_min_max_vars(full_inp, mn, mx, num_bits=num_bits).numpy()
    x = tf.function(
        lambda v, lo, hi: tf.quantization.fake_quant_with_min_max_vars(v, lo, hi, num_bits=num_bits),
        jit_compile=True)(full_inp, mn, mx).numpy()
    diff_idx = np.where(~np.isclose(e, x))[0]
    print(f"\n  num_bits={num_bits}: {len(diff_idx)} mismatches out of {len(full_inp.numpy())}")
    for idx in diff_idx:
        print(f"    inp={full_inp.numpy()[idx]:6.2f}: eager={e[idx]:8.4f}, xla={x[idx]:8.4f}, diff={abs(e[idx]-x[idx]):.6f}")

# === Other affected ops ===
print("\n--- fake_quant_with_min_max_args ---")
e2 = tf.quantization.fake_quant_with_min_max_args(inp, min=-0.5, max=2.5, num_bits=4).numpy()
x2 = tf.function(
    lambda v: tf.quantization.fake_quant_with_min_max_args(v, min=-0.5, max=2.5, num_bits=4),
    jit_compile=True)(inp).numpy()
print(f"  eager: {e2}")
print(f"  xla:   {x2}")
print(f"  diff:  {e2 - x2}")

print("\n--- fake_quant_with_min_max_vars_per_channel ---")
inp_ch = tf.constant(np.array([[[-0.6, 2.6, -0.55],
                                  [2.55, -0.5, 2.5]]]).astype(np.float32))
mn_ch = tf.constant([-0.5, -0.5, -0.5], dtype=tf.float32)
mx_ch = tf.constant([2.5, 2.5, 2.5], dtype=tf.float32)
e3 = tf.quantization.fake_quant_with_min_max_vars_per_channel(inp_ch, mn_ch, mx_ch, num_bits=4).numpy()
x3 = tf.function(
    lambda v, lo, hi: tf.quantization.fake_quant_with_min_max_vars_per_channel(v, lo, hi, num_bits=4),
    jit_compile=True)(inp_ch, mn_ch, mx_ch).numpy()
print(f"  eager: {e3.flatten()}")
print(f"  xla:   {x3.flatten()}")

# === Another triggering range ===
print("\n--- Range [-0.1, 0.1], 4-bit ---")
inp4 = tf.constant([-0.11, -0.1, -0.09, 0.09, 0.1, 0.11], dtype=tf.float32)
mn4, mx4 = tf.constant(-0.1, tf.float32), tf.constant(0.1, tf.float32)
e4 = tf.quantization.fake_quant_with_min_max_vars(inp4, mn4, mx4, num_bits=4).numpy()
x4 = tf.function(
    lambda v, lo, hi: tf.quantization.fake_quant_with_min_max_vars(v, lo, hi, num_bits=4),
    jit_compile=True)(inp4, mn4, mx4).numpy()
print(f"  eager: {e4}")
print(f"  xla:   {x4}")

# === Assertions ===
# 4-bit, range [-0.5, 2.5]: eager clamps to [-0.6, 2.4], XLA to [-0.4, 2.6]
z = tf.constant([2.579], dtype=tf.float32)
e_z = tf.quantization.fake_quant_with_min_max_vars(z, mn, mx, num_bits=4).numpy()[0]
x_z = tf.function(
    lambda v: tf.quantization.fake_quant_with_min_max_vars(v, mn, mx, num_bits=4),
    jit_compile=True)(z).numpy()[0]
assert np.isclose(e_z, 2.4), f"Eager should give 2.4, got {e_z}"
assert np.isclose(x_z, 2.6), f"XLA should give 2.6 (bug), got {x_z}"
assert not np.isclose(e_z, x_z), "Eager and XLA should disagree"

print("\n[CONFIRMED] fake_quant eager/XLA mismatch due to nudge rounding difference.")
print("  Eager rounds half-integers away from zero (std::round).")
print("  XLA rounds half-integers to even (banker's rounding).")
print("  Result: off-by-one quantization step for affected ranges.")
