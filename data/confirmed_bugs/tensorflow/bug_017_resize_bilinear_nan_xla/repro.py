#!/usr/bin/env python3
"""
Bug: XLA propagates NaN to entire channel in tf.image.resize (bilinear)

When a single pixel contains NaN, bilinear interpolation should only
produce NaN in output pixels whose interpolation window includes that
NaN pixel. Eager mode correctly localizes NaN propagation.

Under XLA (jit_compile=True), NaN propagates to ALL pixels in the
channel, even those whose interpolation window doesn't include the
NaN source pixel. This suggests XLA's bilinear resize lowering computes
a global intermediate (e.g., normalizing by total weight) that gets
poisoned by a single NaN.

Oracle: eager vs XLA (jit_compile=True)
Root cause: XLA's bilinear resize implementation likely uses a global
            accumulation that doesn't isolate NaN pixels.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

# 2x2 image with one NaN pixel in channel 0
# channel 0: [[1.0, NaN], [7.0, 10.0]]
# channel 1: [[2.0, 5.0], [8.0, 11.0]]  (no NaN)
# channel 2: [[3.0, 6.0], [9.0, 12.0]]  (no NaN)
img = tf.constant([[[[1.0, 2.0, 3.0], [float('nan'), 5.0, 6.0]],
                     [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]], dtype=tf.float32)

def resize_fn(x):
    return tf.image.resize(x, (4, 4), method='bilinear')

eager = resize_fn(img).numpy()
xla = tf.function(resize_fn, jit_compile=True)(img).numpy()

print("Image shape: (1, 2, 2, 3) → resize to (1, 4, 4, 3)")
print("NaN at position [0, 0, 1, 0] (row=0, col=1, channel=0)")

print("\nChannel 0 (contains NaN source pixel):")
print("  Eager:")
print(f"  {eager[0,:,:,0]}")
print("  XLA:")
print(f"  {xla[0,:,:,0]}")

eager_nan_count = np.isnan(eager[0,:,:,0]).sum()
xla_nan_count = np.isnan(xla[0,:,:,0]).sum()
print(f"\n  Eager NaN pixel count in channel 0: {eager_nan_count}")
print(f"  XLA NaN pixel count in channel 0:   {xla_nan_count}")

print("\nChannel 1 (NO NaN source):")
print("  Eager:")
print(f"  {eager[0,:,:,1]}")
print("  XLA:")
print(f"  {xla[0,:,:,1]}")

eager_c1_nan = np.isnan(eager[0,:,:,1]).sum()
xla_c1_nan = np.isnan(xla[0,:,:,1]).sum()
print(f"\n  Eager NaN pixel count in channel 1: {eager_c1_nan}")
print(f"  XLA NaN pixel count in channel 1:   {xla_c1_nan}")

# Key assertions
assert eager_nan_count < 16, "Eager should localize NaN (not all 16 pixels)"
assert xla_nan_count == 16, "XLA propagates NaN to ALL 16 pixels in channel 0"
assert eager_c1_nan == 0, "Eager: channel 1 should have no NaN"
assert xla_c1_nan == 0, "XLA: channel 1 should have no NaN"

print("\n[CONFIRMED] XLA over-propagates NaN in bilinear resize")
print("  Eager correctly isolates NaN to affected interpolation windows")
print("  XLA spreads NaN to the entire channel")
