#!/usr/bin/env python3
"""
Bug: unsorted_segment_min/max produces WRONG results when segment
     contains NaN at certain positions — wrong under ANY NaN semantics.

min([1.0, NaN, 3.0]) returns 3.0, but:
  - IEEE 754 (NaN-propagating): should return NaN
  - fmin (NaN-suppressing): should return 1.0
  - 3.0 is wrong under BOTH interpretations

Root cause: CPU segment kernel uses raw `acc < x ? acc : x` comparison.
  IEEE 754: NaN < anything = false, anything < NaN = false
  So when NaN enters as x, acc is replaced by NaN (correct min lost).
  When NaN is acc, it gets replaced by the next value (NaN discarded).
  tf.reduce_min and tf.minimum handle NaN correctly; only segment ops broken.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

ids3 = tf.constant([0, 0, 0], dtype=tf.int32)
ids4 = tf.constant([0, 0, 0, 0], dtype=tf.int32)

print("\nunsorted_segment_min — NaN at different positions:")
print(f"{'data':>35s}  {'result':>8s}  {'fmin_correct':>12s}  {'status':>8s}")
for data in [[1.0, float('nan'), 3.0],
             [float('nan'), 1.0, 3.0],
             [1.0, 3.0, float('nan')],
             [3.0, float('nan'), 1.0],
             [1.0, float('nan'), 3.0, 2.0]]:
    d = tf.constant(data, dtype=tf.float32)
    ids = ids3 if len(data) == 3 else ids4
    r = tf.math.unsorted_segment_min(d, ids, 1).numpy()[0]
    correct = np.nanmin(data)
    ok = "OK" if (r == correct or np.isnan(r)) else "WRONG"
    print(f"  {str(data):>33s}  {r:>8.1f}  {correct:>12.1f}  {ok:>8s}")

print("\nunsorted_segment_max — NaN at different positions:")
print(f"{'data':>35s}  {'result':>8s}  {'fmax_correct':>12s}  {'status':>8s}")
for data in [[3.0, float('nan'), 1.0],
             [1.0, float('nan'), 3.0],
             [float('nan'), 1.0, 3.0],
             [3.0, float('nan'), 1.0, 2.0]]:
    d = tf.constant(data, dtype=tf.float32)
    ids = ids3 if len(data) == 3 else ids4
    r = tf.math.unsorted_segment_max(d, ids, 1).numpy()[0]
    correct = np.nanmax(data)
    ok = "OK" if (r == correct or np.isnan(r)) else "WRONG"
    print(f"  {str(data):>33s}  {r:>8.1f}  {correct:>12.1f}  {ok:>8s}")

# Demonstrate: tf.minimum handles NaN correctly, but segment kernel does not
print("\nRoot cause — segment kernel uses raw comparison, not tf.minimum:")
a = tf.constant([1.0], dtype=tf.float32)
b = tf.constant([float('nan')], dtype=tf.float32)
print(f"  tf.minimum(1.0, NaN) = {tf.minimum(a, b).numpy()[0]}  (NaN — correct)")
print(f"  tf.minimum(NaN, 1.0) = {tf.minimum(b, a).numpy()[0]}  (NaN — correct)")
print(f"  tf.reduce_min([1, NaN, 3]) = {tf.reduce_min([1.0, float('nan'), 3.0]).numpy()}  (NaN — correct)")
print(f"  segment_min([1, NaN, 3])   = {tf.math.unsorted_segment_min(tf.constant([1.0, float('nan'), 3.0]), tf.constant([0,0,0]), 1).numpy()[0]}  (3.0 — WRONG)")

# Also affects sorted segment ops (same underlying reduction)
print("\nsegment_min (sorted) — same bug:")
seg_ids_sorted = tf.constant([0, 0, 0], dtype=tf.int32)
for data in [[1.0, float('nan'), 3.0], [3.0, float('nan'), 1.0]]:
    d = tf.constant(data, dtype=tf.float32)
    r = tf.math.segment_min(d, seg_ids_sorted).numpy()[0]
    correct = np.nanmin(data)
    ok = "OK" if (r == correct or np.isnan(r)) else "WRONG"
    print(f"  segment_min({data}) = {r:.1f}  (fmin={correct:.1f})  {ok}")

print("\nsegment_max (sorted) — same bug:")
for data in [[3.0, float('nan'), 1.0], [1.0, float('nan'), 3.0]]:
    d = tf.constant(data, dtype=tf.float32)
    r = tf.math.segment_max(d, seg_ids_sorted).numpy()[0]
    correct = np.nanmax(data)
    ok = "OK" if (r == correct or np.isnan(r)) else "WRONG"
    print(f"  segment_max({data}) = {r:.1f}  (fmax={correct:.1f})  {ok}")

# Key assertion
result = tf.math.unsorted_segment_min(
    tf.constant([1.0, float('nan'), 3.0], dtype=tf.float32),
    tf.constant([0, 0, 0], dtype=tf.int32), 1).numpy()[0]
assert result == 3.0, f"Expected 3.0 (the bug), got {result}"
assert result != 1.0, "If this fails, the bug is fixed"

print("\n[CONFIRMED] segment_min/max and unsorted_segment_min/max produce wrong")
print("  results when NaN is at certain positions. Wrong under ANY NaN semantics.")
