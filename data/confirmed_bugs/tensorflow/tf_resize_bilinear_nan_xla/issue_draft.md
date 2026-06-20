# `tf.image.resize` bilinear: XLA bridge over-propagates NaN to entire channel

## Summary

When a single input pixel contains NaN, `tf.image.resize` with bilinear interpolation should only produce NaN in output pixels whose interpolation window includes that pixel. Eager mode correctly localizes NaN propagation. Under XLA (`jit_compile=True`), NaN propagates to **all** pixels in the channel.

## Reproduction

```python
import tensorflow as tf
import numpy as np

# 2x2 image, 3 channels. One NaN at [0, 0, 1, 0] (row=0, col=1, channel=0)
img = tf.constant([[[[1.0, 2.0, 3.0], [float('nan'), 5.0, 6.0]],
                     [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]], dtype=tf.float32)

def resize_fn(x):
    return tf.image.resize(x, (4, 4), method='bilinear')

eager = resize_fn(img).numpy()
xla = tf.function(resize_fn, jit_compile=True)(img).numpy()

print("Channel 0 NaN count — Eager:", np.isnan(eager[0,:,:,0]).sum())
# 9 — only pixels whose interpolation window includes the NaN source

print("Channel 0 NaN count — XLA:", np.isnan(xla[0,:,:,0]).sum())
# 16 — ALL pixels in the channel are NaN
```

Reproduces on TF 2.22.0-dev20260503 (nightly), CPU.

## Details

| | Eager | XLA |
|---|---|---|
| Channel 0 NaN count (of 16) | 9 | 16 |
| Channel 1 NaN count (no NaN source) | 0 | 0 |
| Channel 2 NaN count (no NaN source) | 0 | 0 |

## Root cause

The TF-XLA bridge implementation of bilinear resize (`tensorflow/compiler/tf2xla/kernels/image_resize_ops.cc`) uses `xla::Gather` + `xla::DotGeneral` for the weighted sum. This batched dot-product approach causes a single NaN to poison the entire channel, unlike the eager kernel's per-pixel computation.

## Expected behavior

XLA-compiled resize should match eager: NaN should only appear in output pixels whose interpolation window includes the NaN source pixel.

## Environment

- TensorFlow 2.22.0-dev20260503
- CPU only
