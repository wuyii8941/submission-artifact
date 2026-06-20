"""Bug 035: XLA FusedBatchNormV3 computes negative variance → NaN output.

Root cause: XLA's BatchNormExpander uses E[x²]-E[x]² (numerically unstable)
instead of E[(x-mean)²] (stable). With offset ≥ 1e4 (NCHW) or 1e5 (multi-channel
NHWC), catastrophic cancellation produces NEGATIVE variance → sqrt → NaN.
"""
import tensorflow as tf
import numpy as np

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    np.random.seed(42)

    # === Test 1: NHWC, 1 channel, offset=1e8 (original finding) ===
    print("=== NHWC 1ch offset=1e8 ===")
    x = np.random.randn(4, 8, 8, 1).astype(np.float32) + 1e8
    x_t = tf.constant(x)
    sc1 = np.ones(1, dtype=np.float32)
    of1 = np.zeros(1, dtype=np.float32)
    mi1 = np.zeros(1, dtype=np.float32)
    vi1 = np.ones(1, dtype=np.float32)

    eager_r = tf.raw_ops.FusedBatchNormV3(
        x=x_t, scale=sc1, offset=of1, mean=mi1, variance=vi1,
        is_training=True, data_format='NHWC')
    print(f"Eager: NaN={np.isnan(eager_r.y.numpy()).sum()}/256")

    @tf.function(jit_compile=True)
    def fbn_xla(x):
        return tf.raw_ops.FusedBatchNormV3(
            x=x, scale=sc1, offset=of1, mean=mi1, variance=vi1,
            is_training=True, data_format='NHWC')
    xla_r = fbn_xla(x_t)
    print(f"XLA:   NaN={np.isnan(xla_r.y.numpy()).sum()}/256")
    print(f"XLA variance: {xla_r.batch_variance.numpy()}")

    assert np.isnan(eager_r.y.numpy()).sum() == 0
    assert xla_r.batch_variance.numpy()[0] < 0
    assert np.isnan(xla_r.y.numpy()).sum() == 256

    # === Test 2: NCHW, 3 channels, offset=1e4 (breaks at much lower offset!) ===
    print("\n=== NCHW 3ch offset=1e4 ===")
    x_nhwc = np.random.randn(4, 8, 8, 3).astype(np.float32) + 1e4
    x_nchw = np.transpose(x_nhwc, (0, 3, 1, 2))
    x_t2 = tf.constant(x_nchw)
    sc3 = np.ones(3, dtype=np.float32)
    of3 = np.zeros(3, dtype=np.float32)
    mi3 = np.zeros(3, dtype=np.float32)
    vi3 = np.ones(3, dtype=np.float32)

    eager_r2 = tf.raw_ops.FusedBatchNormV3(
        x=x_t2, scale=sc3, offset=of3, mean=mi3, variance=vi3,
        is_training=True, data_format='NCHW')

    @tf.function(jit_compile=True)
    def fbn_nchw_xla(x):
        return tf.raw_ops.FusedBatchNormV3(
            x=x, scale=sc3, offset=of3, mean=mi3, variance=vi3,
            is_training=True, data_format='NCHW')
    xla_r2 = fbn_nchw_xla(x_t2)

    print(f"Eager: NaN={np.isnan(eager_r2.y.numpy()).sum()}/{x_nchw.size}")
    print(f"XLA:   NaN={np.isnan(xla_r2.y.numpy()).sum()}/{x_nchw.size}")
    print(f"XLA variance: {xla_r2.batch_variance.numpy()}")

    assert np.isnan(eager_r2.y.numpy()).sum() == 0
    assert (xla_r2.batch_variance.numpy() < 0).any()

    # === Test 3: Gradient breaks at offset=1e7 ===
    print("\n=== Gradient at offset=1e7 ===")
    x_g = tf.constant(np.random.randn(4, 8, 8, 1).astype(np.float32) + 1e7)

    with tf.GradientTape() as tape:
        tape.watch(x_g)
        r = tf.raw_ops.FusedBatchNormV3(
            x=x_g, scale=sc1, offset=of1, mean=mi1, variance=vi1,
            is_training=True, data_format='NHWC')
        loss = tf.reduce_sum(r.y)
    eager_grad = tape.gradient(loss, x_g)
    print(f"Eager grad NaN: {np.isnan(eager_grad.numpy()).sum()}")

    @tf.function(jit_compile=True)
    def fbn_grad_xla(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            r = tf.raw_ops.FusedBatchNormV3(
                x=x, scale=sc1, offset=of1, mean=mi1, variance=vi1,
                is_training=True, data_format='NHWC')
            loss = tf.reduce_sum(r.y)
        return tape.gradient(loss, x)
    xla_grad = fbn_grad_xla(x_g)
    print(f"XLA   grad NaN: {np.isnan(xla_grad.numpy()).sum()}")

    assert np.isnan(eager_grad.numpy()).sum() == 0
    assert np.isnan(xla_grad.numpy()).sum() > 0

    print("\nBUG CONFIRMED: XLA FusedBatchNormV3 uses E[x²]-E[x]² formula")
    print("Negative variance via catastrophic cancellation at offset ≥ 1e4 (NCHW)")
