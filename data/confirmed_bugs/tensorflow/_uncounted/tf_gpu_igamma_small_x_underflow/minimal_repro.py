import tensorflow as tf


print("TensorFlow:", tf.__version__)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

a = tf.constant([0.1], dtype=tf.float32)
x = tf.constant([1e-38], dtype=tf.float32)


def lower(a, x):
    return tf.math.igamma(a, x)


def upper(a, x):
    return tf.math.igammac(a, x)


print("a:", a.numpy())
print("x:", x.numpy())

print("igamma eager:   ", lower(a, x).numpy())
print("igamma grappler:", tf.function(lower, jit_compile=False)(a, x).numpy())
print("igamma xla:     ", tf.function(lower, jit_compile=True)(a, x).numpy())

print("igammac eager:   ", upper(a, x).numpy())
print("igammac grappler:", tf.function(upper, jit_compile=False)(a, x).numpy())
print("igammac xla:     ", tf.function(upper, jit_compile=True)(a, x).numpy())
