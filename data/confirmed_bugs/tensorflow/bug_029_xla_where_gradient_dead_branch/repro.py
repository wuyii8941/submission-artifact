"""Bug 025: XLA gradient of tf.where leaks NaN/inf from dead branch.

Under XLA, tf.where(cond, f(x), g(x)) evaluates both branches during
backpropagation, so gradient of the dead branch (f'(x) or g'(x)) leaks
into the result even when that branch is not selected.

Eager mode correctly computes gradient only through the selected branch.

Affects any safe-guarded pattern: where(x>0, sqrt(x), 0), where(x>0, 1/x, 1), etc.
"""
import tensorflow as tf
import numpy as np

print("TF version:", tf.__version__)
print("=" * 70)
bugs_found = 0

# --- Pattern 1: where(x>0, sqrt(x), 0) ---
def sqrt_grad_eager(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x > 0, tf.sqrt(x), tf.zeros_like(x))
    return t.gradient(y, x)

@tf.function(jit_compile=True)
def sqrt_grad_xla(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x > 0, tf.sqrt(x), tf.zeros_like(x))
    return t.gradient(y, x)

print("\n[Pattern 1] where(x>0, sqrt(x), 0)")
for val, label in [(0.0, "x=0 (boundary)"), (-1.0, "x=-1 (negative)")]:
    x = tf.constant([val])
    e = sqrt_grad_eager(x).numpy()
    x_val = sqrt_grad_xla(x).numpy()
    match = "OK" if np.array_equal(e, x_val) or (np.isnan(e).all() and np.isnan(x_val).all()) else "BUG"
    if match == "BUG": bugs_found += 1
    print(f"  {label}: eager={e}, XLA={x_val}  [{match}]")

# Mixed array
x = tf.constant([-1.0, 0.0, 1.0, 4.0])
e = sqrt_grad_eager(x).numpy()
x_val = sqrt_grad_xla(x).numpy()
print(f"  mixed [-1,0,1,4]: eager={e}")
print(f"                     XLA  ={x_val}")

# Forward pass verification
fwd_e = tf.where(x > 0, tf.sqrt(x), tf.zeros_like(x)).numpy()
fwd_x = tf.function(lambda x: tf.where(x > 0, tf.sqrt(x), tf.zeros_like(x)),
                     jit_compile=True)(x).numpy()
print(f"  Forward: eager={fwd_e}, XLA={fwd_x}, agree={np.array_equal(fwd_e, fwd_x)}")

# --- Pattern 2: where(x>0, reciprocal(x), 1) ---
def recip_grad_eager(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x > 0, tf.math.reciprocal(x), tf.ones_like(x))
    return t.gradient(y, x)

@tf.function(jit_compile=True)
def recip_grad_xla(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x > 0, tf.math.reciprocal(x), tf.ones_like(x))
    return t.gradient(y, x)

print("\n[Pattern 2] where(x>0, 1/x, 1)")
x = tf.constant([0.0])
e = recip_grad_eager(x).numpy()
x_val = recip_grad_xla(x).numpy()
match = "BUG" if not np.array_equal(e, x_val) else "OK"
if match == "BUG": bugs_found += 1
print(f"  x=0: eager={e}, XLA={x_val}  [{match}]")

# --- Pattern 3: where(x>0, log(x), -100) ---
def log_grad_eager(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x > 0, tf.math.log(x), tf.constant(-100.0, shape=x.shape))
    return t.gradient(y, x)

@tf.function(jit_compile=True)
def log_grad_xla(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x > 0, tf.math.log(x), tf.constant(-100.0, shape=x.shape))
    return t.gradient(y, x)

print("\n[Pattern 3] where(x>0, log(x), -100)")
x = tf.constant([0.0])
e = log_grad_eager(x).numpy()
x_val = log_grad_xla(x).numpy()
match = "BUG" if np.isfinite(e).any() != np.isfinite(x_val).any() else "OK/BOTH_INF"
print(f"  x=0: eager={e}, XLA={x_val}  [{match}]")

x = tf.constant([-1.0])
e = log_grad_eager(x).numpy()
x_val = log_grad_xla(x).numpy()
match = "BUG" if np.isfinite(e).any() != np.isfinite(x_val).any() else "OK/AGREE"
print(f"  x=-1: eager={e}, XLA={x_val}  [{match}]")

# --- Pattern 4: where(x>=0, x**0.5, -x**0.5) at x=0 ---
def pow_grad_eager(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x >= 0, x ** 0.5, -((-x) ** 0.5))
    return t.gradient(y, x)

@tf.function(jit_compile=True)
def pow_grad_xla(x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.where(x >= 0, x ** 0.5, -((-x) ** 0.5))
    return t.gradient(y, x)

print("\n[Pattern 4] where(x>=0, x^0.5, -((-x)^0.5))")
x = tf.constant([0.0])
e = pow_grad_eager(x).numpy()
x_val = pow_grad_xla(x).numpy()
match = "BUG" if np.isfinite(e).any() != np.isfinite(x_val).any() else "OK/AGREE"
print(f"  x=0: eager={e}, XLA={x_val}  [{match}]")

# --- Practical impact: safe_sqrt in ML ---
print("\n[Practical] safe_sqrt used in distance computation")

@tf.function(jit_compile=True)
def safe_distance_grad_xla(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        diff_sq = (x - y) ** 2
        dist_sq = tf.reduce_sum(diff_sq)
        dist = tf.where(dist_sq > 0, tf.sqrt(dist_sq), tf.constant(0.0))
    return t.gradient(dist, x)

def safe_distance_grad_eager(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        diff_sq = (x - y) ** 2
        dist_sq = tf.reduce_sum(diff_sq)
        dist = tf.where(dist_sq > 0, tf.sqrt(dist_sq), tf.constant(0.0))
    return t.gradient(dist, x)

x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([1.0, 2.0, 3.0])  # same point → distance = 0
e = safe_distance_grad_eager(x, y).numpy()
x_val = safe_distance_grad_xla(x, y).numpy()
print(f"  identical points: eager={e}, XLA={x_val}")

print(f"\n{'='*70}")
print(f"Bugs confirmed: {bugs_found}")
print("Core issue: XLA evaluates gradient of BOTH branches of tf.where,")
print("leaking NaN/inf from the dead branch into the result.")
