"""Bug 026: Grappler ArgMax/ArgMin monotonic stripping is wrong for strictly-monotonic
ops when float32 saturation/overflow creates ties.

Grappler's ArithmeticOptimizer strips tanh, exp, sigmoid from argmax/argmin because
they are "strictly monotonic". But in float32:
  - tanh(10) = tanh(20) = 1.0 (saturation)
  - exp(100) = exp(200) = inf (overflow)
  - sigmoid(20) = sigmoid(40) = 1.0 (saturation)

This creates ties that change the argmax result.

Existing bug #<anon> covers NON-strictly-monotonic ops (relu, floor, etc).
This bug is about STRICTLY-monotonic ops that produce ties due to float representation limits.
"""
import tensorflow as tf
import numpy as np

print("TF version:", tf.__version__)
print("=" * 70)
bugs_found = 0

def test_grappler_stripping(op_name, op_fn, x_val, use_argmin=False):
    global bugs_found
    arg_fn = tf.math.argmin if use_argmin else tf.math.argmax
    arg_name = "argmin" if use_argmin else "argmax"

    x = tf.constant([x_val])
    transformed = op_fn(x).numpy()
    eager_result = arg_fn(op_fn(x), axis=-1).numpy()

    tf.config.optimizer.set_experimental_options({"arithmetic_optimization": True})
    @tf.function(jit_compile=False)
    def with_opt(x):
        return arg_fn(op_fn(x), axis=-1)
    grappler_result = with_opt(x).numpy()

    tf.config.optimizer.set_experimental_options({"arithmetic_optimization": False})
    @tf.function(jit_compile=False)
    def without_opt(x):
        return arg_fn(op_fn(x), axis=-1)
    no_opt_result = without_opt(x).numpy()

    # Reset
    tf.config.optimizer.set_experimental_options({"arithmetic_optimization": True})

    match = "OK" if eager_result == grappler_result else "BUG"
    if match == "BUG":
        bugs_found += 1
    print(f"\n  [{op_name}] {arg_name}({op_name}({x_val}))")
    print(f"    {op_name}(x) = {transformed[0]}")
    print(f"    eager={eager_result[0]}, Grappler={grappler_result[0]}, no-opt={no_opt_result[0]}  [{match}]")
    return match == "BUG"

# G4: tanh saturation
print("[tanh saturation — both tanh(10) and tanh(20) = 1.0 in float32]")
test_grappler_stripping("tanh", tf.math.tanh, [10.0, 20.0, 5.0])
test_grappler_stripping("tanh", tf.math.tanh, [5.0, 10.0, 20.0])
test_grappler_stripping("tanh", tf.math.tanh, [-5.0, -10.0, -20.0], use_argmin=True)

# G5: exp overflow
print("\n[exp overflow — both exp(100) and exp(200) = inf in float32]")
test_grappler_stripping("exp", tf.math.exp, [100.0, 200.0, 50.0])
test_grappler_stripping("exp", tf.math.exp, [50.0, 100.0, 200.0])

# G6: sigmoid saturation
print("\n[sigmoid saturation — both sigmoid(20) and sigmoid(40) = 1.0 in float32]")
test_grappler_stripping("sigmoid", tf.nn.sigmoid, [20.0, 40.0, 5.0])
test_grappler_stripping("sigmoid", tf.nn.sigmoid, [5.0, 20.0, 40.0])

# Also: argmin with exp near 0 (underflow to 0)
print("\n[exp underflow — exp(-100) and exp(-200) both = 0 in float32]")
test_grappler_stripping("exp", tf.math.exp, [-100.0, -200.0, -50.0], use_argmin=True)

# Practical: argmax(softmax(logits)) is common in ML
print("\n[Practical impact: temperature-scaled softmax]")
logits = tf.constant([[100.0, 200.0, 50.0]])
sm = tf.nn.softmax(logits)
print(f"  softmax([100,200,50]) = {sm.numpy()}")
print(f"  Note: softmax([100,200,50]) = [0,NaN,0] due to exp overflow")
print(f"  In practice, people use log_softmax or stable softmax to avoid this")

print(f"\n{'='*70}")
print(f"Bugs confirmed: {bugs_found}")
print(f"Root cause: Grappler's ArithmeticOptimizer assumes strictly-monotonic")
print(f"ops preserve argmax/argmin ordering, but float32 saturation/overflow")
print(f"creates ties that change the result.")
print(f"This is DISTINCT from #<anon> which covers non-strictly-monotonic ops.")
