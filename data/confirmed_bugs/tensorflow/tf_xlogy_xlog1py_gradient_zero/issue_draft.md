## `tf.math.xlogy` and `tf.math.xlog1py` gradient w.r.t. x is incorrectly zero when x=0

**Filed:** [upstream issue #119476 (anonymized)]
**Status:** `type:bug`, `comp:ops`, assigned <upstream maintainer>

### Summary

`tf.math.xlogy(x, y)` and `tf.math.xlog1py(x, y)` return incorrect gradient `0` w.r.t. `x` when `x=0` and `y > 0`. The correct gradients are `log(y)` and `log(1+y)` respectively. PyTorch's `torch.xlogy` correctly returns `log(y)` in this case.

### Reproduction

```python
import tensorflow as tf

# xlogy
x = tf.constant([0.0, 0.0, 1.0], dtype=tf.float64)
y = tf.constant([2.0, 5.0, 2.0], dtype=tf.float64)

with tf.GradientTape() as tape:
    tape.watch(x)
    out = tf.math.xlogy(x, y)
g = tape.gradient(out, x)
print("TF xlogy grad:", g.numpy())   # [0.         0.         0.69314718]
# Correct:                            # [0.69314718 1.60943791 0.69314718]

# xlog1py
x2 = tf.constant([0.0, 0.0, 1.0], dtype=tf.float64)
y2 = tf.constant([1.0, 4.0, 1.0], dtype=tf.float64)

with tf.GradientTape() as tape:
    tape.watch(x2)
    out2 = tf.math.xlog1py(x2, y2)
g2 = tape.gradient(out2, x2)
print("TF xlog1py grad:", g2.numpy())  # [0.         0.         0.69314718]
# Correct:                              # [0.69314718 1.60943791 0.69314718]
```

### PyTorch comparison

PyTorch correctly returns the gradient at x=0:

```python
import torch
x = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
y = torch.tensor(2.0, dtype=torch.float64)
torch.xlogy(x, y).backward()
print(x.grad)  # tensor(0.6931, dtype=torch.float64)  -- correct: log(2)
```

### Root cause

Both functions are defined piecewise: `xlogy(0, y) = 0` to handle `0 * log(0) = 0`. The gradient w.r.t. `x` is `log(y)` for all `x` including `x = 0` (when `y > 0`):

```
d/dx xlogy(x, y)|_{x=0} = lim_{h→0} [h·log(y) - 0] / h = log(y)
```

The implementation applies a zero-mask (from the `x == 0` special case in the forward pass) to the gradient as well, but should only apply it to the function value, not the derivative w.r.t. `x`. The gradient w.r.t. `y` is unaffected (correctly returns `x/y = 0` when `x = 0`).

### Impact

This creates a dead zone where the optimizer receives zero gradient and cannot update the parameter through zero. Affects:
- KL divergence computations where class probabilities are zero
- Cross-entropy losses with zero-weighted components  
- Mixture models where component weights pass through zero during optimization

### Environment

- TensorFlow: 2.22.0-dev20260508
- OS: Ubuntu 20.04
- Affects both CPU and GPU
