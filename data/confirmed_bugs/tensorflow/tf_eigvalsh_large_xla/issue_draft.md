# `tf.linalg.eigvalsh` returns wrong eigenvalues under XLA for large-magnitude matrices

## Summary

`tf.linalg.eigvalsh` under XLA (`jit_compile=True`) returns incorrect eigenvalues for symmetric matrices with large-magnitude entries (≥ 1e19 in float32). For a rank-1 matrix `[[v, v], [v, v]]` with eigenvalues `[0, 2v]`, XLA returns `[v, v]` (the diagonal elements). Eager mode and NumPy return the correct result.

## Reproduction

```python
import tensorflow as tf
import numpy as np

v = 1e20
m = tf.constant([[v, v], [v, v]], dtype=tf.float32)

print(tf.linalg.eigvalsh(m).numpy())
# [0.0, 2e+20]  — eager: correct

print(tf.function(tf.linalg.eigvalsh, jit_compile=True)(m).numpy())
# [1e+20, 1e+20]  — XLA: wrong (returns diagonal, not eigenvalues)

print(np.linalg.eigvalsh(m.numpy()))
# [0.0, 2e+20]  — NumPy: correct
```

Reproduces on TF 2.22.0-dev20260503 (nightly), CPU.

## Scope

The bug starts at v ≈ 1e19 for 2x2 matrices and affects both rank-deficient and full-rank matrices:

| Matrix | Expected eigenvalues | XLA result |
|---|---|---|
| `[[1e19, 1e19], [1e19, 1e19]]` | `[0, 2e19]` | `[1e19, 1e19]` |
| `[[1e20, 5e19], [5e19, 1e20]]` | `[5e19, 1.5e20]` | `[1e20, 1e20]` |
| `[[1e20, 1e20, 1e20], ...]` (3x3 rank-1) | `[0, 0, 3e20]` | `[1e20, 1e20, 1e20]` |

For v < 1e19 (2x2), results match between eager and XLA.

## Why 1e19 is not extreme

`1e19` is well within float32 range (max ≈ 3.4e38). The correct eigenvalues `[0, 2e19]` are also perfectly representable. Matrices with entries of this magnitude appear in scientific computing (e.g., physical simulations with SI units).

## Root cause

XLA's Jacobi eigenvalue iteration likely fails to converge when off-diagonal elements are comparable to diagonal elements at high magnitude. The convergence check may use an absolute tolerance that is too loose for large values, causing the algorithm to stop after zero iterations and return the diagonal.

## Environment

- TensorFlow 2.22.0-dev20260503
- CPU only
