# `tf.linalg.eigvalsh` / `eigh` returns wrong results for complex input under XLA

> **Status: CONFIRMED** — Filed as [upstream issue <anonymized>], labeled `type:bug` + `comp:xla`; [community reproduced](<upstream issue, anonymized for review>)

## Summary

`tf.linalg.eigvalsh` and `tf.linalg.eigh` return **completely wrong eigenvalues** for complex Hermitian matrices when compiled with XLA (`jit_compile=True`). Consecutive pairs of correct real eigenvalues are packed into (real, imag) parts of complex output values. The identity matrix demonstrates the issue:

```python
import tensorflow as tf

I = tf.constant([[1+0j, 0+0j], [0+0j, 1+0j]], dtype=tf.complex64)

print(tf.linalg.eigvalsh(I).numpy())
# [1.+0.j 1.+0.j]  — correct

print(tf.function(tf.linalg.eigvalsh, jit_compile=True)(I).numpy())
# [1.+1.j  8935.+4.6e-41j]  — WRONG: eigenvalue pair (1,1) packed as 1+1j
```

The eigenvalues of the identity matrix are [1, 1]. XLA returns [1+1j, garbage].

## Reproduction

```python
import tensorflow as tf
import numpy as np

# Test 1: identity matrix — eigenvalues should be [1, 1]
I = tf.constant([[1+0j, 0+0j], [0+0j, 1+0j]], dtype=tf.complex64)
print(tf.function(tf.linalg.eigvalsh, jit_compile=True)(I).numpy())
# [1.+1.j  ...]  — WRONG

# Test 2: diagonal matrix — eigenvalues should be [2, 5]
D = tf.constant([[2+0j, 0+0j], [0+0j, 5+0j]], dtype=tf.complex64)
print(tf.function(tf.linalg.eigvalsh, jit_compile=True)(D).numpy())
# [2.+5.j  0.+0.j]  — WRONG: (2,5) packed as 2+5j

# Test 3: off-diagonal Hermitian — eigenvalues should be [0, 2]
H = tf.constant([[1+0j, 0+1j], [0-1j, 1+0j]], dtype=tf.complex64)
print(tf.function(tf.linalg.eigvalsh, jit_compile=True)(H).numpy())
# [0.+2.j  0.+0.j]  — WRONG: (0,2) packed as 0+2j
```

All cases work correctly in eager mode. Tested on TF 2.22.0-dev20260503.

## Scope

The packing pattern is consistent across all matrix sizes and values tested:

| Matrix | Correct eigenvalues | XLA result |
|--------|-------------------|------------|
| 2×2 identity | [1, 1] | [1+1j, garbage] |
| diag(2, 5) | [2, 5] | [2+5j, 0] |
| [[1,i],[-i,1]] | [0, 2] | [0+2j, 0] |
| random 4×4 Hermitian | [0.62, 2.45, 4.52, 13.96] | [0.62+2.45j, 4.52+13.96j, 0, 0] |

### Affected:
- `tf.linalg.eigvalsh` and `tf.linalg.eigh` (both eigenvalues and eigenvectors)
- Both `complex64` and `complex128` input
- Both CPU XLA and GPU XLA
- Real symmetric (`float32`/`float64`) input works correctly

## Root cause

The XLA implementation computes the correct `n` real eigenvalues and stores them sequentially in a float buffer. However, the output is typed as `complex64` (matching the input dtype), which causes each pair of consecutive real eigenvalues to be reinterpreted as the (real, imaginary) parts of a single complex number.

For `n` eigenvalues:
- Buffer contains: `[λ₁, λ₂, λ₃, λ₄, ...]` as `n` floats
- Read as `complex64`: `[(λ₁+λ₂j), (λ₃+λ₄j), ...]` — wrong
- Correct interpretation: `[(λ₁+0j), (λ₂+0j), (λ₃+0j), (λ₄+0j), ...]`

Note: returning `complex64` for complex input is the intended output dtype (eager does this too), but the buffer layout is wrong under XLA.

## Impact

This makes `eigvalsh` and `eigh` completely unusable under XLA for any complex-valued input. Affected use cases include quantum mechanics, signal processing, and any application using Hermitian eigendecomposition with `jit_compile=True`.

## Environment

- TensorFlow 2.22.0-dev20260503
- CPU and GPU XLA both affected
- Eager execution is correct
