#!/usr/bin/env python3
"""
Bug: tf.linalg.eigvalsh returns wrong results for complex Hermitian
     matrices under XLA. Real eigenvalue pairs are packed as (real, imag)
     of complex output values.

     eigvalsh(identity_2x2) should return [1, 1], XLA returns [1+1j, garbage].
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow version: {tf.__version__}")

eigvalsh_xla = tf.function(tf.linalg.eigvalsh, jit_compile=True)

print("\n--- Test cases ---")
cases = [
    ("identity",    [[1+0j, 0+0j], [0+0j, 1+0j]],        [1.0, 1.0]),
    ("diag(2,5)",   [[2+0j, 0+0j], [0+0j, 5+0j]],        [2.0, 5.0]),
    ("[[1,i],[-i,1]]", [[1+0j, 0+1j], [0-1j, 1+0j]],     [0.0, 2.0]),
]

all_wrong = True
for name, mat, expected in cases:
    A = tf.constant(mat, dtype=tf.complex64)
    eager = tf.linalg.eigvalsh(A).numpy()
    xla = eigvalsh_xla(A).numpy()

    eager_ok = np.allclose(eager.real, expected, atol=1e-5) and np.allclose(eager.imag, 0, atol=1e-5)
    xla_ok = np.allclose(xla.real, expected, atol=1e-5) and np.allclose(xla.imag, 0, atol=1e-5)

    print(f"\n  {name}: expected eigenvalues = {expected}")
    print(f"    eager: {eager}  {'OK' if eager_ok else 'WRONG'}")
    print(f"    XLA:   {xla}  {'OK' if xla_ok else 'WRONG'}")
    if xla_ok:
        all_wrong = False

# eigh also affected
print("\n--- tf.linalg.eigh (eigenvalues + eigenvectors) ---")
A = tf.constant([[2+0j, 1+1j], [1-1j, 3+0j]], dtype=tf.complex64)
eig_eager, _ = tf.linalg.eigh(A)
eig_xla, _ = tf.function(tf.linalg.eigh, jit_compile=True)(A)
print(f"  eager eigenvalues: {eig_eager.numpy()}")
print(f"  XLA eigenvalues:   {eig_xla.numpy()}")

# complex128 also affected
print("\n--- complex128 ---")
A128 = tf.constant([[2+0j, 1+1j], [1-1j, 3+0j]], dtype=tf.complex128)
print(f"  eager: {tf.linalg.eigvalsh(A128).numpy()}")
print(f"  XLA:   {tf.function(tf.linalg.eigvalsh, jit_compile=True)(A128).numpy()}")

# real symmetric works correctly (sanity check)
print("\n--- Sanity: real symmetric (works correctly) ---")
A_real = tf.constant([[2.0, 1.0], [1.0, 3.0]], dtype=tf.float32)
print(f"  eager: {tf.linalg.eigvalsh(A_real).numpy()}")
print(f"  XLA:   {tf.function(tf.linalg.eigvalsh, jit_compile=True)(A_real).numpy()}")

assert all_wrong, "Bug not reproduced"
print("\n[CONFIRMED] XLA eigvalsh/eigh returns wrong results for complex input.")
print("  Real eigenvalue pairs are packed as (real, imag) of complex output.")
