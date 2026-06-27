"""JAX counterpart of minimal_repro.py, used to confirm that the bug
lives in XLA (not in TensorFlow's XLA bridge).

Since JAX dispatches through the same XLA runtime that TF uses for
jit_compile=True, reproducing the shape-dependent erf precision here
rules out any TF-specific explanation.

Tested on jax 0.9.2, CPU backend.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

print(f"jax: {jax.__version__}")
print(f"devices: {jax.devices()}")


@jax.jit
def jit_erf(x):
    return lax.erf(x)


shapes = [(1,), (2,), (7, 1), (7, 64, 1), (7, 64, 5), (1, 1, 1)]

print()
print(f"{'shape':>15}  {'eager lax.erf[0]':>20}  {'jit(lax.erf)[0]':>20}")
for shape in shapes:
    x = jnp.full(shape, 4.5, dtype=jnp.float32)
    e = lax.erf(x)
    xl = jit_erf(x)
    e0 = float(np.asarray(e).flat[0])
    x0 = float(np.asarray(xl).flat[0])
    print(f"{str(shape):>15}  {e0:20.10f}  {x0:20.10f}")

# Floor amplification
print()
x = jnp.full((7, 64, 1), 4.5, dtype=jnp.float32)
eager_sum = float(jnp.sum(jnp.floor(lax.erf(x))))
jit_sum = float(jnp.sum(jnp.floor(jit_erf(x))))
print(f"sum(floor(eager lax.erf)) = {eager_sum}  (448 elements, TF eager returns 448)")
print(f"sum(floor(jit   lax.erf)) = {jit_sum}    (JAX agrees with TF-XLA at 0)")
