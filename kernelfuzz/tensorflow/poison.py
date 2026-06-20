"""Adversarial input generation for TF fuzzer.

Injects adversarial input values into randomly generated programs to trigger
NaN-handling, Inf-handling, and extreme-value kernel paths.

Poison modes:
  none           -- standard normal
  light/heavy/extreme -- 1%/5%/20% poisoned values (NaN, Inf, extreme)
  zero_dim       -- one axis replaced with 0
  huge/tiny      -- magnitude scaling (1e20 / 1e-30)
  size_1_inner   -- innermost axis forced to 1 (degenerate reduction)
  simd_boundary  -- innermost axis in {7,8,9,...,128,129}
  mag_huge_size1 -- size_1_inner + huge magnitude
  cancellation   -- values in 1e-8 range (expm1 sweet spot)
"""
import numpy as np


def poison_array(arr, rng, poison_rate=0.0):
    """Replace `poison_rate` fraction of elements with adversarial values."""
    if poison_rate <= 0:
        return arr
    flat = arr.flatten()
    n = flat.size
    n_poison = int(n * poison_rate)
    if n_poison == 0:
        return arr
    idx = rng.choice(n, size=n_poison, replace=False)
    pool = [np.nan, np.inf, -np.inf, 1e20, -1e20, 1e-10, 0.0, -0.0, 1.0, -1.0, 1e-30]
    flat[idx] = rng.choice(pool, size=n_poison).astype(arr.dtype)
    return flat.reshape(arr.shape)


def _reshape_with_size_1_innermost(shape, rng):
    shape = list(shape)
    if len(shape) == 0:
        return [1]
    if len(shape) == 1:
        return [shape[0], 1]
    shape[-1] = 1
    return shape


def _reshape_to_simd_boundary(shape, rng):
    candidates = [7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    shape = list(shape)
    if len(shape) == 0:
        return [int(rng.choice(candidates))]
    shape[-1] = int(rng.choice(candidates))
    return shape


def gen_poisoned_input(shape, dtype_str, seed, poison_mode="none"):
    """Generate input array with optional poisoning."""
    rng = np.random.default_rng(seed)
    is_complex = "complex" in dtype_str
    np_dtype = {
        "tf.float32": np.float32,
        "tf.float64": np.float64,
        "tf.float16": np.float16,
        "tf.bfloat16": np.float32,
        "tf.complex64": np.complex64,
        "tf.complex128": np.complex128,
    }.get(dtype_str, np.float32)

    if poison_mode == "zero_dim":
        shape = list(shape)
        if len(shape) > 1:
            shape[rng.choice(len(shape))] = 0
        arr = np.zeros(tuple(shape), dtype=np_dtype)
        return arr

    if poison_mode == "size_1_inner":
        shape = _reshape_with_size_1_innermost(shape, rng)
    elif poison_mode == "simd_boundary":
        shape = _reshape_to_simd_boundary(shape, rng)
    elif poison_mode == "mag_huge_size1":
        shape = _reshape_with_size_1_innermost(shape, rng)

    if is_complex:
        real = rng.standard_normal(shape).astype(np.float32 if np_dtype == np.complex64 else np.float64)
        imag = rng.standard_normal(shape).astype(real.dtype)
        if poison_mode == "tiny":
            real *= 1e-20
            imag *= 1e-20
        elif poison_mode == "cancellation":
            real = real * np.float64(1e-8)
            imag = imag * np.float64(1e-8)
        arr = (real + 1j * imag).astype(np_dtype)
    else:
        arr = rng.standard_normal(shape).astype(np_dtype)

    if poison_mode == "cancellation" and not is_complex:
        arr = (arr * 1e-8).astype(np_dtype)
    elif poison_mode == "light":
        arr = poison_array(arr, rng, 0.01)
    elif poison_mode == "heavy":
        arr = poison_array(arr, rng, 0.05)
    elif poison_mode == "extreme":
        arr = poison_array(arr, rng, 0.2)
    elif poison_mode == "huge":
        arr = (arr * 1e20).astype(np_dtype)
    elif poison_mode == "tiny":
        arr = (arr * 1e-30).astype(np_dtype)
    elif poison_mode == "mag_huge_size1":
        arr = (arr * 1e20).astype(np_dtype)

    return arr


POISON_MODES_V2 = ["none"] * 70 + ["light"] * 10 + ["heavy"] * 5 + \
                  ["huge"] * 5 + ["tiny"] * 5 + ["zero_dim"] * 3 + ["extreme"] * 2

POISON_MODES_V3 = (
    ["none"] * 35
    + ["light"] * 5
    + ["heavy"] * 3
    + ["huge"] * 7
    + ["tiny"] * 3
    + ["zero_dim"] * 3
    + ["extreme"] * 2
    + ["size_1_inner"] * 20
    + ["simd_boundary"] * 15
    + ["mag_huge_size1"] * 7
    + ["cancellation"] * 5
)


def sample_poison_mode(rng, preset="v2"):
    """Sample a poison mode from preset distribution."""
    modes = POISON_MODES_V3 if preset == "v3" else POISON_MODES_V2
    return rng.choice(modes)
