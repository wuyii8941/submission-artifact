"""
Pattern Migration Test: PyTorch confirmed bug patterns → JAX/XLA
Tests whether the same optimization pitfalls exist in XLA.
"""
import jax
import jax.numpy as jnp
import numpy as np
import sys

jax.config.update("jax_enable_x64", True)

PASS = 0
FAIL = 0

def report(name, eager_val, jit_val, atol=1e-6):
    global PASS, FAIL
    if isinstance(eager_val, jnp.ndarray):
        eager_val = float(eager_val)
    if isinstance(jit_val, jnp.ndarray):
        jit_val = float(jit_val)
    diff = abs(eager_val - jit_val)
    status = "PASS" if diff <= atol else f"**FAIL** diff={diff:.2e}"
    if diff > atol:
        FAIL += 1
        print(f"  [DIVERGE] {name}: eager={eager_val:.10f} jit={jit_val:.10f} diff={diff:.2e}")
    else:
        PASS += 1
    return diff > atol


# ===========================================================================
# Pattern 1: fp64 softmax at boundary shapes (PyTorch bug_005 analog)
# Bug_005: Triton persistent reduction with masked fp64 at dim≥513, odd
# XLA also does reduction fusion — does it have similar precision bugs?
# ===========================================================================
print("\n=== Pattern 1: fp64 softmax at boundary shapes ===")

def softmax_fp64(x):
    return jax.nn.softmax(x, axis=-1).sum()

for shape in [(2, 513), (2, 1025), (4, 511), (4, 513), (1, 1023), (1, 1025),
              (3, 65), (3, 129), (3, 257), (8, 2049)]:
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, shape, dtype=jnp.float64)
    eager_out = softmax_fp64(x)
    jit_out = jax.jit(softmax_fp64)(x)
    report(f"softmax_fp64 shape={shape}", eager_out, jit_out, atol=1e-10)


# ===========================================================================
# Pattern 2: cumsum precision at adversarial shapes (PyTorch bug_003 analog)
# Bug_003: cumsum + relu + matmul with shapes 257, 129
# ===========================================================================
print("\n=== Pattern 2: cumsum at adversarial shapes ===")

def cumsum_matmul(x, w):
    y = jnp.cumsum(x, axis=-1)
    y = jax.nn.relu(y)
    return jnp.matmul(y, w).sum()

for seq in [127, 128, 129, 255, 256, 257, 513, 1025]:
    key = jax.random.PRNGKey(7)
    x = jax.random.normal(key, (4, seq), dtype=jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(8), (seq, 32), dtype=jnp.float32)
    eager_out = cumsum_matmul(x, w)
    jit_out = jax.jit(cumsum_matmul)(x, w)
    report(f"cumsum_matmul seq={seq}", eager_out, jit_out, atol=1e-3)


# ===========================================================================
# Pattern 3: layernorm manual at boundary shapes (D1 fusion analog)
# Inductor fuses mean/var/normalize — does XLA's fusion cause precision loss?
# ===========================================================================
print("\n=== Pattern 3: manual layernorm at boundaries ===")

def manual_layernorm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / jnp.sqrt(var + eps)).sum()

for hidden in [63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025]:
    key = jax.random.PRNGKey(99)
    x = jax.random.normal(key, (4, 32, hidden), dtype=jnp.float32)
    eager_out = manual_layernorm(x)
    jit_out = jax.jit(manual_layernorm)(x)
    report(f"layernorm hidden={hidden}", eager_out, jit_out, atol=1e-2)


# ===========================================================================
# Pattern 4: reduction chain (online softmax pattern, D1)
# Multiple dependent reductions that may be fused differently
# ===========================================================================
print("\n=== Pattern 4: chained reductions ===")

def chained_reduce(x):
    m = x.max(axis=-1, keepdims=True)
    e = jnp.exp(x - m)
    s = e.sum(axis=-1, keepdims=True)
    out = e / s
    return out.sum()

for shape in [(2, 513), (2, 1025), (4, 2049), (1, 4097), (8, 8193)]:
    key = jax.random.PRNGKey(123)
    x = jax.random.normal(key, shape, dtype=jnp.float64)
    eager_out = chained_reduce(x)
    jit_out = jax.jit(chained_reduce)(x)
    report(f"chained_reduce shape={shape}", eager_out, jit_out, atol=1e-10)


# ===========================================================================
# Pattern 5: gather/scatter patterns (D4 cross-op dependency)
# ===========================================================================
print("\n=== Pattern 5: gather + reduce ===")

def gather_reduce(x, indices):
    gathered = x[indices]
    return gathered.sum(axis=-1).mean()

for n in [128, 129, 256, 257, 512, 513, 1024, 1025]:
    key = jax.random.PRNGKey(55)
    x = jax.random.normal(key, (n, 64), dtype=jnp.float32)
    indices = jax.random.randint(jax.random.PRNGKey(56), (32,), 0, n)
    eager_out = gather_reduce(x, indices)
    jit_out = jax.jit(gather_reduce)(x, indices)
    report(f"gather_reduce n={n}", eager_out, jit_out, atol=1e-4)


# ===========================================================================
# Pattern 6: adaptive avgpool equivalent + reduction (PyTorch bug_006 analog)
# Bug_006: stride rounding in multi-batch reduction
# ===========================================================================
print("\n=== Pattern 6: multi-batch reduction with reshape ===")

def reshape_reduce(x):
    b, c, h, w = x.shape
    # Simulate adaptive_avg_pool: reshape then mean
    y = x.reshape(b, c, h // 2, 2, w // 2, 2)
    y = y.mean(axis=(3, 5))
    return y.sum()

for shape in [(2, 16, 8, 8), (2, 16, 64, 64), (2, 32, 65, 65), (4, 16, 130, 130)]:
    # Only test shapes where h,w are even for the reshape
    h, w = shape[2], shape[3]
    if h % 2 != 0 or w % 2 != 0:
        continue
    key = jax.random.PRNGKey(77)
    x = jax.random.normal(key, shape, dtype=jnp.float32)
    eager_out = reshape_reduce(x)
    jit_out = jax.jit(reshape_reduce)(x)
    report(f"reshape_reduce shape={shape}", eager_out, jit_out, atol=1e-2)


# ===========================================================================
# Pattern 7: var_mean fused (D1 reduction fusion)
# ===========================================================================
print("\n=== Pattern 7: var + mean combined ===")

def var_mean_combine(x):
    v = x.var(axis=-1)
    m = x.mean(axis=-1)
    return (v * m).sum()

for hidden in [127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025]:
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (8, 16, hidden), dtype=jnp.float32)
    eager_out = var_mean_combine(x)
    jit_out = jax.jit(var_mean_combine)(x)
    report(f"var_mean hidden={hidden}", eager_out, jit_out, atol=1e-1)


# ===========================================================================
# Pattern 8: matmul + epilogue (D2 GEMM epilogue fusion)
# ===========================================================================
print("\n=== Pattern 8: matmul + bias + activation ===")

def matmul_epilogue(x, w, b):
    y = jnp.matmul(x, w) + b
    y = jax.nn.gelu(y)
    return y.sum()

for k in [63, 64, 65, 127, 128, 129, 255, 256, 257]:
    key = jax.random.PRNGKey(33)
    x = jax.random.normal(key, (4, 32, k), dtype=jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(34), (k, 64), dtype=jnp.float32)
    b = jax.random.normal(jax.random.PRNGKey(35), (64,), dtype=jnp.float32)
    eager_out = matmul_epilogue(x, w, b)
    jit_out = jax.jit(matmul_epilogue)(x, w, b)
    report(f"matmul_epilogue k={k}", eager_out, jit_out, atol=1e-1)


# ===========================================================================
# Pattern 9: fp64 cumsum last == sum (invariant check under XLA)
# ===========================================================================
print("\n=== Pattern 9: cumsum invariant (last element == sum) ===")

def cumsum_invariant(x):
    cs = jnp.cumsum(x, axis=-1)
    last = cs[..., -1]
    total = x.sum(axis=-1)
    return (last - total).abs().max()

for n in [127, 128, 129, 255, 256, 257, 513, 1025, 2049, 4097]:
    key = jax.random.PRNGKey(11)
    x = jax.random.normal(key, (4, n), dtype=jnp.float64)
    eager_out = cumsum_invariant(x)
    jit_out = jax.jit(cumsum_invariant)(x)
    # The invariant itself should be ~0; if JIT breaks it that's a bug
    if float(jit_out) > 1e-10:
        print(f"  [INVARIANT VIOLATED] cumsum n={n}: jit invariant_diff={float(jit_out):.2e}")
        FAIL += 1
    else:
        PASS += 1


# ===========================================================================
# Pattern 10: transpose + matmul (D3 layout propagation)
# ===========================================================================
print("\n=== Pattern 10: transpose + matmul ===")

def transpose_matmul(x, w):
    y = x.transpose(0, 2, 1)  # (B, H, S)
    z = jnp.matmul(y, w)
    return z.sum()

for s, h in [(63, 64), (64, 63), (65, 64), (128, 129), (129, 128), (257, 256)]:
    key = jax.random.PRNGKey(44)
    x = jax.random.normal(key, (4, s, h), dtype=jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(45), (h, 32), dtype=jnp.float32)
    eager_out = transpose_matmul(x, w)
    jit_out = jax.jit(transpose_matmul)(x, w)
    report(f"transpose_matmul s={s} h={h}", eager_out, jit_out, atol=1e-1)


print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    print("⚠️  DIVERGENCES FOUND — investigate above failures")
else:
    print("All patterns pass — no cross-framework bugs detected")
