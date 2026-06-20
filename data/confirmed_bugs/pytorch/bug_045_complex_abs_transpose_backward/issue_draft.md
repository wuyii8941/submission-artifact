# `torch.compile` backward crashes for `abs()`/`angle()` on transposed complex tensors

## 🐛 Describe the bug

`torch.compile(backend="inductor")` crashes during backward when `.abs()` or `.angle()` is applied to a **transposed (permuted) complex tensor**. The error is a stride assertion failure in Inductor's backward codegen:

```
AssertionError: expected size 8==8, stride 1==4 at dim=0; expected size 4==4, stride 8==1 at dim=1
Error in op: torch.ops.aten.mul.Tensor
```

The forward pass works fine. Eager mode and `aot_eager` handle this correctly — the bug is **Inductor-specific**, in the backward graph compilation for complex abs/angle on non-contiguous (permuted-stride) inputs.

## Minimal reproducer

```python
import torch

x = torch.randn(4, 8, dtype=torch.complex64, device="cuda", requires_grad=True)

# Eager: works
(x.T.abs().sum()).backward()
print(x.grad.shape)  # torch.Size([4, 8])

# Compiled: crashes on backward
x.grad = None
torch.compile(lambda x: x.T.abs().sum(), backend="inductor")(x).backward()
```

## Error message

```
AssertionError: expected size 8==8, stride 1==4 at dim=0; expected size 4==4, stride 8==1 at dim=1
Error in op: torch.ops.aten.mul.Tensor
This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
```

## Backend isolation

| Backend | Forward | Backward |
|---------|---------|----------|
| `eager` | ✅ | ✅ |
| `aot_eager` | ✅ | ✅ |
| `inductor` | ✅ | ❌ **crashes** |

Since `aot_eager` works, the bug is in **Inductor's backward codegen**, not in AOT Autograd tracing.

## Affected operations

Complex operations that decompose into element-wise ops involving the original tensor's layout:

```python
x = torch.randn(4, 8, dtype=torch.complex64, device="cuda", requires_grad=True)

x.T.abs().sum().backward()     # CRASH
x.T.angle().sum().backward()   # CRASH
x.T.conj().abs().sum().backward()  # CRASH (conj + abs)

# Also with permute:
x.permute(1, 0).abs().sum().backward()  # CRASH
x.reshape(8, 4).T.abs().sum().backward()  # CRASH

# Transposed strides propagate through pointwise ops:
(x.T ** 2).abs().sum().backward()   # CRASH — x.T**2 inherits transposed strides
(x.T + 0).abs().sum().backward()    # CRASH — x.T+0 also non-contiguous
(x.T * 1).abs().sum().backward()    # CRASH

# .mH (conjugate transpose): CRASH
x.mH.abs().sum().backward()   # CRASH

# Practical: STFT output is Fortran-contiguous complex
s = torch.stft(signal, n_fft=64, window=window, return_complex=True)
s.abs().sum().backward()       # CRASH
```

## Non-triggering patterns

```python
# Real tensors: OK (only complex is affected)
x_real = torch.randn(4, 8, device="cuda", requires_grad=True)
x_real.T.abs().sum().backward()  # OK

# .real / .imag (no decomposition needed): OK
x.T.real.sum().backward()  # OK
x.T.imag.sum().backward()  # OK

# Manual abs² (avoids the problematic decomposition): OK
(x.T.real**2 + x.T.imag**2).sum().backward()  # OK

# x * x.conj() (different decomposition path): OK
(x.T * x.T.conj()).real.sum().backward()  # OK

# Non-transposed strides (strided but not permuted): OK
x[:, ::2].abs().sum().backward()  # OK
x[::2].abs().sum().backward()     # OK
x.flip(0).abs().sum().backward()  # OK

# Contiguous complex: OK
x.contiguous().abs().sum().backward()  # OK

# Explicit .contiguous() before abs fixes it:
x.T.contiguous().abs().sum().backward()  # OK

# Forward-only (no backward): OK
with torch.no_grad():
    torch.compile(lambda x: x.T.abs())(x)  # OK
```

## Root cause analysis

The trigger condition is **permuted strides on complex tensors + abs/angle backward**:

1. `complex_tensor.T` produces a tensor with transposed (reversed) strides — e.g., shape `(8, 4)` with strides `(1, 8)` instead of the expected contiguous `(4, 1)`.
2. `.abs()` on complex decomposes into operations on the real/imaginary parts.
3. In the **backward graph**, Inductor generates code that multiplies gradients back through these decomposed ops. It appears to assume the intermediate tensor follows contiguous stride order, but the actual tensor has transposed strides.
4. The stride assertion in `aten.mul.Tensor` catches this mismatch.

The key distinction: **strided-but-not-permuted** views (like `x[:, ::2]`) don't crash because the stride ordering is preserved (just scaled). Only **permuted** strides (where `stride[i] > stride[j]` for `i > j`) trigger the bug.

## Practical impact

This pattern is common in signal processing, physics, and scientific computing:
- `torch.stft()` returns Fortran-contiguous complex tensors → `.abs()` for magnitude spectrum crashes
- LAPACK-backed operations (`linalg.solve`, `linalg.eigh`, `linalg.cholesky`, `linalg.qr`, `linalg.inv`) all return Fortran-contiguous complex results → `.abs()` on their output crashes
- Complex matrix operations where `.T` or `.mH` is used before taking magnitude
- Spectral analysis pipelines: `fft → transpose → abs → loss → backward`

```python
# LAPACK ops affected:
torch.linalg.solve(A, b).abs().sum().backward()    # CRASH
torch.linalg.eigh(A)[1].abs().sum().backward()     # CRASH (eigenvectors)
torch.linalg.cholesky(A).abs().sum().backward()    # CRASH
torch.linalg.qr(A)[0].abs().sum().backward()       # CRASH (Q matrix)
torch.linalg.inv(A).abs().sum().backward()          # CRASH

# Exception: SVD works (different backward implementation path)
torch.linalg.svd(A)[0].abs().sum().backward()      # OK
```

## Workaround

Make the tensor contiguous before applying `.abs()`:

```python
# Instead of: x.T.abs().sum()
x.T.contiguous().abs().sum()  # OK (extra copy)

# Or use manual decomposition:
(x.T.real**2 + x.T.imag**2).sqrt().sum()  # OK
```

## Versions

- PyTorch: 2.13.0.dev20260513+cu126
- Python: 3.11
- CUDA: 12.6
- GPU: Tesla T4
