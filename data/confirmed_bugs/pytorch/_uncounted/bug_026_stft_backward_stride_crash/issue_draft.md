# `torch.compile` crashes on `torch.stft(...).abs().sum().backward()` with stride assertion error

## Summary

`torch.compile(lambda x: torch.stft(x, ...).abs().sum(), backend="inductor").backward()` crashes with `AssertionError` on a stride mismatch. The generated Inductor backward code expects the STFT backward output (`_fft_c2r`) to be C-contiguous (row-major), but the actual cuFFT kernel produces Fortran-contiguous (column-major) output.

- Affects **all n_fft sizes** (tested: 64, 128, 256, 512, 1024).
- `aot_eager` backend works correctly — the bug is **Inductor-specific**.
- **CUDA only** — CPU Inductor does not crash.
- Triggered by `.sum()` or `.sum(dim=-1).sum()` but NOT by `.mean()` or `.norm()`.
- External tangent (non-stride-0, via `.backward(grad)`) does NOT crash.

## Reproducer

```python
import torch

x = torch.randn(2, 1024, device='cuda', requires_grad=True)
window = torch.hann_window(256, device='cuda')

# Eager — works
x_e = x.detach().clone().requires_grad_(True)
torch.stft(x_e, 256, hop_length=64, win_length=256,
           window=window, return_complex=True).abs().sum().backward()
print("Eager: OK")

# Compiled — crashes
torch._dynamo.reset()
x_c = x.detach().clone().requires_grad_(True)
torch.compile(
    lambda x, w: torch.stft(x, 256, hop_length=64, win_length=256,
                             window=w, return_complex=True).abs().sum()
)(x_c, window).backward()
# AssertionError: expected size 129==129, stride 1==17 at dim=1;
#                 expected size 17==17, stride 129==1 at dim=2
```

## Error

```
AssertionError: expected size 129==129, stride 1==17 at dim=1; expected size 17==17, stride 129==1 at dim=2
```

The generated wrapper code contains:
```python
assert_size_stride(buf12, (2, 129, 17), (2193, 17, 1))
```
but the actual tensor has strides `(2193, 1, 129)` — the frequency and time dimensions are transposed.

## Root cause

The meta/fake kernel for the STFT backward op (`_fft_c2r` or similar) reports C-contiguous output strides, but the actual cuFFT backward kernel produces Fortran-contiguous output (frequency dimension contiguous). Inductor trusts the meta kernel's stride prediction to generate `assert_size_stride` checks and plan the memory layout of downstream kernels. When the actual output strides don't match, the assertion fails.

This only manifests when Inductor generates a backward kernel that asserts the strides of the STFT backward output. The `.sum()` reduction pattern triggers this code path, while `.mean()` and `.norm()` take different backward decomposition paths that don't hit the assertion.

## Characterization

### Backend comparison
```
aot_eager:   OK
inductor:    CRASH
```

### CPU vs CUDA
```
cpu:   OK
cuda:  CRASH
```

### n_fft sweep (all crash)
```
n_fft=64:    CRASH
n_fft=128:   CRASH
n_fft=256:   CRASH
n_fft=512:   CRASH
n_fft=1024:  CRASH
```

### Reduction pattern sensitivity
```
.sum():              CRASH
.mean():             OK
.norm():             OK
.sum(dim=-1).sum():  CRASH
```

## Environment

```
PyTorch: 2.7.0.dev20250310+cu124
GPU:     Tesla T4, sm_75
CUDA:    12.2
OS:      Linux 5.4.0-42-generic
```

cc <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer>
