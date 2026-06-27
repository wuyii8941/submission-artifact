## [inductor] `torch.compile` crashes on `x[torch.randperm(n)]` — pattern matcher missing `slice_shape` input

### 🐛 Describe the bug

`torch.compile(backend='inductor')` crashes when a `torch.randperm` result is used for fancy indexing. The crash occurs in the pattern matcher during `joint_graph_passes`:

```
RuntimeError: Not all inputs to pattern found in match.kwargs.
Perhaps one of the inputs is unused?
argnames=['x', 'slice_shape'], match.kwargs={'x': arg0_1}
```

`aot_eager` works fine — the crash is inductor-specific, in `torch/_inductor/pattern_matcher.py`.

### Reproducer

```python
import torch

def fn(x):
    idx = torch.randperm(x.shape[0], device=x.device)
    return x[idx]

x = torch.randn(8, 64, device="cuda")

# Eager: OK
print(fn(x).shape)  # torch.Size([8, 64])

# aot_eager: OK
torch._dynamo.reset()
print(torch.compile(fn, backend="aot_eager")(x).shape)  # torch.Size([8, 64])

# Inductor: CRASH
torch._dynamo.reset()
print(torch.compile(fn, backend="inductor")(x).shape)  # RuntimeError
```

### Crash location

```
File "torch/_inductor/fx_passes/joint_graph.py", line 693, in joint_graph_passes
  .apply_graph_pass(patterns.apply)
File "torch/_inductor/pattern_matcher.py", line 2152, in apply
  if is_match(m) and guard_or_false(entry.extra_check(m)):
File "torch/_inductor/pattern_matcher.py", line 1530, in check_fn
  raise RuntimeError("Not all inputs to pattern found in match.kwargs. ...")
```

A pattern rule expects both `x` and `slice_shape` in `match.kwargs`, but only `x` is present. The `slice_shape` input is not captured by the pattern match.

### Trigger conditions

- `torch.randperm(n, device=...)` followed by fancy indexing `x[idx]`
- Inductor backend only (`aot_eager` is fine)
- All tensor shapes and dimensions tested (1D, 2D, 3D; from 8 to 1024 elements)
- Both CUDA and CPU
- Both forward-only and forward+backward
- Not hardware-dependent

### Versions

- PyTorch: 2.13.0.dev20260513+cu126
- GPU: Tesla T4 (SM 7.5), also reproducible on CPU
- OS: Linux

cc <upstream maintainer> <upstream maintainer>
