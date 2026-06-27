# `torch.compile` silently drops higher-order grad graph metadata from `autograd.grad(create_graph=True)` outputs

> GitHub issue: <upstream issue, anonymized for review>
>
> Maintainer confirmation: <upstream issue, anonymized for review>
>
> Status as of 2026-04-28: reproduced on main; labeled `module: correctness (silent)`,
> `module: double backwards`, `module: aotdispatch`, `oncall: pt2`, and `high priority`.
> The maintainer comment states that the issue is not a regression and produces a
> silently incorrect result. PR <upstream PR, anonymized for review>
> converts the silent incorrectness into an explicit unsupported-double-backward
> failure; full support requires additional changes.

## 🐛 Describe the bug

When using `torch.autograd.grad(..., create_graph=True)` inside a `torch.compile`d function, the returned gradient tensor can silently lose its higher-order gradient graph metadata for some ops (`mm`, `matmul`, `addmm`, `relu`, `leaky_relu`). In eager mode the same code returns a gradient tensor with `requires_grad=True` and a valid `grad_fn`, while the compiled version returns a tensor with `requires_grad=False` and `grad_fn=None`.

This is surprising because `create_graph=True` asks autograd to construct the derivative graph for higher-order differentiation. If this pattern is currently unsupported by `torch.compile` / AOTAutograd for some operators, it would be helpful to either preserve eager-compatible behavior or fail explicitly instead of returning a tensor that looks like a normal value but no longer participates in the expected gradient graph.

The silent loss of graph metadata breaks double/higher-order backward, which is required by:
- **MAML** (Model-Agnostic Meta-Learning) — inner-loop gradient update must be differentiable
- **WGAN-GP** — gradient penalty requires gradients of gradients
- **Physics-informed neural networks** — PDE losses involve gradients of network outputs

This also reproduces with `backend='aot_eager'`, so it does not appear to be Inductor-specific.

## Minimal reproducer

```python
import torch

x = torch.randn(4, 8, device='cuda', requires_grad=True)
w = torch.randn(8, 8, device='cuda', requires_grad=True)

def f(x, w):
    y = torch.mm(x, w).sum()
    grad_x, = torch.autograd.grad(y, x, create_graph=True)
    return grad_x.sum()

# Eager: requires_grad=True, grad_fn=<SumBackward0>
r_eager = f(x, w)
print(r_eager.requires_grad)  # True ✅

# Compile: requires_grad=False, grad_fn=None
torch._dynamo.reset()
r_compile = torch.compile(f)(x.clone().detach().requires_grad_(True),
                              w.clone().detach().requires_grad_(True))
print(r_compile.requires_grad)  # False

# Consequence: higher-order backward fails later
r_compile.backward()
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

## Real-world impact: WGAN-GP gradient penalty

```python
import torch
import torch.nn as nn

G = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64)).cuda()
D = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()

def gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    grad, = torch.autograd.grad(d_interp.sum(), interp, create_graph=True)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

real = torch.randn(8, 64, device='cuda')
fake = G(torch.randn(8, 16, device='cuda'))

# Eager: works
gp = gradient_penalty(D, real, fake.detach())
gp.backward()  # ✅

# Compile: fails
torch._dynamo.reset()
gp = torch.compile(gradient_penalty)(D, real, fake.detach())
gp.backward()  # ❌ RuntimeError
```

## Affected ops (systematic test, 22 ops)

| Op | Eager `requires_grad` | Compile `requires_grad` | Status |
|----|----------------------|------------------------|--------|
| `torch.mm` | True | **False** | differs from eager |
| `x @ w` (matmul 2D) | True | **False** | differs from eager |
| `torch.addmm` | True | **False** | differs from eager |
| `torch.relu` | True | **False** | differs from eager |
| `F.leaky_relu` | True | **False** | differs from eager |
| `F.linear` (no bias) | True | True | ✅ OK |
| `F.linear` (with bias) | True | True | ✅ OK |
| `torch.sigmoid` | True | True | ✅ OK |
| `torch.tanh` | True | True | ✅ OK |
| `F.gelu` | True | True | ✅ OK |
| `F.silu` | True | True | ✅ OK |
| `x * y` | True | True | ✅ OK |
| `x / y` | True | True | ✅ OK |
| `x ** 2` | True | True | ✅ OK |
| `x.exp()` | True | True | ✅ OK |
| `x.sin()` | True | True | ✅ OK |
| `F.layer_norm` | True | True | ✅ OK |
| `F.softmax` | True | True | ✅ OK |
| `torch.mv` | True | True | ✅ OK |
| `x.sum()` | False | False | ✅ OK |
| `x.mean()` | False | False | ✅ OK |
| `(x @ w).norm()` | True | True | ✅ OK |

Note: `F.linear` works correctly despite using `mm`/`addmm` internally — the decomposition path in AOT Autograd differs.

## Trigger conditions

| Condition | Required? | Notes |
|-----------|-----------|-------|
| `create_graph=True` | **Yes** | `create_graph=False` correctly returns `requires_grad=False` |
| Op | Specific | `mm`, `matmul`, `addmm`, `relu`, `leaky_relu` |
| Backend | Any | Reproduces with both `inductor` and `aot_eager` |
| Device | Any | Tested on CUDA (CPU blocked by unrelated C++20 compiler issue) |
| Input dims | Any | 2D matmul is sufficient to trigger |

## Error message

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The issue only becomes visible when `.backward()` is called on the result. The forward computation and first-order gradient values appear correct; the surprising part is that `requires_grad` / `grad_fn` metadata is lost silently even though `create_graph=True` was requested.

## Versions

```
PyTorch: 2.13.0.dev20260425+cu126
Python: 3.11.15
OS: Linux-5.4.0-42-generic-x86_64-with-glibc2.31
CUDA: 12.6
GPU: Tesla T4
```

cc <upstream maintainer> <upstream maintainer> <upstream maintainer> <upstream maintainer>
