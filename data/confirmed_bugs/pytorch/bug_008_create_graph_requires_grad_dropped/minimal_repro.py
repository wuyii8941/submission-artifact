"""
Bug: torch.compile drops requires_grad from torch.autograd.grad(create_graph=True) output.

In eager mode, autograd.grad(..., create_graph=True) returns a tensor that participates
in the computation graph (requires_grad=True, has grad_fn). Under torch.compile, the
returned gradient tensor has requires_grad=False and no grad_fn, breaking double backward.

This affects any algorithm that needs higher-order gradients:
  - MAML (meta-learning inner loop)
  - WGAN-GP (gradient penalty)
  - Physics-informed neural networks (PDE loss on gradients)
"""
import torch

x = torch.randn(4, 8, device='cuda', requires_grad=True)
w = torch.randn(8, 8, device='cuda', requires_grad=True)

def f(x, w):
    y = torch.mm(x, w).sum()
    grad_x, = torch.autograd.grad(y, x, create_graph=True)
    return grad_x.sum()

# === Eager: works ===
r_eager = f(
    x.clone().detach().requires_grad_(True),
    w.clone().detach().requires_grad_(True),
)
print(f"Eager:   requires_grad={r_eager.requires_grad}, grad_fn={r_eager.grad_fn}")
# Eager:   requires_grad=True, grad_fn=<SumBackward0 ...>

# === Compile: broken ===
torch._dynamo.reset()
x2 = x.clone().detach().requires_grad_(True)
w2 = w.clone().detach().requires_grad_(True)
r_compile = torch.compile(f)(x2, w2)
print(f"Compile: requires_grad={r_compile.requires_grad}, grad_fn={r_compile.grad_fn}")
# Compile: requires_grad=False, grad_fn=None

# Double backward fails under compile
try:
    r_compile.backward()
    print(f"Compile backward: OK, w2.grad={w2.grad is not None}")
except RuntimeError as e:
    print(f"Compile backward FAILS: {e}")
# Compile backward FAILS: element 0 of tensors does not require grad ...
