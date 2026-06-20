"""
Bug 044: torch.compile backward crashes when complex tensor is multiplied/divided
by a Python complex literal.

Inductor codegen receives a raw Python `complex` object in the backward graph
but tries to call .get_name() on it (treating it as an IR node).

Minimal: x * (2.0 + 1.0j) where x is complex and requires_grad.
"""
import torch
import torch._dynamo

x = torch.randn(4, dtype=torch.complex64, device='cuda', requires_grad=True)

def fn(x):
    return x * (2.0 + 1.0j)

# Eager: works
fn(x).abs().sum().backward()
print(f"Eager: OK (grad={x.grad[:2]})")
x.grad = None

# aot_eager: works
torch._dynamo.reset()
x2 = x.detach().clone().requires_grad_(True)
torch.compile(fn, backend='aot_eager')(x2).abs().sum().backward()
print(f"aot_eager: OK")

# inductor: crashes
torch._dynamo.reset()
x3 = x.detach().clone().requires_grad_(True)
try:
    torch.compile(fn, backend='inductor')(x3).abs().sum().backward()
    print(f"inductor: OK")
except Exception as e:
    print(f"inductor: CRASH - {type(e).__name__}: {str(e)[:80]}")

# Workaround: wrap in tensor
print("\nWorkaround: x * torch.tensor(2.0+1.0j)")
torch._dynamo.reset()
x4 = x.detach().clone().requires_grad_(True)
def fn_fix(x):
    return x * torch.tensor(2.0 + 1.0j, device=x.device)
torch.compile(fn_fix, backend='inductor')(x4).abs().sum().backward()
print(f"  OK (grad={x4.grad[:2]})")
