"""
Bug 042: torch.compile backward crashes when output is expand() of computed intermediate.

AOT Autograd functionalization fails to clone stride-0 output grad before backward write.
Crashes with aot_eager (not Inductor-specific).
"""
import torch
import torch._dynamo

def fn(x):
    return x.sum(dim=0, keepdim=True).expand_as(x)

x = torch.randn(4, 8, device="cuda", requires_grad=True)

# Eager: works
fn(x).sum().backward()
print(f"Eager: OK (grad shape={x.grad.shape})")

# aot_eager: crashes (proves AOT Autograd is the bug layer)
x2 = x.detach().clone().requires_grad_(True)
torch._dynamo.reset()
try:
    torch.compile(fn, backend="aot_eager")(x2).sum().backward()
    print(f"aot_eager: OK")
except RuntimeError as e:
    print(f"aot_eager: CRASH - {str(e)[:80]}")

# inductor: crashes
x3 = x.detach().clone().requires_grad_(True)
torch._dynamo.reset()
try:
    torch.compile(fn, backend="inductor")(x3).sum().backward()
    print(f"inductor: OK")
except RuntimeError as e:
    print(f"inductor: CRASH - {str(e)[:80]}")
