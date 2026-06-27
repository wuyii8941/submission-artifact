"""
Bug 045: torch.compile backward crashes for complex .abs()/.angle() on transposed tensors.

Inductor backward codegen produces incorrect stride assertions when complex
abs/angle is applied to a transposed (permuted) complex tensor.

Minimal: x.T.abs().sum().backward() where x is complex and requires_grad.
"""
import torch
import torch._dynamo

x = torch.randn(4, 8, dtype=torch.complex64, device="cuda", requires_grad=True)

def fn(x):
    return x.T.abs().sum()

# Eager: works
fn(x).backward()
print(f"Eager: OK (grad shape={x.grad.shape})")
x.grad = None

# aot_eager: works
torch._dynamo.reset()
torch.compile(fn, backend="aot_eager")(x).backward()
print(f"aot_eager: OK (grad shape={x.grad.shape})")
x.grad = None

# inductor: crashes
torch._dynamo.reset()
try:
    torch.compile(fn, backend="inductor")(x).backward()
    print(f"inductor: OK")
except Exception as e:
    print(f"inductor: CRASH - {type(e).__name__}: {str(e)[:100]}")

# Also crashes with .angle()
x.grad = None
torch._dynamo.reset()
try:
    torch.compile(lambda x: x.T.angle().sum(), backend="inductor")(x).backward()
    print(f"angle: OK")
except Exception as e:
    print(f"angle: CRASH - {type(e).__name__}")

# Forward-only works fine:
torch._dynamo.reset()
with torch.no_grad():
    out = torch.compile(fn, backend="inductor")(x)
    print(f"\nForward-only: OK (result={out.item():.4f})")
