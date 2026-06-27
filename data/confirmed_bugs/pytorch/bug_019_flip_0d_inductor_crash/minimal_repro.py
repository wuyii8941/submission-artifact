"""
Bug 043: torch.compile crashes on flip() applied to 0-dimensional tensors.

Inductor codegen indexes into empty dimension list for scalar tensors.
aot_eager handles it correctly (no-op).
"""
import torch
import torch._dynamo

x = torch.tensor(3.14, device="cuda")

# Eager: works (flip on scalar is no-op)
print(f"Eager: x.flip(0) = {x.flip(0).item()}")

# aot_eager: works
torch._dynamo.reset()
try:
    result = torch.compile(lambda x: x.flip(0), backend="aot_eager")(x)
    print(f"aot_eager: OK ({result.item()})")
except Exception as e:
    print(f"aot_eager: CRASH - {type(e).__name__}")

# inductor: crashes
torch._dynamo.reset()
try:
    result = torch.compile(lambda x: x.flip(0), backend="inductor")(x)
    print(f"inductor: OK ({result.item()})")
except Exception as e:
    print(f"inductor: CRASH - {type(e).__name__}: {str(e)[:80]}")
