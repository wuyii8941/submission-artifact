"""
Bug: torch.compile + torch._higher_order_ops.map generates invalid C++ wrapper code.

torch.compile(fullgraph=True) with Inductor crashes when compiling a function
using torch._higher_order_ops.map. The generated C++ wrapper code is syntactically
invalid and fails g++ compilation.

- Eager mode: works fine
- aot_eager: works fine
- Inductor: CppCompileError

Filed: <upstream issue, anonymized>
Tested on: PyTorch 2.13.0.dev20260501+cu126, Tesla T4 (sm_75)
"""
import torch
import torch.nn.functional as F

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

torch.manual_seed(42)
xs = torch.randn(8, 64, device='cuda')

def body_fn(x):
    return F.gelu(x)

def fn(xs):
    return torch._higher_order_ops.map(body_fn, xs).sum()

# === Eager: OK ===
print("\n=== Eager ===")
out = fn(xs)
print(f"OK: result={out.item():.6f}")

# === aot_eager: OK ===
print("\n=== aot_eager ===")
torch._dynamo.reset()
fn_aot = torch.compile(fn, backend='aot_eager', fullgraph=True)
try:
    out = fn_aot(xs)
    print(f"OK: result={out.item():.6f}")
except Exception as e:
    print(f"CRASH — {type(e).__name__}: {str(e)[:200]}")

# === Inductor: CRASH ===
print("\n=== Inductor ===")
torch._dynamo.reset()
fn_c = torch.compile(fn, fullgraph=True)
try:
    out = fn_c(xs)
    print(f"OK (BUG FIXED): result={out.item():.6f}")
except Exception as e:
    print(f"CRASH — {type(e).__name__}: {str(e)[:200]}")
