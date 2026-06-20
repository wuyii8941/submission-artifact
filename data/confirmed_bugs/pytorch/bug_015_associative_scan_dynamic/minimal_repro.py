"""
Bug: torch._higher_order_ops.associative_scan crashes under torch.compile(dynamic=True)
for inputs with ndim >= 2.

Static compilation works fine. 1D inputs with dynamic=True also work.
Only ndim >= 2 + dynamic=True triggers the crash.

Error: LoweringException: RuntimeError: Unable to generate code for
associative_scan op, because there are lifted arguments

PyTorch: 2.13.0.dev20260425+cu126
"""
import torch

print(f"PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name()}")

def combine_fn(a, b):
    return a + b

# --- 1D: works with both static and dynamic ---
x1d = torch.randn(8, device='cuda')

torch._dynamo.reset()
r = torch.compile(lambda x: torch._higher_order_ops.associative_scan(combine_fn, x, dim=0), dynamic=True)(x1d)
print(f"1D dynamic=True: OK, shape={r.shape}")

# --- 2D static: works ---
x2d = torch.randn(4, 8, device='cuda')

torch._dynamo.reset()
r = torch.compile(lambda x: torch._higher_order_ops.associative_scan(combine_fn, x, dim=0))(x2d)
print(f"2D dynamic=False: OK, shape={r.shape}")

# --- 2D dynamic: CRASHES ---
torch._dynamo.reset()
try:
    r = torch.compile(lambda x: torch._higher_order_ops.associative_scan(combine_fn, x, dim=0), dynamic=True)(x2d)
    print(f"2D dynamic=True: OK, shape={r.shape}")
except Exception as e:
    print(f"2D dynamic=True: CRASH — {type(e).__name__}: {e}")

# --- 3D dynamic: also crashes ---
x3d = torch.randn(2, 4, 8, device='cuda')
torch._dynamo.reset()
try:
    r = torch.compile(lambda x: torch._higher_order_ops.associative_scan(combine_fn, x, dim=1), dynamic=True)(x3d)
    print(f"3D dynamic=True dim=1: OK, shape={r.shape}")
except Exception as e:
    print(f"3D dynamic=True dim=1: CRASH — {type(e).__name__}")
