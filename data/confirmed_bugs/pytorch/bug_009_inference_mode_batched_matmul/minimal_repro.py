"""
Bug: torch.compile crashes on batched matmul (3D+ input) inside torch.inference_mode().

2D matmul inside inference_mode works fine.
Batched matmul without inference_mode works fine.
Only the combination of batched matmul + inference_mode triggers the crash.

Error: BackendCompilerFailed: RuntimeError: <weakref to UntypedStorage>
  (sometimes: INTERNAL ASSERT FAILED at PythonFallbackKernel.cpp:137)

PyTorch: 2.13.0.dev20260425+cu126
"""
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

def test(desc, fn, *args):
    torch._dynamo.reset()
    try:
        r = torch.compile(fn)(*args)
        print(f"  {desc}: OK")
    except Exception as e:
        print(f"  {desc}: CRASH — {type(e).__name__}")

x2d = torch.randn(4, 8, device='cuda')
x3d = torch.randn(2, 4, 8, device='cuda')
w = torch.randn(8, 8, device='cuda')

print("--- 2D matmul ---")
test("no inference_mode", lambda x, w: (x @ w).sum(), x2d, w)
test("inference_mode", lambda x, w: torch.inference_mode()(lambda: (x @ w).sum())(), x2d, w)

print("--- 3D matmul (batched) ---")
test("no inference_mode", lambda x, w: (x @ w).sum(), x3d, w)

# This is the crash:
print("--- 3D matmul + inference_mode (BUG) ---")
torch._dynamo.reset()
try:
    def program(x, w):
        with torch.inference_mode():
            return (x @ w).sum()
    r = torch.compile(program)(x3d, w)
    print(f"  OK: {r.item():.2f}")
except Exception as e:
    print(f"  CRASH: {type(e).__name__}: {e}")
