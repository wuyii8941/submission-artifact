"""
Bug 030: torch.compile + torch.while_loop backward crashes with stride mismatch.

torch.compile(fullgraph=True) with Inductor crashes when running backward through
torch.while_loop. The carried input gets stride (0, 0) while the body output has
stride (64, 1), causing UncapturedHigherOrderOpError.

- Eager mode: works fine
- aot_eager backend: works fine
- Inductor forward-only (no grad): works fine
- Inductor with backward: CRASHES

Error: UncapturedHigherOrderOpError: Expected carried_inputs and body_output to have
same metadata but found: pair[1] differ in 'stride: (0, 0) vs (64, 1)'

Tested on: PyTorch 2.13.0.dev20260501+cu126, Tesla T4 (sm_75)
"""
import torch

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

torch.manual_seed(42)

def cond_fn(i, x):
    return i < 5

def body_fn(i, x):
    return i + 1, x * 0.9 + 0.1

def fn(x):
    _, result = torch.while_loop(
        cond_fn, body_fn,
        (torch.tensor(0, device='cuda'), x)
    )
    return result.sum()

# === Eager: OK ===
print("\n=== Eager ===")
x = torch.randn(4, 64, device='cuda', requires_grad=True)
loss = fn(x)
loss.backward()
print(f"OK: grad_max={x.grad.abs().max().item():.6f}")
x.grad = None

# === aot_eager: OK ===
print("\n=== aot_eager ===")
torch._dynamo.reset()
fn_aot = torch.compile(fn, backend='aot_eager', fullgraph=True)
loss = fn_aot(x)
loss.backward()
print(f"OK: grad_max={x.grad.abs().max().item():.6f}")
x.grad = None

# === Inductor forward-only: OK ===
print("\n=== Inductor forward-only (no grad) ===")
torch._dynamo.reset()
x_nograd = torch.randn(4, 64, device='cuda')
fn_fwd = torch.compile(fn, fullgraph=True)
try:
    out = fn_fwd(x_nograd)
    print(f"OK: result={out.item():.6f}")
except Exception as e:
    print(f"CRASH — {type(e).__name__}: {str(e)[:200]}")

# === Inductor with backward: CRASH ===
print("\n=== Inductor with backward ===")
torch._dynamo.reset()
fn_c = torch.compile(fn, fullgraph=True)
try:
    loss = fn_c(x)
    loss.backward()
    print(f"OK (BUG FIXED): grad_max={x.grad.abs().max().item():.6f}")
except Exception as e:
    print(f"CRASH — {type(e).__name__}: {str(e)[:200]}")
