"""
Bug 022: torch.compile silently zeroes x.grad for diagonal_scatter + sum backward.

diagonal_scatter(x, src, offset).sum().backward() under torch.compile produces
x.grad = all zeros, while eager correctly produces x.grad = 1 everywhere except
the diagonal (which was overwritten by src).

Root cause (inductor-specific, aot_eager is correct):
  The backward graph for `diagonal_scatter(x, src).sum()` contains:
    expand(tangent_scalar, [N, N])  →  clone  →  diagonal_scatter(clone, zeros)
  Inductor materializes clone(expand(...)) as a (1,1)-element buffer with stride (0,0)
  instead of a proper contiguous (N,N) tensor. The subsequent diagonal_scatter writes
  zeros only to memory offset 0, then the whole buffer is reinterpret_tensor'd to (N,N)
  with stride (0,0) — so x.grad reads the single zeroed element everywhere.

  Bug_007 (slice_scatter) had a similar symptom (zero grad) but different root cause:
  bug_007 was in AOT decomposition (meta_slice_scatter preserving zero strides).
  This bug is in Inductor's materialization of clone(expand(...)).

Trigger conditions:
  - torch.diagonal_scatter + any sum-like reduction (sum, sum(dim), add(scalar)+sum)
  - backward pass
  - Inserting a pointwise op between diagonal_scatter and sum (e.g. relu, mul)
    breaks the fusion and avoids the bug

Tested on: PyTorch 2.13.0.dev20260425+cu126, Tesla T4 (sm_75)
Affects: CUDA, float32 and float64, all shapes and offsets
"""
import torch

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

def test(desc, x_shape, src_shape, offset=0, device='cuda', dtype=torch.float32):
    torch._dynamo.reset()

    x_e = torch.randn(x_shape, device=device, dtype=dtype, requires_grad=True)
    s_e = torch.randn(src_shape, device=device, dtype=dtype, requires_grad=True)
    out_e = torch.diagonal_scatter(x_e, s_e, offset).sum()
    out_e.backward()

    torch._dynamo.reset()
    x_c = x_e.detach().clone().requires_grad_(True)
    s_c = s_e.detach().clone().requires_grad_(True)
    out_c = torch.compile(lambda x, s: torch.diagonal_scatter(x, s, offset).sum())(x_c, s_c)
    out_c.backward()
    torch._dynamo.reset()

    fwd_diff = abs(out_e.item() - out_c.item())
    x_diff = (x_e.grad - x_c.grad).abs().max().item()
    s_diff = (s_e.grad - s_c.grad).abs().max().item()

    status = "BUG" if x_diff > 1e-4 else "OK"
    print(f"  {status} {desc}: x.grad diff={x_diff:.1e}, "
          f"eager x.grad.sum={x_e.grad.sum().item():.0f}, "
          f"compiled x.grad.sum={x_c.grad.sum().item():.0f}")

print("\n=== Basic cases ===")
test("4x4 offset=0", (4, 4), (4,), offset=0)
test("8x8 offset=0", (8, 8), (8,), offset=0)
test("8x8 offset=2", (8, 8), (6,), offset=2)
test("8x8 offset=-3", (8, 8), (5,), offset=-3)

print("\n=== Non-square ===")
test("4x8 offset=0", (4, 8), (4,), offset=0)
test("8x4 offset=0", (8, 4), (4,), offset=0)

print("\n=== Larger sizes ===")
test("64x64", (64, 64), (64,))
test("128x128", (128, 128), (128,))
test("256x256", (256, 256), (256,))

print("\n=== dtype ===")
test("8x8 fp64", (8, 8), (8,), dtype=torch.float64)

print("\n=== Workaround: insert pointwise op to break fusion ===")
torch._dynamo.reset()
x = torch.randn(8, 8, device='cuda', requires_grad=True)
s = torch.randn(8, device='cuda', requires_grad=True)
out = torch.compile(lambda x, s: torch.diagonal_scatter(x, s, 0).relu().sum())(x, s)
out.backward()
print(f"  With relu: x.grad.sum = {x.grad.sum().item():.0f} (expected ~27-30, not 0)")

print("\n=== Confirm: aot_eager is correct (bug is inductor-specific) ===")
torch._dynamo.reset()
x1 = torch.randn(8, 8, device='cuda', requires_grad=True)
s1 = torch.randn(8, device='cuda', requires_grad=True)
out1 = torch.diagonal_scatter(x1, s1, 0).sum()
out1.backward()

torch._dynamo.reset()
x2 = x1.detach().clone().requires_grad_(True)
s2 = s1.detach().clone().requires_grad_(True)
out2 = torch.compile(lambda x, s: torch.diagonal_scatter(x, s, 0).sum(), backend='aot_eager')(x2, s2)
out2.backward()
torch._dynamo.reset()

print(f"  aot_eager x.grad diff = {(x1.grad - x2.grad).abs().max().item():.6e}")
print(f"  -> aot_eager is {'CORRECT' if (x1.grad - x2.grad).abs().max().item() < 1e-6 else 'WRONG'}, bug is inductor-specific")
