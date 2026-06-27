"""
bug_007 candidate: torch.compile silently zeroes the gradient w.r.t. `x` in
the pattern `slice_scatter(x, y, dim=k, start, end).sum(dim=k)`.

Forward is correct (matches eager to fp32 noise level).
Backward gradient w.r.t. `x` is silently dropped to all zeros.
Backward gradient w.r.t. `y` is correct.

Found by GraphFuzzerV4 primary backward oracle on iter 19/20 of a smoke
test, on PyTorch 2.11.0 / Triton 3.6.0 / Tesla T4. The fact that the
new oracle landed it within 20 iterations suggests this pattern is
relatively common — any model that does in-place buffer updates via
slice_scatter (e.g. KV-cache, ring buffers, autoregressive decoders)
plus a reduction over the same dim will silently train with broken
gradients.

Mathematical expectation:
    out = slice_scatter(x, y, dim=1, start=0, end=6)  # [B, 13, H]
    L   = out.sum(dim=1)                              # [B, H]
    dL/dx[b, i, h] = upstream[b, h]  if i ∈ [6, 13)   (positions kept from x)
                  = 0                if i ∈ [0, 6)    (positions overwritten by y)
    dL/dy[b, j, h] = upstream[b, h]  for j ∈ [0, 6)

Eager produces dL/dx with the correct mask (~140.6 absolute sum on the
seed below). Inductor produces dL/dx ≡ 0 — the gradient is fully dropped.

Pattern characterization (see characterize section below):
  ✗ slice_scatter(x, y, dim=K, ...).sum(dim=K)  → bug FIRES (x.grad wrong)
  ✓ slice_scatter(x, y, dim=K, ...).sum(dim=K') → clean (different dim is fine)

i.e. the bug only fires when reduction is over the SAME dim as the slice.
"""
import torch

print(f"PyTorch: {torch.__version__}")
print(f"Triton:  {__import__('triton').__version__}")
print(f"GPU:     {torch.cuda.get_device_name(0)}")
print()


def program(x, y):
    out = torch.slice_scatter(x, y, dim=1, start=0, end=6)
    return out.sum(dim=1)


print("=" * 70)
print("Test 1: forward — must match (sanity)")
print("=" * 70)
torch.manual_seed(0)
x = torch.randn(4, 13, 33, dtype=torch.float32, device='cuda')
y = torch.randn(4, 6, 33, dtype=torch.float32, device='cuda')
eager_out = program(x.clone(), y.clone())
torch._dynamo.reset()
ind_out = torch.compile(program, backend="inductor")(x.clone(), y.clone())
print(f"  fwd max_diff: {(eager_out - ind_out).abs().max().item():.4e}")

print()
print("=" * 70)
print("Test 2: backward — gradient w.r.t. x is silently zeroed by inductor")
print("=" * 70)
torch.manual_seed(0)
x_e = torch.randn(4, 13, 33, dtype=torch.float32, device='cuda', requires_grad=True)
y_e = torch.randn(4, 6, 33, dtype=torch.float32, device='cuda', requires_grad=True)
out_e = program(x_e, y_e)
g = torch.randn_like(out_e)
out_e.backward(g)

torch.manual_seed(0)
x_i = torch.randn(4, 13, 33, dtype=torch.float32, device='cuda', requires_grad=True)
y_i = torch.randn(4, 6, 33, dtype=torch.float32, device='cuda', requires_grad=True)
torch._dynamo.reset()
out_i = torch.compile(program, backend="inductor")(x_i, y_i)
out_i.backward(g)

print(f"  eager x.grad sum:  {x_e.grad.sum().item():.4f}")
print(f"  ind   x.grad sum:  {x_i.grad.sum().item():.4f}      <-- WRONG (should match eager)")
print(f"  eager y.grad sum:  {y_e.grad.sum().item():.4f}")
print(f"  ind   y.grad sum:  {y_i.grad.sum().item():.4f}")
print()
print(f"  x.grad max_diff:   {(x_e.grad - x_i.grad).abs().max().item():.4e}")
print(f"  y.grad max_diff:   {(y_e.grad - y_i.grad).abs().max().item():.4e}")
print()
# Sanity: print just the [6:13, 0] slice to see the actual values
print("  eager x.grad[0, 6:13, 0]:", x_e.grad[0, 6:13, 0].tolist())
print("  ind   x.grad[0, 6:13, 0]:", x_i.grad[0, 6:13, 0].tolist())
print("  -> eager has nonzero values for positions kept from x;")
print("     inductor produced all zeros — the gradient is being dropped.")

print()
print("=" * 70)
print("Test 3: characterize — bug only fires when sum dim == slice dim")
print("=" * 70)

def go(prog, name):
    torch.manual_seed(0)
    x_e = torch.randn(4, 13, 33, dtype=torch.float32, device='cuda', requires_grad=True)
    y_e = torch.randn(4, 6, 33, dtype=torch.float32, device='cuda', requires_grad=True)
    out_e = prog(x_e, y_e)
    g = torch.randn_like(out_e)
    out_e.backward(g)

    torch.manual_seed(0)
    x_i = torch.randn(4, 13, 33, dtype=torch.float32, device='cuda', requires_grad=True)
    y_i = torch.randn(4, 6, 33, dtype=torch.float32, device='cuda', requires_grad=True)
    torch._dynamo.reset()
    c = torch.compile(prog, backend="inductor")
    out_i = c(x_i, y_i)
    out_i.backward(g)

    fwd = (out_e - out_i).abs().max().item()
    xg = (x_e.grad - x_i.grad).abs().max().item()
    yg = (y_e.grad - y_i.grad).abs().max().item()
    flag = " <-- WRONG" if xg > 1e-3 else ""
    print(f"  {name:55s} fwd={fwd:.1e} x.grad_diff={xg:.4e}{flag}")

go(lambda x, y: torch.slice_scatter(x, y, dim=1, start=0, end=6).sum(dim=1),
   "dim=1 start=0 end=6 .sum(dim=1)")
go(lambda x, y: torch.slice_scatter(x, y, dim=1, start=0, end=6).sum(dim=2),
   "dim=1 start=0 end=6 .sum(dim=2)")
go(lambda x, y: torch.slice_scatter(x, y[:, :3], dim=1, start=5, end=8).sum(dim=1),
   "dim=1 start=5 end=8 .sum(dim=1)")
go(lambda x, y: torch.slice_scatter(x, y, dim=1, start=0, end=6).mean(dim=1),
   "dim=1 start=0 end=6 .mean(dim=1)")
