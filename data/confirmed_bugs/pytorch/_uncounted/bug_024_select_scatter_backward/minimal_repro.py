"""
Bug A: Inductor produces wrong gradient for select_scatter backward
===================================================================
torch.select_scatter(x, src, dim, idx).sum().backward() gives wrong
gradient for x under torch.compile(backend="inductor").

The compiled gradient for x is all-ones, while eager correctly has
a zero at the scattered index position.

Key characteristics:
  - Inductor-specific (aot_eager backend is correct)
  - Affects both CPU and CUDA Inductor paths
  - Only triggers when select_scatter feeds DIRECTLY into sum
    (any intermediate pointwise op like relu/sigmoid prevents it)
  - Affects ALL shapes (2D, 3D+), ALL dims, ALL dtypes
  - The src gradient is correct; only x gradient is wrong

Root cause: Inductor's fusion of select_scatter + sum in backward
mishandles the stride-0 expanded tangent. The backward formula is:
  grad_x = select_scatter(tangent, zeros_like(src), dim, idx)
When tangent is an expand with stride-0 (from sum backward), Inductor
materializes it as all-ones instead of applying the zero at idx.

Same vulnerability family as:
  - pytorch#<anon> (slice_scatter backward stride-0)
  - pytorch#<anon> (diagonal_scatter backward stride-0)

Severity: SILENT CORRECTNESS — wrong gradient, no error or warning
"""
import torch
import torch._dynamo


def minimal_repro():
    """
    Five-line minimal reproducer.
    """
    torch._dynamo.reset()
    x = torch.randn(4, 8, device='cuda', requires_grad=True)
    src = torch.randn(4, device='cuda', requires_grad=True)

    # Eager
    x_e, s_e = x.detach().clone().requires_grad_(True), src.detach().clone().requires_grad_(True)
    torch.select_scatter(x_e, s_e, dim=1, index=3).sum().backward()

    # Compiled (Inductor)
    torch._dynamo.reset()
    x_c, s_c = x.detach().clone().requires_grad_(True), src.detach().clone().requires_grad_(True)
    torch.compile(lambda x, s: torch.select_scatter(x, s, dim=1, index=3).sum())(x_c, s_c).backward()

    print("x.grad[0]:")
    print(f"  eager:    {x_e.grad[0].tolist()}")
    print(f"  compiled: {x_c.grad[0].tolist()}")
    print(f"  expected: ones with ZERO at index=3")
    print(f"\nx.grad max_diff = {(x_e.grad - x_c.grad).abs().max().item()}")
    print(f"src.grad diff   = {(s_e.grad - s_c.grad).abs().max().item()}")
    assert (x_e.grad - x_c.grad).abs().max().item() > 0.5, "Bug may have been fixed"
    print("\n*** BUG CONFIRMED ***")


def test_inductor_vs_aot_eager():
    """Pinpoints the bug to Inductor (not AOTAutograd)."""
    print("\n=== Backend comparison ===")
    for backend in ['aot_eager', 'inductor']:
        torch._dynamo.reset()
        x_e = torch.randn(4, 8, device='cuda', requires_grad=True)
        s_e = torch.randn(4, device='cuda', requires_grad=True)
        fn = lambda x, s: torch.select_scatter(x, s, 1, 3).sum()
        fn(x_e, s_e).backward()

        torch._dynamo.reset()
        x_c = x_e.detach().clone().requires_grad_(True)
        s_c = s_e.detach().clone().requires_grad_(True)
        torch.compile(fn, backend=backend)(x_c, s_c).backward()

        diff = (x_e.grad - x_c.grad).abs().max().item()
        print(f"  {backend:12s}: x.grad diff = {diff:.6e}  {'BUG' if diff > 0.5 else 'OK'}")


def test_cpu_and_cuda():
    """Affects both CPU and CUDA Inductor."""
    print("\n=== CPU vs CUDA ===")
    for device in ['cpu', 'cuda']:
        torch._dynamo.reset()
        x_e = torch.randn(4, 8, device=device, requires_grad=True)
        s_e = torch.randn(4, device=device, requires_grad=True)
        fn = lambda x, s: torch.select_scatter(x, s, 1, 3).sum()
        fn(x_e, s_e).backward()

        torch._dynamo.reset()
        x_c = x_e.detach().clone().requires_grad_(True)
        s_c = s_e.detach().clone().requires_grad_(True)
        torch.compile(fn, backend='inductor')(x_c, s_c).backward()

        diff = (x_e.grad - x_c.grad).abs().max().item()
        print(f"  {device:5s}: x.grad diff = {diff:.6e}  {'BUG' if diff > 0.5 else 'OK'}")


def test_intermediate_op_prevents_bug():
    """Any pointwise op between select_scatter and sum prevents the bug."""
    print("\n=== Intermediate ops ===")
    patterns = [
        ("select_scatter → sum",         lambda x, s: torch.select_scatter(x, s, 1, 3).sum()),
        ("select_scatter → relu → sum",  lambda x, s: torch.select_scatter(x, s, 1, 3).relu().sum()),
        ("select_scatter → *2+1 → sum",  lambda x, s: (torch.select_scatter(x, s, 1, 3) * 2 + 1).sum()),
        ("select_scatter → sigmoid → sum",lambda x, s: torch.select_scatter(x, s, 1, 3).sigmoid().sum()),
        ("select_scatter → clamp → sum", lambda x, s: torch.select_scatter(x, s, 1, 3).clamp(-1, 1).sum()),
    ]
    for name, fn in patterns:
        torch._dynamo.reset()
        x_e = torch.randn(4, 8, device='cuda', requires_grad=True)
        s_e = torch.randn(4, device='cuda', requires_grad=True)
        fn(x_e, s_e).backward()

        torch._dynamo.reset()
        x_c = x_e.detach().clone().requires_grad_(True)
        s_c = s_e.detach().clone().requires_grad_(True)
        torch.compile(fn)(x_c, s_c).backward()

        diff = (x_e.grad - x_c.grad).abs().max().item()
        print(f"  {name:40s} diff={diff:.6e}  {'BUG' if diff > 0.5 else 'OK'}")


def test_sweep():
    """Sweep shapes, dims, dtypes."""
    print("\n=== Shape/dim/dtype sweep ===")
    configs = [
        ((4, 8), 1, 3, torch.float32),
        ((8, 16), 0, 4, torch.float32),
        ((3, 7, 13), 1, 5, torch.float32),
        ((3, 7, 13), 2, 6, torch.float32),
        ((2, 16, 32), 1, 5, torch.float32),  # 3D (BERT-like mask replace)
        ((4, 8), 1, 3, torch.float64),
        ((4, 8), 1, 3, torch.float16),
    ]
    for shape, dim, idx, dtype in configs:
        torch._dynamo.reset()
        src_shape = list(shape); del src_shape[dim]

        x_e = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
        s_e = torch.randn(src_shape, device='cuda', dtype=dtype, requires_grad=True)
        fn = lambda x, s, d=dim, i=idx: torch.select_scatter(x, s, d, i).sum()
        fn(x_e, s_e).backward()

        torch._dynamo.reset()
        x_c = x_e.detach().clone().requires_grad_(True)
        s_c = s_e.detach().clone().requires_grad_(True)
        torch.compile(fn)(x_c, s_c).backward()

        diff = (x_e.grad - x_c.grad).abs().max().item()
        print(f"  shape={str(shape):18s} dim={dim} idx={idx} dtype={str(dtype):15s}  diff={diff:.1e}  {'BUG' if diff > 0.5 else 'OK'}")


if __name__ == '__main__':
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.get_device_name(0)}\n")

    minimal_repro()
    test_inductor_vs_aot_eager()
    test_cpu_and_cuda()
    test_intermediate_op_prevents_bug()
    test_sweep()
