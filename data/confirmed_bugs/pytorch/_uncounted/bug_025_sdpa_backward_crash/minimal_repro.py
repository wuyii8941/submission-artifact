"""
Bug 025: SDPA backward crashes Inductor when head_dim is not a multiple of 8
=============================================================================
torch.compile(F.scaled_dot_product_attention(q,k,v).sum()).backward() crashes
with IndexError in sdpa_constraint when head_dim % 8 != 0.

The crash is in Inductor's sdpa_constraint (lowering.py): the stride-0
tangent from sum().backward() gets unwrap_view()'d to a 0-d scalar, and
get_size()[-1] fails on the empty size list.

Key characteristics:
  - Inductor-specific (aot_eager backend is correct)
  - Only when SDPA output feeds DIRECTLY into sum (stride-0 tangent)
  - Only when head_dim % 8 != 0 (alignment boundary)
  - Any intermediate op (matmul, norm, partial reduction) prevents it
  - head_dim=1 is special-cased and does not crash

Severity: CRASH — prevents compilation of valid code
"""
import torch
import torch._dynamo
import torch.nn.functional as F


def minimal_repro():
    """Five-line minimal reproducer."""
    torch._dynamo.reset()
    q = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)
    k = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)
    v = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)

    # Eager — works fine
    q_e = q.detach().clone().requires_grad_(True)
    k_e = k.detach().clone().requires_grad_(True)
    v_e = v.detach().clone().requires_grad_(True)
    F.scaled_dot_product_attention(q_e, k_e, v_e).sum().backward()
    print("Eager backward: OK")

    # Compiled (Inductor) — crashes
    torch._dynamo.reset()
    q_c = q.detach().clone().requires_grad_(True)
    k_c = k.detach().clone().requires_grad_(True)
    v_c = v.detach().clone().requires_grad_(True)
    try:
        torch.compile(
            lambda q, k, v: F.scaled_dot_product_attention(q, k, v).sum()
        )(q_c, k_c, v_c).backward()
        print("Compiled backward: OK (bug may be fixed)")
    except Exception as e:
        print(f"Compiled backward: CRASH\n  {type(e).__name__}: {str(e)[:120]}")
        print("\n*** BUG CONFIRMED ***")


def test_backend_comparison():
    """Pinpoints the bug to Inductor."""
    print("\n=== Backend comparison ===")
    for backend in ['aot_eager', 'inductor']:
        torch._dynamo.reset()
        q = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)
        k = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)
        v = torch.randn(2, 4, 8, 33, device='cuda', requires_grad=True)
        try:
            torch.compile(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v).sum(),
                backend=backend
            )(q, k, v).backward()
            print(f"  {backend:12s}: OK")
        except Exception as e:
            print(f"  {backend:12s}: CRASH ({type(e).__name__})")


def test_alignment_sweep():
    """Crashes for head_dim % 8 != 0, OK for multiples of 8."""
    print("\n=== Head dim alignment sweep ===")
    for hd in [1, 2, 4, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128, 129, 256]:
        torch._dynamo.reset()
        q = torch.randn(1, 1, 4, hd, device='cuda', requires_grad=True)
        k = torch.randn(1, 1, 4, hd, device='cuda', requires_grad=True)
        v = torch.randn(1, 1, 4, hd, device='cuda', requires_grad=True)
        try:
            torch.compile(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v).sum(),
                backend='inductor'
            )(q, k, v).backward()
            print(f"  head_dim={hd:4d} (mod8={hd%8}): OK")
        except:
            print(f"  head_dim={hd:4d} (mod8={hd%8}): CRASH")


def test_intermediate_ops():
    """Intermediate ops prevent the crash by producing non-stride-0 tangent."""
    print("\n=== Intermediate ops (head_dim=33) ===")
    hd = 33
    patterns = [
        ("sdpa.sum()",            lambda q, k, v: F.scaled_dot_product_attention(q, k, v).sum()),
        ("sdpa.mean()",           lambda q, k, v: F.scaled_dot_product_attention(q, k, v).mean()),
        ("sdpa.norm()",           lambda q, k, v: F.scaled_dot_product_attention(q, k, v).norm()),
        ("sdpa.sum(dim=-1).sum()",lambda q, k, v: F.scaled_dot_product_attention(q, k, v).sum(dim=-1).sum()),
        ("(sdpa @ W).sum()",      None),  # handled separately
    ]
    for name, fn in patterns:
        if fn is None:
            continue
        torch._dynamo.reset()
        q = torch.randn(2, 4, 8, hd, device='cuda', requires_grad=True)
        k = torch.randn(2, 4, 8, hd, device='cuda', requires_grad=True)
        v = torch.randn(2, 4, 8, hd, device='cuda', requires_grad=True)
        try:
            torch.compile(fn, backend='inductor')(q, k, v).backward()
            print(f"  {name:35s}: OK")
        except Exception as e:
            print(f"  {name:35s}: CRASH")

    # matmul after
    torch._dynamo.reset()
    q = torch.randn(2, 4, 8, hd, device='cuda', requires_grad=True)
    k = torch.randn(2, 4, 8, hd, device='cuda', requires_grad=True)
    v = torch.randn(2, 4, 8, hd, device='cuda', requires_grad=True)
    w = torch.randn(hd, 16, device='cuda', requires_grad=True)
    try:
        torch.compile(
            lambda q, k, v, w: (F.scaled_dot_product_attention(q, k, v) @ w).sum(),
            backend='inductor'
        )(q, k, v, w).backward()
        print(f"  {'(sdpa @ W).sum()':35s}: OK")
    except:
        print(f"  {'(sdpa @ W).sum()':35s}: CRASH")


if __name__ == '__main__':
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.get_device_name(0)}\n")

    minimal_repro()
    test_backend_comparison()
    test_alignment_sweep()
    test_intermediate_ops()
