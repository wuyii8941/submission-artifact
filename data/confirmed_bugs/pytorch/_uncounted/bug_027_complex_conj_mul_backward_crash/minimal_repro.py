"""
Bug 027: Complex conj mul backward crashes Inductor
====================================================
torch.compile((z * z.conj()).real.sum()).backward() crashes with
"view is not supported for conjugate view tensors when converting dtype"

Pattern: any computation involving z * z.conj() (or z.conj() * z)

Key characteristics:
  - Inductor-specific (aot_eager backend is correct)
  - All sizes (size-independent)
  - z.conj().abs() alone works — crash only when conj view enters multiplication
  - z.abs()**2 (equivalent computation) works — workaround available
  - Forward works; only backward crashes

Severity: CRASH — prevents compilation of valid code
"""
import torch
import torch._dynamo


def minimal_repro():
    torch._dynamo.reset()
    x = torch.randn(4, 8, device='cuda', requires_grad=True)
    y = torch.randn(4, 8, device='cuda', requires_grad=True)

    def fn(x, y):
        z = torch.complex(x, y)
        return (z * z.conj()).real.sum()

    # Eager
    x_e = x.detach().clone().requires_grad_(True)
    y_e = y.detach().clone().requires_grad_(True)
    fn(x_e, y_e).backward()
    print("Eager: OK")

    # Compiled
    torch._dynamo.reset()
    x_c = x.detach().clone().requires_grad_(True)
    y_c = y.detach().clone().requires_grad_(True)
    try:
        torch.compile(fn, backend='inductor')(x_c, y_c).backward()
        print("Compiled: OK (bug may be fixed)")
    except Exception as e:
        print(f"Compiled: CRASH\n  {type(e).__name__}: {str(e)[:120]}")
        print("\n*** BUG CONFIRMED ***")


def test_variants():
    print("\n=== Variant patterns ===")
    patterns = [
        ("z*z.conj().real.sum()",   lambda x, y: (torch.complex(x, y) * torch.complex(x, y).conj()).real.sum()),
        ("z.conj()*z.real.sum()",   lambda x, y: (torch.complex(x, y).conj() * torch.complex(x, y)).real.sum()),
        ("(z*z.conj()).abs.sum()",   lambda x, y: (torch.complex(x, y) * torch.complex(x, y).conj()).abs().sum()),
        ("z.conj().abs().sum()",     lambda x, y: torch.complex(x, y).conj().abs().sum()),
        ("z.abs()**2.sum()",         lambda x, y: (torch.complex(x, y).abs() ** 2).sum()),
    ]
    for name, fn in patterns:
        torch._dynamo.reset()
        x = torch.randn(4, 33, device='cuda', requires_grad=True)
        y = torch.randn(4, 33, device='cuda', requires_grad=True)
        try:
            xe, ye = x.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(True)
            fn(xe, ye).backward()
            torch._dynamo.reset()
            xc, yc = x.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(True)
            torch.compile(fn, backend='inductor')(xc, yc).backward()
            print(f"  {name:35s}: OK")
        except:
            print(f"  {name:35s}: CRASH")


def test_backends():
    print("\n=== Backend comparison ===")
    for backend in ['aot_eager', 'inductor']:
        torch._dynamo.reset()
        x = torch.randn(4, 8, device='cuda', requires_grad=True)
        y = torch.randn(4, 8, device='cuda', requires_grad=True)
        try:
            fn = lambda x, y: (torch.complex(x, y) * torch.complex(x, y).conj()).real.sum()
            torch.compile(fn, backend=backend)(x, y).backward()
            print(f"  {backend:12s}: OK")
        except:
            print(f"  {backend:12s}: CRASH")


if __name__ == '__main__':
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.get_device_name(0)}\n")
    minimal_repro()
    test_variants()
    test_backends()
