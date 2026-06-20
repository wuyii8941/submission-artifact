"""
Bug 028: Python complex literal (1j) crashes Inductor code generation
=====================================================================
torch.compile(lambda x: torch.exp(1j * x).abs().sum()).backward() crashes
with AttributeError: 'complex' object has no attribute 'get_name'

Inductor's code generator encounters the Python complex literal 1j
and tries to call IR methods on it.

Key characteristics:
  - Inductor-specific (aot_eager backend is correct)
  - Forward works (with performance warning); only backward crashes
  - All sizes, both CPU and CUDA
  - Workaround: use torch.tensor(1j) instead of literal 1j

Severity: CRASH — prevents compilation of valid code
"""
import torch
import torch._dynamo


def minimal_repro():
    torch._dynamo.reset()
    x = torch.randn(4, 8, device='cuda', requires_grad=True)

    def fn(x):
        return torch.exp(1j * x).abs().sum()

    # Eager
    x_e = x.detach().clone().requires_grad_(True)
    fn(x_e).backward()
    print("Eager: OK")

    # Compiled
    torch._dynamo.reset()
    x_c = x.detach().clone().requires_grad_(True)
    try:
        torch.compile(fn, backend='inductor')(x_c).backward()
        print("Compiled: OK (bug may be fixed)")
    except Exception as e:
        print(f"Compiled: CRASH\n  {type(e).__name__}: {str(e)[:120]}")
        print("\n*** BUG CONFIRMED ***")


def test_backends():
    print("\n=== Backend comparison ===")
    for backend in ['aot_eager', 'inductor']:
        torch._dynamo.reset()
        x = torch.randn(4, 8, device='cuda', requires_grad=True)
        try:
            torch.compile(lambda x: torch.exp(1j * x).abs().sum(),
                         backend=backend)(x).backward()
            print(f"  {backend:12s}: OK")
        except:
            print(f"  {backend:12s}: CRASH")


def test_workaround():
    """Using torch.tensor(1j) instead of literal 1j."""
    print("\n=== Workaround test ===")
    torch._dynamo.reset()
    x = torch.randn(4, 8, device='cuda', requires_grad=True)
    j = torch.tensor(1j, device='cuda')
    def fn(x, j):
        return torch.exp(j * x).abs().sum()
    try:
        torch.compile(fn, backend='inductor')(x, j).backward()
        print("  torch.tensor(1j) workaround: OK")
    except:
        print("  torch.tensor(1j) workaround: CRASH")


if __name__ == '__main__':
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.get_device_name(0)}\n")
    minimal_repro()
    test_backends()
    test_workaround()
