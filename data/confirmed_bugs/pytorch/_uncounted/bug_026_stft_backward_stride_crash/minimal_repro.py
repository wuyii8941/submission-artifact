"""
Bug 026: STFT backward crashes Inductor with stride assertion error
===================================================================
torch.compile(torch.stft(...).abs().sum()).backward() crashes with
AssertionError on stride mismatch: the meta kernel predicts C-contiguous
output but the actual cuFFT backward kernel produces Fortran-contiguous.

Key characteristics:
  - Inductor-specific (aot_eager backend is correct)
  - CUDA only (CPU Inductor is correct)
  - All n_fft sizes (64, 128, 256, 512, 1024) affected
  - Triggered by .sum() or .sum(dim=-1).sum(), NOT by .mean() or .norm()
  - Non-stride-0 external tangent (manual .backward(grad)) does NOT crash

Severity: CRASH — prevents compilation of valid code
"""
import torch
import torch._dynamo


def minimal_repro():
    """Minimal reproducer."""
    torch._dynamo.reset()
    x = torch.randn(2, 1024, device='cuda', requires_grad=True)
    window = torch.hann_window(256, device='cuda')

    # Eager — works
    x_e = x.detach().clone().requires_grad_(True)
    torch.stft(x_e, 256, hop_length=64, win_length=256,
               window=window, return_complex=True).abs().sum().backward()
    print(f"Eager backward: OK, grad shape={x_e.grad.shape}")

    # Compiled — crashes
    torch._dynamo.reset()
    x_c = x.detach().clone().requires_grad_(True)
    try:
        torch.compile(
            lambda x, w: torch.stft(x, 256, hop_length=64, win_length=256,
                                    window=w, return_complex=True).abs().sum()
        )(x_c, window).backward()
        print("Compiled backward: OK (bug may be fixed)")
    except AssertionError as e:
        print(f"Compiled backward: CRASH\n  AssertionError: {str(e)[:120]}")
        print("\n*** BUG CONFIRMED ***")


def test_backend_comparison():
    print("\n=== Backend comparison ===")
    for backend in ['aot_eager', 'inductor']:
        torch._dynamo.reset()
        x = torch.randn(2, 1024, device='cuda', requires_grad=True)
        window = torch.hann_window(256, device='cuda')
        try:
            torch.compile(
                lambda x, w: torch.stft(x, 256, hop_length=64, win_length=256,
                                        window=w, return_complex=True).abs().sum(),
                backend=backend
            )(x, window).backward()
            print(f"  {backend:12s}: OK")
        except Exception as e:
            print(f"  {backend:12s}: CRASH ({type(e).__name__})")


def test_nfft_sweep():
    print("\n=== n_fft sweep ===")
    for N in [64, 128, 256, 512, 1024]:
        torch._dynamo.reset()
        sig_len = max(1024, N * 4)
        x = torch.randn(2, sig_len, device='cuda', requires_grad=True)
        window = torch.hann_window(N, device='cuda')
        try:
            torch.compile(
                lambda x, w, N=N: torch.stft(x, N, hop_length=N//4, win_length=N,
                                             window=w, return_complex=True).abs().sum(),
                backend='inductor'
            )(x, window).backward()
            print(f"  n_fft={N:5d}: OK")
        except Exception as e:
            print(f"  n_fft={N:5d}: CRASH")


def test_reduction_patterns():
    print("\n=== Reduction patterns ===")
    patterns = {
        'sum':          lambda s: s.sum(),
        'mean':         lambda s: s.mean(),
        'norm':         lambda s: s.norm(),
        'sum_dim+sum':  lambda s: s.sum(dim=-1).sum(),
    }
    for name, reduce_fn in patterns.items():
        torch._dynamo.reset()
        x = torch.randn(2, 1024, device='cuda', requires_grad=True)
        window = torch.hann_window(256, device='cuda')
        def fn(x, w, reduce_fn=reduce_fn):
            s = torch.stft(x, 256, hop_length=64, win_length=256,
                           window=w, return_complex=True).abs()
            return reduce_fn(s)
        try:
            torch.compile(fn, backend='inductor')(x, window).backward()
            print(f"  {name:20s}: OK")
        except:
            print(f"  {name:20s}: CRASH")


def test_cpu_vs_cuda():
    print("\n=== CPU vs CUDA ===")
    for device in ['cpu', 'cuda']:
        torch._dynamo.reset()
        x = torch.randn(2, 1024, device=device, requires_grad=True)
        window = torch.hann_window(256, device=device)
        try:
            torch.compile(
                lambda x, w: torch.stft(x, 256, hop_length=64, win_length=256,
                                        window=w, return_complex=True).abs().sum(),
                backend='inductor'
            )(x, window).backward()
            print(f"  {device:5s}: OK")
        except:
            print(f"  {device:5s}: CRASH")


if __name__ == '__main__':
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.get_device_name(0)}\n")

    minimal_repro()
    test_backend_comparison()
    test_nfft_sweep()
    test_reduction_patterns()
    test_cpu_vs_cuda()
