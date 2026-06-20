"""
Bug 031: torch.compile crashes on cumprod backward for long sequences.

Reproducer: RetNet-style learnable retention mechanism.
RetNet (Retentive Network) uses cumulative product of learned retention rates
along the sequence dimension. When seq_len >= 8193, torch.compile crashes
with NotImplementedError in TritonSplitDimKernel.

cumsum backward works fine at the same dimension — only cumprod is affected.
"""
import torch
import torch._dynamo
import torch._inductor
torch._inductor.config.force_disable_caches = True

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# RetNet-style retention: learned decay rates along sequence
seq_len = 8193
batch, d_model = 2, 64

x = torch.randn(batch, seq_len, d_model, device='cuda', requires_grad=True)
gamma = torch.nn.Parameter(torch.sigmoid(torch.randn(d_model, device='cuda')))

def retention_forward(x, gamma):
    """Compute RetNet-style retention: weight each position by cumulative decay."""
    decay = gamma.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, d_model)
    retention = torch.cumprod(decay, dim=1)  # shape: (B, T, D)
    return (x * retention).sum()

# ---- Eager: works ----
x_e = x.clone().detach().requires_grad_(True)
g_e = gamma.clone().detach().requires_grad_(True)
y_e = retention_forward(x_e, g_e)
y_e.backward()
print(f"Eager OK: x.grad norm={x_e.grad.norm():.4f}, gamma.grad norm={g_e.grad.norm():.4f}")

# ---- Compiled: crashes ----
torch._dynamo.reset()
x_c = x.clone().detach().requires_grad_(True)
g_c = gamma.clone().detach().requires_grad_(True)
compiled = torch.compile(retention_forward, backend='inductor')
y_c = compiled(x_c, g_c)
try:
    y_c.backward()
    grad_diff = (x_e.grad - x_c.grad).abs().max().item()
    print(f"Compiled OK: grad_diff={grad_diff:.6f}")
except Exception as e:
    print(f"Compiled CRASH: {type(e).__name__}: {e}")
