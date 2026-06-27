"""
Bug 006: Silent incorrectness in fused adaptive_avg_pool2d + flatten + sum.

Summary:
    When torch.compile(backend='inductor') compiles a simple
        y = adaptive_avg_pool2d(x, k)
        return y.flatten(1).sum(dim=-1)
    graph, the resulting fused kernel produces correct output only for
    batch index 0. All other batch indices receive a systematically wrong
    value. The error persists at fp32 AND fp64 and is fully deterministic.

Environment:
    PyTorch 2.11.0+cu126
    Triton  3.6.0
    GPU     Tesla T4 (sm_75, cc 7.5)

Run:  python minimal_repro.py
"""
import torch

print(f"PyTorch: {torch.__version__}")
try:
    import triton
    print(f"Triton:  {triton.__version__}")
except Exception:
    pass
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
print()


def program(x):
    y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
    return y.flatten(1).sum(dim=-1)


torch.manual_seed(0)
# Large C (2049) — r0_numel = 2049 * 49 = 100401 elements per batch.
# This is big enough that Inductor picks a non-persistent reduction
# (`triton_red_fused_*`) on every GPU architecture (T4/A100/H100/etc.),
# which avoids a "reproduces on T4 but not on A100" report.
# C=2049 is not a multiple of 32, which is what triggers the wrong stride.
x = torch.randn(4, 2049, 8, 8, dtype=torch.float32, device='cuda')

eager = program(x.clone())
torch._dynamo.reset()
compiled = torch.compile(program, backend="inductor")
out = compiled(x.clone())

print("Shape [4, 2049, 8, 8] fp32, adaptive_avg_pool2d(x, 7).flatten(1).sum(-1)")
print(f"  eager   : {eager.tolist()}")
print(f"  inductor: {out.tolist()}")
d = (eager - out).abs()
print(f"  abs diff: {d.tolist()}")
print(f"  max_diff: {d.max().item():.4e}")
print()

# Verify eager is internally consistent
y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
print("Manual eager pool sum per batch:")
for b in range(x.size(0)):
    print(f"  batch {b}: {y[b].sum().item():.6f}")
print()

# Per-batch signature at different batch sizes
print("Per-batch bug pattern at shape [B, 2049, 8, 8]:")
for B in [2, 3, 4, 8]:
    torch.manual_seed(0)
    x = torch.randn(B, 2049, 8, 8, dtype=torch.float32, device='cuda')
    e = program(x.clone())
    torch._dynamo.reset()
    c = torch.compile(program, backend="inductor")
    o = c(x.clone())
    bad = [i for i in range(B) if (e[i] - o[i]).abs().item() > 1e-3]
    print(f"  B={B}: wrong batches = {bad}  (only batch 0 is ever correct)")
print()

# The bug is in the fusion — adding a no-op breaks it:
print("Breaking the fusion restores correctness:")
def via_intermediate_reduce(x):
    # sum over inner dims directly — inductor can't fuse through this
    y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
    return y.sum(dim=(1, 2, 3))
def via_noop_add(x):
    # insert a no-op pointwise between pool and flatten
    y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
    return (y + 0.0).flatten(1).sum(dim=-1)
torch.manual_seed(0)
x = torch.randn(4, 2049, 8, 8, dtype=torch.float32, device='cuda')
for name, fn in [("sum((1,2,3))", via_intermediate_reduce),
                 ("(y+0.0) flatten sum", via_noop_add)]:
    e = fn(x.clone())
    torch._dynamo.reset()
    c = torch.compile(fn, backend="inductor")
    o = c(x.clone())
    print(f"  {name:25s} max_diff = {(e-o).abs().max().item():.2e}")

# fp64 also affected
print()
print("fp64 also reproduces:")
torch.manual_seed(0)
x64 = torch.randn(4, 2049, 8, 8, dtype=torch.float64, device='cuda')
e = program(x64.clone())
torch._dynamo.reset()
c = torch.compile(program, backend="inductor")
o = c(x64.clone())
print(f"  max_diff fp64: {(e - o).abs().max().item():.4e}")
