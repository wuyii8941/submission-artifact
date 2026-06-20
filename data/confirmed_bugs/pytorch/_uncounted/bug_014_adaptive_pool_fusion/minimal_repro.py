"""
Bug 014: torch.compile produces wrong results for adaptive_avg_pool2d + flatten + sum
when the channel dimension is not a power of 2.

Inductor fuses adaptive_avg_pool2d with downstream flatten+sum into a single kernel,
but the fused kernel computes incorrect results when C is not a power of 2
(e.g., C=33, 50, 65, 100, 129, 277).

The pool alone is correct. The fusion introduces the error.

Reproduces on: PyTorch 2.11.0+cu126, 2.12.0.dev20260410+cu126 (nightly)
Device: CUDA only (CPU is correct)
"""
import torch


def test(desc, x):
    def f(x):
        return torch.nn.functional.adaptive_avg_pool2d(x, 7).flatten(1).sum(dim=-1)

    eager = f(x)

    torch._dynamo.reset()
    compiled = torch.compile(f, backend='inductor')(x)
    torch._dynamo.reset()

    diff = (eager - compiled).abs().max().item()
    status = "\u2705" if diff < 1e-6 else "\u274c"
    print(f"{status} {desc}")
    print(f"   eager   = {eager.tolist()}")
    print(f"   inductor= {compiled.tolist()}")
    print(f"   diff = {diff}")
    print()


print("=== Channel sweep (H=8, output_size=7) ===")
torch.manual_seed(42)
for C in [32, 33, 64, 65, 128, 129]:
    test(f"C={C}", torch.randn(2, C, 8, 8, dtype=torch.float64, device='cuda'))

print("=== Pool alone is correct ===")
torch.manual_seed(42)
x = torch.randn(2, 33, 8, 8, dtype=torch.float64, device='cuda')

def pool_only(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, 7)

eager_pool = pool_only(x)
torch._dynamo.reset()
compiled_pool = torch.compile(pool_only, backend='inductor')(x)
torch._dynamo.reset()
print(f"Pool only diff: {(eager_pool - compiled_pool).abs().max().item()}")
print()

print("=== Fused pool+flatten+sum is wrong ===")
def fused(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, 7).flatten(1).sum(dim=-1)

eager_fused = fused(x)
torch._dynamo.reset()
compiled_fused = torch.compile(fused, backend='inductor')(x)
torch._dynamo.reset()
print(f"Fused diff: {(eager_fused - compiled_fused).abs().max().item()}")
