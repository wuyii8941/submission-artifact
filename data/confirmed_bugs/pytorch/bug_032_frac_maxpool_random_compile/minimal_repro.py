"""
Bug 032: fractional_max_pool2d returns wrong results under torch.compile
when _random_samples is not explicitly provided.

torch.compile(backend='inductor') produces DIFFERENT pooling indices than
eager mode for fractional_max_pool2d when the random samples are generated
internally (default behavior). The compiled version generates different
random pooling regions, causing it to select wrong elements.

With explicit _random_samples: results match (diff=0.0)
Without _random_samples: results differ (diff=2.03, wrong indices)

This is different from bug_006 (#179931) which was about stride alignment
in flatten+sum fusion. Bug_006 is FIXED. This bug is about random state
handling in the compiled graph.

Root cause: Dynamo/Inductor does not properly trace the internal random
sample generation inside fractional_max_pool2d, causing the compiled
version to use different pooling regions than eager.

Tested on: PyTorch 2.13.0.dev20260501+cu126, Tesla T4
"""

import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Setup
torch.manual_seed(42)
x = torch.randn(1, 3, 32, 32, device='cuda')

def fn(x):
    return torch.nn.functional.fractional_max_pool2d(x, 3, output_size=(8, 8))[0]

# Eager
eager = fn(x)

# Compiled
torch._dynamo.reset()
compiled = torch.compile(fn, backend='inductor')(x)

diff = (eager.float() - compiled.float()).abs().max().item()
print(f"\nWithout _random_samples:")
print(f"  max_diff = {diff:.4f}")
print(f"  BUG: {'YES' if diff > 0.01 else 'NO'}")

# Verify: with explicit _random_samples, it works
torch._dynamo.reset()
samples = torch.rand(1, 3, 2, device='cuda')

def fn_explicit(x):
    return torch.nn.functional.fractional_max_pool2d(
        x, 3, output_size=(8, 8), _random_samples=samples
    )[0]

eager2 = fn_explicit(x)
compiled2 = torch.compile(fn_explicit, backend='inductor')(x)
diff2 = (eager2.float() - compiled2.float()).abs().max().item()
print(f"\nWith explicit _random_samples:")
print(f"  max_diff = {diff2:.6f}")
print(f"  BUG: {'YES' if diff2 > 0.01 else 'NO'}")
