"""
Bug: Inductor sink_cat_after_pointwise pass crashes on torch.cat(axis=)

torch.cat supports both `dim` and `axis` as keyword arguments (they are
aliases in NumPy-compatible API). However, the Inductor pre-grad pass
`sink_cat_after_pointwise` defines a helper `cat_args(tensors, dim=0)`
that only accepts `dim`. When the FX graph captures `torch.cat([...], axis=N)`,
the node's kwargs contain {'axis': N}, which is passed via **node.kwargs
to cat_args() and crashes.

Trigger condition:
  torch.cat([...], axis=N) where the cat output is consumed by a pointwise
  unary op (relu or tanh) — this triggers the sink_cat_after_pointwise pass.

Error:
  BackendCompilerFailed: backend='inductor' raised:
  TypeError: sink_cat_after_pointwise.<locals>.cat_args() got an unexpected
  keyword argument 'axis'

Location: torch/_inductor/fx_passes/pre_grad.py:729-732
  def cat_args(tensors, dim=0):  # <-- doesn't accept 'axis'
      return tensors, dim
  tensors, dim = cat_args(*node.args, **node.kwargs)  # <-- crash

Fix: Change cat_args to also accept axis:
  def cat_args(tensors, dim=0, axis=None):
      if axis is not None:
          dim = axis
      return tensors, dim

Related: GitHub issue #171979 (opened Jan 8, 2026, still unfixed as of May 22, 2026)
"""
import torch
import torch._dynamo

print(f"PyTorch: {torch.__version__}")

def fn(x):
    a = x.sin()
    b = x.cos()
    # axis= keyword (NumPy-compatible alias for dim=)
    c = torch.cat([a, b], axis=1)
    # Pointwise op after cat triggers sink_cat_after_pointwise pass
    return c.relu()

x = torch.randn(4, 8, device="cuda")

# Eager: works correctly
out_eager = fn(x)
print(f"Eager: shape={out_eager.shape}, sum={out_eager.sum().item():.4f}")

# Compiled: crashes
torch._dynamo.reset()
try:
    compiled_fn = torch.compile(fn)
    out_compiled = compiled_fn(x.clone())
    print(f"Compiled: shape={out_compiled.shape} (BUG NOT TRIGGERED)")
except Exception as e:
    print(f"Compiled CRASH: {e}")
    print("\nBug confirmed: sink_cat_after_pointwise doesn't handle axis= keyword")

# Workaround: use dim= instead of axis=
print("\nWorkaround: use dim= instead of axis=")
def fn_fixed(x):
    a = x.sin()
    b = x.cos()
    c = torch.cat([a, b], dim=1)
    return c.relu()

torch._dynamo.reset()
compiled_fixed = torch.compile(fn_fixed)
out_fixed = compiled_fixed(x.clone())
print(f"Fixed: shape={out_fixed.shape}, matches_eager={(out_eager - out_fixed).abs().max().item() < 1e-6}")
