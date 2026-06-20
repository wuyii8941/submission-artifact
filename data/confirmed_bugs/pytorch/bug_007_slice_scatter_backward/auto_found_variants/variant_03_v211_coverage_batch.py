"""bug_007 family — batch of 6 confirmed variants auto-found by V4 coverage run
(graph_exp_v211_coverage, 2026-04-21, 680+ iter with coverage-guided fuzzing).

All 6 variants confirmed on PyTorch 2.11.0+cu126 (nightly 2026-04-10).
All reproduce on default torch.compile (no special config needed).
All reproduce on CPU and GPU, fp32 and fp64.
Diff is always exactly 1.0 (binary on/off error).

Variants found (from FIRE entries in output_v211_coverage.log):
  1. bwd_slice_scatter      fp64  b3s33h255   combo_kernels           line 81   NOVEL
  2. d10_slice_scatter_dtype fp64  b3s13h64    combo_pointwise_only    line 985
  3. d10_slice_scatter_dtype fp64  b1s255h65   freezing+dyn            line 2698 NOVEL
  4. bwd_slice_scatter       fp32  b15s129h33  combo_pointwise_only    line 7617 NOVEL
  5. d5a_slice_scatter_bwd   fp32  b7s257h255  emulate_precision+dyn   line 8017 NOVEL
  6. select_scatter also affected (verified, same root cause)

Verification script below tests all 6 + select_scatter on default config.
"""
import torch

torch._dynamo.reset()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def verify(name, fn, inputs):
    torch._dynamo.reset()
    ei = [t.clone().detach().to(DEVICE).requires_grad_(True) for t in inputs]
    ci = [t.clone().detach().to(DEVICE).requires_grad_(True) for t in inputs]
    fn(*ei).backward()
    torch._dynamo.reset()
    torch.compile(fn)(*ci).backward()
    torch._dynamo.reset()
    max_diff = max(
        (e.grad - c.grad).abs().max().item()
        for e, c in zip(ei, ci) if e.grad is not None and c.grad is not None
    )
    status = "BUG" if max_diff > 0.5 else "ok"
    print(f"  {name}: diff={max_diff:.4g}  [{status}]")

print(f"PyTorch {torch.__version__} | {DEVICE}\n")

# Variant 1: bwd_slice_scatter fp64 (line 81)
verify("v1 bwd_slice_scatter fp64 3x33x255",
       lambda x, y: torch.slice_scatter(x, y, dim=1, start=0, end=11).sum(),
       (torch.randn(3, 33, 255, dtype=torch.float64),
        torch.randn(3, 11, 255, dtype=torch.float64)))

# Variant 2: d10_slice_scatter_dtype fp64 (line 985)
verify("v2 d10_slice_scatter fp64 3x13x64",
       lambda x, y: torch.slice_scatter(x, y, dim=1, start=0, end=6).sum(dim=1).sum(),
       (torch.randn(3, 13, 64, dtype=torch.float64),
        torch.randn(3, 6, 64, dtype=torch.float64)))

# Variant 3: d10_slice_scatter_dtype fp64 dyn (line 2698)
verify("v3 d10_slice_scatter fp64 1x255x65",
       lambda x, y: torch.slice_scatter(x, y, dim=1, start=0, end=127).sum(dim=1).sum(),
       (torch.randn(1, 255, 65, dtype=torch.float64),
        torch.randn(1, 127, 65, dtype=torch.float64)))

# Variant 4: bwd_slice_scatter fp32 (line 7617)
verify("v4 bwd_slice_scatter fp32 15x129x33",
       lambda x, y: torch.slice_scatter(x, y, dim=1, start=0, end=43).sum(),
       (torch.randn(15, 129, 33, dtype=torch.float32),
        torch.randn(15, 43, 33, dtype=torch.float32)))

# Variant 5: d5a_slice_scatter_backward fp32 (line 8017)
verify("v5 d5a_slice_scatter fp32 7x257x255",
       lambda x, src: x.slice_scatter(src, dim=1, start=0, end=85).sum(),
       (torch.zeros(7, 257, 255, dtype=torch.float32),
        torch.randn(7, 85, 255, dtype=torch.float32)))

# Also: select_scatter (same root cause)
verify("v6 select_scatter fp64 4x8",
       lambda x, s: torch.select_scatter(x, s, dim=1, index=2).sum(),
       (torch.randn(4, 8, dtype=torch.float64),
        torch.randn(4, dtype=torch.float64)))
