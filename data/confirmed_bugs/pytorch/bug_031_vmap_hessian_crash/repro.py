#!/usr/bin/env python3
# === PROVENANCE: Class B — issue-body lifted ===============================
# Code lifted from upstream issue body via web extraction. Not guaranteed
# byte-identical to the original issue post; verify against upstream before
# relying. See BUG_MAPPING.md > Reproducer Provenance Audit.
# Verified on PyTorch 2.13.0.dev20260609+cu126 / Tesla T4: hessian path
# crashes with TorchRuntimeError on the fake-tensor call_function for
# _autograd_grad; the nested-jacrev workaround compiles cleanly.
# ==========================================================================
"""
Bug: torch.compile(torch.vmap(torch.func.hessian(f))) crashes with
TorchRuntimeError — fake-tensor propagation breaks during Dynamo when
jacfwd is used as the outer transform with nested differentiation, vmap,
and compile.

Upstream issue: <upstream issue/PR, anonymized> (open, triaged)

A nested-jacrev workaround compiles cleanly, which isolates the bug to the
hessian (jacfwd-outer) path.
"""
import torch

torch.manual_seed(42)
x = torch.randn(4, 8, device='cuda', dtype=torch.float64)

def f(x):
    return (x ** 4).sum()

# Crashes: hessian = jacfwd(jacrev(f))
try:
    compiled = torch.compile(torch.vmap(torch.func.hessian(f)), fullgraph=True)
    result = compiled(x)
    print(f"[hessian path] unexpected: no crash, result shape = {result.shape}")
except Exception as e:
    print(f"[hessian path] CRASH: {type(e).__name__}: {str(e)[:300]}")

# Workaround: nested jacrev (jacrev ∘ jacrev) compiles cleanly.
compiled_ok = torch.compile(
    torch.vmap(torch.func.jacrev(torch.func.jacrev(f))),
    fullgraph=True,
)
result_ok = compiled_ok(x)
print(f"[nested jacrev workaround] OK, result shape = {result_ok.shape}")
