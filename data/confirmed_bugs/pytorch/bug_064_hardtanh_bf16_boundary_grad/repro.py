#!/usr/bin/env python3
# === PROVENANCE: Class B — issue-body lifted ===============================
# Code lifted from upstream issue body via web extraction. Not guaranteed
# byte-identical to the original issue post; verify against upstream before
# relying. See BUG_MAPPING.md > Reproducer Provenance Audit.
# Verified to run on PyTorch 2.13.0.dev20260609+cu126 / Tesla T4: on T4 the
# bf16 compile path is skipped ("Tesla T4 does not support bfloat16
# compilation natively, skipping"), so eager == compiled grad on this GPU.
# To trigger the divergence reproduce on sm_80+ (A100, H100, RTX 30xx+).
# ==========================================================================
"""
Bug: F.hardtanh compiled backward silently drops gradient at bf16 boundary values.

Upstream issue: <upstream issue/PR, anonymized> (closed/fixed)

The eager CUDA kernel compares the bf16 input against the original float scalar,
while the compiled decomposition compares in bf16 precision. So input values that
equal the bf16 representation of bounds (e.g. bf16(0.7) = 0.69921875) are
misclassified as "outside the clamp range" by the compiled path, producing a
zero gradient where eager passes through 1.

~477 of 930 non-exact bound values trigger this bug (~51%).

Oracle: eager backward vs torch.compile backward (bf16 tensor, bound = bf16 of float).
"""
import torch
import torch.nn.functional as F

# bf16(0.7) = 0.69921875 — input exactly equals bf16 cast of upper bound.
x = torch.tensor([0.69921875], dtype=torch.bfloat16,
                 device='cuda', requires_grad=True)
F.hardtanh(x, min_val=-0.7, max_val=0.7).backward()
eager_grad = x.grad.clone()
print(f"eager grad:    {eager_grad}")  # tensor([1.])

x2 = torch.tensor([0.69921875], dtype=torch.bfloat16,
                  device='cuda', requires_grad=True)
torch.compile(lambda y: F.hardtanh(y, min_val=-0.7, max_val=0.7))(x2).backward()
compiled_grad = x2.grad.clone()
print(f"compiled grad: {compiled_grad}")  # tensor([0.]) <- WRONG

assert torch.allclose(eager_grad.float(), compiled_grad.float()), \
    f"BUG REPRODUCED: eager={eager_grad}, compiled={compiled_grad}"
