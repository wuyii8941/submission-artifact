#!/usr/bin/env python3
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
