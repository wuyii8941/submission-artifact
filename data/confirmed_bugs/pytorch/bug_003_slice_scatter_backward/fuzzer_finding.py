# GraphFuzzerV2 Bug  [D10]
# type=backward_only P3 config=inductor_backward
# divergence=CATASTROPHIC max_diff=2.8421573638916016 vr=1.0
# config_preset=max_fusion_low dynamic=True
# error: [BACKWARD_ONLY] forward clean but gradient diverges: grad_max=2.84e+00
# program: d10_slice_scatter_dtype|torch.float32|b4s13h33|cfg=max_fusion_low|dyn

import torch

_x = torch.randn(4, 13, 33, dtype=torch.float32)
_y = torch.randn(4, 6, 33, dtype=torch.float32)
_program_inputs = (_x, _y)

def program(x, y):
    out = torch.slice_scatter(x, y, dim=1, start=0, end=6)
    return out.sum(dim=1)
