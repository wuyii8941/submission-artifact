# GraphFuzzerV2 Bug  [D5]
# type=inductor_diff P3 config=inductor_default
# divergence=CATASTROPHIC max_diff=0.1749114990234375 vr=1.0
# config_preset=aggressive_fusion dynamic=False
# error: eager vs inductor(default) max_diff=1.75e-01 vr=1.0000 [CATASTROPHIC|inductor-specific]
# program: d5_dtype_cast_bf16_round|torch.float64|b8s30h61|cfg=aggressive_fusion

import torch

_x = torch.randn(8, 30, 61, dtype=torch.float32)
_w = torch.randn(61, 61, dtype=torch.float32)
_program_inputs = (_x, _w)

def program(x, w):
    x = torch.matmul(x, w)
    x = x.to(torch.bfloat16).float()
    x = x.contiguous()
    x = x * torch.sigmoid(x)
    return x.sum(dim=1)
