# GraphFuzzerV2 Bug  [D5]
# type=inductor_diff P3 config=inductor_default
# divergence=CATASTROPHIC max_diff=0.3272247314453125 vr=0.8820754716981132
# config_preset=aggressive_fusion dynamic=False
# error: eager vs inductor(default) max_diff=3.27e-01 vr=0.8821 [CATASTROPHIC|inductor-specific]
# program: d5_dtype_cast_bf16_round|torch.float32|b8s69h53|cfg=aggressive_fusion

import torch

_x = torch.randn(8, 69, 53, dtype=torch.float32)
_w = torch.randn(53, 53, dtype=torch.float32)
_program_inputs = (_x, _w)

def program(x, w):
    x = torch.matmul(x, w)
    x = x.to(torch.bfloat16).float()
    x = x * torch.sigmoid(x)
    return x.sum(dim=1)
