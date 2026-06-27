# GraphFuzzerV2 Bug  [D15]
# type=dynamic_crash P2 config=inductor_dynamic
# divergence=CATASTROPHIC max_diff=None vr=None
# config_preset=freezing dynamic=True
# error: LoweringException: RuntimeError: Unable to generate code for associative_scan op,
#        because there are lifted arguments
#        target: associative_scan
#        args[0]: Subgraph(name='associative_scan_combine_graph...')
# program: d15_scan_mul|torch.float64|b2s64h65|cfg=freezing|dyn

import torch

_x = torch.randn(2, 64, 65, dtype=torch.float64).clamp(-2, 2)
_program_inputs = (_x,)

def program(x):
    def combine_fn(a, b):
        return a * b
    return torch._higher_order_ops.associative_scan(combine_fn, x, dim=1)
