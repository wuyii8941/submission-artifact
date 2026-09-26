#!/usr/bin/env python3
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pylib'))
import torch
from torch.testing._internal.common_methods_invocations import op_db

t0 = time.time()
rows = []
for op in op_db:
    row = {'op': op.name, 'variant': op.variant_test_name}
    try:
        samples = []
        for i, smp in enumerate(op.sample_inputs('cpu', torch.float32, requires_grad=False)):
            if i >= 20:
                break
            samples.append(smp)
        s = max(samples, key=lambda x: x.input.numel() if isinstance(x.input, torch.Tensor) else 0)
        args = [s.input, *s.args]
        tensors = [a for a in args if isinstance(a, torch.Tensor)] + [v for v in s.kwargs.values() if isinstance(v, torch.Tensor)]
        torch.manual_seed(0); out1 = op.op(*args, **s.kwargs)
        torch.manual_seed(1); out2 = op.op(*args, **s.kwargs)
        single = isinstance(out1, torch.Tensor)
        row.update(
            status='ok',
            n_tensor_inputs=len(tensors),
            input_float=isinstance(s.input, torch.Tensor) and s.input.is_floating_point(),
            output_float=single and out1.is_floating_point(),
            same_dtype=single and isinstance(s.input, torch.Tensor) and out1.dtype == s.input.dtype,
            same_shape=single and isinstance(s.input, torch.Tensor) and out1.shape == s.input.shape,
            output_smaller=single and isinstance(s.input, torch.Tensor) and out1.numel() < s.input.numel(),
            deterministic=bool(single and isinstance(out2, torch.Tensor) and out1.shape == out2.shape and
                               torch.equal(torch.nan_to_num(out1), torch.nan_to_num(out2))),
        )
    except Exception as e:
        row.update(status='error', error=f'{type(e).__name__}: {str(e)[:100]}')
    rows.append(row)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else 'pt_entry_metadata.json', 'w'), indent=0)
print(f'{len(rows)} ops, ok {sum(r["status"] == "ok" for r in rows)}, {time.time() - t0:.0f}s')
