#!/usr/bin/env python3
"""Re-run sample triggers to verify reproducibility and understand root causes."""
import torch
import torch._dynamo
import os
import re
import sys
from collections import defaultdict

torch._dynamo.config.suppress_errors = False
DEVICE = 'cuda'
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_4h_v3", "triggers")

def verify_trigger(path):
    """Re-run a trigger and return (reproducible, max_diff, error)."""
    with open(path) as f:
        text = f.read()

    code_section = text.split('# ---')[0].strip()

    # Skip nnsmith (can't re-run without the framework)
    if 'NNSmith seed_i=' in text:
        return None, None, 'nnsmith_skip'

    # Build namespace
    ns = {'torch': torch}
    try:
        exec(code_section, ns)
    except Exception as e:
        return None, None, f'parse_error: {e}'

    program = ns.get('program')
    inputs = ns.get('_program_inputs')
    if program is None or inputs is None:
        return None, None, 'no_program'

    # Move inputs to device
    inputs_dev = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            inputs_dev.append(inp.clone().detach().to(DEVICE))
        else:
            inputs_dev.append(inp)
    inputs_dev = tuple(inputs_dev)

    # Eager
    torch._dynamo.reset()
    try:
        if any(isinstance(inp, torch.Tensor) and inp.requires_grad for inp in inputs):
            inputs_eager = tuple(
                inp.clone().detach().requires_grad_(True).to(DEVICE) if isinstance(inp, torch.Tensor) and inp.requires_grad
                else (inp.clone().detach().to(DEVICE) if isinstance(inp, torch.Tensor) else inp)
                for inp in inputs
            )
            eager_out = program(*inputs_eager)
            if isinstance(eager_out, torch.Tensor) and eager_out.requires_grad:
                eager_out.backward(torch.ones_like(eager_out))
                eager_grads = [inp.grad.clone() for inp in inputs_eager if isinstance(inp, torch.Tensor) and inp.grad is not None]
            else:
                eager_grads = []
        else:
            eager_out = program(*inputs_dev)
            eager_grads = []
    except Exception as e:
        return None, None, f'eager_error: {e}'

    # Compiled
    torch._dynamo.reset()
    try:
        if any(isinstance(inp, torch.Tensor) and inp.requires_grad for inp in inputs):
            inputs_comp = tuple(
                inp.clone().detach().requires_grad_(True).to(DEVICE) if isinstance(inp, torch.Tensor) and inp.requires_grad
                else (inp.clone().detach().to(DEVICE) if isinstance(inp, torch.Tensor) else inp)
                for inp in inputs
            )
            compiled = torch.compile(program, backend='inductor')
            comp_out = compiled(*inputs_comp)
            if isinstance(comp_out, torch.Tensor) and comp_out.requires_grad:
                comp_out.backward(torch.ones_like(comp_out))
                comp_grads = [inp.grad.clone() for inp in inputs_comp if isinstance(inp, torch.Tensor) and inp.grad is not None]
            else:
                comp_grads = []
        else:
            compiled = torch.compile(program, backend='inductor')
            comp_out = compiled(*inputs_dev)
            comp_grads = []
    except Exception as e:
        return True, 0.0, f'compile_crash: {type(e).__name__}: {str(e)[:200]}'

    # Compare outputs
    max_diff = 0.0
    if isinstance(eager_out, torch.Tensor) and isinstance(comp_out, torch.Tensor):
        diff = (eager_out.float() - comp_out.float()).abs().max().item()
        max_diff = max(max_diff, diff)

    # Compare gradients
    for eg, cg in zip(eager_grads, comp_grads):
        diff = (eg.float() - cg.float()).abs().max().item()
        max_diff = max(max_diff, diff)

    return True, max_diff, None


def main():
    # Sample triggers from each mode and BUG category
    for mode in ["targeted_seed42", "random_seed42", "whitefox_seed42"]:
        mode_dir = os.path.join(RESULTS, mode)
        if not os.path.isdir(mode_dir):
            continue

        # Parse all
        triggers = []
        for f in sorted(os.listdir(mode_dir)):
            if f.endswith('.py'):
                path = os.path.join(mode_dir, f)
                with open(path) as fh:
                    text = fh.read()

                dtype_match = re.search(r'dtype=torch\.(\w+)', text)
                dtype = dtype_match.group(1) if dtype_match else 'float32'
                m = re.search(r'# bug_type: (\S+)', text)
                bug_type = m.group(1) if m else ''
                m = re.search(r'# max_diff: (\S+)', text)
                max_diff = float(m.group(1)) if m else 0

                code = text.split('# ---')[0]
                has_reciprocal = 'reciprocal' in code
                has_softmax = 'softmax' in code or 'log_softmax' in code
                has_grad = 'requires_grad=True' in code
                has_exp = 'exp' in code
                has_scan = 'cumsum' in code or 'cumprod' in code

                # Quick classification
                if dtype == 'bfloat16' and bug_type == 'compile_crash':
                    continue  # skip bf16 hw incompatible
                if dtype in ('float16', 'bfloat16'):
                    continue  # skip all fp16
                if 'matmul' in code and 'exp' in code and max_diff > 100:
                    continue  # skip matmul+exp

                triggers.append({
                    'path': path,
                    'filename': f,
                    'dtype': dtype,
                    'bug_type': bug_type,
                    'max_diff': max_diff,
                    'has_reciprocal': has_reciprocal,
                    'has_softmax': has_softmax,
                    'has_grad': has_grad,
                    'has_exp': has_exp,
                    'has_scan': has_scan,
                })

        # Group and sample
        groups = defaultdict(list)
        for t in triggers:
            if t['bug_type'] == 'compile_crash':
                groups['compile_crash'].append(t)
            elif t['has_reciprocal'] and t['has_grad']:
                groups['reciprocal_backward'].append(t)
            elif t['has_reciprocal']:
                groups['reciprocal_forward'].append(t)
            elif t['has_softmax'] and t['has_grad']:
                groups['softmax_backward'].append(t)
            elif t['has_softmax']:
                groups['softmax_forward'].append(t)
            elif t['has_scan']:
                groups['scan'].append(t)
            elif t['has_grad']:
                groups['backward_other'].append(t)
            elif t['has_exp'] and t['max_diff'] > 100:
                groups['exp_overflow'].append(t)
            else:
                groups['other_diff'].append(t)

        print(f"\n{'='*70}")
        print(f"  VERIFYING {mode}")
        print(f"{'='*70}")

        for group_name, group_items in sorted(groups.items()):
            # Take up to 3 samples from each group
            samples = group_items[:3]
            print(f"\n  [{group_name}] ({len(group_items)} total, verifying {len(samples)})")

            for t in samples:
                torch._dynamo.reset()
                repro, diff, error = verify_trigger(t['path'])
                status = 'CRASH' if error and 'compile_crash' in error else \
                         f'DIFF={diff:.6g}' if diff is not None else \
                         f'ERROR: {error}'
                orig_diff = t['max_diff']
                match = '✓' if (diff is not None and diff > 0.01) or (error and 'compile_crash' in error) else '✗'
                print(f"    {match} {t['filename']}: orig={orig_diff}, now={status}")


if __name__ == '__main__':
    main()
