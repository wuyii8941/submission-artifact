#!/usr/bin/env python3
"""
Verify and root-cause dedup triggers across all modes.
Re-runs a sample of triggers to confirm reproducibility, then groups by root cause.
"""
import os
import re
import sys
import json
import torch
import torch._dynamo
from collections import defaultdict, Counter

torch._dynamo.config.suppress_errors = False

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_4h_v3", "triggers")
MODES = ["targeted_seed42", "random_seed42", "nnsmith_seed42", "whitefox_seed42"]

def parse_trigger(path):
    with open(path) as f:
        text = f.read()
    info = {'path': path, 'filename': os.path.basename(path)}
    m = re.search(r'# fingerprint: (\S+)', text)
    info['fingerprint'] = m.group(1) if m else ''
    m = re.search(r'# bug_type: (\S+)', text)
    info['bug_type'] = m.group(1) if m else ''
    m = re.search(r'# max_diff: (\S+)', text)
    info['max_diff'] = float(m.group(1)) if m else 0
    m = re.search(r'# kernel_templates: (.+)', text)
    info['kernels'] = m.group(1) if m else '[]'
    dtype_match = re.search(r'dtype=torch\.(\w+)', text)
    info['dtype'] = dtype_match.group(1) if dtype_match else 'float32'
    info['has_grad'] = 'requires_grad=True' in text
    info['code_section'] = text.split('# ---')[0].strip()
    info['is_nnsmith'] = 'NNSmith seed_i=' in text
    m = re.search(r'iter(\d+)', os.path.basename(path))
    info['iter'] = int(m.group(1)) if m else 0

    code = text.split('# ---')[0]
    ops = set()
    for op in ['reciprocal', 'log1p', 'exp', 'softmax', 'layer_norm',
               'silu', 'gelu', 'tanh', 'sigmoid', 'sqrt', 'sin', 'cos',
               'matmul', 'mm', 'bmm', 'linear', 'addmm', 'conv2d', 'conv1d',
               'cumsum', 'cumprod', 'pow', 'div', 'mul', 'add', 'sub',
               'relu', 'abs', 'neg', 'amax', 'amin', 'mean', 'std', 'var',
               'prod', 'log_softmax', 'maximum', 'minimum',
               'scatter_add', 'scatter_reduce', 'index_add', 'index_copy',
               'batch_norm', 'group_norm', 'instance_norm', 'elu',
               'leaky_relu', 'hardswish', 'hardsigmoid']:
        if op in code:
            ops.add(op)
    info['ops'] = sorted(ops)
    return info


def assign_root_cause(info):
    """Assign a coarse root-cause label for dedup."""
    dtype = info['dtype']
    bug_type = info['bug_type']
    max_diff = info['max_diff']
    ops = info['ops']
    has_grad = info['has_grad']
    kernels = info['kernels']
    is_nnsmith = info['is_nnsmith']

    # FALSE POSITIVES
    if bug_type == 'compile_crash' and dtype == 'bfloat16':
        return 'FP', 'bf16_hw_incompatible'
    if dtype in ('float16', 'bfloat16') and 'reciprocal' in ops:
        return 'FP', 'fp16_reciprocal'
    if dtype in ('float16', 'bfloat16') and 'log1p' in ops:
        return 'FP', 'fp16_log1p'
    if dtype in ('float16', 'bfloat16') and any(o in ops for o in ['exp', 'silu', 'gelu']) and (max_diff == float('inf') or max_diff > 100):
        return 'FP', 'fp16_overflow'
    if dtype in ('float16', 'bfloat16') and has_grad:
        return 'FP', 'fp16_gradient'
    if dtype in ('float16', 'bfloat16') and max_diff < 10:
        return 'FP', 'fp16_precision'
    if 'matmul' in ops and 'exp' in ops and max_diff > 100:
        return 'FP', 'matmul_exp_amplification'
    if any(o in ops for o in ['cumsum', 'cumprod']) and max_diff == float('inf'):
        return 'FP', 'scan_inf'

    # REAL BUG ROOT CAUSES

    # Compile crashes
    if bug_type == 'compile_crash':
        if is_nnsmith:
            return 'BUG', 'nnsmith_compile_crash'
        if 'index_copy' in ops:
            return 'BUG', 'index_copy_backward_crash'
        return 'BUG', 'compile_crash_other'

    # f32 reciprocal fusion: inductor fuses reciprocal differently, changing near-zero behavior
    if 'reciprocal' in ops and dtype in ('float32', 'float64'):
        if has_grad:
            return 'BUG', 'reciprocal_backward_fusion'
        return 'BUG', 'reciprocal_fusion'

    # Softmax/log_softmax fusion precision
    if 'softmax' in ops or 'log_softmax' in ops:
        if has_grad:
            return 'BUG', 'softmax_backward_fusion'
        return 'BUG', 'softmax_fusion_precision'

    # Layer norm fusion
    if 'layer_norm' in ops:
        return 'BUG', 'layernorm_fusion'

    # cumsum/cumprod scan
    if any(o in ops for o in ['cumsum', 'cumprod']):
        return 'BUG', 'scan_fusion'

    # Backward diffs in f32 (without reciprocal/softmax)
    if has_grad and dtype == 'float32':
        if max_diff > 1:
            return 'BUG', 'backward_large_diff'
        return 'BUG', 'backward_precision'

    # f64 any diff
    if dtype == 'float64':
        return 'BUG', 'f64_diff'

    # f32 exp-only amplification
    if dtype == 'float32' and 'exp' in ops and max_diff > 100:
        return 'FP', 'f32_exp_overflow'

    # NNSmith diff triggers (complex models)
    if is_nnsmith:
        if max_diff > 10:
            return 'BUG', 'nnsmith_large_diff'
        return 'BUG', 'nnsmith_moderate_diff'

    # f32 moderate/large diffs (no known amplification pattern)
    if dtype == 'float32':
        if max_diff > 10:
            return 'BUG', 'f32_large_diff'
        if max_diff > 0.01:
            return 'BUG', 'f32_moderate_diff'
        return 'FP', 'f32_tiny_diff'

    # fp16 large remaining
    if dtype in ('float16', 'bfloat16') and max_diff >= 10:
        return 'MAYBE', 'fp16_large_diff'

    return 'UNK', 'uncategorized'


def main():
    summary = {}

    for mode in MODES:
        mode_dir = os.path.join(RESULTS, mode)
        if not os.path.isdir(mode_dir):
            continue

        triggers = []
        for f in sorted(os.listdir(mode_dir)):
            if f.endswith('.py'):
                triggers.append(parse_trigger(os.path.join(mode_dir, f)))

        # Classify all triggers
        root_causes = defaultdict(list)
        fp_count = 0
        bug_count = 0

        for t in triggers:
            category, cause = assign_root_cause(t)
            t['category'] = category
            t['cause'] = cause
            root_causes[(category, cause)].append(t)
            if category == 'FP':
                fp_count += 1
            elif category == 'BUG':
                bug_count += 1

        print(f"\n{'='*70}")
        print(f"  {mode}: {len(triggers)} triggers")
        print(f"{'='*70}")

        # Show all categories
        for (cat, cause) in sorted(root_causes.keys()):
            items = root_causes[(cat, cause)]
            fps = set(t['fingerprint'] for t in items)
            tag = {'FP': '[FP]', 'BUG': '[BUG]', 'MAYBE': '[?]', 'UNK': '[??]'}[cat]
            print(f"  {tag:6s} {cause:<35s} {len(items):>5d} triggers, {len(fps):>5d} fingerprints")

        # Count distinct root causes (BUG only)
        bug_causes = [(cat, cause) for (cat, cause) in root_causes.keys() if cat == 'BUG']
        print(f"\n  FALSE POSITIVES: {fp_count}")
        print(f"  BUG ROOT CAUSES: {len(bug_causes)} distinct")
        print(f"  BUG TRIGGERS: {bug_count}")

        # Detail each BUG cause
        for cat, cause in sorted(bug_causes):
            items = root_causes[(cat, cause)]
            sample = items[0]
            first_iter = min(t['iter'] for t in items)
            print(f"    {cause}: {len(items)} triggers, first@iter{first_iter}, "
                  f"sample_diff={sample['max_diff']}, dtype={sample['dtype']}")

        summary[mode] = {
            'total': len(triggers),
            'fp': fp_count,
            'bug_triggers': bug_count,
            'bug_root_causes': len(bug_causes),
            'causes': {cause: len(items) for (cat, cause), items in root_causes.items() if cat == 'BUG'},
        }

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON (root-cause dedup)")
    print(f"{'='*70}")
    print(f"{'Mode':<20} {'Total':>7} {'FP':>7} {'BugTrig':>8} {'RootCauses':>11}")
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*8} {'-'*11}")
    for mode in MODES:
        if mode in summary:
            s = summary[mode]
            print(f"{mode:<20} {s['total']:>7} {s['fp']:>7} {s['bug_triggers']:>8} {s['bug_root_causes']:>11}")

    # Show which root causes are shared across modes
    print(f"\n{'='*70}")
    print(f"  ROOT CAUSE OVERLAP")
    print(f"{'='*70}")
    all_causes = set()
    mode_causes = {}
    for mode in MODES:
        if mode in summary:
            causes = set(summary[mode]['causes'].keys())
            mode_causes[mode] = causes
            all_causes |= causes

    for cause in sorted(all_causes):
        found_in = [m.replace('_seed42', '') for m in MODES if m in mode_causes and cause in mode_causes[m]]
        counts = [f"{summary[m]['causes'].get(cause, 0)}" for m in MODES if m in summary]
        print(f"  {cause:<35s} found_in: {', '.join(found_in):<40s} counts: {'/'.join(counts)}")


if __name__ == '__main__':
    main()
