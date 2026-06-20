#!/usr/bin/env python3
"""Classify triggers into root-cause categories for manual analysis."""
import os
import re
import json
from collections import defaultdict, Counter

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_4h_v3", "triggers")
MODES = ["targeted_seed42", "random_seed42", "nnsmith_seed42", "whitefox_seed42"]

def parse_trigger(path):
    with open(path) as f:
        text = f.read()
    info = {}
    info['path'] = path
    info['code'] = text

    m = re.search(r'# fingerprint: (\S+)', text)
    info['fingerprint'] = m.group(1) if m else ''
    m = re.search(r'# bug_type: (\S+)', text)
    info['bug_type'] = m.group(1) if m else ''
    m = re.search(r'# max_diff: (\S+)', text)
    info['max_diff'] = float(m.group(1)) if m else 0
    m = re.search(r'# kernel_templates: (.+)', text)
    info['kernels'] = m.group(1) if m else ''

    # Extract dtype
    dtype_match = re.search(r'dtype=torch\.(\w+)', text)
    info['dtype'] = dtype_match.group(1) if dtype_match else 'float32'

    # Check for requires_grad (backward)
    info['has_grad'] = 'requires_grad=True' in text

    # Check for .sum() at end (backward pattern)
    info['has_sum'] = '.sum()' in text

    # Extract operations used
    ops = []
    for op in ['reciprocal', 'log1p', 'exp', 'softmax', 'layer_norm',
               'silu', 'gelu', 'tanh', 'sigmoid', 'sqrt', 'sin', 'cos',
               'matmul', 'mm', 'bmm', 'linear', 'addmm', 'conv2d', 'conv1d',
               'cumsum', 'cumprod', 'pow', 'div', 'mul', 'add', 'sub',
               'relu', 'abs', 'neg', 'amax', 'amin', 'mean', 'std', 'var',
               'prod', 'log_softmax', 'maximum', 'minimum',
               'scatter_add', 'scatter_reduce', 'index_add', 'index_copy',
               'batch_norm', 'group_norm', 'instance_norm',
               'max_pool', 'avg_pool', 'adaptive']:
        if op in text.split('# ---')[0]:  # only in code, not comments
            ops.append(op)
    info['ops'] = ops

    return info

def classify_root_cause(info):
    """Classify into root-cause categories."""
    code = info['code'].split('# ---')[0]
    dtype = info['dtype']
    bug_type = info['bug_type']
    max_diff = info['max_diff']
    ops = info['ops']
    has_grad = info['has_grad']

    # compile_crash is always interesting
    if bug_type == 'compile_crash':
        return 'compile_crash'

    # --- NOISE categories ---

    # fp16 with reciprocal near zero -> amplification noise
    if dtype in ('float16', 'bfloat16') and 'reciprocal' in ops:
        return 'NOISE_fp16_reciprocal'

    # fp16 with exp/silu/gelu (overflow)
    if dtype in ('float16', 'bfloat16') and any(o in ops for o in ['exp', 'silu', 'gelu', 'sigmoid']):
        if max_diff == float('inf') or max_diff > 100:
            return 'NOISE_fp16_overflow'

    # fp16 with log1p (NaN from negative inputs)
    if dtype in ('float16', 'bfloat16') and 'log1p' in ops:
        return 'NOISE_fp16_log1p'

    # fp16 general small diffs
    if dtype in ('float16', 'bfloat16') and max_diff < 10:
        return 'NOISE_fp16_precision'

    # softmax fusion precision (float32)
    if dtype == 'float32' and 'softmax' in ops and max_diff < 1:
        return 'NOISE_softmax_precision'

    # layer_norm fusion precision (float32)
    if dtype == 'float32' and 'layer_norm' in ops and max_diff < 1:
        return 'NOISE_layernorm_precision'

    # matmul + exp amplification
    if 'matmul' in ops and 'exp' in ops and max_diff > 100:
        return 'NOISE_matmul_exp'

    # reciprocal near-zero amplification (float32)
    if dtype == 'float32' and 'reciprocal' in ops and max_diff < 10:
        return 'NOISE_f32_reciprocal'

    # cumsum/cumprod inf propagation
    if any(o in ops for o in ['cumsum', 'cumprod']) and max_diff == float('inf'):
        return 'NOISE_scan_inf'

    # Very small float32 diffs (< 0.1) - likely numerical noise
    if dtype == 'float32' and max_diff < 0.1 and not has_grad:
        return 'NOISE_f32_small_diff'

    # --- POTENTIALLY REAL categories ---

    # backward with grad differences
    if has_grad and bug_type in ('SUSPICIOUS_diff', 'CATASTROPHIC_diff'):
        if dtype in ('float16', 'bfloat16'):
            return 'NOISE_fp16_backward'
        return 'REAL_backward_diff'

    # scan precision issues (cumsum/cumprod) with moderate diffs
    if any(o in ops for o in ['cumsum', 'cumprod']) and dtype == 'float32':
        return 'REAL_scan_precision'

    # Large float32 diffs (high confidence real)
    if dtype == 'float32' and max_diff > 10:
        return 'REAL_high_diff_f32'

    # Moderate float32 diffs
    if dtype == 'float32' and max_diff > 0.01:
        return 'REAL_moderate_diff_f32'

    # float64 diffs (should be very precise, any diff is suspicious)
    if dtype == 'float64':
        return 'REAL_f64_diff'

    # fp16 remaining large diffs
    if dtype in ('float16', 'bfloat16') and max_diff >= 10:
        return 'MAYBE_fp16_large_diff'

    return 'UNCATEGORIZED'


def main():
    for mode in MODES:
        mode_dir = os.path.join(RESULTS, mode)
        if not os.path.isdir(mode_dir):
            continue

        triggers = []
        for f in sorted(os.listdir(mode_dir)):
            if f.endswith('.py'):
                triggers.append(parse_trigger(os.path.join(mode_dir, f)))

        # Classify
        categories = defaultdict(list)
        for t in triggers:
            cat = classify_root_cause(t)
            categories[cat].append(t)

        print(f"\n{'='*60}")
        print(f"  {mode}: {len(triggers)} triggers")
        print(f"{'='*60}")

        # Summary
        noise_count = 0
        real_count = 0
        for cat in sorted(categories.keys()):
            items = categories[cat]
            prefix = cat.split('_')[0]
            if prefix == 'NOISE':
                noise_count += len(items)
            elif prefix in ('REAL', 'MAYBE'):
                real_count += len(items)

            # Count unique fingerprints in this category
            fps = set(t['fingerprint'] for t in items)
            print(f"  {cat}: {len(items)} triggers, {len(fps)} unique fingerprints")

            # Show a sample
            sample = items[0]
            print(f"    sample: max_diff={sample['max_diff']}, dtype={sample['dtype']}, "
                  f"ops={sample['ops'][:4]}, type={sample['bug_type']}")

        print(f"\n  NOISE: {noise_count} | POTENTIALLY REAL: {real_count} | "
              f"compile_crash: {len(categories.get('compile_crash', []))}")

        # Root cause dedup for REAL categories
        print(f"\n  --- Root-cause dedup (REAL categories) ---")
        real_fingerprints = set()
        for cat, items in categories.items():
            if cat.startswith('REAL') or cat == 'compile_crash':
                for t in items:
                    real_fingerprints.add(t['fingerprint'])
        print(f"  Total unique fingerprints in REAL+crash: {len(real_fingerprints)}")

        # Group by kernel template pattern (coarser dedup)
        kernel_groups = defaultdict(list)
        for cat, items in categories.items():
            if cat.startswith('REAL') or cat == 'compile_crash':
                for t in items:
                    # Extract base kernel pattern (remove fused_ prefix variations)
                    kernels = t['kernels']
                    # Simplify: sort kernel names
                    kernel_groups[kernels].append(t)

        print(f"  Unique kernel template combinations: {len(kernel_groups)}")


if __name__ == '__main__':
    main()
