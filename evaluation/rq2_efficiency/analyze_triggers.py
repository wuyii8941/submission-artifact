#!/usr/bin/env python3
"""
Root-cause analysis of triggers: re-run samples, classify, dedup by root cause.
Outputs a summary of distinct bugs per mode.
"""
import os
import re
import json
import sys
from collections import defaultdict, Counter

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

    # Extract ops
    code = text.split('# ---')[0]
    ops = set()
    for op in ['reciprocal', 'log1p', 'exp', 'softmax', 'layer_norm',
               'silu', 'gelu', 'tanh', 'sigmoid', 'sqrt', 'sin', 'cos',
               'matmul', 'mm', 'bmm', 'linear', 'addmm', 'conv2d', 'conv1d',
               'cumsum', 'cumprod', 'pow', 'div', 'mul', 'add', 'sub',
               'relu', 'abs', 'neg', 'amax', 'amin', 'mean', 'std', 'var',
               'prod', 'log_softmax', 'maximum', 'minimum',
               'scatter_add', 'scatter_reduce', 'index_add', 'index_copy',
               'batch_norm', 'group_norm', 'instance_norm',
               'max_pool', 'avg_pool', 'adaptive',
               'leaky_relu', 'elu', 'hardswish', 'hardsigmoid',
               'index_select', 'gather', 'scatter',
               'slice_scatter', 'select_scatter', 'as_strided_scatter']:
        if op in code:
            ops.add(op)
    info['ops'] = sorted(ops)

    # Extract iter number for timeline
    m = re.search(r'iter(\d+)', os.path.basename(path))
    info['iter'] = int(m.group(1)) if m else 0

    return info


def root_cause_classify(info):
    """Classify to root-cause bucket."""
    dtype = info['dtype']
    bug_type = info['bug_type']
    max_diff = info['max_diff']
    ops = info['ops']
    has_grad = info['has_grad']
    kernels = info['kernels']

    # --- FALSE POSITIVES (not real bugs) ---

    # bf16 on T4 (sm_75) -> PTX compile error, hardware limitation
    if bug_type == 'compile_crash' and dtype == 'bfloat16':
        return 'FP:bf16_hw_incompatible'

    # fp16/bf16 reciprocal near-zero amplification
    if dtype in ('float16', 'bfloat16') and 'reciprocal' in ops:
        return 'FP:fp16_reciprocal_amplification'

    # fp16/bf16 log1p on negative inputs -> NaN
    if dtype in ('float16', 'bfloat16') and 'log1p' in ops:
        return 'FP:fp16_log1p_nan'

    # fp16/bf16 exp/silu/gelu overflow
    if dtype in ('float16', 'bfloat16') and any(o in ops for o in ['exp', 'silu', 'gelu']) and (max_diff == float('inf') or max_diff > 100):
        return 'FP:fp16_activation_overflow'

    # fp16 backward diffs (gradient amplification)
    if dtype in ('float16', 'bfloat16') and has_grad:
        return 'FP:fp16_gradient_noise'

    # fp16/bf16 small precision diffs
    if dtype in ('float16', 'bfloat16') and max_diff < 10:
        return 'FP:fp16_precision_noise'

    # matmul+exp amplification (large inputs -> exp overflow -> diff)
    if 'matmul' in ops and 'exp' in ops and max_diff > 100:
        return 'FP:matmul_exp_amplification'

    # cumsum/cumprod inf propagation
    if any(o in ops for o in ['cumsum', 'cumprod']) and max_diff == float('inf'):
        return 'FP:scan_inf_propagation'

    # --- REAL BUG CANDIDATES ---

    # compile_crash (not bf16) - actual compiler crashes
    if bug_type == 'compile_crash':
        if 'index_copy' in ops:
            return 'BUG:index_copy_backward_crash'
        if 'scatter' in ops or 'index_add' in ops:
            return 'BUG:scatter_compile_crash'
        if 'prod' in ops:
            return 'BUG:prod_compile_crash'
        return f'BUG:compile_crash_other'

    # f32 reciprocal diffs (fusion changes reciprocal behavior)
    if dtype == 'float32' and 'reciprocal' in ops:
        if has_grad:
            return 'BUG:reciprocal_backward_fusion'
        if max_diff > 1:
            return 'BUG:reciprocal_fusion_diff'
        return 'FP:f32_reciprocal_small'

    # softmax/log_softmax precision in fusion
    if 'softmax' in ops or 'log_softmax' in ops:
        if has_grad:
            return 'BUG:softmax_backward_fusion'
        if max_diff > 0.01:
            return 'BUG:softmax_fusion_precision'
        return 'FP:softmax_tiny_diff'

    # layer_norm fusion
    if 'layer_norm' in ops:
        if max_diff > 0.1:
            return 'BUG:layernorm_fusion_diff'
        return 'FP:layernorm_tiny_diff'

    # cumsum/cumprod precision
    if any(o in ops for o in ['cumsum', 'cumprod']):
        if dtype == 'float32' and max_diff > 1:
            return 'BUG:scan_fusion_precision'
        return 'FP:scan_small_diff'

    # backward diffs (f32)
    if has_grad and dtype == 'float32':
        if max_diff > 1:
            return 'BUG:backward_fusion_diff'
        if max_diff > 0.01:
            return 'BUG:backward_precision'
        return 'FP:backward_tiny_diff'

    # f64 diffs (any diff is suspicious)
    if dtype == 'float64':
        return 'BUG:f64_diff'

    # f32 large diffs without known amplification patterns
    if dtype == 'float32' and max_diff > 10:
        if 'exp' in ops:
            return 'FP:f32_exp_overflow'
        return 'BUG:f32_large_diff'

    # f32 moderate diffs
    if dtype == 'float32' and max_diff > 0.01:
        if 'exp' in ops and max_diff > 1:
            return 'FP:f32_exp_amplification'
        return 'BUG:f32_moderate_diff'

    # f32 small diffs
    if dtype == 'float32' and max_diff <= 0.01:
        return 'FP:f32_tiny_diff'

    # fp16 large diffs remaining
    if dtype in ('float16', 'bfloat16') and max_diff >= 10:
        return 'MAYBE:fp16_large_diff'

    return 'UNCATEGORIZED'


def main():
    all_results = {}

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
            cat = root_cause_classify(t)
            t['root_cause'] = cat
            categories[cat].append(t)

        print(f"\n{'='*70}")
        print(f"  {mode}: {len(triggers)} triggers")
        print(f"{'='*70}")

        fp_total = 0
        bug_total = 0
        maybe_total = 0
        bug_details = {}

        for cat in sorted(categories.keys()):
            items = categories[cat]
            prefix = cat.split(':')[0]
            fps = set(t['fingerprint'] for t in items)

            if prefix == 'FP':
                fp_total += len(items)
                tag = '  [FP]'
            elif prefix == 'BUG':
                bug_total += len(items)
                tag = '  [BUG]'
                bug_details[cat] = items
            elif prefix == 'MAYBE':
                maybe_total += len(items)
                tag = '  [?]'
            else:
                tag = '  [??]'

            print(f"{tag} {cat}: {len(items)} triggers, {len(fps)} fingerprints")

        print(f"\n  Summary: FP={fp_total}, BUG candidates={bug_total}, MAYBE={maybe_total}")

        # Root-cause dedup for BUG categories
        print(f"\n  --- BUG root-cause details ---")
        distinct_bugs = 0
        for cat in sorted(bug_details.keys()):
            items = bug_details[cat]
            # Further dedup within category by kernel template pattern
            kernel_groups = defaultdict(list)
            for t in items:
                kernel_groups[t['kernels']].append(t)

            print(f"\n  {cat}: {len(items)} triggers -> {len(kernel_groups)} kernel groups")
            distinct_bugs += len(kernel_groups)

            # Show first trigger from each kernel group
            for kg, kg_items in sorted(kernel_groups.items(), key=lambda x: -len(x[1]))[:3]:
                sample = kg_items[0]
                print(f"    [{len(kg_items)}x] diff={sample['max_diff']}, ops={sample['ops'][:5]}, "
                      f"dtype={sample['dtype']}, file={sample['filename']}")

        print(f"\n  === Distinct BUG groups (by kernel template): {distinct_bugs} ===")
        all_results[mode] = {
            'total': len(triggers),
            'fp': fp_total,
            'bug_candidates': bug_total,
            'distinct_bugs': distinct_bugs,
            'maybe': maybe_total,
        }

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Mode':<20} {'Total':>8} {'FP':>8} {'BUG':>8} {'Distinct':>10} {'Maybe':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for mode in MODES:
        if mode in all_results:
            r = all_results[mode]
            print(f"{mode:<20} {r['total']:>8} {r['fp']:>8} {r['bug_candidates']:>8} "
                  f"{r['distinct_bugs']:>10} {r['maybe']:>8}")


if __name__ == '__main__':
    main()
