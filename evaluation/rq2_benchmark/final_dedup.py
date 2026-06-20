#!/usr/bin/env python3
"""
Final root-cause dedup across all 4 modes.

Root-cause categories (distinct bugs):
1. reciprocal_fusion: reciprocal(near_zero) in fused kernel produces different rounding
2. reciprocal_backward_fusion: same as above, in backward pass (gradient through reciprocal)
3. softmax_fusion_precision: softmax/log_softmax loses precision in fused kernel
4. softmax_backward_fusion: softmax backward fusion precision
5. scan_fusion: cumsum/cumprod accumulation order changes in fused kernel
6. index_copy_backward_crash: index_copy backward causes AssertionError in inductor
7. backward_exp_fusion: backward through exp-like activations has precision diff
8. fp16_roundtrip_fusion: to(fp16).float() intermediate precision lost in fusion
9. f64_reciprocal_fusion: reciprocal fusion in float64

False positives excluded:
- bf16 on sm_75 hardware → PTX compile error (not compiler bug)
- fp16 precision noise (inherent low precision)
- matmul+exp / exp+exp overflow amplification
- dropout / random ops in whitefox
- uninitialized tensor (torch.Tensor()) in whitefox
- scan inf propagation (cumsum of values that overflow)
"""
import os
import re
import json
from collections import defaultdict

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
    m = re.search(r'iter(\d+)', os.path.basename(path))
    info['iter'] = int(m.group(1)) if m else 0

    code = text.split('# ---')[0]
    info['code'] = code
    info['is_nnsmith'] = 'NNSmith seed_i=' in text

    dtype_match = re.search(r'dtype=torch\.(\w+)', code)
    info['dtype'] = dtype_match.group(1) if dtype_match else 'float32'
    info['has_grad'] = 'requires_grad=True' in code

    has = lambda pat: bool(re.search(pat, code))
    info['has_reciprocal'] = has(r'reciprocal')
    info['has_softmax'] = has(r'softmax|log_softmax')
    info['has_scan'] = has(r'cumsum|cumprod')
    info['has_exp'] = has(r'torch\.exp|\.exp\(')
    info['has_index_copy'] = has(r'index_copy')
    info['has_fp16_roundtrip'] = has(r'to\(torch\.float16\)\.float\(\)')
    info['has_dropout'] = has(r'dropout|Dropout')
    info['has_random'] = has(r'\.normal_\(\)|torch\.rand\b|torch\.Tensor\(')
    info['has_matmul'] = has(r'matmul|mm\b|bmm|linear|addmm')
    info['has_layer_norm'] = has(r'layer_norm')
    info['has_batch_norm'] = has(r'batch_norm|BatchNorm')

    return info


def classify_final(t):
    """Return (category, root_cause) where category is FP/BUG/UNKNOWN."""
    dtype = t['dtype']
    bt = t['bug_type']
    diff = t['max_diff']
    grad = t['has_grad']

    # === FALSE POSITIVES ===

    # bf16 hardware incompatible
    if bt == 'compile_crash' and dtype == 'bfloat16':
        return 'FP', 'bf16_hw_incompatible'

    # Whitefox: dropout/random ops → non-deterministic
    if t['has_dropout'] or t['has_random']:
        return 'FP', 'nondeterministic_ops'

    # fp16/bf16 any precision issue
    if dtype in ('float16', 'bfloat16'):
        return 'FP', 'fp16_precision'

    # fp16 round-trip with exp overflow
    if t['has_fp16_roundtrip'] and t['has_exp'] and diff > 100:
        return 'FP', 'fp16_roundtrip_exp_overflow'

    # matmul+exp overflow (f32)
    if t['has_matmul'] and t['has_exp'] and diff > 100:
        return 'FP', 'matmul_exp_overflow'

    # exp+exp double overflow
    if t['has_exp'] and 'exp' in t['code'] and t['code'].count('exp') >= 2 and diff > 100:
        return 'FP', 'double_exp_overflow'

    # scan inf propagation
    if t['has_scan'] and diff == float('inf'):
        return 'FP', 'scan_inf'

    # === REAL BUGS ===

    # NNSmith - can't decompose further without re-running the models
    if t['is_nnsmith']:
        if bt == 'compile_crash':
            return 'BUG', 'nnsmith_compile_crash'
        return 'BUG', 'nnsmith_diff'

    # Compile crashes (non-bf16)
    if bt == 'compile_crash':
        if t['has_index_copy']:
            return 'BUG', 'index_copy_backward_crash'
        return 'BUG', 'other_compile_crash'

    # Reciprocal fusion
    if t['has_reciprocal']:
        if grad:
            return 'BUG', 'reciprocal_backward_fusion'
        return 'BUG', 'reciprocal_fusion'

    # Softmax/log_softmax fusion
    if t['has_softmax']:
        if grad:
            return 'BUG', 'softmax_backward_fusion'
        return 'BUG', 'softmax_fusion'

    # Layer norm fusion
    if t['has_layer_norm']:
        return 'BUG', 'layernorm_fusion'

    # Scan fusion
    if t['has_scan']:
        return 'BUG', 'scan_fusion'

    # fp16 round-trip fusion (non-overflow)
    if t['has_fp16_roundtrip']:
        return 'BUG', 'fp16_roundtrip_fusion'

    # Backward diffs (no reciprocal/softmax)
    if grad:
        return 'BUG', 'backward_fusion_diff'

    # f64 diff (any diff in f64 is suspicious)
    if dtype == 'float64':
        if t['has_reciprocal']:
            return 'BUG', 'reciprocal_fusion'
        return 'BUG', 'f64_fusion_diff'

    # Remaining f32 with exp and large diff → likely amplification FP
    if t['has_exp'] and diff > 10:
        return 'FP', 'exp_amplification'

    # Remaining f32 moderate diffs
    if diff > 0.01:
        return 'BUG', 'f32_fusion_precision'

    # Very small f32 diffs
    return 'FP', 'f32_tiny_diff'


def main():
    all_mode_results = {}

    for mode in MODES:
        mode_dir = os.path.join(RESULTS, mode)
        if not os.path.isdir(mode_dir):
            continue

        triggers = []
        for f in sorted(os.listdir(mode_dir)):
            if f.endswith('.py'):
                triggers.append(parse_trigger(os.path.join(mode_dir, f)))

        # Classify
        buckets = defaultdict(list)
        for t in triggers:
            cat, cause = classify_final(t)
            t['final_cat'] = cat
            t['final_cause'] = cause
            buckets[(cat, cause)].append(t)

        fp_count = sum(len(v) for (c, _), v in buckets.items() if c == 'FP')
        bug_count = sum(len(v) for (c, _), v in buckets.items() if c == 'BUG')
        bug_causes = sorted(set(cause for (c, cause) in buckets.keys() if c == 'BUG'))

        print(f"\n{'='*70}")
        print(f"  {mode}: {len(triggers)} total triggers")
        print(f"{'='*70}")

        for (cat, cause) in sorted(buckets.keys()):
            items = buckets[(cat, cause)]
            tag = {'FP': 'FP', 'BUG': '** BUG **', 'UNKNOWN': '??'}[cat]
            first_iter = min(t['iter'] for t in items)
            print(f"  {tag:>10s}  {cause:<35s}  {len(items):>5d} triggers  first@iter{first_iter}")

        print(f"\n  FP: {fp_count} | BUG: {bug_count} | Distinct root causes: {len(bug_causes)}")
        for cause in bug_causes:
            items = buckets[('BUG', cause)]
            first_iter = min(t['iter'] for t in items)
            print(f"    -> {cause} ({len(items)} triggers, first@iter{first_iter})")

        all_mode_results[mode] = {
            'total': len(triggers),
            'fp': fp_count,
            'bug_triggers': bug_count,
            'bug_causes': bug_causes,
            'bug_details': {cause: {
                'count': len(buckets[('BUG', cause)]),
                'first_iter': min(t['iter'] for t in buckets[('BUG', cause)]),
            } for cause in bug_causes},
        }

    # === CROSS-MODE COMPARISON ===
    print(f"\n{'='*70}")
    print(f"  FINAL CROSS-MODE COMPARISON (root-cause dedup)")
    print(f"{'='*70}")

    all_causes = sorted(set(
        cause for r in all_mode_results.values() for cause in r['bug_causes']
    ))

    # Header
    short_names = {'targeted_seed42': 'targeted', 'random_seed42': 'random',
                   'nnsmith_seed42': 'nnsmith', 'whitefox_seed42': 'whitefox'}
    print(f"\n  {'Root Cause':<35s}", end='')
    for mode in MODES:
        if mode in all_mode_results:
            print(f"  {short_names[mode]:>10s}", end='')
    print()
    print(f"  {'-'*35}", end='')
    for _ in MODES:
        print(f"  {'-'*10}", end='')
    print()

    for cause in all_causes:
        print(f"  {cause:<35s}", end='')
        for mode in MODES:
            if mode in all_mode_results:
                details = all_mode_results[mode]['bug_details']
                if cause in details:
                    print(f"  {details[cause]['count']:>10d}", end='')
                else:
                    print(f"  {'-':>10s}", end='')
        print()

    # Summary row
    print(f"  {'-'*35}", end='')
    for _ in MODES:
        print(f"  {'-'*10}", end='')
    print()
    print(f"  {'TOTAL DISTINCT ROOT CAUSES':<35s}", end='')
    for mode in MODES:
        if mode in all_mode_results:
            print(f"  {len(all_mode_results[mode]['bug_causes']):>10d}", end='')
    print()
    print(f"  {'TOTAL BUG TRIGGERS':<35s}", end='')
    for mode in MODES:
        if mode in all_mode_results:
            print(f"  {all_mode_results[mode]['bug_triggers']:>10d}", end='')
    print()
    print(f"  {'TOTAL FP':<35s}", end='')
    for mode in MODES:
        if mode in all_mode_results:
            print(f"  {all_mode_results[mode]['fp']:>10d}", end='')
    print()

    # Time to first discovery
    print(f"\n\n  TIME TO FIRST DISCOVERY (iteration number)")
    print(f"  {'Root Cause':<35s}", end='')
    for mode in MODES:
        if mode in all_mode_results:
            print(f"  {short_names[mode]:>10s}", end='')
    print()
    print(f"  {'-'*35}", end='')
    for _ in MODES:
        print(f"  {'-'*10}", end='')
    print()
    for cause in all_causes:
        print(f"  {cause:<35s}", end='')
        for mode in MODES:
            if mode in all_mode_results:
                details = all_mode_results[mode]['bug_details']
                if cause in details:
                    print(f"  {details[cause]['first_iter']:>10d}", end='')
                else:
                    print(f"  {'-':>10s}", end='')
        print()


if __name__ == '__main__':
    main()
