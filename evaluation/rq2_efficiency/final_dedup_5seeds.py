#!/usr/bin/env python3
"""Root-cause dedup across seeds 42 & 123, then union per mode."""
import os
import re
import json
from collections import defaultdict

TRIGGERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_5seeds", "triggers")
MODES = ["targeted", "random", "nnsmith", "whitefox"]
SEEDS = [42, 123]


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
    dtype = t['dtype']
    bt = t['bug_type']
    diff = t['max_diff']
    grad = t['has_grad']

    # === FALSE POSITIVES ===
    if bt == 'compile_crash' and dtype == 'bfloat16':
        return 'FP', 'bf16_hw_incompatible'
    if t['has_dropout'] or t['has_random']:
        return 'FP', 'nondeterministic_ops'
    if dtype in ('float16', 'bfloat16'):
        return 'FP', 'fp16_precision'
    if t['has_fp16_roundtrip'] and t['has_exp'] and diff > 100:
        return 'FP', 'fp16_roundtrip_exp_overflow'
    if t['has_matmul'] and t['has_exp'] and diff > 100:
        return 'FP', 'matmul_exp_overflow'
    if t['has_exp'] and 'exp' in t['code'] and t['code'].count('exp') >= 2 and diff > 100:
        return 'FP', 'double_exp_overflow'
    if t['has_scan'] and diff == float('inf'):
        return 'FP', 'scan_inf'

    # === REAL BUGS ===
    if t['is_nnsmith']:
        if bt == 'compile_crash':
            return 'BUG', 'nnsmith_compile_crash'
        return 'BUG', 'nnsmith_diff'

    if bt == 'compile_crash':
        if t['has_index_copy']:
            return 'BUG', 'index_copy_backward_crash'
        return 'BUG', 'other_compile_crash'

    if t['has_reciprocal']:
        if grad:
            return 'BUG', 'reciprocal_backward_fusion'
        return 'BUG', 'reciprocal_fusion'

    if t['has_softmax']:
        if grad:
            return 'BUG', 'softmax_backward_fusion'
        return 'BUG', 'softmax_fusion'

    if t['has_layer_norm']:
        return 'BUG', 'layernorm_fusion'

    if t['has_scan']:
        return 'BUG', 'scan_fusion'

    if t['has_fp16_roundtrip']:
        return 'BUG', 'fp16_roundtrip_fusion'

    if grad:
        return 'BUG', 'backward_fusion_diff'

    if dtype == 'float64':
        if t['has_reciprocal']:
            return 'BUG', 'reciprocal_fusion'
        return 'BUG', 'f64_fusion_diff'

    if t['has_exp'] and diff > 10:
        return 'FP', 'exp_amplification'

    if diff > 0.01:
        return 'BUG', 'f32_fusion_precision'

    return 'FP', 'f32_tiny_diff'


def main():
    # Per-mode, per-seed results
    results = {}  # (mode, seed) -> {bug_causes: set, ...}

    for mode in MODES:
        for seed in SEEDS:
            key = f"{mode}_seed{seed}"
            mode_dir = os.path.join(TRIGGERS, key)
            if not os.path.isdir(mode_dir):
                print(f"  SKIP {key} (no trigger dir)")
                continue

            triggers = []
            for f in sorted(os.listdir(mode_dir)):
                if f.endswith('.py'):
                    triggers.append(parse_trigger(os.path.join(mode_dir, f)))

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
            print(f"  {key}: {len(triggers)} triggers -> {bug_count} bugs, {fp_count} FP, {len(bug_causes)} root causes")
            print(f"{'='*70}")

            for (cat, cause) in sorted(buckets.keys()):
                items = buckets[(cat, cause)]
                tag = 'FP' if cat == 'FP' else '** BUG **'
                first_iter = min(t['iter'] for t in items)
                print(f"  {tag:>10s}  {cause:<35s}  {len(items):>5d} triggers  first@iter{first_iter}")

            results[key] = {
                'total': len(triggers),
                'fp': fp_count,
                'bug_triggers': bug_count,
                'bug_causes': set(bug_causes),
                'bug_details': {cause: {
                    'count': len(buckets[('BUG', cause)]),
                    'first_iter': min(t['iter'] for t in buckets[('BUG', cause)]),
                } for cause in bug_causes},
            }

    # === UNION ACROSS SEEDS PER MODE ===
    print(f"\n\n{'='*70}")
    print(f"  UNION ACROSS SEEDS (per mode)")
    print(f"{'='*70}")

    mode_union = {}
    for mode in MODES:
        union_causes = set()
        total_triggers = 0
        total_bugs = 0
        total_fp = 0
        cause_details = {}
        for seed in SEEDS:
            key = f"{mode}_seed{seed}"
            if key not in results:
                continue
            r = results[key]
            union_causes |= r['bug_causes']
            total_triggers += r['total']
            total_bugs += r['bug_triggers']
            total_fp += r['fp']
            for cause, det in r['bug_details'].items():
                if cause not in cause_details:
                    cause_details[cause] = {'total_count': 0, 'seeds_found': []}
                cause_details[cause]['total_count'] += det['count']
                cause_details[cause]['seeds_found'].append(seed)

        mode_union[mode] = {
            'union_causes': sorted(union_causes),
            'total_triggers': total_triggers,
            'total_bugs': total_bugs,
            'total_fp': total_fp,
            'cause_details': cause_details,
        }

        print(f"\n  {mode}: {len(union_causes)} root causes (union of seeds {SEEDS})")
        print(f"    Total: {total_triggers} triggers, {total_bugs} bugs, {total_fp} FP")
        for cause in sorted(union_causes):
            det = cause_details[cause]
            seeds_str = ','.join(str(s) for s in det['seeds_found'])
            print(f"    -> {cause:<35s} {det['total_count']:>5d} triggers  seeds=[{seeds_str}]")

    # === CROSS-MODE COMPARISON TABLE ===
    print(f"\n\n{'='*70}")
    print(f"  CROSS-MODE COMPARISON (root-cause dedup, union of seeds {SEEDS})")
    print(f"{'='*70}")

    all_causes = sorted(set(
        c for m in mode_union.values() for c in m['union_causes']
    ))

    print(f"\n  {'Root Cause':<35s}", end='')
    for mode in MODES:
        print(f"  {mode:>10s}", end='')
    print()
    print(f"  {'-'*35}", end='')
    for _ in MODES:
        print(f"  {'-'*10}", end='')
    print()

    for cause in all_causes:
        print(f"  {cause:<35s}", end='')
        for mode in MODES:
            det = mode_union[mode]['cause_details']
            if cause in det:
                print(f"  {det[cause]['total_count']:>10d}", end='')
            else:
                print(f"  {'-':>10s}", end='')
        print()

    print(f"  {'-'*35}", end='')
    for _ in MODES:
        print(f"  {'-'*10}", end='')
    print()
    print(f"  {'DISTINCT ROOT CAUSES':<35s}", end='')
    for mode in MODES:
        print(f"  {len(mode_union[mode]['union_causes']):>10d}", end='')
    print()
    print(f"  {'TOTAL BUG TRIGGERS':<35s}", end='')
    for mode in MODES:
        print(f"  {mode_union[mode]['total_bugs']:>10d}", end='')
    print()
    print(f"  {'TOTAL FP':<35s}", end='')
    for mode in MODES:
        print(f"  {mode_union[mode]['total_fp']:>10d}", end='')
    print()

    # === UNIQUE TO EACH MODE ===
    print(f"\n\n  UNIQUE ROOT CAUSES (found ONLY by this mode):")
    for mode in MODES:
        others = set()
        for m2 in MODES:
            if m2 != mode:
                others |= set(mode_union[m2]['union_causes'])
        unique = set(mode_union[mode]['union_causes']) - others
        print(f"    {mode}: {sorted(unique) if unique else '(none)'}")


if __name__ == '__main__':
    main()
