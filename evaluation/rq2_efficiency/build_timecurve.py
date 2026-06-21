#!/usr/bin/env python3
"""
Build deduplicated bug-discovery-over-time data for all 4 modes.
Uses the same FP classification as final_dedup.py to filter noise.
Outputs CSV + summary for plotting.
"""
import os
import re
import json
import csv
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_4h_v3")
MODES = ["targeted_seed42", "random_seed42", "nnsmith_seed42", "whitefox_seed42"]


def parse_trigger_meta(path):
    """Parse just enough metadata from trigger file for classification."""
    with open(path) as f:
        text = f.read()
    info = {}
    m = re.search(r'# fingerprint: (\S+)', text)
    info['fingerprint'] = m.group(1) if m else ''
    m = re.search(r'# bug_type: (\S+)', text)
    info['bug_type'] = m.group(1) if m else ''
    m = re.search(r'# max_diff: (\S+)', text)
    info['max_diff'] = float(m.group(1)) if m else 0
    m = re.search(r'iter(\d+)', os.path.basename(path))
    info['iter'] = int(m.group(1)) if m else 0

    code = text.split('# ---')[0]
    dtype_match = re.search(r'dtype=torch\.(\w+)', code)
    info['dtype'] = dtype_match.group(1) if dtype_match else 'float32'
    info['is_nnsmith'] = 'NNSmith seed_i=' in text
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
    info['exp_count'] = len(re.findall(r'torch\.exp|\.exp\(', code))

    return info


def is_false_positive(t):
    """Return True if this trigger is a false positive."""
    dtype = t['dtype']
    bt = t['bug_type']
    diff = t['max_diff']

    if bt == 'compile_crash' and dtype == 'bfloat16':
        return True
    if t['has_dropout'] or t['has_random']:
        return True
    if dtype in ('float16', 'bfloat16'):
        return True
    if t['has_fp16_roundtrip'] and t['has_exp'] and diff > 100:
        return True
    if t['has_matmul'] and t['has_exp'] and diff > 100:
        return True
    if t['has_exp'] and t['exp_count'] >= 2 and diff > 100:
        return True
    if t['has_scan'] and diff == float('inf'):
        return True
    if t['has_exp'] and diff > 10 and not t['has_reciprocal'] and not t['has_softmax'] and not t['has_scan']:
        if not t['has_grad'] and not t['is_nnsmith']:
            return True
    if dtype == 'float32' and diff <= 0.01 and not t['has_grad'] and not t['is_nnsmith'] and bt != 'compile_crash':
        return True
    return False


def get_root_cause(t):
    """Get root cause label for a non-FP trigger."""
    if t['is_nnsmith']:
        if t['bug_type'] == 'compile_crash':
            return 'nnsmith_compile_crash'
        return 'nnsmith_diff'

    if t['bug_type'] == 'compile_crash':
        if t['has_index_copy']:
            return 'index_copy_backward_crash'
        return 'other_compile_crash'

    if t['has_reciprocal']:
        if t['has_grad']:
            return 'reciprocal_backward_fusion'
        return 'reciprocal_fusion'

    if t['has_softmax']:
        if t['has_grad']:
            return 'softmax_backward_fusion'
        return 'softmax_fusion'

    if t['has_layer_norm']:
        return 'layernorm_fusion'

    if t['has_scan']:
        return 'scan_fusion'

    if t['has_fp16_roundtrip']:
        return 'fp16_roundtrip_fusion'

    if t['has_grad']:
        return 'backward_fusion_diff'

    if t['dtype'] == 'float64':
        return 'f64_fusion_diff'

    return 'f32_fusion_precision'


def main():
    # For each mode: parse triggers, classify, get discovery time
    all_data = {}

    for mode in MODES:
        trigger_dir = os.path.join(RESULTS, "triggers", mode)
        ckpt_file = os.path.join(RESULTS, f"{mode}.ckpt")

        if not os.path.isdir(trigger_dir):
            continue

        # Parse all triggers
        triggers = []
        for f in sorted(os.listdir(trigger_dir)):
            if f.endswith('.py'):
                t = parse_trigger_meta(os.path.join(trigger_dir, f))
                t['fp'] = is_false_positive(t)
                if not t['fp']:
                    t['root_cause'] = get_root_cause(t)
                triggers.append(t)

        # Load timeseries for time mapping (iter -> time_sec)
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        ts = ckpt.get('timeseries', [])
        iter_to_time = {}
        for p in ts:
            iter_to_time[p['iter']] = p['time_sec']

        # Build time curve: at each trigger's iteration, count cumulative real bugs
        real_triggers = [t for t in triggers if not t['fp']]
        real_triggers.sort(key=lambda t: t['iter'])

        # Root-cause dedup over time
        seen_causes = set()
        seen_fingerprints = set()
        curve = []  # (time_sec, cumulative_root_causes, cumulative_fingerprints)

        for t in real_triggers:
            # Find closest time
            best_time = 0
            for it in sorted(iter_to_time.keys()):
                if it <= t['iter']:
                    best_time = iter_to_time[it]
                else:
                    break

            seen_causes.add(t['root_cause'])
            seen_fingerprints.add(t['fingerprint'])
            curve.append({
                'time_sec': best_time,
                'iter': t['iter'],
                'root_causes': len(seen_causes),
                'fingerprints': len(seen_fingerprints),
                'latest_cause': t['root_cause'],
            })

        all_data[mode] = {
            'total_triggers': len(triggers),
            'fp_count': sum(1 for t in triggers if t['fp']),
            'real_count': len(real_triggers),
            'root_causes': sorted(seen_causes),
            'curve': curve,
            'max_time': max(iter_to_time.values()) if iter_to_time else 0,
        }

    # Output CSV for plotting
    csv_path = os.path.join(RESULTS, 'bug_discovery_curves.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'time_sec', 'iter', 'root_causes', 'fingerprints', 'latest_cause'])
        for mode in MODES:
            if mode not in all_data:
                continue
            short = mode.replace('_seed42', '')
            # Add initial point
            writer.writerow([short, 0, 0, 0, 0, ''])
            for pt in all_data[mode]['curve']:
                writer.writerow([short, f"{pt['time_sec']:.1f}", pt['iter'],
                                pt['root_causes'], pt['fingerprints'], pt['latest_cause']])
            # Add final point at max time
            if all_data[mode]['curve']:
                last = all_data[mode]['curve'][-1]
                max_t = all_data[mode]['max_time']
                if max_t > last['time_sec']:
                    writer.writerow([short, f"{max_t:.1f}", '', last['root_causes'],
                                    last['fingerprints'], ''])

    print(f"Saved: {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  DEDUPLICATED BUG DISCOVERY SUMMARY")
    print(f"{'='*70}")

    for mode in MODES:
        if mode not in all_data:
            continue
        d = all_data[mode]
        short = mode.replace('_seed42', '')
        print(f"\n  {short}: {d['total_triggers']} total, {d['fp_count']} FP, "
              f"{d['real_count']} real triggers, {len(d['root_causes'])} root causes")
        print(f"  Max time: {d['max_time']:.0f}s ({d['max_time']/3600:.1f}h)")
        print(f"  Root causes: {', '.join(d['root_causes'])}")

        # Time to first N root causes
        curve = d['curve']
        for n in [1, 2, 3, 5]:
            for pt in curve:
                if pt['root_causes'] >= n:
                    print(f"    Time to {n} root causes: {pt['time_sec']:.0f}s "
                          f"(iter {pt['iter']}, cause: {pt['latest_cause']})")
                    break
            else:
                print(f"    Time to {n} root causes: not reached")

    # Cross-mode table
    all_causes = sorted(set(
        c for d in all_data.values() for c in d['root_causes']
    ))

    print(f"\n{'='*70}")
    print(f"  ROOT CAUSE CROSS-MODE TABLE")
    print(f"{'='*70}")
    shorts = {m: m.replace('_seed42', '') for m in MODES}
    header = f"  {'Root Cause':<35s}"
    for m in MODES:
        if m in all_data:
            header += f"  {shorts[m]:>10s}"
    print(header)
    print(f"  {'-'*35}" + "".join(f"  {'-'*10}" for m in MODES if m in all_data))

    for cause in all_causes:
        row = f"  {cause:<35s}"
        for m in MODES:
            if m in all_data:
                if cause in all_data[m]['root_causes']:
                    # Count triggers for this cause
                    count = sum(1 for pt in all_data[m]['curve'] if pt['latest_cause'] == cause)
                    row += f"  {'Y':>10s}"
                else:
                    row += f"  {'-':>10s}"
        print(row)

    print(f"  {'-'*35}" + "".join(f"  {'-'*10}" for m in MODES if m in all_data))
    row = f"  {'TOTAL':<35s}"
    for m in MODES:
        if m in all_data:
            row += f"  {len(all_data[m]['root_causes']):>10d}"
    print(row)


if __name__ == '__main__':
    main()
