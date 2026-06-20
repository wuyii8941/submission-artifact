#!/usr/bin/env python3
"""Plot KD and root-cause discovery curves for RQ2 seed 42."""
import csv
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

TRIGGERS = "results_5seeds/triggers"
CKPT_DIR = "results_5seeds"
MODES = ["targeted", "random", "nnsmith", "whitefox"]
MODE_LABELS = {
    "targeted": "KernelFuzz (targeted)",
    "random": "Random",
    "nnsmith": "NNSmith",
    "whitefox": "WhiteFox",
}
MODE_COLORS = {
    "targeted": "#E63946",
    "random": "#457B9D",
    "nnsmith": "#2A9D8F",
    "whitefox": "#E9C46A",
}
SEED = 42


def parse_trigger(path):
    with open(path) as f:
        text = f.read()
    info = {'filename': os.path.basename(path)}
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
    info['has_reciprocal'] = has(r'reciprocal|rsqrt')
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
    info['has_noncontig'] = has(r'transpose|permute|\.t\(\)|as_strided|narrow')
    info['has_conditional'] = has(r'torch\.where|torch\.clamp')
    info['has_view'] = has(r'\.view\(|\.reshape\(')
    info['has_erf'] = has(r'torch\.erf')
    info['has_floor_ceil'] = has(r'torch\.floor|torch\.ceil|torch\.round|torch\.trunc')
    info['has_hardswish'] = has(r'hardswish|hardsigmoid')
    info['has_softplus'] = has(r'softplus')
    return info


def classify(t):
    dtype = t['dtype']
    bt = t['bug_type']
    diff = t['max_diff']
    grad = t['has_grad']
    if bt == 'compile_crash' and dtype == 'bfloat16':
        return 'FP', 'bf16_hw'
    if t['has_dropout'] or t['has_random']:
        return 'FP', 'nondeterministic'
    if dtype in ('float16', 'bfloat16'):
        return 'FP', 'low_precision'
    if t['has_fp16_roundtrip'] and t['has_exp'] and diff > 100:
        return 'FP', 'fp16_exp_overflow'
    if t['has_matmul'] and t['has_exp'] and diff > 100:
        return 'FP', 'matmul_exp_overflow'
    if t['has_exp'] and diff > 10:
        return 'FP', 'exp_amplification'
    if t['has_scan'] and diff == float('inf'):
        return 'FP', 'scan_inf'
    if t['is_nnsmith']:
        if bt == 'compile_crash':
            return 'BUG', 'nnsmith_compile_crash'
        return 'BUG', 'nnsmith_diff'
    if bt == 'compile_crash':
        if t['has_index_copy']:
            return 'BUG', 'index_copy_crash'
        return 'BUG', 'other_compile_crash'
    if t['has_reciprocal']:
        if grad: return 'BUG', 'reciprocal_backward'
        return 'BUG', 'reciprocal_fusion'
    if t['has_softmax']:
        if grad: return 'BUG', 'softmax_backward'
        return 'BUG', 'softmax_fusion'
    if t['has_layer_norm']:
        return 'BUG', 'layernorm_fusion'
    if t['has_scan']:
        return 'BUG', 'scan_fusion'
    if t['has_fp16_roundtrip']:
        return 'BUG', 'fp16_roundtrip'
    if t['has_noncontig']:
        if grad: return 'BUG', 'noncontig_backward'
        return 'BUG', 'noncontig_fusion'
    if t['has_conditional']:
        return 'BUG', 'conditional_fusion'
    if t['has_view']:
        return 'BUG', 'view_fusion'
    if t['has_erf']:
        return 'BUG', 'erf_fusion'
    if t['has_floor_ceil']:
        return 'BUG', 'floor_ceil_fusion'
    if t['has_hardswish']:
        return 'BUG', 'hardswish_fusion'
    if t['has_softplus']:
        return 'BUG', 'softplus_fusion'
    if grad:
        return 'BUG', 'backward_diff'
    if dtype == 'float64':
        return 'BUG', 'f64_fusion'
    if diff > 0.01:
        return 'BUG', 'f32_precision'
    return 'FP', 'tiny_diff'


def build_rootcause_curve(mode):
    """Build (time_sec, n_root_causes) curve from triggers + checkpoint timeseries."""
    d = os.path.join(TRIGGERS, f'{mode}_seed{SEED}')
    if not os.path.isdir(d):
        return [], []

    triggers = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.py'):
            triggers.append(parse_trigger(os.path.join(d, f)))

    bug_triggers = []
    for t in triggers:
        cat, cause = classify(t)
        if cat == 'BUG':
            bug_triggers.append((t['iter'], cause))

    bug_triggers.sort(key=lambda x: x[0])

    ckpt = os.path.join(CKPT_DIR, f'{mode}_seed{SEED}.ckpt')
    ts = json.load(open(ckpt)).get('timeseries', [])
    iter_to_time = {}
    for p in ts:
        iter_to_time[p['iter']] = p['time_sec']

    def iter_to_sec(it):
        if it in iter_to_time:
            return iter_to_time[it]
        closest = min(iter_to_time.keys(), key=lambda x: abs(x - it))
        if closest == 0:
            return 0
        return iter_to_time[closest] * it / closest

    causes_seen = set()
    curve_time = [0.0]
    curve_count = [0]
    for it, cause in bug_triggers:
        causes_seen.add(cause)
        t = iter_to_sec(it)
        curve_time.append(t)
        curve_count.append(len(causes_seen))

    if ts:
        final_t = ts[-1]['time_sec']
        curve_time.append(final_t)
        curve_count.append(len(causes_seen))

    return curve_time, curve_count


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.85), sharex=True)

    # --- Left: KD over time ---
    kd_csv = 'data_kd.csv'
    if os.path.exists(kd_csv):
        rows = list(csv.DictReader(open(kd_csv)))
        times = [float(r['time_h']) for r in rows]
        for mode in ["targeted", "random", "nnsmith"]:
            values = [float(r[mode]) for r in rows]
            ax1.plot(times, values, color=MODE_COLORS[mode], label=MODE_LABELS[mode],
                     linewidth=2.0 if mode == "targeted" else 1.7)
    else:
        for mode in ["targeted", "random", "nnsmith"]:
            ckpt = os.path.join(CKPT_DIR, f'{mode}_seed{SEED}.ckpt')
            if not os.path.exists(ckpt):
                continue
            ts = json.load(open(ckpt)).get('timeseries', [])
            times = [p['time_sec'] / 3600 for p in ts]
            kds = [p['kernel_diversity'] for p in ts]
            ax1.plot(times, kds, color=MODE_COLORS[mode], label=MODE_LABELS[mode],
                     linewidth=2.0 if mode == "targeted" else 1.7)

    ax1.set_xlabel('Time (hours)', fontsize=8.5)
    ax1.set_ylabel('Artifact signatures $\\mathcal{D}(t)$', fontsize=8.5)
    ax1.set_title('(a) Artifact diversity', fontsize=9.5, pad=4)
    ax1.grid(True, alpha=0.22, linewidth=0.6)
    ax1.set_xlim(0, 4)
    ax1.tick_params(axis='both', labelsize=8)

    # --- Right: Root causes over time ---
    root_csv = 'data_rootcause.csv'
    if os.path.exists(root_csv):
        rows = list(csv.DictReader(open(root_csv)))
        times = [float(r['time_h']) for r in rows]
        for mode in MODES:
            values = [float(r[mode]) for r in rows]
            ax2.step(times, values, where='post', color=MODE_COLORS[mode],
                     label=MODE_LABELS[mode], linewidth=2.0 if mode == "targeted" else 1.7)
            final = int(values[-1])
            ax2.text(4.03, values[-1], str(final), va='center', ha='left',
                     fontsize=7.5, color=MODE_COLORS[mode])
    else:
        for mode in MODES:
            times, counts = build_rootcause_curve(mode)
            if not times:
                continue
            times_h = [t / 3600 for t in times]
            ax2.step(times_h, counts, where='post', color=MODE_COLORS[mode],
                     label=MODE_LABELS[mode], linewidth=2.0 if mode == "targeted" else 1.7)
            ax2.text(4.03, counts[-1], str(counts[-1]), va='center', ha='left',
                     fontsize=7.5, color=MODE_COLORS[mode])

    ax2.set_xlabel('Time (hours)', fontsize=8.5)
    ax2.set_ylabel('Distinct root causes', fontsize=8.5)
    ax2.set_title('(b) Root-cause discovery', fontsize=9.5, pad=4)
    ax2.grid(True, alpha=0.22, linewidth=0.6)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-0.2, 15)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.04), columnspacing=1.4,
               handlelength=2.0)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.19, top=0.82, wspace=0.32)
    out = 'rq2_kd_rootcause_seed42.pdf'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved to {out}')

    out_png = 'rq2_kd_rootcause_seed42.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved to {out_png}')


if __name__ == '__main__':
    main()
