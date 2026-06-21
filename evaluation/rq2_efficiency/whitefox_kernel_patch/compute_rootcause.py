#!/usr/bin/env python3
"""Compute root-cause counts for the patched WhiteFox runs at 2h and 4h.
Reuses final_dedup_5seeds.classify_final() for the per-trigger BUG/FP +
root-cause label, then dedups per seed by cause."""
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_dedup_5seeds import classify_final, parse_trigger  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
TRIGGER_ROOT = os.path.join(RESULT_DIR, 'triggers')
SEEDS = [42, 123, 456, 789, 1024]


def main():
    rc_2h, rc_4h = [], []
    union_4h = set()
    union_2h = set()
    per_seed_dump = []

    for seed in SEEDS:
        trigger_dir = os.path.join(TRIGGER_ROOT, f'whitefox_seed{seed}')
        log_file = os.path.join(RESULT_DIR, f'whitefox_seed{seed}.log')
        if not os.path.isdir(trigger_dir):
            print(f'seed={seed}: SKIP (no trigger dir)')
            continue
        if not os.path.exists(log_file):
            print(f'seed={seed}: SKIP (no log)')
            continue

        # Find iter at 2h checkpoint from timeseries
        with open(log_file) as f:
            json_lines = [l.strip() for l in f if l.strip().startswith('{')]
        r = json.loads(json_lines[-1])
        ts = r.get('timeseries', [])
        iter_at_2h = max(
            (e['iter'] for e in ts if e.get('time_sec', 0) <= 7200),
            default=0,
        )

        bugs_4h_causes = set()
        bugs_2h_causes = set()
        triggers_in = sorted(os.listdir(trigger_dir))
        n_total, n_bug, n_fp = 0, 0, 0
        for fname in triggers_in:
            if not fname.endswith('.py'):
                continue
            m = re.search(r'iter(\d+)', fname)
            if not m:
                continue
            it = int(m.group(1))
            info = parse_trigger(os.path.join(trigger_dir, fname))
            cat, cause = classify_final(info)
            n_total += 1
            if cat == 'BUG':
                n_bug += 1
                bugs_4h_causes.add(cause)
                if it <= iter_at_2h:
                    bugs_2h_causes.add(cause)
            else:
                n_fp += 1

        rc_2h.append(len(bugs_2h_causes))
        rc_4h.append(len(bugs_4h_causes))
        union_4h |= bugs_4h_causes
        union_2h |= bugs_2h_causes
        per_seed_dump.append({
            'seed': seed,
            'iter_at_2h': iter_at_2h,
            'iter_4h_total': ts[-1].get('iter') if ts else 0,
            'triggers': n_total,
            'bugs': n_bug,
            'fp': n_fp,
            'root_causes_2h': sorted(bugs_2h_causes),
            'root_causes_4h': sorted(bugs_4h_causes),
        })
        print(
            f'seed={seed}: iter@2h={iter_at_2h}, triggers={n_total} '
            f'(bug={n_bug}, fp={n_fp}), '
            f'rc@2h={len(bugs_2h_causes)} {sorted(bugs_2h_causes)}, '
            f'rc@4h={len(bugs_4h_causes)} {sorted(bugs_4h_causes)}',
        )

    print()
    if rc_2h:
        m_2h = statistics.mean(rc_2h)
        s_2h = statistics.stdev(rc_2h) if len(rc_2h) > 1 else 0
        m_4h = statistics.mean(rc_4h)
        s_4h = statistics.stdev(rc_4h) if len(rc_4h) > 1 else 0
        print(f'WhiteFox root causes:')
        print(f'  per-run mean: 2h={m_2h:.1f}+/-{s_2h:.1f}, 4h={m_4h:.1f}+/-{s_4h:.1f}')
        print(f'  per-run values: 2h={rc_2h}, 4h={rc_4h}')
        print(f'  cross-seed union: 2h={len(union_2h)} {sorted(union_2h)}, '
              f'4h={len(union_4h)} {sorted(union_4h)}')

    out_path = os.path.join(RESULT_DIR, 'rootcause_summary.json')
    with open(out_path, 'w') as f:
        json.dump({
            'per_seed': per_seed_dump,
            'mean_2h': statistics.mean(rc_2h) if rc_2h else 0,
            'mean_4h': statistics.mean(rc_4h) if rc_4h else 0,
            'std_2h': statistics.stdev(rc_2h) if len(rc_2h) > 1 else 0,
            'std_4h': statistics.stdev(rc_4h) if len(rc_4h) > 1 else 0,
            'union_2h': sorted(union_2h),
            'union_4h': sorted(union_4h),
        }, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
