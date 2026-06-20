#!/usr/bin/env python3
"""Extract cross-seed union curve of root causes vs time for WhiteFox.
Produces step-function points for the Figure (a) tikz curve."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_dedup_5seeds import classify_final, parse_trigger  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
TRIGGER_ROOT = os.path.join(RESULT_DIR, 'triggers')
SEEDS = [42, 123, 456, 789, 1024]


def iter_to_time_map(log_file):
    """Map iteration number -> elapsed_seconds for one seed."""
    with open(log_file) as f:
        ts = json.loads(
            [l for l in f if l.strip().startswith('{')][-1]
        ).get('timeseries', [])
    # ts: list of {iter, time_sec}
    return [(e['iter'], e['time_sec']) for e in ts]


def time_at_iter(it_time_pairs, target_iter):
    """Interpolate time at given iter using checkpoint pairs."""
    if not it_time_pairs:
        return 0
    # Linear between bracketing checkpoints
    prev_iter, prev_time = 0, 0
    for it, t in it_time_pairs:
        if it >= target_iter:
            if it == prev_iter:
                return t
            frac = (target_iter - prev_iter) / (it - prev_iter)
            return prev_time + frac * (t - prev_time)
        prev_iter, prev_time = it, t
    # Past last checkpoint: extrapolate from rate
    return prev_time


def main():
    # Gather (time_h, cause) discovery events across all seeds
    events = []
    for seed in SEEDS:
        trigger_dir = os.path.join(TRIGGER_ROOT, f'whitefox_seed{seed}')
        log_file = os.path.join(RESULT_DIR, f'whitefox_seed{seed}.log')
        it_time = iter_to_time_map(log_file)

        for fname in sorted(os.listdir(trigger_dir)):
            if not fname.endswith('.py'):
                continue
            m = re.search(r'iter(\d+)', fname)
            if not m:
                continue
            it = int(m.group(1))
            info = parse_trigger(os.path.join(trigger_dir, fname))
            cat, cause = classify_final(info)
            if cat == 'BUG':
                t_sec = time_at_iter(it_time, it)
                events.append((t_sec / 3600.0, cause, seed, it))

    # Sort events by time
    events.sort(key=lambda x: x[0])

    # Build cumulative union curve
    union = set()
    curve = [(0.0, 0)]  # (time_h, union_size)
    for t_h, cause, seed, it in events:
        if cause not in union:
            union.add(cause)
            # Step function: keep previous size up to this point
            curve.append((t_h, len(union) - 1))
            curve.append((t_h, len(union)))
            print(f't={t_h:.2f}h: union grew to {len(union)} (new: {cause}, seed={seed}, iter={it})')

    # Extend to 4h
    curve.append((4.0, len(union)))

    # Print tikz coordinates
    print()
    print('Tikz coordinates for panel (a) WhiteFox step curve:')
    coords = ' -- '.join(f'({t:.2f},{n})' for t, n in curve)
    print(coords)

    print(f'\nFinal union: {len(union)} causes')


if __name__ == '__main__':
    main()
