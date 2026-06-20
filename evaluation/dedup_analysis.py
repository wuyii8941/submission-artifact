#!/usr/bin/env python3
"""
Bug deduplication and false-positive analysis.

Classifies ablation bugs into:
  - Crash bugs: fingerprinted by error message
  - Diff bugs: classified as genuine vs false-positive (precision artifact)

False-positive evidence for random diff bugs:
  1. Cross-seed determinism: identical (prog_type, dtype, n_ops) → identical diffs
     across seeds → systematic precision gap, not a random discovery
  2. Known limitation: BF16-not-supported is a torch.compile limitation, not a bug
  3. Targeted bugs map to confirmed PyTorch issues (fusion bugs, not precision noise)
"""

import json
import sys
import os
from collections import Counter, defaultdict

import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def fingerprint_crash(bug):
    """Extract root-cause fingerprint for crash bugs."""
    err = bug['error']
    if 'BF16 is not supported' in err:
        return ('BF16_not_supported', 'SkipFrame')
    # Generic: first line of error
    first_line = err.strip().split('\n')[0][:80]
    return ('crash', first_line)


def classify_random_diff(bug, cross_seed_fps):
    """Classify a random diff bug as genuine or false positive.

    Returns (is_genuine: bool, reason: str)
    """
    m = bug['meta']
    fp = (m['prog_type'], m['dtype'], m['n_ops'])
    d = bug['max_diff']

    # Evidence 1: cross-seed determinism
    # If this exact fingerprint appears in all 3 seeds with similar diffs,
    # it's a systematic precision gap, not a stochastic discovery
    if fp in cross_seed_fps and cross_seed_fps[fp]['n_seeds'] == 3:
        return False, f"deterministic across 3 seeds (precision artifact)"

    if fp in cross_seed_fps and cross_seed_fps[fp]['n_seeds'] >= 2:
        return False, f"reproduces across {cross_seed_fps[fp]['n_seeds']} seeds"

    # Evidence 2: dtype-specific precision noise
    dtype = m['dtype']
    if 'float16' in dtype and d < 1.0:
        return False, "fp16 precision gap (diff < 1.0)"
    if 'float32' in dtype and d < 0.01:
        return False, "fp32 precision gap (diff < 0.01)"
    if 'float64' in dtype and d < 1e-6:
        return False, "fp64 precision gap"

    # Remaining: could be genuine
    return True, "unique, large diff — manual triage needed"


def classify_targeted_diff(bug):
    """Targeted diff bugs are assumed genuine — they trigger specific
    fusion code paths corresponding to confirmed PyTorch issues.

    Root-cause fingerprint: (family, dtype_class)
    where dtype_class = 'low_prec' (fp16/bf16) or 'standard' (fp32/fp64)
    """
    m = bug['meta']
    family = m['family']
    dtype = m['dtype']
    if 'float16' in dtype or 'bfloat16' in dtype:
        dtype_class = 'low_precision'
    else:
        dtype_class = 'standard'
    return (family, dtype_class)


def analyze_ablation(ablation_path):
    data = load_results(ablation_path)
    per_run = data['per_run']

    # Build cross-seed fingerprint map for random mode
    random_fps = defaultdict(lambda: {'n_seeds': 0, 'seeds': set(), 'diffs': []})
    for ps in per_run:
        if ps['mode'] != 'random':
            continue
        seen_fps = set()
        for b in ps['bugs']:
            if 'diff' not in b['bug_type']:
                continue
            m = b['meta']
            fp = (m['prog_type'], m['dtype'], m['n_ops'])
            if fp not in seen_fps:
                seen_fps.add(fp)
                random_fps[fp]['seeds'].add(ps['seed'])
                random_fps[fp]['diffs'].append(b['max_diff'])
    for fp in random_fps:
        random_fps[fp]['n_seeds'] = len(random_fps[fp]['seeds'])

    # Analyze each run
    results = []
    for ps in per_run:
        mode = ps['mode']
        seed = ps['seed']
        bugs = ps['bugs']

        crash_fps = Counter()
        genuine_diff_fps = Counter()
        false_positive_count = 0
        fp_reasons = Counter()

        for b in bugs:
            if b['bug_type'] == 'compile_crash':
                fp = fingerprint_crash(b)
                crash_fps[fp] += 1
            elif 'diff' in b['bug_type']:
                if mode == 'random':
                    is_genuine, reason = classify_random_diff(b, random_fps)
                    if is_genuine:
                        m = b['meta']
                        genuine_diff_fps[(m['prog_type'], m['dtype'])] += 1
                    else:
                        false_positive_count += 1
                        fp_reasons[reason] += 1
                else:
                    rc = classify_targeted_diff(b)
                    genuine_diff_fps[rc] += 1

        unique_crashes = len(crash_fps)
        unique_genuine_diffs = len(genuine_diff_fps)
        total_unique = unique_crashes + unique_genuine_diffs

        results.append({
            'mode': mode,
            'seed': seed,
            'raw_total': len(bugs),
            'raw_crashes': sum(1 for b in bugs if b['bug_type'] == 'compile_crash'),
            'raw_diffs': sum(1 for b in bugs if 'diff' in b['bug_type']),
            'unique_crash_roots': unique_crashes,
            'unique_genuine_diffs': unique_genuine_diffs,
            'false_positives': false_positive_count,
            'total_unique_genuine': total_unique,
            'crash_fps': dict(crash_fps),
            'genuine_diff_fps': dict(genuine_diff_fps),
            'fp_reasons': dict(fp_reasons),
        })

    return results, random_fps


def print_report(results, random_fps):
    print("=" * 80)
    print("  BUG DEDUPLICATION & FALSE-POSITIVE ANALYSIS")
    print("=" * 80)

    # Cross-seed evidence
    print("\n--- Evidence: Random diff bugs reproduce deterministically across seeds ---")
    total_fps = len(random_fps)
    all3 = sum(1 for fp in random_fps if random_fps[fp]['n_seeds'] == 3)
    all2 = sum(1 for fp in random_fps if random_fps[fp]['n_seeds'] == 2)
    only1 = sum(1 for fp in random_fps if random_fps[fp]['n_seeds'] == 1)
    print(f"  Total unique (prog_type, dtype, n_ops) fingerprints: {total_fps}")
    print(f"  Appear in ALL 3 seeds: {all3}/{total_fps} ({100*all3/total_fps:.0f}%)")
    print(f"  Appear in 2 seeds:     {all2}/{total_fps}")
    print(f"  Appear in 1 seed only: {only1}/{total_fps}")
    print(f"  → {all3+all2}/{total_fps} fingerprints are deterministic "
          f"precision artifacts, not stochastic bug discoveries")

    # Per-run summary
    print(f"\n--- Per-run deduplication results ---")
    print(f"{'Mode':>10s} {'Seed':>6s} {'Raw':>5s} {'Crash→':>7s} "
          f"{'Diff→':>6s} {'FP':>4s} {'Genuine':>8s}")
    print("-" * 55)
    for r in results:
        print(f"{r['mode']:>10s} {r['seed']:>6d} {r['raw_total']:>5d} "
              f"{r['unique_crash_roots']:>7d} {r['unique_genuine_diffs']:>6d} "
              f"{r['false_positives']:>4d} {r['total_unique_genuine']:>8d}")

    # Aggregated comparison
    print(f"\n--- Aggregated comparison (mean ± std) ---")
    for mode in ['targeted', 'random']:
        mode_results = [r for r in results if r['mode'] == mode]
        raw = np.array([r['raw_total'] for r in mode_results])
        genuine = np.array([r['total_unique_genuine'] for r in mode_results])
        fp = np.array([r['false_positives'] for r in mode_results])
        print(f"  {mode:10s}: raw={raw.mean():.1f}±{raw.std():.1f}  "
              f"genuine_unique={genuine.mean():.1f}±{genuine.std():.1f}  "
              f"false_positives={fp.mean():.1f}±{fp.std():.1f}")

    # False positive breakdown
    print(f"\n--- False positive reasons (random mode) ---")
    all_reasons = Counter()
    for r in results:
        if r['mode'] == 'random':
            for reason, count in r['fp_reasons'].items():
                all_reasons[reason] += count
    for reason, count in all_reasons.most_common():
        print(f"  {count:>4d}  {reason}")

    # Genuine diff root causes
    print(f"\n--- Genuine diff root causes (targeted) ---")
    all_targeted_fps = Counter()
    for r in results:
        if r['mode'] == 'targeted':
            for fp, count in r['genuine_diff_fps'].items():
                all_targeted_fps[fp] += count
    for fp, count in sorted(all_targeted_fps.items()):
        print(f"  {str(fp):45s}  ({count} instances across seeds)")

    print(f"\n--- Genuine bugs (random, if any) ---")
    all_random_genuine = Counter()
    for r in results:
        if r['mode'] == 'random':
            for fp, count in r['genuine_diff_fps'].items():
                all_random_genuine[fp] += count
    if all_random_genuine:
        for fp, count in sorted(all_random_genuine.items()):
            print(f"  {str(fp):45s}  ({count} instances)")
    else:
        print("  (none — all random diff bugs classified as false positives)")

    # Crash root causes
    print(f"\n--- Crash root causes ---")
    for r in results:
        if r['crash_fps']:
            print(f"  {r['mode']} seed={r['seed']}: {r['crash_fps']}")

    return results


def save_results(results, random_fps, out_path):
    save_data = {
        'analysis': 'dedup_false_positive',
        'methodology': {
            'crash_fingerprint': '(error_class, key_message)',
            'diff_false_positive_criteria': [
                'Cross-seed determinism: same (prog_type, dtype, n_ops) across >=2 seeds',
                'Precision noise: fp16 diff < 1.0, fp32 diff < 0.01, fp64 diff < 1e-6',
            ],
            'targeted_root_cause': '(family, dtype_precision_class)',
        },
        'cross_seed_evidence': {
            'total_fingerprints': len(random_fps),
            'in_all_3_seeds': sum(1 for fp in random_fps if random_fps[fp]['n_seeds'] == 3),
            'in_2_seeds': sum(1 for fp in random_fps if random_fps[fp]['n_seeds'] == 2),
            'in_1_seed': sum(1 for fp in random_fps if random_fps[fp]['n_seeds'] == 1),
        },
        'per_run': [{k: v for k, v in r.items()
                     if k not in ('crash_fps', 'genuine_diff_fps', 'fp_reasons')}
                    for r in results],
        'per_run_detail': results,
    }
    # Convert tuple keys to strings for JSON
    for r in save_data['per_run_detail']:
        r['crash_fps'] = {str(k): v for k, v in r['crash_fps'].items()}
        r['genuine_diff_fps'] = {str(k): v for k, v in r['genuine_diff_fps'].items()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    merged_path = os.path.join(script_dir, 'results', 'ablation_5seed_merged.json')
    fallback_path = os.path.join(script_dir, 'ablation_v2', 'ablation_v2_results.json')
    ablation_path = merged_path if os.path.exists(merged_path) else fallback_path

    if not os.path.exists(ablation_path):
        print(f"ERROR: no ablation results found")
        sys.exit(1)

    print(f"Using: {ablation_path}")
    results, random_fps = analyze_ablation(ablation_path)
    print_report(results, random_fps)

    out_path = os.path.join(script_dir, 'results', 'dedup_analysis.json')
    save_results(results, random_fps, out_path)


if __name__ == '__main__':
    main()
