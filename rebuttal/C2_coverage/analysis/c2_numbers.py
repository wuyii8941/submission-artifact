#!/usr/bin/env python3
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'c2_runs'
ARMS = ('deepopfuzz', 'random_boundary', 'random')
SEEDS = (42, 123, 456)
DEC = ('persistent_reduction', 'cooperative_reduction', 'split_reduction', 'split_scan')
SCOPE = ('_inductor/codegen/triton.py', '_inductor/choices.py', '_inductor/codegen/triton_split_scan.py',
         '_inductor/select_algorithm.py', '_inductor/kernel/mm.py', '_inductor/kernel/bmm.py', '_inductor/kernel/conv.py')


def records(arm, seed):
    return [json.loads(l) for l in open(RUN / f'{arm}_seed{seed}.jsonl')]


def switch_points(rs):
    progs, at = defaultdict(set), 0
    for r in rs:
        hit = False
        for d in DEC:
            v = r['decisions'].get(d)
            for e in (v or {}).get('adjacent') or []:
                hit = True
                progs[d + ':' + '|'.join(e.split('|')[:-1])].add(r['code_sha1'])
        at += hit
    return at, len(progs), sum(len(p) >= 10 for p in progs.values())


out = {'arms': list(ARMS), 'seeds': list(SEEDS), 'targets': 43, 'runs': []}
for seed in SEEDS:
    for arm in ARMS:
        rs = records(arm, seed)
        reached = set().union(*(set(r.get('targets_observed') or []) for r in rs))
        at, points, points10 = switch_points(rs)
        out['runs'].append({'seed': seed, 'arm': arm, 'programs': len(rs), 'targets_reached': len(reached),
                            'programs_at_switch_point': at, 'share_at_switch_point': round(at / len(rs), 5),
                            'switch_points': points, 'switch_points_ge10_programs': points10})

cov = json.load(open(HERE / 'results' / 'coverage_per_file.json'))
total = {}
for pts in cov.values():
    for rep in pts.values():
        for f, sm in rep.items():
            total[f] = max(total.get(f, 0), sm['num_branches'])
denom = sum(n for f, n in total.items() if f.endswith(SCOPE))
for row in out['runs']:
    pts = cov[f"{row['arm']}_seed{row['seed']}"]
    n = max(pts, key=int)
    covered = sum(sm['covered_branches'] for f, sm in pts[n].items() if f.endswith(SCOPE))
    row['coverage_programs'] = int(n)
    row['selection_code_branches'] = [covered, denom]
    row['selection_code_branch_coverage'] = round(covered / denom, 6)

mean = lambda arm, k: statistics.mean(r[k] for r in out['runs'] if r['arm'] == arm)
out['mean'] = {arm: {'targets_reached_share': round(mean(arm, 'targets_reached') / 43, 4),
                     'share_at_switch_point': round(mean(arm, 'share_at_switch_point'), 5),
                     'selection_code_branch_coverage': round(mean(arm, 'selection_code_branch_coverage'), 6)}
               for arm in ARMS}
(HERE / 'results').mkdir(exist_ok=True)
json.dump(out, open(HERE / 'results' / 'c2_numbers.json', 'w'), indent=1)
print(json.dumps(out['mean'], indent=1))
