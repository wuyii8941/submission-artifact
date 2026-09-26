#!/usr/bin/env python3
import json, math, os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'a1_runs')
SEEDS, POLICIES = (42, 123, 456), ('feedback', 'uniform')


def l25(hits, targets):
    s = sorted(int(hits.get(t, 0)) for t in targets)
    return statistics.mean(s[:math.ceil(len(s) / 4)])


rec = {(s, p): json.load(open(os.path.join(RUNS, f'a1_{p}_seed{s}_14400s_43t4.json'))) for s in SEEDS for p in POLICIES}
targets = rec[(42, 'feedback')]['targets']
snaps = defaultdict(list)
path = os.path.join(RUNS, 'attempt_snapshots.jsonl')
if os.path.exists(path):
    for line in open(path):
        r = json.loads(line); snaps[(r['seed'], r['policy'])].append(r)

first = rec[(42, 'feedback')]
out = {'targets': len(targets), 'torch_version': first['torch_version'], 'device': first['cuda_device'],
       'time_budget_seconds': first['time_budget_seconds'], 'seeds': list(SEEDS), 'runs': []}
val = defaultdict(list)
for s in SEEDS:
    for p in POLICIES:
        r = rec[(s, p)]; h = r['observed_hits_per_target']
        v = l25(h, targets); val[p].append(v)
        out['runs'].append({'seed': s, 'policy': p, 'tests': r['iterations_completed'],
                            'targets_hit_ge1': sum(int(h.get(t, 0)) >= 1 for t in targets),
                            'targets_hit_ge10': sum(int(h.get(t, 0)) >= 10 for t in targets), 'l25': round(v, 1)})
out['l25_mean'] = {p: round(statistics.mean(val[p]), 1) for p in POLICIES}
out['final_l25_gain'] = {'mean': round(statistics.mean(val['feedback']) / statistics.mean(val['uniform']) - 1, 4),
                         'per_seed': [round(f / u - 1, 4) for f, u in zip(val['feedback'], val['uniform'])]}
if snaps:
    rows = []
    for s in SEEDS:
        common = min(rec[(s, p)]['iterations_completed'] for p in POLICIES)
        row = {}
        for p in POLICIES:
            final = rec[(s, p)]
            row[p] = final if final['iterations_completed'] == common else max(
                (r for r in snaps[(s, p)] if r['iterations_completed'] <= common), key=lambda r: r['iterations_completed'])
        f, u = (l25(row[p]['observed_hits_per_target'], targets) for p in POLICIES)
        rows.append({'seed': s, 'common_tests': common, 'feedback_tests': row['feedback']['iterations_completed'],
                     'uniform_tests': row['uniform']['iterations_completed'], 'feedback_l25': round(f, 1),
                     'uniform_l25': round(u, 1), 'gain': round(f / u - 1, 4)})
    out['equal_count_l25_gain'] = {'per_seed': rows, 'mean': round(statistics.mean(r['gain'] for r in rows), 4)}
curve = []
for t in list(range(300, 14400, 300)) + [14400]:
    point = {'seconds': t}
    for p in POLICIES:
        vals = []
        for s in SEEDS:
            if t == 14400:
                row = rec[(s, p)]
            else:
                rows = sorted((r for r in snaps[(s, p)] if r['elapsed_seconds'] <= t), key=lambda r: r['elapsed_seconds'])
                row = rows[-1] if rows else None
            if row:
                vals.append(l25(row['observed_hits_per_target'], targets))
        point[p] = round(statistics.mean(vals), 1) if vals else None
        point[p + '_seeds'] = len(vals)
    curve.append(point)
out['l25_over_time'] = curve
out['time_until_all_targets_confirmed'] = {}
for p in POLICIES:
    hit = [next(x for x in rec[(s, p)]['timeseries'] if x['intended_target_coverage'] >= len(targets)) for s in SEEDS]
    secs = [x['time_seconds'] for x in hit]
    out['time_until_all_targets_confirmed'][p] = {
        'tests_per_seed': [x['iteration'] for x in hit], 'seconds_per_seed': [round(v, 1) for v in secs],
        'seconds_mean': round(statistics.mean(secs), 1), 'seconds_sd': round(statistics.stdev(secs), 1)}
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'results', 'summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
