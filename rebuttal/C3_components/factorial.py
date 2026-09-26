#!/usr/bin/env python3
import ast, glob, itertools, json, os, re, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.environ.get('C3_RUNS', os.path.join(HERE, 'data', 'runs'))


class _Rules:
    def __init__(self, path):
        tree = ast.parse(open(path).read())
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('parse_trigger', 'classify')]
        ns = {'re': re, 'os': os}
        exec(compile(ast.Module(body=fns, type_ignores=[]), path, 'exec'), ns)
        self.parse_trigger, self.classify = ns['parse_trigger'], ns['classify']


R = _Rules(os.path.join(REPO, 'evaluation', 'rq2_efficiency', 'plot_kd_rootcause.py'))
N = {42: 22437, 123: 13445, 456: 13621}
SEEDS = (42, 123, 456)
CELLS = {'none': ('L0', (0, 0, 0)), 'E': ('L1', (1, 0, 0)), 'C': ('C', (0, 1, 0)), 'P': ('P', (0, 0, 1)),
         'E+C': ('L2', (1, 1, 0)), 'E+P': ('EP', (1, 0, 1)), 'C+P': ('CP', (0, 1, 1)), 'E+C+P': ('Full', (1, 1, 1))}
NAMES = ('Path entries', 'Conditions', 'Composition constraints')


def categories(tag, seed):
    best = {}
    for f in glob.glob(os.path.join(RUNS, f'{tag}_seed{seed}_part*', 'triggers', '*', '*.py')):
        t = R.parse_trigger(f)
        fp = re.search(r'# fingerprint: (\S+)', open(f).read())
        k = fp.group(1) if fp else f
        if k not in best or t['iter'] < best[k][0]['iter']:
            best[k] = (t, f)
    cats = {}
    for t, f in best.values():
        if t['iter'] >= N[seed]:
            continue
        t2 = dict(t); t2['has_view'] = bool(re.search(r'\.view\(|\.reshape\(', t['code'].replace('.reshape(-1)', '')))
        c, cause = R.classify(t2)
        if c == 'BUG':
            cats.setdefault(cause, []).append(os.path.relpath(f, RUNS))
    return cats


def effects(y):
    out = {}
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(3), k):
            sign = lambda f: 1 if sum(1 - f[i] for i in combo) % 2 == 0 else -1
            per = [statistics.mean(y[(c, s)] for c, (_, f) in CELLS.items() if sign(f) == 1)
                   - statistics.mean(y[(c, s)] for c, (_, f) in CELLS.items() if sign(f) == -1) for s in SEEDS]
            out['×'.join('ECP'[i] for i in combo)] = per
    return out


def main():
    if not glob.glob(os.path.join(RUNS, '*_seed*_part*')):
        raise SystemExit(f'no runs under {RUNS}')
    sets = {(c, s): categories(tag, s) for c, (tag, _) in CELLS.items() for s in SEEDS}
    y = {k: len(v) for k, v in sets.items()}
    mean = lambda c: statistics.mean(y[(c, s)] for s in SEEDS)
    out = {'pytorch': '2.2.0', 'triton': '2.2.0', 'device': 'Tesla T4', 'seeds': list(SEEDS),
           'programs_per_seed': [N[s] for s in SEEDS],
           'benchmark_bugs': {c: {'per_seed': [y[(c, s)] for s in SEEDS], 'mean': round(mean(c), 2)} for c in CELLS},
           'component_added': {}, 'effects': {}}
    for i, name in enumerate(NAMES):
        letter = 'ECP'[i]; others = [x for x in 'ECP' if x != letter]
        key = lambda parts: '+'.join(x for x in 'ECP' if x in parts) or 'none'
        d = lambda base: round(mean(key(base + [letter])) - mean(key(base)), 2)
        row = {'alone': d([]), f'with_{others[0]}': d([others[0]]), f'with_{others[1]}': d([others[1]]),
               'with_both': d(others)}
        row['average'] = round(statistics.mean(row.values()), 2)
        out['component_added'][name] = row
    for k, per in effects(y).items():
        out['effects'][k] = {'per_seed': [round(v, 2) for v in per], 'mean': round(statistics.mean(per), 2),
                             'sd': round(statistics.stdev(per), 2)}
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'results.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
