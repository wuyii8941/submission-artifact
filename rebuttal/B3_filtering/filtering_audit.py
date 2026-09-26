#!/usr/bin/env python3
import ast
import json
import os
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path(os.environ.get('B3_DATA', HERE / 'data')) / 'rq2_runs'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
SEEDS = (42, 123, 456)
MODES = (('targeted', 'DeepOpFuzz'), ('random', 'Random'))


def load_classifier():
    source = REPO / 'evaluation' / 'rq2_efficiency' / 'plot_kd_rootcause.py'
    module = ast.parse(source.read_text())
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name in ('parse_trigger', 'classify')]
    namespace = {'re': re, 'os': __import__('os')}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['parse_trigger'], namespace['classify']


def main():
    parse_trigger, classify = load_classifier()
    rows, fp_reasons = [], Counter()
    for mode, name in MODES:
        for seed in SEEDS:
            d = json.loads((RUNS / f'{mode}_seed{seed}.json').read_text())
            assert d['mode'] == mode and d['seed'] == seed and d['time_budget'] == 14400
            assert d['raw_triggers'] == d['bugs_found'] == len(d['bugs'])
            labels = {}
            for path in (RUNS / 'triggers' / f'{mode}_seed{seed}').glob('*.py'):
                match = re.search(r'# fingerprint: (\S+)', path.read_text())
                assert match and match.group(1) not in labels, path
                labels[match.group(1)] = classify(parse_trigger(str(path)))
            assert len(labels) == d['unique_bugs'] and set(labels) == set(d['bug_fingerprints'])
            fp = [reason for category, reason in labels.values() if category == 'FP']
            if mode == 'targeted':
                fp_reasons.update(fp)
            rows.append({'generator': name, 'seed': seed, 'attempts': d['iterations_completed'],
                         'candidate_failures': d['raw_triggers'], 'deduplicated_candidates': d['unique_bugs'],
                         'bug_labeled': len(labels) - len(fp), 'fp_labeled': len(fp)})
    cols = ('attempts', 'candidate_failures', 'deduplicated_candidates', 'bug_labeled', 'fp_labeled')
    mean = {name: {c: round(sum(r[c] for r in rows if r['generator'] == name) / len(SEEDS), 1) for c in cols}
            for _, name in MODES}
    for _, name in MODES:
        fp = sum(r['fp_labeled'] for r in rows if r['generator'] == name)
        dedup = sum(r['deduplicated_candidates'] for r in rows if r['generator'] == name)
        mean[name]['filtered_as_false_positives'] = round(fp / dedup, 3)
    low, total = fp_reasons['low_precision'], sum(fp_reasons.values())
    out = {'pytorch': '2.2.0', 'seeds': list(SEEDS), 'budget_seconds': 14400,
           'classifier': 'evaluation/rq2_efficiency/plot_kd_rootcause.py', 'runs': rows, 'mean_over_seeds': mean,
           'deepopfuzz_false_positives': {'total': total, 'fp16_bf16_numerical': low, 'share': round(low / total, 3)}}
    (OUT / 'filter_audit.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
