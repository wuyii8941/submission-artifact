#!/usr/bin/env python3
import json, os, re
import oracle


def rec(bug, mode):
    f = os.path.join(oracle.run_dir(oracle.ORACLE[bug][0], mode), bug + '.json')
    r = json.load(open(f))
    so = open(f[:-5] + '.stdout.txt').read()
    k = re.search(r"'kernels': (\d+)", r['per_op_stats'] or '')
    return r, so, int(k.group(1)) if k else None


rows = []
for bug, (build, _, _) in oracle.ORACLE.items():
    assert oracle.manifests(bug, 'baseline') and oracle.manifests(bug, 'nested_whole'), bug
    r, so, kern = rec(bug, 'per_op')
    _, so_w, _ = rec(bug, 'nested_whole')
    new = [l for l in so.splitlines() if re.search(r'CRASH|Error', l) and l not in so_w.splitlines()]
    if oracle.manifests(bug, 'per_op'):
        cls = 'not composition-dependent'
    elif r['rc'] == 0 and kern and not new:
        cls = 'composition-dependent'
    else:
        cls = 'not applicable (per-operator run failed: ' + ((r['error'] or new[0]).strip()[:60]) + ')'
    rows.append((bug, build, kern, cls))
count = lambda p: sum(c.startswith(p) for *_, c in rows)
out = {'bugs': 31, 'runtime_bugs_reproduced': len(rows), 'composition_dependent': count('composition'),
       'not_composition_dependent': count('not composition'), 'not_applicable': count('not applicable'),
       'excluded': len(oracle.EXCLUDED),
       'per_bug': [{'bug': b, 'build': bd, 'per_op_kernels': k, 'result': c} for b, bd, k, c in rows],
       'excluded_bugs': [{'bug': b, 'reason': why} for b, why in oracle.EXCLUDED.items()]}
os.makedirs(os.path.join(oracle.HERE, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(oracle.HERE, 'results', 'results.json'), 'w'), indent=1)
print({k: v for k, v in out.items() if not isinstance(v, list)})
