#!/usr/bin/env python3
import collections, json, sys

FLOATS = ['float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128']
cand = json.load(open('tf_candidates.json'))
sig = {o['op']: o['signature'] for o in cand['ops']}


def raw_op_pool():
    funnel, pool = collections.Counter(), []
    for o in cand['ops']:
        if 'GPU' not in o['devices']:
            funnel['no GPU kernel'] += 1
        elif not set(o['gpu_dtypes']) & set(FLOATS):
            funnel['no float/complex GPU kernel'] += 1
        elif o['stateful']:
            funnel['stateful (random, resource, I/O)'] += 1
        else:
            funnel['pass'] += 1
            pool.append(o)
    return funnel, pool


def family(r):
    sigs = {sig.get(o, 'other') for o in r['core_ops']}
    same = r['out_shape'] == [4, 16]
    if 'linalg' in sigs:
        return 'linalg'
    if 'reduction' in sigs or 'scan' in sigs:
        return 'normalize' if same and r['coupling'] == 'row' else 'reduction'
    if same and r['coupling'] == 'row':
        return 'normalize'
    if same and r['coupling'] == 'local' and 'Cast' not in r['core_ops']:
        return 'arithmetic' if r['binary'] else 'activation'
    return 'structural'


def families():
    trace = json.load(open('tf_trace.json'))
    fam = collections.defaultdict(list)
    for r in trace:
        if r['status'] == 'ok':
            fam[family(r)].append(r['op'])
    json.dump({f: sorted(v) for f, v in fam.items()}, open('tf_families.json', 'w'), indent=1)
    print('families:', {f: len(v) for f, v in fam.items()})


def catalog():
    funnel, pool = raw_op_pool()
    fams = json.load(open('tf_families.json'))
    counted = [o for o in pool if o['signature'] != 'other']
    share = {d: sum(d in o['gpu_dtypes'] for o in counted) / len(counted) for d in FLOATS}
    dtypes = [d for d in FLOATS if share[d] > 0 and d not in ('float16', 'bfloat16')]
    paths = cand['execution_paths']
    seams = {'eager_vs_grappler': f"tf.function applies Grappler ({len(paths['grappler_optimizers'])} RewriterConfig optimizers)",
             'eager_vs_xla': paths['xla'], 'cpu_vs_gpu': 'device placement (CPU, GPU)',
             'backward': paths['gradient'], 'mixed_precision': f"policies {paths['mixed_precision_policies'][1:]}"}
    branches = json.load(open('tf_shape_branches.json'))
    consts = sorted({b['constant'] for b in branches})
    regimes = {'standard': 'no size branch taken specially',
               'degenerate': [f"{b['file']}:{b['line']} if ({b['condition']})" for b in branches
                              if b['constant'] in ('0', '1') and b['op'] == '=='],
               'boundary': [f"{b['file']}:{b['line']} if ({b['condition']})" for b in branches
                            if b['constant'] not in ('0', '1', '2', '3')],
               'magnitude': 'value regime for the numeric oracles (overflow, underflow, cancellation); not a size branch'}
    out = {'tf_version': cand['tf_version'], 'raw_op_funnel': dict(funnel), 'raw_op_pool_size': len(pool),
           'families': fams, 'gpu_dtype_share': {d: round(share[d], 2) for d in FLOATS}, 'dtype_classes': dtypes,
           'execution_seams': seams, 'size_branch_constants': consts, 'input_regimes': regimes}
    json.dump(out, open('tf_targets.json', 'w'), indent=1)
    print(f"raw-op pool {len(pool)} of {len(cand['ops'])} | families {({f: len(v) for f, v in fams.items()})}")
    print(f"dtype classes {dtypes} | seams {list(seams)} | size-branch constants {consts}")


if __name__ == '__main__':
    {'families': families, 'catalog': catalog}[sys.argv[1]]()
