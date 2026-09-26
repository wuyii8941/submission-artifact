#!/usr/bin/env python3
import glob, json, collections
probe = {(r['op'], r['variant']): r for f in sorted(glob.glob('pt_entries_[0-9]*.json')) for r in json.load(open(f))}
meta = {(r['op'], r['variant']): r for r in json.load(open('pt_entry_metadata.json'))}
GROUP_OF = {'extern_gemm': 'extern_gemm', 'extern_conv': 'extern_conv', 'scan': 'scan', 'split_scan': 'scan',
            'reduction': 'reduction', 'persistent_reduction': 'reduction', 'cooperative_reduction': 'reduction'}
PRIORITY = ['extern_gemm', 'extern_conv', 'scan', 'reduction']


def group(kinds):
    groups = {GROUP_OF[k] for k in kinds if k in GROUP_OF}
    if 'aten_fallback' in kinds or (not groups and kinds != ['pointwise']):
        return None
    if not groups:
        return 'pointwise'
    return groups.pop() if len(groups) == 1 else None


def verdict(key):
    p, m = probe[key], meta.get(key, {})
    name = key[0]
    if p['status'] != 'ok':
        return 'R1', None
    if name.startswith('_') or name.startswith('jiterator') or name.startswith('torch.ops'):
        return 'R2', None
    g = group(p['kinds'])
    if g is None:
        return 'R3', None
    if m.get('status') != 'ok' or not (m['input_float'] and m['output_float'] and m['same_dtype']) \
            or m['n_tensor_inputs'] > 3 or (g in ('pointwise', 'scan') and not m['same_shape']):
        return 'R4', g
    if not m['deterministic']:
        return 'R5', g
    return 'pass', g


results = {k: verdict(k) for k in probe}
funnel = collections.Counter(v[0] for v in results.values())
pools = collections.defaultdict(set)
for (name, _), (v, g) in results.items():
    if v == 'pass':
        pools[g].add(name)
out = {'funnel': {k: funnel[k] for k in ('R1', 'R2', 'R3', 'R4', 'R5', 'pass')},
       'pools': {g: sorted(v) for g, v in pools.items()}, 'pool_sizes': {g: len(v) for g, v in pools.items()},
       'distinct_operators': len(set().union(*pools.values()))}
json.dump(out, open('pt_entry_pools.json', 'w'), indent=1)
print('OpInfo entries:', len(probe), '| stopped at rule:', {k: funnel[k] for k in ('R1', 'R2', 'R3', 'R4', 'R5')},
      '| pass:', funnel['pass'])
print('pool sizes:', out['pool_sizes'], '| distinct operators:', out['distinct_operators'])
