#!/usr/bin/env python3
import glob, importlib.util, itertools, json, os, re, sys, collections

ART = os.environ.get('ARTIFACT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
KF = os.path.join(ART, 'kernelfuzz')
sys.path.insert(0, KF)


def load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace('/', '_')[:-3], os.path.join(KF, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


L = []
w = L.append
R = {}

ss = load('kernel/state_space.py').KernelStateSpace
tg = load('targeted_generator.py')
gops = load('generation/ops.py')
gcond = load('generation/conditions.py')
derived = json.load(open('pt_targets_derived.json'))
w('# Derived targets vs. submitted artifact\n')
w(f"Artifact: this repository. PyTorch {derived['torch_version']}.\n")
w('## PyTorch/Inductor\n')
fam_a, fam_d = ss.KERNEL_FAMILIES, derived['families']
R['pt_families'] = {'artifact': fam_a, 'derived': fam_d, 'same': sorted(fam_a) == sorted(fam_d),
                    'feature_flag': derived.get('feature_flag_families', []), 'no_generator': derived['no_generator']}
w(f"**Families.** Derived {len(fam_d)}, artifact {len(fam_a)}; identical: {sorted(fam_a) == sorted(fam_d)}. "
  f"Behind a documented feature flag (off by default): {derived.get('feature_flag_families', [])}. "
  f"No generator: {derived['no_generator']}.\n")

rows_all = [r for f in sorted(glob.glob('pt_entries_[0-9]*.json')) for r in json.load(open(f))]
rows_all += json.load(open('pt_composition.json')) + [r for r in json.load(open('pt_dtypes.json'))['support'] if 'views_agree' in r]
checked = [r['views_agree'] for r in rows_all if 'views_agree' in r]
R['tracer_views_agree'] = [sum(checked), len(checked)]
tpl = json.load(open('pt_composition_templates.json'))
w(f"**Matcher.** In the entry, composition and dtype probes, the value `_get_heuristic` returned and the kernel "
  f"attributes read by the artifact's worker named the same implementation in {sum(checked)}/{len(checked)} "
  f"compilations.\n")

st = json.load(open('pt_target_states.json'))
der_st = {tuple(x) for x in st['states']}
art_st = {(f, d, nf) for f, d, _, nf in ss().get_uncovered_states()}
miss = sorted(art_st - der_st)
miss_gpu = all(d not in st['gpu_dtypes'].get(f, []) for f, d, _ in miss)
R['pt_target_states'] = {'derived': len(der_st), 'artifact': len(art_st), 'derived_subset': der_st <= art_st,
                         'artifact_not_derived': [list(x) for x in miss], 'all_missing_not_produced_on_gpu': miss_gpu}
w(f"**Target states.** Formed on {st['device']} (step 09: family × dtype × fusion-group size): {len(der_st)}. "
  f"The artifact's state space (`KernelStateSpace`) has {len(art_st)}; every formed state is in it: {der_st <= art_st}. "
  f"Its other {len(miss)} states use dtypes for which this GPU produces no kernel "
  f"({sorted({d for _, d, _ in miss})}; step 08): {miss_gpu}.\n")

pools = json.load(open('pt_entry_pools.json'))
FAM_GROUP = {'pointwise': 'pointwise', 'reduction': 'reduction', 'persistent_reduction': 'reduction',
             'cooperative_reduction': 'reduction', 'scan': 'scan', 'split_scan': 'scan',
             'extern_gemm': 'extern_gemm', 'extern_conv': 'extern_conv'}
OPINFO = {'relu': 'nn.functional.relu', 'silu': 'nn.functional.silu', 'gelu': 'nn.functional.gelu',
          'linear': 'nn.functional.linear', 'layer_norm': 'nn.functional.layer_norm',
          'conv2d': 'nn.functional.conv2d', 'conv1d': 'nn.functional.conv1d'}


def entry_check(family_ops):
    rows = []
    for fam, p in family_ops.items():
        items = p['unary'] + [b[0] for b in p['binary']] if isinstance(p, dict) else [x[0] for x in p]
        for it in items:
            name = OPINFO.get(it.split('.')[-1], it.split('.')[-1])
            rows.append((fam, it, name in pools['pools'].get(FAM_GROUP[fam], [])))
    return rows


w(f"**Entries.** Rule-derived pools: {pools['pool_sizes']} ({pools['distinct_operators']} distinct operators, "
  f"from {sum(pools['funnel'].values())} OpInfo samples; stopped at rules {dict((k, v) for k, v in pools['funnel'].items() if k != 'pass')}).\n")
for label, fo in (('targeted_generator.FAMILY_OPS', tg.FAMILY_OPS), ('generation/ops.FAMILY_OPS', gops.FAMILY_OPS)):
    rows = entry_check(fo)
    inside = sum(r[2] for r in rows)
    R.setdefault('pt_entries', {})[label] = {'inside': inside, 'total': len(rows), 'outside': [r[:2] for r in rows if not r[2]]}
    w(f"- `{label}`: {inside}/{len(rows)} entries inside the derived pool of their family"
      + (f"; outside: {[r[:2] for r in rows if not r[2]]}" if inside < len(rows) else '') + '.')
w('')

cond = json.load(open('pt_conditions.json'))
R['pt_switch_points'] = {k: v['switch'] for k, v in cond.items() if 'switch' in v}
w('**Conditions.** Measured switch points (last value of the old implementation | first of the new): '
  + '; '.join(f"{k} {v['switch'][0]}|{v['switch'][1]}" for k, v in cond.items() if 'switch' in v) + '.\n')

comp = json.load(open('pt_composition.json'))
fused = {(r['first'], r['second']): r['fused'] for r in comp}
R['pt_composition'] = {'fuse': [f'{a}->{b}' for (a, b), f in fused.items() if f], 'separate': [f'{a}->{b}' for (a, b), f in fused.items() if not f]}
w('**Composition.** Pairs that fuse into one kernel: '
  + ', '.join(f'{a}→{b}' for (a, b), f in fused.items() if f) + '; separate: '
  + ', '.join(f'{a}→{b}' for (a, b), f in fused.items() if not f) + '.\n')
for name, label in (('view', 'Through a view (A, flatten, B)'), ('fan_out_merge', 'Fan-out and merge (A(x) + B(x))')):
    rows = [r for r in tpl['templates'] if r['template'] == name]
    w(f"{label}: fuse: " + ', '.join(f"{r['first']}→{r['second']}" for r in rows if r['fused'])
      + '; separate: ' + (', '.join(f"{r['first']}→{r['second']}" for r in rows if not r['fused']) or 'none') + '.\n')
ve = tpl['view_entries']
joined = sorted(r['op'] + (f"[{r['variant']}]" if r['variant'] else '') for r in ve if r['fused'])
w(f"Shape-changing pointwise entries (every entry rule holds except the same-shape clause of R4): {len(joined)} of "
  f"{len(ve)} join a reduction through a view in one kernel: {', '.join(joined)}.\n")
src = open(os.path.join(KF, 'generation', 'compose.py')).read()
pool_op = 'nn.functional.' + re.search(r'def gen_pool_flatten_reduce.*?torch\.nn\.functional\.(\w+)\(', src, re.S).group(1)
tfused = {(r['template'], r['first'], r['second']): r['fused'] for r in tpl['templates']}
PATTERN = {'linear_chain': ('direct pointwise→pointwise', fused[('pointwise', 'pointwise')]),
           'reduce_after_pw': ('direct pointwise→reduction', fused[('pointwise', 'reduction')]),
           'branch_merge': ('fan-out and merge pointwise+pointwise', tfused[('fan_out_merge', 'pointwise', 'pointwise')]),
           'pool_flatten_reduce': (f'{pool_op} (a shape-changing entry) through a view into a reduction',
                                   pool_op in joined)}
R['pt_composition_patterns'] = {}
w('Composition patterns of the artifact (`generation/ops.COMPOSITION_PATTERNS`):\n')
for p in gops.COMPOSITION_PATTERNS:
    if p in PATTERN:
        desc, ok = PATTERN[p]
        R['pt_composition_patterns'][p] = {'derived_as': desc, 'fuses': ok}
        w(f"- `{p}`: {desc}; fuses into one kernel: {ok}")
    else:
        R['pt_composition_patterns'][p] = {'derived_as': None}
        w(f"- `{p}`: the backward context (a program context, not a composition of operators)")
w('')

ks = load('tensorflow/kernel_state.py')
gg = load('tensorflow/graph_gen.py')
tft = json.load(open('tf_targets.json'))
w(f"\n## TensorFlow/XLA ({tft['tf_version']})\n")
der_fam = {op: f for f, ops in tft['families'].items() for op in ops}
art_fam = ks._OP_TO_FAMILY
agree = sum(der_fam.get(o) == f for o, f in art_fam.items())
conf = collections.Counter((f, der_fam.get(o, 'not traced')) for o, f in art_fam.items() if der_fam.get(o) != f)
R['tf_families'] = {'agree': agree, 'total': len(art_fam), 'differences': {f'{a}->{b}': n for (a, b), n in conf.items()}}
w(f"**Operator families.** {agree}/{len(art_fam)} artifact operators get the same family from the rules; differences "
  f"(artifact → rule): {dict(R['tf_families']['differences'])}.\n")
w(f"**Dtype classes.** Derived {tft['dtype_classes']}; artifact {ks.DTYPE_CLASSES}; identical: "
  f"{sorted(tft['dtype_classes']) == sorted(ks.DTYPE_CLASSES)}. GPU registration share: {tft['gpu_dtype_share']}.\n")
w(f"**Execution seams.** Derived {list(tft['execution_seams'])}; artifact {ks.EXECUTION_SEAMS}; identical: "
  f"{sorted(tft['execution_seams']) == sorted(ks.EXECUTION_SEAMS)}.\n")
w(f"**Input regimes.** Derived {list(tft['input_regimes'])}; artifact {ks.SHAPE_REGIMES}; identical: "
  f"{sorted(tft['input_regimes']) == sorted(ks.SHAPE_REGIMES)}. Size-branch constants in the reduction kernels: "
  f"{tft['size_branch_constants']}.\n")

R['tf_catalog'] = {'dtype_classes_identical': sorted(tft['dtype_classes']) == sorted(ks.DTYPE_CLASSES),
                   'execution_seams_identical': sorted(tft['execution_seams']) == sorted(ks.EXECUTION_SEAMS),
                   'input_regimes_identical': sorted(tft['input_regimes']) == sorted(ks.SHAPE_REGIMES)}
tfs = json.load(open('tf_target_states.json'))
art_tf = ks.TFKernelStateSpace()
R['tf_target_states'] = {'combinations': tfs['combinations'], 'derived': tfs['target_states'], 'artifact': art_tf.total_reachable,
                         'same': {tuple(x) for x in tfs['states']} == set(art_tf.reachable_states)}
json.dump(R, open('REPORT.json', 'w'), indent=1, default=list)
print(json.dumps(R, indent=1, default=list))
