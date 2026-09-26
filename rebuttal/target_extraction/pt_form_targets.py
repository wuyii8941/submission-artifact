#!/usr/bin/env python3
import ast, json, os

ART = os.environ.get('ARTIFACT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
FUSION_SIZES = ['1', '2-3', '4+']
GROUP = {'pointwise': 'pointwise', 'reduction': 'reduction', 'persistent_reduction': 'reduction',
         'cooperative_reduction': 'reduction', 'scan': 'scan', 'split_scan': 'scan',
         'extern_gemm': 'extern_gemm', 'extern_conv': 'extern_conv'}


def generator_input_dtypes():
    tree = ast.parse(open(os.path.join(ART, 'kernelfuzz', 'kernel', 'coverage.py')).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == '_DTYPES' for t in node.targets):
            return [ast.literal_eval(e).replace('torch.', '') for e in node.value.elts]
    raise SystemExit('_DTYPES not found in kernelfuzz/kernel/coverage.py')


def generator_family_dtypes():
    tree = ast.parse(open(os.path.join(ART, 'kernelfuzz', 'targeted_generator.py')).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_targeted_program')
    out = {}

    def families(test):
        if (isinstance(test, ast.Compare) and getattr(test.left, 'id', None) == 'family' and len(test.ops) == 1):
            if isinstance(test.ops[0], ast.Eq):
                return [ast.literal_eval(test.comparators[0])]
            if isinstance(test.ops[0], ast.In):
                return list(ast.literal_eval(test.comparators[0]))
        return None

    for node in ast.walk(fn):
        if isinstance(node, ast.If) and families(node.test):
            for sub in node.body:
                for n in ast.walk(sub):
                    if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Constant)
                            and isinstance(n.test.ops[0], ast.In) and getattr(n.test.comparators[0], 'id', None) == 'dtype_str'):
                        for f in families(node.test):
                            out.setdefault(f, set()).add(n.test.left.value)
    return {f: sorted(d) for f, d in out.items()}


fams = json.load(open('pt_targets_derived.json'))['families']
dt = json.load(open('pt_dtypes.json'))
comp = json.load(open('pt_composition.json'))
gpu = {}
for r in dt['support']:
    if r['reached']:
        gpu.setdefault(r['family'], set()).add(r['dtype'])
fusing = {g for r in comp if r['fused'] and 'pointwise' in (r['first'], r['second'])
          for g in (r['first'], r['second'])}
gen_in, gen_fam = generator_input_dtypes(), generator_family_dtypes()
states, per_family = [], {}
for f in fams:
    dtypes = sorted(gpu.get(f, set()) & (set(gen_in) | set(gen_fam.get(f, []))))
    sizes = FUSION_SIZES if GROUP[f] in fusing else FUSION_SIZES[:1]
    per_family[f] = {'dtypes': dtypes, 'sizes': sizes}
    states += [[f, d, s] for d in dtypes for s in sizes]
out = {'device': dt['device'], 'compute_capability': dt['compute_capability'], 'fusion_sizes': FUSION_SIZES,
       'fusing_groups': sorted(fusing), 'gpu_dtypes': {f: sorted(v) for f, v in gpu.items()},
       'generator_input_dtypes': gen_in, 'generator_family_dtypes': gen_fam, 'per_family': per_family, 'states': states}
json.dump(out, open('pt_target_states.json', 'w'), indent=1)
print(f"{len(states)} target states on {dt['device']}; generator dtypes {gen_in} + {gen_fam}; fusing groups {sorted(fusing)}")
