#!/usr/bin/env python3
import ast, itertools, json, os

ART = os.environ.get('ARTIFACT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
SRC = os.path.join('kernelfuzz', 'tensorflow', 'kernel_state.py')


def reachability():
    path = os.path.join(ART, SRC)
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_is_reachable']
    ns = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), path, 'exec'), ns)
    return ns['_is_reachable']


t = json.load(open('tf_targets.json'))
dims = [sorted(t['families']), list(t['dtype_classes']), list(t['execution_seams']), list(t['input_regimes'])]
ok = reachability()
combos = list(itertools.product(*dims))
states = [list(c) for c in combos if ok(*c)]
out = {'dimensions': {'families': dims[0], 'dtype_classes': dims[1], 'execution_modes': dims[2], 'input_regimes': dims[3]},
       'combinations': len(combos), 'reachability': SRC.replace(os.sep, '/') + ':_is_reachable', 'target_states': len(states),
       'states': states}
json.dump(out, open('tf_target_states.json', 'w'), indent=1)
print(len(combos), 'combinations,', len(states), 'target states')
