#!/usr/bin/env python3
import ast, importlib, inspect, json, os, pkgutil, sys, time
t0 = time.time()
import torch, torch._inductor as inductor
from torch._inductor import config, ir, lowering
ROOT = os.path.dirname(inductor.__file__)

def returns_of(fn_node):
    return sorted({n.value.value for n in ast.walk(fn_node)
                   if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)})

def config_reads(fn_node):
    reads = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Attribute):
            parts, x = [], n
            while isinstance(x, ast.Attribute):
                parts.append(x.attr); x = x.value
            if isinstance(x, ast.Name) and x.id == 'config':
                reads.add('.'.join(reversed(parts)))
    return sorted(reads)

def config_default(path):
    obj = config
    try:
        for p in path.split('.'):
            obj = getattr(obj, p)
        return obj if isinstance(obj, (bool, int, float, str, type(None))) else repr(type(obj))
    except AttributeError:
        return None

heuristics, predicates = [], {}
for rel in ('codegen/triton.py', 'codegen/triton_split_scan.py', 'codegen/triton_combo_kernel.py', 'select_algorithm.py', 'choices.py', 'codegen/simd.py'):
    tree = ast.parse(open(os.path.join(ROOT, rel)).read())
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name == '_get_heuristic':
                heuristics.append({'class': cls.name, 'file': rel, 'returns': returns_of(fn), 'line': fn.lineno})
            if fn.name.startswith('should_use_'):
                reads = config_reads(fn)
                predicates.setdefault(fn.name, {'file': rel, 'class': cls.name, 'line': fn.lineno,
                                               'config_reads': {r: config_default(r) for r in reads}})
def subclasses(c):
    out = []
    for s in c.__subclasses__():
        out.append(s); out += subclasses(s)
    return out
loops = sorted({c.__name__ for c in subclasses(ir.Loops) if c.__module__ == ir.__name__})
from torch._inductor.select_algorithm import ExternKernelChoice, TritonTemplate
externs, templates = [], []
import torch._inductor.kernel as kpkg
for m in pkgutil.walk_packages(kpkg.__path__, kpkg.__name__ + '.'):
    try:
        mod = importlib.import_module(m.name)
    except Exception as e:
        continue
    for name, obj in vars(mod).items():
        if isinstance(obj, ExternKernelChoice):
            externs.append({'module': m.name.split('.')[-1], 'name': getattr(obj, 'name', name), 'kernel': str(getattr(obj, 'kernel', ''))[:60]})
        elif isinstance(obj, TritonTemplate):
            templates.append({'module': m.name.split('.')[-1], 'name': getattr(obj, 'name', name)})
externs = sorted({json.dumps(e, sort_keys=True) for e in externs}); externs = [json.loads(e) for e in externs]
templates = sorted({json.dumps(e, sort_keys=True) for e in templates}); templates = [json.loads(e) for e in templates]
fallbacks = sorted({str(k) for k in lowering.fallbacks})
backends = sorted(p[:-3] if p.endswith('.py') else p for p in os.listdir(os.path.join(ROOT, 'codegen'))
                  if any(k in p for k in ('cpp', 'halide', 'mps', 'pallas', 'mtia', 'xpu', 'rocm', 'cutlass', 'cutedsl', 'nv_universal_gemm', 'cuda')))
out = {'torch_version': torch.__version__, 'source_root': 'torch/_inductor',
       'kernel_heuristics': heuristics, 'decision_predicates': predicates, 'ir_loop_nodes': loops,
       'extern_kernel_choices': externs, 'triton_templates': templates,
       'aten_fallback_ops': fallbacks, 'non_default_codegens': backends,
       'config': {k: config_default(k) for k in ('max_autotune', 'combo_kernels', 'triton.persistent_reductions',
                                                   'triton.cooperative_reductions', 'triton.decompose_sort_ops')},
       'seconds': round(time.time() - t0, 1)}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'candidates.json', 'w'), indent=1)
print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}, indent=1))
