#!/usr/bin/env python3
import json, os, re, shutil, subprocess, sys, tempfile, time

PRELUDE = r'''
import os, sys, contextlib, runpy, torch
from torch.fx.passes.split_module import split_module
from torch._inductor.compile_fx import compile_fx
from torch._inductor import metrics
_orig_compile = torch.compile
_stats = {"wrapper_calls": 0, "backend_calls": 0, "ops": []}

def _fake_inputs(example_inputs):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    mode = torch._guards.detect_fake_mode(example_inputs) or FakeTensorMode(allow_non_fake_inputs=True)
    fake = [mode.from_tensor(t) if isinstance(t, torch.Tensor) and not isinstance(t, FakeTensor) else t
            for t in example_inputs]
    return mode, fake

def _capture(split, example_inputs):
    mode, fake = _fake_inputs(example_inputs)
    captured = {}
    class Cap(torch.fx.Interpreter):
        def call_module(self, target, args, kwargs):
            captured.setdefault(target, list(args))
            return super().call_module(target, args, kwargs)
    with mode:
        Cap(split).run(*fake)
    return captured

class _Compiled(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, *args):
        return self.fn(*args)

def _patches(mode, options, dynamic):
    cfg = {}
    if mode is not None:
        cfg.update(torch._inductor.list_mode_options(mode, dynamic))
    if options:
        cfg.update(options)
    return cfg

def _make_backend(cfg):
    def per_op_backend(gm, example_inputs):
        _stats["backend_calls"] += 1
        ids = {}
        for n in gm.graph.nodes:
            if n.op in ("call_function", "call_method", "call_module"):
                ids[n] = len(ids)
        _stats["ops"].append(len(ids))
        with torch._inductor.config.patch(cfg):
            if len(ids) <= 1 or os.environ.get("PER_OP_SPLIT") == "0":
                return compile_fx(gm, example_inputs)
            split = split_module(gm, None, lambda n: ids.get(n, 0), keep_original_order=True)
            captured = _capture(split, example_inputs)
            for name, sub in list(split.named_children()):
                setattr(split, name, _Compiled(compile_fx(sub, captured[name])))
        return split
    return per_op_backend

def _compile(model=None, *args, backend="inductor", mode=None, options=None, dynamic=None, **kw):
    if backend != "inductor":
        return _orig_compile(model, *args, backend=backend, mode=mode, options=options, dynamic=dynamic, **kw)
    _stats["wrapper_calls"] += 1
    outer = {k: v for k, v in kw.items() if k in ("fullgraph", "disable")}
    if dynamic is not None:
        outer["dynamic"] = dynamic
    b = _make_backend(_patches(mode, options, dynamic))
    if model is None:
        return lambda m: _orig_compile(m, backend=b, **outer)
    return _orig_compile(model, backend=b, **outer)

torch.compile = _compile
import atexit
atexit.register(lambda: sys.stderr.write("[per_op] stats " + repr(dict(_stats, kernels=metrics.generated_kernel_count)) + "\n"))
'''

MARKERS = re.compile(r'BUG CONFIRMED|BUG REPRODUCED|MISMATCH|\*\*\* BUG', re.I)


def canonical(bug_dir):
    for c in ('minimal_repro.py', 'repro.py', 'reproducer.py', 'reproduce.py'):
        if os.path.exists(os.path.join(bug_dir, c)):
            return c
    py = [f for f in os.listdir(bug_dir) if f.endswith('.py')]
    return py[0] if len(py) == 1 else None


def main():
    py, bug_dir, mode, out = sys.argv[1:5]
    script = canonical(bug_dir)
    os.makedirs(os.environ.get('ATTR_SCRATCH') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch'), exist_ok=True)
    scratch = tempfile.mkdtemp(prefix='attr_', dir=os.environ.get('ATTR_SCRATCH') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch'))
    work = os.path.join(scratch, 'bug')
    shutil.copytree(bug_dir, work)
    env = dict(os.environ, PYTHONNOUSERSITE='1', TMPDIR=scratch, TORCHINDUCTOR_CACHE_DIR=os.path.join(scratch, 'ind'),
               TRITON_CACHE_DIR=os.path.join(scratch, 'tri'), CUDA_CACHE_PATH=os.path.join(scratch, 'nv'),
               TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', XDG_CACHE_HOME=os.path.join(scratch, 'xdg'))
    if mode == 'nested_whole':
        env['PER_OP_SPLIT'] = '0'
    base = ('import runpy, sys, atexit, torch\nfrom torch._inductor import metrics\n'
            'atexit.register(lambda: sys.stderr.write("[per_op] stats " + repr({"kernels": metrics.generated_kernel_count}) + "\\n"))\n')
    code = (PRELUDE if mode in ('per_op', 'nested_whole') else base) + \
        f'sys.argv = [{script!r}]\nrunpy.run_path({script!r}, run_name="__main__")\n'
    t0 = time.time()
    try:
        p = subprocess.run([py, '-c', code], cwd=work, env=env, capture_output=True, text=True, timeout=600)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = 124, (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or ''), 'TIMEOUT'
    err_lines = [l for l in se.splitlines() if re.match(r'^\w[\w.]*(Error|Exception|Fault)\b', l)]
    rec = {'bug': os.path.basename(bug_dir.rstrip('/')), 'script': script, 'mode': mode, 'rc': rc,
           'marker': bool(MARKERS.search(so)), 'error': err_lines[-1][:300] if err_lines else None,
           'per_op_stats': next((l for l in se.splitlines() if l.startswith('[per_op] stats')), None),
           'seconds': round(time.time() - t0, 1), 'stdout_tail': so[-1500:], 'stderr_tail': se[-1500:]}
    json.dump(rec, open(out, 'w'), indent=1)
    open(out[:-5] + '.stdout.txt', 'w').write(so)
    open(out[:-5] + '.stderr.txt', 'w').write(se)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
