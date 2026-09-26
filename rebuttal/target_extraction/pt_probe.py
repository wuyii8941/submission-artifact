#!/usr/bin/env python3
import json, os, signal, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pylib'))
import torch
import pt_tracer

pt_tracer.install()
torch._inductor.config.force_disable_caches = True
DEV = 'cuda'


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def observe(fn, *args, **kwargs):
    pt_tracer.reset()
    torch._dynamo.reset()
    signal.alarm(120)
    try:
        torch.compile(fn, backend='inductor', fullgraph=False)(*args, **kwargs)
        torch.cuda.synchronize()
    finally:
        signal.alarm(0)
    kinds, agree = pt_tracer.observed()
    return kinds, agree, list(pt_tracer.kernel_names)


def entries(shard, nshards, out):
    from torch.testing._internal.common_methods_invocations import op_db
    torch._inductor.config.unroll_reductions_threshold = 1
    rows = []
    for i, op in enumerate(op_db):
        if i % nshards != shard:
            continue
        row = {'op': op.name, 'variant': op.variant_test_name}
        try:
            if torch.float32 not in op.supported_dtypes(DEV):
                row['status'] = 'no_cuda_fp32'
            else:
                samples = []
                for j, smp in enumerate(op.sample_inputs(DEV, torch.float32, requires_grad=False)):
                    if j >= 20:
                        break
                    samples.append(smp)
                s = max(samples, key=lambda x: x.input.numel() if isinstance(x.input, torch.Tensor) else 0)
                kinds, agree, _ = observe(lambda *a, **k: op.op(*a, **k), s.input, *s.args, **s.kwargs)
                row.update(status='ok', kinds=kinds, views_agree=agree)
        except Exception as e:
            row.update(status='error', error=f'{type(e).__name__}: {str(e)[:120]}')
        rows.append(row)
        json.dump(rows, open(out, 'w'), indent=0)


def bisect(make, lo, hi, predicate):
    def test(n):
        fn, args = make(n)
        return predicate(observe(fn, *args)[0])
    assert not test(lo) and test(hi), (lo, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if test(mid) else (mid, hi)
    return [lo, hi]


def row_sum(rows):
    return lambda n: (lambda x: x.sum(dim=-1), (torch.randn(rows, n, device=DEV),))


def conditions(out):
    non_persistent = lambda k: 'reduction' in k and 'persistent_reduction' not in k
    probes = {
        'persistent_inner': ('persistent -> non-persistent reduction, inner, rows=128 (no split)',
                             row_sum(128), 2, 65536, non_persistent),
        'persistent_outer': ('persistent -> non-persistent reduction, outer, cols=64',
                             lambda n: (lambda x: x.sum(dim=0), (torch.randn(n, 64, device=DEV),)), 2, 65536,
                             non_persistent),
        'split_scan_length': ('scan -> split scan, length, rows=4',
                              lambda n: (lambda x: torch.cumsum(x, dim=1), (torch.randn(4, n, device=DEV),)), 16, 65536,
                              lambda k: 'split_scan' in k),
        'split_reduction': ('single reduction -> split reduction (two kernels), inner, rows=1',
                            row_sum(1), 2048, 1 << 17, lambda k: 'reduction' in k and 'persistent_reduction' in k),
        'split_scan_rows': ('split scan -> scan, rows, length=16384',
                            lambda n: (lambda x: torch.cumsum(x, dim=1), (torch.randn(n, 16384, device=DEV),)), 1, 256,
                            lambda k: 'split_scan' not in k),
    }
    results = {}
    for key, (desc, make, lo, hi, pred) in probes.items():
        try:
            results[key] = {'description': desc, 'switch': bisect(make, lo, hi, pred)}
        except Exception as e:
            results[key] = {'description': desc, 'error': f'{type(e).__name__}: {str(e)[:120]}'}
        print(key, results[key], flush=True)
    torch._inductor.config.triton.cooperative_reductions = True
    for rows in (1, 2, 4):
        key = f'cooperative_rows{rows}'
        desc = f'reduction -> cooperative reduction, length, rows={rows}, feature flag on'
        try:
            results[key] = {'description': desc,
                            'switch': bisect(row_sum(rows), 1024, 1 << 20, lambda k: 'cooperative_reduction' in k)}
        except Exception as e:
            results[key] = {'description': desc, 'error': f'{type(e).__name__}: {str(e)[:120]}'}
        print(key, results[key], flush=True)
    json.dump(results, open(out, 'w'), indent=1)


def composition(out):
    x = torch.randn(8, 512, device=DEV); w = torch.randn(512, 256, device=DEV)
    img = torch.randn(2, 8, 16, 16, device=DEV); k = torch.randn(8, 8, 3, 3, device=DEV)
    conv = lambda a: torch.nn.functional.conv2d(a, k, padding=1)
    pairs = {
        ('pointwise', 'pointwise'): (lambda a: torch.sin(a).exp(), x),
        ('pointwise', 'reduction'): (lambda a: torch.sin(a).sum(dim=-1), x),
        ('reduction', 'pointwise'): (lambda a: a.sum(dim=-1).sin(), x),
        ('pointwise', 'scan'): (lambda a: torch.cumsum(torch.sin(a), dim=1), x),
        ('scan', 'pointwise'): (lambda a: torch.cumsum(a, dim=1).sin(), x),
        ('reduction', 'reduction'): (lambda a: a.sum(dim=-1).sum(), x),
        ('pointwise', 'extern_gemm'): (lambda a: torch.sin(a) @ w, x),
        ('extern_gemm', 'pointwise'): (lambda a: (a @ w).sin(), x),
        ('pointwise', 'extern_conv'): (lambda a: conv(a.relu()), img),
        ('extern_conv', 'pointwise'): (lambda a: conv(a).relu(), img),
    }
    results = []
    for (first, second), (fn, arg) in pairs.items():
        kinds, agree, names = observe(fn, arg)
        triton = [n for n in names if not n.startswith('extern')]
        extern = [n for n in names if n.startswith('extern')]
        fused = not extern and len(triton) == 1
        results.append({'first': first, 'second': second, 'fused': fused, 'triton_kernels': len(triton),
                        'extern_calls': len(extern), 'kinds': kinds, 'views_agree': agree})
        print(first, '->', second, 'fused' if fused else 'separate', kinds, flush=True)
    json.dump(results, open(out, 'w'), indent=1)


def fused_row(fn, *args):
    kinds, agree, names = observe(fn, *args)
    triton = [n for n in names if not n.startswith('extern')]
    extern = [n for n in names if n.startswith('extern')]
    return {'fused': not extern and len(triton) == 1, 'triton_kernels': len(triton), 'extern_calls': len(extern),
            'kinds': kinds, 'views_agree': agree}


def templates(out):
    import glob
    torch._inductor.config.unroll_reductions_threshold = 1
    x4 = torch.randn(2, 8, 16, 16, device=DEV)
    rep = {'pointwise': lambda t: t.sin(), 'reduction': lambda t: t.sum(dim=-1),
           'scan': lambda t: torch.cumsum(t, dim=-1)}
    keep = {'pointwise': lambda t: t.sin(), 'reduction': lambda t: t.sum(dim=-1, keepdim=True),
            'scan': lambda t: torch.cumsum(t, dim=-1)}
    rows = []
    for a in rep:
        for b in rep:
            for name, fn in (('view', lambda t, a=a, b=b: rep[b](rep[a](t).flatten(1))),
                             ('fan_out_merge', lambda t, a=a, b=b: keep[a](t) + keep[b](t))):
                row = {'template': name, 'first': a, 'second': b}
                try:
                    row.update(fused_row(fn, x4))
                except Exception as e:
                    row.update(fused=False, error=f'{type(e).__name__}: {str(e)[:120]}')
                rows.append(row)
                print(name, a, '->', b, 'fused' if row['fused'] else 'separate', flush=True)
    from torch.testing._internal.common_methods_invocations import op_db
    probe = {(r['op'], r['variant']): r for f in sorted(glob.glob('pt_entries_[0-9]*.json')) for r in json.load(open(f))}
    meta = {(r['op'], r['variant']): r for r in json.load(open('pt_entry_metadata.json'))}
    def shape_changing(k):
        p, m = probe[k], meta.get(k, {})
        return (p['status'] == 'ok' and p['kinds'] == ['pointwise'] and not k[0].startswith(('_', 'jiterator', 'torch.ops'))
                and m.get('status') == 'ok' and m['input_float'] and m['output_float'] and m['same_dtype']
                and m['n_tensor_inputs'] <= 3 and not m['same_shape'] and m['deterministic'])
    wanted = {k for k in probe if shape_changing(k)}
    view_rows = []
    for op in op_db:
        if (op.name, op.variant_test_name) not in wanted:
            continue
        row = {'op': op.name, 'variant': op.variant_test_name}
        try:
            samples = []
            for j, smp in enumerate(op.sample_inputs(DEV, torch.float32, requires_grad=False)):
                if j >= 20:
                    break
                samples.append(smp)
            s = max(samples, key=lambda x: x.input.numel() if isinstance(x.input, torch.Tensor) else 0)
            def joined(*a, **k):
                y = op.op(*a, **k)
                return y.flatten(1).sum(dim=-1) if y.dim() >= 2 else y.sum()
            row.update(fused_row(joined, s.input, *s.args, **s.kwargs))
        except Exception as e:
            row.update(fused=False, error=f'{type(e).__name__}: {str(e)[:120]}')
        view_rows.append(row)
        print(op.name, op.variant_test_name, 'fused' if row['fused'] else 'separate', flush=True)
    json.dump({'templates': rows, 'view_entries': view_rows}, open(out, 'w'), indent=1)


def dtypes(cond_path, out):
    c = json.load(open(cond_path))
    inner = c['persistent_inner']['switch']; scan = c['split_scan_length']['switch']
    coop = c['cooperative_rows1']['switch']
    k = lambda dt: torch.randn(8, 8, 3, 3, device=DEV, dtype=dt)
    rep = {
        'pointwise': lambda dt: (lambda a: a.sin(), (torch.randn(64, 256, device=DEV, dtype=dt),)),
        'persistent_reduction': lambda dt: row_sum_dt(128, inner[0] // 2, dt),
        'reduction': lambda dt: row_sum_dt(128, inner[1] * 2, dt),
        'cooperative_reduction': lambda dt: row_sum_dt(1, coop[1] * 2, dt),
        'scan': lambda dt: (lambda a: torch.cumsum(a, dim=1), (torch.randn(4, scan[0] // 2, device=DEV, dtype=dt),)),
        'split_scan': lambda dt: (lambda a: torch.cumsum(a, dim=1), (torch.randn(4, scan[1] * 2, device=DEV, dtype=dt),)),
        'extern_gemm': lambda dt: (lambda a, b: a @ b, (torch.randn(128, 128, device=DEV, dtype=dt),
                                                         torch.randn(128, 128, device=DEV, dtype=dt))),
        'extern_conv': lambda dt: (lambda a, w: torch.nn.functional.conv2d(a, w, padding=1),
                                   (torch.randn(2, 8, 16, 16, device=DEV, dtype=dt), k(dt))),
    }
    results = {'device': torch.cuda.get_device_name(), 'compute_capability': list(torch.cuda.get_device_capability()),
               'support': []}
    for fam, make in rep.items():
        torch._inductor.config.triton.cooperative_reductions = fam == 'cooperative_reduction'
        for name in ('float16', 'bfloat16', 'float32', 'float64'):
            dt = getattr(torch, name)
            try:
                fn, args = make(dt)
                kinds, agree, _ = observe(fn, *args)
                results['support'].append({'family': fam, 'dtype': name, 'reached': fam in kinds,
                                           'kinds': kinds, 'views_agree': agree})
            except Exception as e:
                results['support'].append({'family': fam, 'dtype': name, 'reached': False,
                                           'error': f'{type(e).__name__}: {str(e)[:100]}'})
            print(fam, name, results['support'][-1].get('reached'), flush=True)
    torch._inductor.config.triton.cooperative_reductions = False
    json.dump(results, open(out, 'w'), indent=1)


def row_sum_dt(rows, n, dt):
    return (lambda x: x.sum(dim=-1), (torch.randn(rows, n, device=DEV, dtype=dt),))


if __name__ == '__main__':
    t0 = time.time()
    mode = sys.argv[1]
    if mode == 'entries':
        entries(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif mode == 'conditions':
        conditions(sys.argv[2])
    elif mode == 'composition':
        composition(sys.argv[2])
    elif mode == 'templates':
        templates(sys.argv[2])
    elif mode == 'dtypes':
        dtypes(sys.argv[2], sys.argv[3])
    print(f'done in {time.time() - t0:.0f}s', flush=True)
