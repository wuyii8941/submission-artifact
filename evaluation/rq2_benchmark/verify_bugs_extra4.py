#!/usr/bin/env python3
"""
Additional bug reproducers found in exhaustive GitHub search (Round 4).
4 new bugs confirmed on PyTorch 2.2.0 beyond the original 44.
Also includes ~30 candidates that were tested and found to be PASS (fixed by 2.2.0).
"""
import torch
import torch.nn as nn
import sys
import traceback
import copy
import math
import numpy as np

try:
    torch._inductor.config.force_disable_caches = True
except AttributeError:
    pass

RESULTS = {}

def test_bug(issue_num, description, test_fn):
    try:
        torch._dynamo.reset()
        result = test_fn()
        if result is None:
            RESULTS[issue_num] = ('SKIP', description, '')
            return
        status, detail = result
        RESULTS[issue_num] = (status, description, detail)
    except Exception as e:
        RESULTS[issue_num] = ('ERROR', description, f'{type(e).__name__}: {str(e)[:200]}')


# ============================================================================
# CONFIRMED BUGS on PyTorch 2.2.0
# ============================================================================

def test_88813():
    """Inductor merges two output tensors into one"""
    if not torch.cuda.is_available():
        return None
    def g(a):
        b = a * 2
        c = a * 2
        return b, c
    x = torch.rand((1000,), device="cuda", requires_grad=True)
    eager_b, eager_c = g(x)
    compiled_g = torch.compile(g)
    comp_b, comp_c = compiled_g(x)
    if comp_b.data_ptr() == comp_c.data_ptr():
        return ('BUG', 'outputs aliased: mutation would corrupt')
    if not (torch.allclose(comp_b, eager_b) and torch.allclose(comp_c, eager_c)):
        return ('BUG', 'values wrong')
    return ('PASS', 'outputs distinct and correct')


def test_94693():
    """compile_fx_inner trailing copy_() incorrect"""
    from torch._inductor.compile_fx import compile_fx_inner
    from torch.fx.experimental.proxy_tensor import make_fx
    def f(arg0_1):
        s1 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0)
        s2 = torch.ops.aten.slice.Tensor(s1, 1, 20, 40)
        add = torch.ops.aten.add.Tensor(s2, 1)
        s5 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0)
        ss = torch.ops.aten.slice_scatter.default(s5, add, 1, 20, 40)
        ss1 = torch.ops.aten.slice_scatter.default(arg0_1, ss, 0, 0)
        s10 = torch.ops.aten.slice.Tensor(ss1, 0, 0)
        s11 = torch.ops.aten.slice.Tensor(s10, 1, 1, 10)
        add1 = torch.ops.aten.add.Tensor(s11, 2)
        s16 = torch.ops.aten.slice.Tensor(ss1, 0, 0)
        ss2 = torch.ops.aten.slice_scatter.default(s16, add1, 1, 2, 11)
        ss3 = torch.ops.aten.slice_scatter.default(ss1, ss2, 0, 0)
        torch.ops.aten.copy_.default(arg0_1, ss3)
        return ()
    x_ref = torch.ones([1, 64]); x_test = torch.ones([1, 64])
    x2 = torch.ones([1, 64]); x3 = torch.ones([1, 64])
    fx_g = make_fx(f)(x2)
    f_compiled = compile_fx_inner(fx_g, [x3])
    f(x_ref); f_compiled([x_test])
    if torch.allclose(x_test, x_ref):
        return ('PASS', 'copy_() results match')
    return ('BUG', f'max diff={torch.abs(x_test - x_ref).max().item()}')


def test_115260():
    """tan output order corruption (CPU)"""
    p0 = torch.tensor([1.0879], dtype=torch.float16)
    class M0(nn.Module):
        def forward(self, *a):
            cat = torch.cat((a[3], a[2], a[1], a[0]), dim=2)
            mx = torch.max(a[4], p0)
            mul = torch.mul(cat, mx)
            return (torch.tan(mul),)
    class M1(nn.Module):
        def forward(self, *a):
            cat = torch.cat((a[3], a[2], a[1], a[0]), dim=2)
            mx = torch.max(a[4], p0)
            mul = torch.mul(cat, mx)
            return (mul, torch.tan(mul))
    torch.manual_seed(42)
    d = [torch.randn(17, 5, 1, 7, dtype=torch.float16) for _ in range(3)]
    d.insert(2, torch.randn(17, 5, 11, 7, dtype=torch.float16))
    d.append(torch.tensor(4.39, dtype=torch.float16))
    torch._dynamo.reset()
    o0 = torch.compile(M0(), fullgraph=True, backend='inductor')(*d)
    torch._dynamo.reset()
    o1 = torch.compile(M1(), fullgraph=True, backend='inductor')(*d)
    if not torch.allclose(o0[0], o1[1], atol=1e-1, rtol=1e-1):
        diff = torch.abs(o0[0].float() - o1[1].float()).max().item()
        return ('BUG', f'tan differs: max diff={diff}')
    return ('PASS', 'outputs match')


def test_115774():
    """mutable attr captured as stale constant"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = torch.tensor([0, 0])
        @torch.compile(backend="eager")
        def forward(self, x, pos):
            self.cache[pos] = x
            return self.cache.clone()
    m = Model()
    m(torch.tensor([2]), torch.tensor([0]))
    m.cache = torch.tensor([0, 0])
    result = m(torch.tensor([1]), torch.tensor([1]))
    expected = torch.tensor([0, 1])
    if torch.equal(result, expected):
        return ('PASS', 'mutable attr correct')
    return ('BUG', f'expected={expected.tolist()} got={result.tolist()}')


# ============================================================================
# CANDIDATES TESTED AND FOUND TO BE PASS (fixed by 2.2.0)
# ============================================================================

def test_93351():
    """pad + gt wrong result float64"""
    def fn(inp):
        v1 = torch.nn.functional.pad(inp, pad=(1, 0))
        return torch.gt(v1, inp)
    x = torch.rand([1, 2, 2, 1], dtype=torch.float64)
    eager = fn(x); comp = torch.compile(fn)(x)
    if torch.equal(eager, comp):
        return ('PASS', 'pad+gt correct')
    return ('BUG', f'mismatch count={int((eager!=comp).sum())}')


def test_93365():
    """cat + mul returns zeros"""
    p0 = torch.tensor([[4.9334, 5.5571]])
    def fn():
        v7 = torch.cat([p0, p0], dim=0)
        return v7, torch.mul(v7, v7)
    eager = fn(); comp = torch.compile(fn)()
    if torch.allclose(eager[1], comp[1]):
        return ('PASS', 'cat+mul correct')
    return ('BUG', f'max diff={torch.abs(eager[1]-comp[1]).max().item()}')


def test_93374():
    """torch.where broadcast wrong"""
    torch.manual_seed(0)
    p1 = torch.rand(13, 7, 3); p0 = torch.rand(1, 1)
    def fn(x):
        return torch.where(x, p1, p0)
    x = torch.tensor([[True]])
    eager = fn(x); comp = torch.compile(fn)(x)
    if torch.allclose(eager, comp):
        return ('PASS', 'where broadcast correct')
    return ('BUG', f'max diff={torch.abs(eager-comp).max().item()}')


def test_93829():
    """neg uint8 + lt wrong"""
    def fn(v3):
        v7 = torch.neg(v3)
        v4 = torch.lt(v7, torch.tensor(4, dtype=torch.uint8))
        return v7, v4
    v3 = torch.tensor(6, dtype=torch.uint8)
    eager = fn(v3); comp = torch.compile(fn)(v3)
    if torch.equal(eager[1], comp[1]):
        return ('PASS', 'neg+lt correct')
    return ('BUG', f'eager={eager[1].item()} comp={comp[1].item()}')


def test_95695():
    """ge int vs float wrong"""
    def fn(inp):
        return torch.ge(inp, 3.5)
    x = torch.tensor([3])
    eager = fn(x); comp = torch.compile(fn)(x)
    if torch.equal(eager, comp):
        return ('PASS', 'ge correct')
    return ('BUG', f'eager={eager.item()} comp={comp.item()}')


def test_96694():
    """abs negative for typecasted tensors"""
    if not torch.cuda.is_available():
        return None
    x = torch.tensor(1, device='cuda')
    def forward():
        return torch.abs((-x).to(dtype=torch.float32))
    eager = forward(); comp = torch.compile(forward)()
    if torch.allclose(eager, comp):
        return ('PASS', 'abs correct')
    return ('BUG', f'eager={eager.item()} comp={comp.item()}')


def test_100466():
    """x * (x >= 0) NaN"""
    torch.manual_seed(420)
    x = torch.rand(3, 3)
    class M(nn.Module):
        def forward(self, x):
            return x * torch.tensor(x >= 0, dtype=torch.float32)
    eager = M()(x); comp = torch.compile(M())(x)
    if torch.allclose(eager, comp, equal_nan=True):
        return ('PASS', 'multiplication correct')
    return ('BUG', f'NaN count={comp.isnan().sum().item()}')


def test_102461():
    """hardshrink wrong result"""
    class M(nn.Module):
        def forward(self, x):
            return torch.nn.functional.hardshrink(torch.sign(x) * x, 0.7)
    x = torch.tensor(np.array([-0.9, 0.9]))
    with torch.no_grad():
        eager = M()(x); comp = torch.compile(M())(x)
    if torch.allclose(eager, comp, atol=1e-5):
        return ('PASS', 'hardshrink correct')
    return ('BUG', f'eager={eager.tolist()} comp={comp.tolist()}')


def test_105264():
    """uint8 add overflow CPU codegen"""
    def f(x):
        return (x + x).to(torch.int16)
    x = torch.tensor(128, dtype=torch.uint8)
    eager = f(x); comp = torch.compile(f)(x)
    if torch.equal(eager, comp):
        return ('PASS', 'uint8 add correct')
    return ('BUG', f'eager={eager.item()} comp={comp.item()}')


def test_97130():
    """shared storage mutation in nested function"""
    def forward():
        a = torch.ones(2)
        b = a.reshape(2)
        c = a[0:2]
        def subfunc():
            b[0] = 2
            if b.sum() >= -1e5:
                pass
        subfunc()
        return c
    eager = forward(); comp = torch.compile(forward)()
    if torch.allclose(eager, comp):
        return ('PASS', 'shared storage correct')
    return ('BUG', f'eager={eager.tolist()} comp={comp.tolist()}')


def test_113641():
    """double inplace clamp on strides wrong"""
    def fn(inp, top, left, height, width):
        output = inp - torch.tensor([left, top, left, top], dtype=inp.dtype)
        output[..., 0::2].clamp_(min=0, max=width)
        output[..., 1::2].clamp_(min=0, max=height)
        return output
    x = torch.tensor([[0.0, 1.0, 10.0, 14.0]])
    eager = fn(x.clone(), 7, 3, 3, 5)
    comp = torch.compile(fn, dynamic=False)(x.clone(), 7, 3, 3, 5)
    if torch.allclose(eager, comp):
        return ('PASS', 'clamp_ correct')
    return ('BUG', f'eager={eager.tolist()} comp={comp.tolist()}')


if __name__ == '__main__':
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    tests = [
        # Confirmed BUG on 2.2.0
        (88813,  "output tensors merged [CUDA]",          test_88813),
        (94693,  "trailing copy_() wrong [CPU]",           test_94693),
        (115260, "tan output order [CPU]",                 test_115260),
        (115774, "mutable attr stale [CPU]",               test_115774),
        # Tested PASS (fixed by 2.2.0)
        (93351,  "pad+gt float64 [CPU]",                   test_93351),
        (93365,  "cat+mul zeros [CPU]",                    test_93365),
        (93374,  "where broadcast [CPU]",                  test_93374),
        (93829,  "neg uint8 lt [CPU]",                     test_93829),
        (95695,  "ge int vs float [CUDA]",                 test_95695),
        (96694,  "abs typecast negative [CUDA]",           test_96694),
        (97130,  "shared storage mutation [CPU]",          test_97130),
        (100466, "x*(x>=0) NaN [CPU]",                    test_100466),
        (102461, "hardshrink wrong [CPU]",                 test_102461),
        (105264, "uint8 add overflow [CPU]",               test_105264),
        (113641, "double clamp_ strides [CPU]",            test_113641),
    ]

    for issue_num, desc, test_fn in tests:
        test_bug(issue_num, desc, test_fn)

    print(f"{'Issue':>8s}  {'Status':>6s}  {'Description':<45s}  Detail")
    print("-" * 110)
    for issue_num in sorted(RESULTS.keys()):
        status, desc, detail = RESULTS[issue_num]
        marker = '***' if status == 'BUG' else '   '
        print(f"#{issue_num:>7d}  {status:>6s}  {desc:<45s}  {detail[:50]} {marker}")

    bugs = [(k, v) for k, v in RESULTS.items() if v[0] == 'BUG']
    passes = [(k, v) for k, v in RESULTS.items() if v[0] == 'PASS']
    errors = [(k, v) for k, v in RESULTS.items() if v[0] == 'ERROR']
    skips = [(k, v) for k, v in RESULTS.items() if v[0] == 'SKIP']

    print()
    print(f"BUG: {len(bugs)}, PASS: {len(passes)}, ERROR: {len(errors)}, SKIP: {len(skips)}")
    if bugs:
        print("\nConfirmed bugs on 2.2.0:")
        for k, v in sorted(bugs):
            print(f"  #{k}: {v[1]}")
