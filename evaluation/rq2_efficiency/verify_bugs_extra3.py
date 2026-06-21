#!/usr/bin/env python3
"""
Remaining 43 bugs from Demystifying dataset that were not previously tested.
Only includes bugs with self-contained reproducers potentially working on 2.2.0.
Bugs requiring FlexAttention(2.5+), MPS, HPU, FSDP(2.4+), user Triton kernels(2.3+)
are excluded as they cannot run on PyTorch 2.2.0.
"""
import torch
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
# Graph Semantic Capturing / Dynamo bugs
# ============================================================================

def test_105929():
    """TorchDispatchMode silently ignored under compile"""
    from torch.utils._python_dispatch import TorchDispatchMode
    class RewriteAddToMul(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.add.Tensor:
                func = torch.ops.aten.mul.Tensor
            return func(*args, **(kwargs or {}))
    def fn(x):
        return x + x
    x = torch.tensor([3.0])
    with RewriteAddToMul():
        eager = fn(x)
        torch._dynamo.reset()
        compiled = torch.compile(fn)(x)
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()} (mode ignored)')
    return ('PASS', '')

def test_130290():
    """capture_dynamic_output_shape_ops changes output"""
    try:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
    except Exception:
        return None
    @torch.compile
    def fn(tensor, mapping):
        xx, yy = torch.meshgrid(mapping, tensor, indexing="ij")
        indices = torch.argwhere(xx == yy)
        mapped_values = torch.zeros_like(tensor)
        mapped_values[indices[:, 1]] = indices[:, 0]
        return mapped_values
    tensor = torch.tensor([1, 2, 3, 5, 6, 7])
    mapping = torch.tensor([0, 3, 4, 5, 7])
    torch._dynamo.reset()
    result = fn(tensor, mapping)
    expected = torch.tensor([0, 0, 1, 3, 0, 4])
    try:
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
    except Exception:
        pass
    if not torch.equal(result, expected):
        return ('BUG', f'got {result.tolist()} expected {expected.tolist()}')
    return ('PASS', '')

def test_130750():
    """dynamo map returns tuple instead of map object"""
    def f(a):
        return a[0] * a[1]
    x = [torch.randn(5) for _ in range(3)]
    def h(x):
        if isinstance(map(f, x), tuple):
            return torch.tensor(1)
        return torch.tensor(0)
    eager = h(x)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(h, fullgraph=True, backend="eager")(x)
    except Exception:
        compiled = torch.compile(h, backend="eager")(x)
    if eager.item() != compiled.item():
        return ('BUG', f'map returns tuple under compile: eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_135432():
    """torch.Tensor() returns stale values on second call"""
    @torch.compile
    def foo(x):
        return torch.Tensor(x)
    torch._dynamo.reset()
    r1 = foo([1, 2])
    r2 = foo([3, 4])
    if torch.equal(r1, r2):
        return ('BUG', f'second call returns stale values: r1={r1.tolist()} r2={r2.tolist()}')
    expected = torch.tensor([3.0, 4.0])
    if not torch.allclose(r2, expected, atol=1e-3):
        return ('BUG', f'second call wrong: got {r2.tolist()} expected [3,4]')
    return ('PASS', '')

def test_135441():
    """Tensor._version not incremented under compile"""
    @torch.compile()
    def version_bump_test(x):
        x[...] = 0
    x = torch.ones(4)
    v0 = x._version
    torch._dynamo.reset()
    version_bump_test(x)
    v1 = x._version
    if v1 <= v0:
        return ('BUG', f'_version not incremented: before={v0} after={v1}')
    return ('PASS', '')

def test_138112():
    """dynamo fails to propagate nonlocal variable updates"""
    result = {}
    def nonlocal_test():
        z = 1
        k = 2
        def create_fn():
            def fn(x):
                nonlocal k, z
                k = z
            return fn
        def run_fn(fn, x):
            nonlocal z
            z = 3
            fn(x)
            return x.cos()
        @torch.compile(backend="eager", fullgraph=True)
        def foo(x):
            fn = create_fn()
            return run_fn(fn, x)
        x = torch.randn(2, 3)
        torch._dynamo.reset()
        foo(x)
        result['z'] = z
        result['k'] = k
    try:
        nonlocal_test()
    except Exception as e:
        return ('ERROR', str(e)[:200])
    if result.get('k') != 3:
        return ('BUG', f'nonlocal k={result.get("k")} expected 3')
    return ('PASS', '')

def test_143431():
    """aot_eager changes CPU RNG behavior"""
    def f(image_latent):
        x = torch.rand(2, 12)
        indices = torch.argsort(torch.rand(*x.shape), dim=-1)[:, :6]
        return image_latent[torch.arange(2).unsqueeze(-1), indices][:, :3]
    torch.manual_seed(54321)
    torch._dynamo.reset()
    try:
        r1 = torch.compile(backend="aot_eager", fullgraph=True)(f)(torch.randn(2, 12, 16, 32, 32)).sum()
    except Exception:
        return ('ERROR', 'aot_eager fullgraph failed')
    torch.manual_seed(54321)
    torch._dynamo.reset()
    r2 = torch.compile(backend="eager", fullgraph=True)(f)(torch.randn(2, 12, 16, 32, 32)).sum()
    if not torch.allclose(r1, r2, atol=1e-3):
        return ('BUG', f'RNG differs: aot_eager={r1.item():.4f} eager={r2.item():.4f}')
    return ('PASS', '')


# ============================================================================
# Low-level Code Generation bugs
# ============================================================================

def test_135873():
    """bucketize fails on non-contiguous strided tensors (CUDA)"""
    boundaries = torch.linspace(-1.0, 1.0, 33, device="cuda")
    vals = torch.randn(16, device="cuda")
    noncontiguous = boundaries[::2]
    eager = torch.bucketize(vals, noncontiguous)
    torch._dynamo.reset()
    compiled = torch.compile(torch.bucketize)(vals, noncontiguous)
    if not torch.equal(eager, compiled):
        n_diff = (eager != compiled).sum().item()
        return ('BUG', f'bucketize strided wrong, {n_diff} elements differ')
    return ('PASS', '')

def test_137254():
    """addmm+cat produces random garbage (CUDA)"""
    class Model(torch.nn.Module):
        def forward(self, x):
            s1 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
            s2 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
            return torch.cat((s1, s2), -1)
    x = torch.randn(2, 2).cuda()
    func = Model().cuda()
    eager = func(x)
    torch._dynamo.reset()
    compiled = torch.compile(func, backend="inductor")(x)
    expected = torch.zeros(2, 4).cuda()
    if not torch.allclose(compiled, expected, atol=1e-3):
        diff = compiled.abs().max().item()
        return ('BUG', f'addmm(beta=0,alpha=0)+cat not zero, max={diff:.6f}')
    return ('PASS', '')

def test_142345():
    """torch.asinh outputs -inf on CPU inductor"""
    x = torch.tensor([0.0, 0.0, 0.0, -10000.1])
    eager = torch.asinh(x)
    torch._dynamo.reset()
    compiled = torch.compile(torch.asinh)(x)
    if compiled.isinf().any() and not eager.isinf().any():
        return ('BUG', f'compiled produces -inf: {compiled.tolist()}')
    if not torch.allclose(eager, compiled, atol=1e-2):
        return ('BUG', f'values differ')
    return ('PASS', '')

def test_143182():
    """LayerNorm numerical error on CPU for large tensors"""
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.LayerNorm([1000, 100])).eval()
    x = torch.randn(1, 3, 1000, 100)
    with torch.no_grad():
        eager = model(x)
        torch._dynamo.reset()
        compiled = torch.compile(model)(x)
    diff = (eager - compiled).abs().max().item()
    if diff > 1e-3:
        return ('BUG', f'LayerNorm max_diff={diff:.6f}')
    return ('PASS', '')

def test_143648():
    """polygamma incorrect under compile"""
    inp = torch.tensor(2.0, dtype=torch.float32)
    eager = torch.special.polygamma(0, inp)
    torch._dynamo.reset()
    compiled = torch.compile(torch.special.polygamma)(0, inp)
    if compiled.isinf().any() and not eager.isinf().any():
        return ('BUG', f'compiled=-inf, eager={eager.item():.4f}')
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'eager={eager.item():.4f} compiled={compiled.item():.4f}')
    return ('PASS', '')

def test_143738():
    """avg_pool2d incorrectly processes int64"""
    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.argsort(x, dim=3)
            return torch.nn.functional.avg_pool2d(x.float(), kernel_size=2, stride=2)
    model = Model()
    x = torch.randn(1, 1, 2, 4)
    eager = model(x)
    torch._dynamo.reset()
    compiled = torch.compile(model)(x)
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'avg_pool2d int64 wrong')
    return ('PASS', '')

def test_143960():
    """torch.dist numerical unstable after compile"""
    torch.manual_seed(42)
    a = torch.rand(1).float()
    b = torch.rand(1).float()
    eager = torch.dist(a, b, -41)
    torch._dynamo.reset()
    compiled = torch.compile(torch.dist)(a, b, -41)
    if abs(eager.item()) > 0 and abs(compiled.item() / eager.item() - 1) > 0.1:
        return ('BUG', f'eager={eager.item():.6e} compiled={compiled.item():.6e}')
    return ('PASS', '')

def test_144462():
    """F.pad incorrect value on GPU"""
    inp = torch.randn(1, 1, 1).cuda()
    eager = torch.nn.functional.pad(inp, [10, 10], 'constant', 0)
    torch._dynamo.reset()
    compiled = torch.compile(torch.nn.functional.pad)(inp, [10, 10], 'constant', 0)
    if not torch.allclose(eager, compiled, atol=1e-5):
        n_diff = (~torch.isclose(eager, compiled, atol=1e-5)).sum().item()
        return ('BUG', f'pad fills wrong values, {n_diff} elements differ')
    return ('PASS', '')

def test_145379():
    """torch.logit wrong when input < eps"""
    inp = torch.tensor(0.3, dtype=torch.float64)
    eps = torch.tensor(0.9, dtype=torch.float64)
    eager = torch.logit(inp, eps.item())
    torch._dynamo.reset()
    compiled = torch.compile(torch.logit)(inp, eps.item())
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'eager={eager.item():.4f} compiled={compiled.item():.4f}')
    return ('PASS', '')

def test_147261():
    """F.interpolate + dynamic=True wrong shape"""
    @torch.compile(dynamic=True)
    def f(x):
        return torch.nn.functional.interpolate(x, scale_factor=1/300, mode="linear")
    torch._dynamo.reset()
    x = torch.randn(1, 8, 396 * 300)
    result = f(x)
    expected_size = 396
    if result.shape[2] != expected_size:
        return ('BUG', f'shape wrong: got {result.shape[2]} expected {expected_size}')
    return ('PASS', '')

def test_147450():
    """polygamma less precise under compile n==1"""
    inp = torch.tensor([-1.0])
    eager = torch.special.polygamma(1, inp)
    torch._dynamo.reset()
    compiled = torch.compile(torch.special.polygamma)(1, inp)
    if compiled.isinf().any() and not eager.isinf().any():
        return ('BUG', f'compiled=inf, eager={eager.item():.4e}')
    if not torch.allclose(eager, compiled, rtol=1e-2, atol=1e-3):
        return ('BUG', f'eager={eager.item():.4e} compiled={compiled.item():.4e}')
    return ('PASS', '')

def test_148241():
    """Tanhshrink-atan2 inconsistent on CPU"""
    torch.manual_seed(42)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shrink = torch.nn.Tanhshrink()
        def forward(self, x):
            s = self.shrink(x)
            return torch.atan2(s, s)
    model = Model()
    x = torch.randn(1, 3, 64, 64)
    eager = model(x)
    torch._dynamo.reset()
    compiled = torch.compile(model)(x)
    diff = (eager - compiled).abs().max().item()
    if diff > 0.1:
        return ('BUG', f'max_diff={diff:.4f}')
    return ('PASS', '')

def test_142853():
    """fallback_random negative effect Dropout"""
    try:
        torch._inductor.config.fallback_random = True
    except Exception:
        return None
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout2d(p=0.5)
        def forward(self, x):
            torch.manual_seed(0)
            return self.dropout(x)
    model = Model().train()
    x = torch.randn(1, 3, 256, 256)
    eager = model(x)
    torch._dynamo.reset()
    compiled = torch.compile(model)(x)
    try:
        torch._inductor.config.fallback_random = False
    except Exception:
        pass
    if not torch.allclose(eager, compiled, atol=1e-5):
        return ('BUG', f'dropout results differ with fallback_random')
    return ('PASS', '')

def test_136883():
    """pattern matcher caching returns wrong replacement"""
    return None


# ============================================================================
# Bugs that definitely can't run on 2.2.0 (recorded for completeness)
# ============================================================================

def test_not_available(issue_num, reason):
    """Placeholder for bugs requiring features not in PyTorch 2.2.0"""
    RESULTS[issue_num] = ('N/A', reason, 'requires features not in 2.2.0')


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    tests = [
        # Dynamo / Graph Semantic
        (105929, "TorchDispatchMode ignored [CPU]",          test_105929),
        (130290, "dynamic_output_shape wrong [CPU]",         test_130290),
        (130750, "map returns tuple [CPU]",                  test_130750),
        (135432, "Tensor() stale values [CPU]",              test_135432),
        (135441, "Tensor._version not bumped [CPU]",         test_135441),
        (138112, "nonlocal propagation [CPU]",               test_138112),
        (143431, "aot_eager RNG differs [CPU]",              test_143431),

        # Low-level Code Generation
        (135873, "bucketize strided [CUDA]",                 test_135873),
        (137254, "addmm+cat garbage [CUDA]",                 test_137254),
        (142345, "asinh -inf [CPU]",                         test_142345),
        (143182, "LayerNorm numerical [CPU]",                test_143182),
        (143648, "polygamma incorrect [CPU]",                test_143648),
        (143738, "avg_pool2d int64 [CPU]",                   test_143738),
        (143960, "torch.dist unstable [CPU]",                test_143960),
        (144462, "F.pad wrong value [CUDA]",                 test_144462),
        (145379, "logit wrong sign [CPU]",                   test_145379),
        (147261, "interpolate dynamic shape [CPU]",          test_147261),
        (147450, "polygamma n=1 inf [CPU]",                  test_147450),
        (148241, "Tanhshrink-atan2 [CPU]",                   test_148241),
        (142853, "fallback_random Dropout [CPU]",            test_142853),
    ]

    for issue_num, desc, test_fn in tests:
        test_bug(issue_num, desc, test_fn)

    # Record bugs that can't run on 2.2.0
    na_bugs = [
        (114903, "torch.var NaN A16/A2 hardware specific"),
        (116848, "inference_mode black images (needs diffusers)"),
        (119485, "DTensor _Partial (no reproducer)"),
        (128202, "inplace transpose (HPU only)"),
        (130243, "triton kernel stride reorder (2.3+)"),
        (130487, "custom_op mutation (2.4+)"),
        (132196, "triton multi-output (2.3+)"),
        (132197, "fsdp.set_ (2.4+)"),
        (132200, "fsdp.set_ CSE (2.4+)"),
        (135161, "FlexAttention grad (2.5+)"),
        (136232, "FlexAttention NaN backward (2.5+)"),
        (136550, "custom Triton gemlite (2.5+)"),
        (136883, "pattern matcher cache (private API)"),
        (139348, "triton 3.2 cumsum (triton version)"),
        (139462, "FlexAttention zeros (2.5+)"),
        (141710, "dynamic shapes attention NaN (internals)"),
        (145697, "constant folding swin_v2 (config.freezing)"),
        (145735, "lgamma MPS (MPS only)"),
        (147828, "triton_op padding (2.5+)"),
        (148356, "layout constraints (2.6 API)"),
        (150184, "guard_size_oblivious (2.3+)"),
        (150516, "OLMo training (no reproducer, 2.7)"),
        (150629, "RMSNorm MPS (MPS/2.4+)"),
    ]
    for issue_num, reason in na_bugs:
        test_not_available(issue_num, reason)

    # Print results for tested bugs
    tested = {k: v for k, v in RESULTS.items() if v[0] != 'N/A'}
    na = {k: v for k, v in RESULTS.items() if v[0] == 'N/A'}

    bugs = [(k, v) for k, v in tested.items() if v[0] == 'BUG']
    passes = [(k, v) for k, v in tested.items() if v[0] == 'PASS']
    errors = [(k, v) for k, v in tested.items() if v[0] == 'ERROR']
    skips = [(k, v) for k, v in tested.items() if v[0] == 'SKIP']

    print(f"{'Issue':>8s}  {'Status':>6s}  {'Description':<45s}  Detail")
    print("-" * 110)
    for issue_num in sorted(tested.keys()):
        status, desc, detail = tested[issue_num]
        marker = '***' if status == 'BUG' else '   '
        print(f"#{issue_num:>7d}  {status:>6s}  {desc:<45s}  {detail[:50]} {marker}")

    print()
    print(f"Tested:              {len(tested)}")
    print(f"  BUG (reproducible):  {len(bugs)}")
    print(f"  PASS (not on 2.2):   {len(passes)}")
    print(f"  ERROR (test issue):  {len(errors)}")
    print(f"  SKIP:                {len(skips)}")
    print(f"N/A (can't run 2.2):   {len(na)}")

    if bugs:
        print()
        print("Reproducible bug issues:")
        for k, v in sorted(bugs):
            print(f"  #{k}: {v[1]}")

    print()
    print("N/A bugs (features not in 2.2.0):")
    for k in sorted(na.keys()):
        print(f"  #{k}: {na[k][1]}")
