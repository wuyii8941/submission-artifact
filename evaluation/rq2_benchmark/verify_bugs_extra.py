#!/usr/bin/env python3
"""
Extra bug reproducers found beyond the Demystifying dataset.
These are from broader GitHub search of torch.compile correctness bugs.
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
# Batch C: Early torch.compile bugs (93xxx series)
# ============================================================================

def test_93283():
    """stack + inplace silu alias corruption (CUDA)"""
    def fn(input):
        rt = torch.stack([input])
        v = torch.nn.functional.silu(input, inplace=True)
        return rt
    x = torch.tensor([1.0, 1.0])
    eager = fn(x.clone())
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x.clone())
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'eager={eager.tolist()} compiled={compiled.tolist()}')
    return ('PASS', '')

def test_93351():
    """pad + gt wrong result for float64 (CPU)"""
    def fn(input):
        v1 = torch.nn.functional.pad(input, pad=(1, 0))
        return torch.gt(v1, input)
    x = torch.rand([1, 2, 2, 1], dtype=torch.float64)
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.equal(eager, compiled):
        n_diff = (eager != compiled).sum().item()
        return ('BUG', f'{n_diff} elements differ')
    return ('PASS', '')

def test_93357():
    """cat + expand on Parameter corrupts values (CUDA)"""
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = torch.nn.parameter.Parameter(
                torch.tensor([4, 4, 5, 5, 4, 4, 4, 7, 4]), requires_grad=False)
        def forward(self):
            v0 = torch.cat([self.p1, self.p1], dim=0)
            v5 = v0.expand(1, 1, 1, 18)
            return v5
    with torch.no_grad():
        model = Model()
        eager = model()
        torch._dynamo.reset()
        compiled = torch.compile(model, fullgraph=True)(  )
    if not torch.equal(eager, compiled):
        return ('BUG', f'values corrupted')
    return ('PASS', '')

def test_93365():
    """cat + mul on global tensor returns zeros (CPU)"""
    p0 = torch.tensor([[4.9334, 5.5571]])
    def fn():
        v7 = torch.cat([p0, p0], dim=0)
        v1 = torch.mul(v7, v7)
        return v7, v1
    eager = fn()
    torch._dynamo.reset()
    compiled = torch.compile(fn)()
    if not torch.allclose(eager[1], compiled[1], atol=1e-3):
        diff = (eager[1] - compiled[1]).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_93374():
    """where with broadcast produces wrong values (CPU)"""
    torch.manual_seed(0)
    p1 = torch.rand(13, 7, 3)
    p0 = torch.rand(1, 1)
    def fn(x):
        return torch.where(x, p1, p0)
    x = torch.tensor([[True]])
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.allclose(eager, compiled, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_93823():
    """masked_fill no-ops instead of filling (CUDA)"""
    def fn(x, y):
        return x.masked_fill(y, 0)
    x = torch.rand(2, 4, 8)
    y = torch.ones(2, 4, 1).bool()
    eager = fn(x, y)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x, y)
    if not torch.allclose(eager, compiled, atol=1e-5):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_93824():
    """inplace reciprocal_ on squeezed view wrong (CPU)"""
    def fn(v0):
        v6 = v0.min(0).values
        v5 = v6.squeeze(1)
        v3 = v5.squeeze(0)
        v2 = v5.reciprocal_()
        return v3
    x = torch.rand(1, 1, 1, 1, 12)
    eager = fn(x.clone())
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x.clone())
    if not torch.allclose(eager, compiled, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_93829():
    """neg + lt on uint8 wrong comparison (CUDA)"""
    def fn(v3):
        v7 = torch.neg(v3)
        v4 = torch.lt(v7, torch.tensor(4, dtype=torch.uint8))
        return v7, v4
    v3 = torch.tensor(6, dtype=torch.uint8)
    eager = fn(v3)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(v3)
    if eager[1].item() != compiled[1].item():
        return ('BUG', f'eager lt={eager[1].item()} compiled lt={compiled[1].item()}')
    return ('PASS', '')

# ============================================================================
# Batch D: Various correctness bugs
# ============================================================================

def test_95695():
    """torch.ge wrong result int vs float scalar (CUDA)"""
    def fn(input):
        return torch.ge(input, 3.5)
    x = torch.tensor([3])
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if eager.item() != compiled.item():
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_98149():
    """pow + cos wrong for uint8 (CUDA)"""
    def fn(x):
        t = x.pow(5)
        return torch.cos(t)
    x = torch.tensor([4], dtype=torch.uint8)
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.allclose(eager.float(), compiled.float(), rtol=1e-2, atol=1e-3):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_99840():
    """ConvTranspose2d + gt + mul NaN (CUDA)"""
    torch.manual_seed(420)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = torch.nn.ConvTranspose2d(3, 18, kernel_size=3, stride=1, padding=1)
            self.y = 0.1
        def forward(self, x):
            out = self.conv_transpose(x)
            out = torch.gt(out, 0)
            out = torch.mul(out, self.y)
            return out
    x = torch.randn(1, 3, 4, 4)
    func = Model()
    res1 = func(x)
    torch._dynamo.reset()
    res2 = torch.compile(func)(x)
    if res2.isnan().any() and not res1.isnan().any():
        return ('BUG', f'compiled NaN, eager ok')
    if not torch.allclose(res1, res2, atol=1e-3):
        diff = (res1 - res2).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_100466():
    """x * (x >= 0) returns NaN (CPU)"""
    torch.manual_seed(420)
    input_tensor = torch.rand(3, 3)
    class Model(torch.nn.Module):
        def forward(self, x):
            return x * torch.tensor(x >= 0, dtype=torch.float32)
    func = Model()
    with torch.no_grad():
        func.eval()
        res1 = func(input_tensor)
        torch._dynamo.reset()
        res2 = torch.compile(func)(input_tensor)
    if res2.isnan().any() and not res1.isnan().any():
        return ('BUG', f'compiled NaN, eager ok')
    if not torch.allclose(res1, res2, atol=1e-3):
        return ('BUG', f'values differ')
    return ('PASS', '')

def test_100794():
    """conv2d + interpolate wrong backward grads (CPU)"""
    torch.manual_seed(0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, kernel_size=4)
        def forward(self, x):
            v2 = self.conv(x)
            return torch.nn.functional.interpolate(v2, size=[1, 1], mode="nearest")
    x = torch.ones(size=[1, 1, 4, 4])
    model = Model()
    model_tmp = copy.deepcopy(model)
    model_tmp.train()
    torch._dynamo.reset()
    compiled = torch.compile(model_tmp, fullgraph=True)
    ret_c = compiled(x)
    grad_c = torch.autograd.grad(ret_c, list(compiled.parameters()),
                                  grad_outputs=torch.ones_like(ret_c), retain_graph=True)
    model.train()
    ret_e = model(x)
    grad_e = torch.autograd.grad(ret_e, list(model.parameters()),
                                  grad_outputs=torch.ones_like(ret_e), retain_graph=True)
    if not torch.allclose(grad_c[0], grad_e[0], atol=1e-3):
        diff = (grad_c[0] - grad_e[0]).abs().max().item()
        return ('BUG', f'grad max_diff={diff:.6f}')
    return ('PASS', '')

def test_108520():
    """int32 overflow differs eager vs compiled (CPU)"""
    def mySum64(x):
        return (x + x).to(torch.int64)
    x = torch.tensor(2147483647, dtype=torch.int32)
    eager = mySum64(x)
    torch._dynamo.reset()
    compiled = torch.compile(mySum64)(x)
    if eager.item() != compiled.item():
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

# ============================================================================
# Batch A: Inductor miscompile bugs (output-order-dependent)
# ============================================================================

def test_96064():
    """reduce-overhead loop indexing wrong (CUDA)"""
    def compute(log_probs):
        lit_weights = torch.stack((log_probs, log_probs), dim=-1).permute(1, 2, 0)
        levels = [torch.tensor([5, 7], device='cuda'), torch.tensor([8], device='cuda')]
        lit_indices = torch.tensor([0, 1, 2, 3, 4, 6], device='cuda')
        id = 9
        node_indices = torch.tensor([[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],
            [[0,0],[0,0]],[[1,2],[3,4]],[[0,0],[0,0]],[[1,4],[9,9]],[[0,5],[6,7]]], device='cuda')
        lit_mask = (torch.tensor([0,1,2,1,2,0], device='cuda'), torch.tensor([1,1,0,0,1,0], device='cuda'))
        data = torch.zeros(id+1, 5, device='cuda')
        data[id] = -1000.0
        data[lit_indices] = lit_weights[lit_mask[0], lit_mask[1]]
        data[levels[0]] = data[node_indices[levels[0]]].sum(-2).logsumexp(-2)
        data[levels[1]] = data[node_indices[levels[1]]].sum(-2).logsumexp(-2)
        return data[levels[-1]]
    log_probs = torch.zeros((5, 10), device='cuda', requires_grad=True)
    eager = compute(log_probs)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(compute, mode='reduce-overhead')(log_probs)
    except Exception:
        compiled = torch.compile(compute)(log_probs)
    if not torch.allclose(eager.exp(), compiled.exp(), atol=1e-2):
        return ('BUG', f'eager={eager.exp().tolist()} compiled={compiled.exp().tolist()}')
    return ('PASS', '')

def test_113632():
    """output order changes sub result (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x):
            gelu = torch.nn.functional.gelu(x)
            sigmoid = torch.sigmoid(gelu)
            sub = torch.sub(gelu, gelu)
            return (sigmoid, sub)
    class Model1(torch.nn.Module):
        def forward(self, x):
            gelu = torch.nn.functional.gelu(x)
            sigmoid = torch.sigmoid(gelu)
            sub = torch.sub(gelu, gelu)
            return (sub, sigmoid)
    x = torch.tensor([5.238, 3.800, 6.932, 3.328], device='cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), backend='inductor')(x)
    if not torch.allclose(out0[1], out1[0], atol=1e-5):
        diff = (out0[1] - out1[0]).abs().max().item()
        return ('BUG', f'sub result depends on output order, max_diff={diff:.6f}')
    return ('PASS', '')

def test_114221():
    """extra cat output corrupts lt result (CUDA)"""
    p0 = torch.tensor(0.0002, dtype=torch.float16, device='cuda')
    class Model0(torch.nn.Module):
        def forward(self, x):
            sigmoid = torch.sigmoid(p0)
            mul = torch.mul(x, sigmoid)
            pad = torch.nn.functional.pad(mul, (-8, 0, 0, 0, 0, 0), 'constant', value=0.5)
            lt = torch.lt(pad, mul)
            cat = torch.cat((pad,), dim=2)
            return (lt, cat)
    class Model1(torch.nn.Module):
        def forward(self, x):
            sigmoid = torch.sigmoid(p0)
            mul = torch.mul(x, sigmoid)
            pad = torch.nn.functional.pad(mul, (-8, 0, 0, 0, 0, 0), 'constant', value=0.5)
            lt = torch.lt(pad, mul)
            return lt
    np.random.seed(42)
    data = np.random.normal(5, 1, size=(1, 57, 8, 9)).astype(np.float16)
    x = torch.from_numpy(data).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), backend='inductor')(x)
    if not torch.equal(out0[0], out1):
        n_diff = (out0[0] != out1).sum().item()
        return ('BUG', f'lt result differs with/without cat, {n_diff} elements')
    return ('PASS', '')

def test_114227():
    """extra branch corrupts sub(gelu,gelu) result (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x0, x1):
            min_1 = x1.min(0)
            min_2 = torch.min(min_1.values, x0)
            mean = min_2.mean(0)
            gelu = torch.nn.functional.gelu(mean)
            sub = torch.sub(gelu, gelu)
            return sub
    class Model1(torch.nn.Module):
        def forward(self, x0, x1):
            min_1 = x1.min(0)
            min_2 = torch.min(min_1.values, x0)
            mean = min_2.mean(0)
            gelu = torch.nn.functional.gelu(mean)
            floor = torch.floor(gelu)
            atan = torch.atan(floor)
            sub = torch.sub(gelu, gelu)
            return (atan, sub)
    x0 = torch.tensor([[5.859193]], device='cuda')
    x1 = torch.tensor(4.8227854, device='cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), backend='inductor')(x0, x1)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), backend='inductor')(x0, x1)
    if not torch.allclose(out0, out1[1], atol=1e-5):
        return ('BUG', f'sub(x,x) != 0 with extra branch: {out1[1].item():.6f}')
    return ('PASS', '')

def test_115159():
    """extra transpose output corrupts sin (CPU)"""
    class Model0(torch.nn.Module):
        def forward(self, x):
            mul = torch.mul(x, x)
            sin = torch.sin(mul)
            pad = torch.nn.functional.pad(mul, (0, 0), 'constant', value=0.5)
            return (pad, sin)
    class Model1(torch.nn.Module):
        def forward(self, x):
            mul = torch.mul(x, x)
            transpose = mul.transpose(1, 0)
            sin = torch.sin(mul)
            pad = torch.nn.functional.pad(mul, (0, 0), 'constant', value=0.5)
            return (transpose, pad, sin)
    np.random.seed(42)
    data = np.random.normal(5, 1, size=(37, 30)).astype(np.float16)
    x = torch.from_numpy(data).to('cpu')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), backend='inductor')(x)
    if not torch.allclose(out0[1], out1[2], atol=1e-3):
        n_diff = (~torch.isclose(out0[1], out1[2], atol=1e-3)).sum().item()
        total = out0[1].numel()
        return ('BUG', f'sin differs with/without transpose output, {n_diff}/{total}')
    return ('PASS', '')

def test_115774():
    """stale buffer after module attr reset (CPU)"""
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = torch.tensor([0, 0])
        @torch.compile(backend="eager")
        def forward(self, x, pos):
            self.cache[pos] = x
            return self.cache
    model = Model()
    r1 = model(torch.tensor([2]), torch.tensor([0]))
    model.cache = torch.tensor([0, 0])
    torch._dynamo.reset()
    r2 = model(torch.tensor([1]), torch.tensor([1]))
    expected = torch.tensor([0, 1])
    if not torch.equal(r2, expected):
        return ('BUG', f'stale buffer: got {r2.tolist()} expected {expected.tolist()}')
    return ('PASS', '')

# ============================================================================
# Batch B: More correctness bugs
# ============================================================================

def test_124357():
    """mutable attribute captured as constant (CUDA)"""
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = -1
            self.cache = torch.ones([2, 3])
        def change_value(self, value):
            self.value = value
        def forward(self, value):
            return self.cache + self.value
    model = ToyModel()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="inductor")
    results = []
    for v in [6, 8, 10]:
        model.change_value(v)
        out = compiled(v)
        expected = model.cache + v
        results.append((v, out.clone(), expected.clone()))
    for v, got, exp in results:
        if not torch.allclose(got, exp, atol=1e-3):
            return ('BUG', f'value={v}: got {got[0,0].item()} expected {exp[0,0].item()}')
    return ('PASS', '')

def test_125999():
    """copy_() inside compile silently no-ops (CUDA)"""
    @torch.compile
    def func(a, b):
        max_scale = torch.max(a, b)
        min_scale = torch.min(a, b)
        new_scale = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
        a.copy_(new_scale)
    a = torch.randn(1, 32, 2, 1026).cuda()
    b = torch.randn(1, 32, 1026).cuda()
    a_before = a.clone()
    torch._dynamo.reset()
    func(a[..., 1, :], b)
    if torch.allclose(a, a_before, atol=1e-5):
        return ('BUG', f'copy_() had no effect')
    return ('PASS', '')

def test_126537():
    """math.trunc casts to wrong dtype (CUDA)"""
    try:
        torch._dynamo.config.capture_scalar_outputs = True
    except Exception:
        pass
    def f(x):
        r = math.sqrt(x.size(0))
        r = r ** 70
        return torch.tensor(math.trunc(r), dtype=torch.float64, device='cuda')
    eager = f(torch.randn(4, device='cuda'))
    torch._dynamo.reset()
    compiled = torch.compile(f, fullgraph=True, dynamic=True)(torch.randn(4, device='cuda'))
    if not torch.allclose(eager, compiled, rtol=1e-3):
        return ('BUG', f'eager={eager.item():.2e} compiled={compiled.item():.2e}')
    return ('PASS', '')


# ============================================================================
# Batch E: Output-order-dependent inductor miscompiles + scatter_reduce
# ============================================================================

def test_113636():
    """argmin result changes when output order swaps (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x):
            div = torch.div(x, x)
            cat = torch.cat((div,), dim=1)
            argmin = div.argmin(0)
            return (cat, argmin)
    class Model1(torch.nn.Module):
        def forward(self, x):
            div = torch.div(x, x)
            cat = torch.cat((div,), dim=1)
            argmin = div.argmin(0)
            return (argmin, cat)
    np.random.seed(42)
    x = torch.from_numpy(np.random.rand(19, 1, 60, 47).astype(np.float16)).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x)
    if not torch.equal(out0[1], out1[0]):
        n_diff = (out0[1] != out1[0]).sum().item()
        return ('BUG', f'argmin differs with output order swap, {n_diff} elements')
    return ('PASS', '')

def test_114020():
    """extra cat output corrupts div via softmax fusion (CUDA)"""
    p0 = torch.nn.Parameter(torch.randn(3, 42, 7, 26, 1, dtype=torch.float16, device='cuda'), requires_grad=False)
    class Model0(torch.nn.Module):
        def forward(self, x):
            mul = torch.mul(x, p0)
            softmax = torch.softmax(mul, dim=0)
            div = torch.div(x, softmax)
            return (div,)
    class Model1(torch.nn.Module):
        def forward(self, x):
            mul = torch.mul(x, p0)
            cat = torch.cat((mul, mul), dim=0)
            softmax = torch.softmax(mul, dim=0)
            div = torch.div(x, softmax)
            return (cat, div)
    np.random.seed(42)
    data = np.random.normal(5, 1, size=(1, 1, 1, 26, 1)).astype(np.float16)
    x = torch.from_numpy(data).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x)
    if not torch.allclose(out0[0], out1[1], rtol=1e-2, atol=1e-2, equal_nan=True):
        has_nan = out1[1].isnan().any().item() and not out0[0].isnan().any().item()
        return ('BUG', f'div corrupted by extra cat output, NaN={has_nan}')
    return ('PASS', '')

def test_114222():
    """lt result changes when output order swaps (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x0, x1):
            floor = torch.floor(x0)
            mul = torch.mul(x1, floor)
            round_1 = torch.round(mul)
            sin = torch.sin(round_1)
            expand = sin.expand(33, 33)
            lt = torch.lt(sin, expand)
            matmul = torch.matmul(sin, expand)
            return (lt, matmul)
    class Model1(torch.nn.Module):
        def forward(self, x0, x1):
            floor = torch.floor(x0)
            mul = torch.mul(x1, floor)
            round_1 = torch.round(mul)
            sin = torch.sin(round_1)
            expand = sin.expand(33, 33)
            lt = torch.lt(sin, expand)
            matmul = torch.matmul(sin, expand)
            return (matmul, lt)
    np.random.seed(42)
    x0 = torch.from_numpy(np.random.normal(5, 1, size=()).astype(np.float16)).to('cuda')
    x1 = torch.from_numpy(np.random.normal(5, 1, size=(33,)).astype(np.float16)).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x0, x1)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x0, x1)
    if not torch.equal(out0[0], out1[1]):
        n_diff = (out0[0] != out1[1]).sum().item()
        return ('BUG', f'lt differs with output order swap, {n_diff} elements')
    return ('PASS', '')

def test_114223():
    """mean result changes when output order swaps (CUDA)"""
    p0 = torch.nn.Parameter(torch.randn(57, 1, 10, 1, 40, dtype=torch.float16, device='cuda'), requires_grad=False)
    class Model0(torch.nn.Module):
        def forward(self, x):
            abs_1 = torch.abs(p0)
            div = torch.div(x, abs_1)
            cat = torch.cat((div,), dim=2)
            transpose = div.transpose(2, 4)
            cat_1 = torch.cat((transpose,), dim=4)
            floor = torch.floor(cat_1)
            floor_1 = torch.floor(floor)
            mean = floor_1.mean(4)
            interpolate = torch.nn.functional.interpolate(floor_1, size=[1, 2, 10], mode='trilinear')
            return (cat, mean, interpolate)
    class Model1(torch.nn.Module):
        def forward(self, x):
            abs_1 = torch.abs(p0)
            div = torch.div(x, abs_1)
            cat = torch.cat((div,), dim=2)
            transpose = div.transpose(2, 4)
            cat_1 = torch.cat((transpose,), dim=4)
            floor = torch.floor(cat_1)
            floor_1 = torch.floor(floor)
            mean = floor_1.mean(4)
            interpolate = torch.nn.functional.interpolate(floor_1, size=[1, 2, 10], mode='trilinear')
            return (interpolate, cat, mean)
    x = torch.tensor(3.273, dtype=torch.float16, device='cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x)
    if not torch.allclose(out0[1], out1[2], atol=1e-2):
        diff = (out0[1] - out1[2]).abs().max().item()
        return ('BUG', f'mean differs with output order swap, max_diff={diff:.6f}')
    return ('PASS', '')

def test_114224():
    """add result changes when output order swaps (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x0, x1):
            ceil = torch.ceil(x0)
            sum_1 = x1.sum(2)
            tan = torch.tan(ceil)
            clip = torch.clip(sum_1, -1.5, 1.5)
            pad = torch.nn.functional.pad(tan, (0, 0), 'constant', value=0.5)
            add = torch.add(sum_1, pad)
            matmul = torch.matmul(pad, clip)
            return (add, matmul)
    class Model1(torch.nn.Module):
        def forward(self, x0, x1):
            ceil = torch.ceil(x0)
            sum_1 = x1.sum(2)
            tan = torch.tan(ceil)
            clip = torch.clip(sum_1, -1.5, 1.5)
            pad = torch.nn.functional.pad(tan, (0, 0), 'constant', value=0.5)
            add = torch.add(sum_1, pad)
            matmul = torch.matmul(pad, clip)
            return (matmul, add)
    np.random.seed(42)
    x0 = torch.from_numpy(np.array([4.55], dtype=np.float16)).to('cuda')
    x1 = torch.from_numpy(np.random.normal(5, 1, size=(1, 51, 1)).astype(np.float16)).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x0, x1)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x0, x1)
    if not torch.allclose(out0[0], out1[1], atol=1e-2):
        diff = (out0[0] - out1[1]).abs().max().item()
        return ('BUG', f'add differs with output order swap, max_diff={diff:.6f}')
    return ('PASS', '')

def test_114225():
    """extra cat output corrupts softmax (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x0, x1):
            mul = torch.mul(x1, x0)
            squeeze = mul.squeeze(1)
            tan = torch.tan(squeeze)
            ceil = torch.ceil(tan)
            softmax = torch.softmax(ceil, dim=0)
            return (softmax,)
    class Model1(torch.nn.Module):
        def forward(self, x0, x1):
            mul = torch.mul(x1, x0)
            cat = torch.cat((x0, x0, mul, mul), dim=2)
            squeeze = mul.squeeze(1)
            tan = torch.tan(squeeze)
            ceil = torch.ceil(tan)
            softmax = torch.softmax(ceil, dim=0)
            return (cat, softmax)
    np.random.seed(42)
    x0 = torch.from_numpy(np.random.normal(5, 1, size=(34, 1, 17)).astype(np.float16)).to('cuda')
    x1 = torch.from_numpy(np.random.normal(5, 1, size=(1,)).astype(np.float16)).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x0, x1)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x0, x1)
    if not torch.allclose(out0[0], out1[1], atol=1e-3):
        diff = (out0[0] - out1[1]).abs().max().item()
        return ('BUG', f'softmax corrupted by extra cat, max_diff={diff:.6f}')
    return ('PASS', '')

def test_114226():
    """output reorder corrupts interpolate (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x0, x1, x2, x3):
            cat = torch.cat((x3, x2, x1, x0), dim=0)
            pad = torch.nn.functional.pad(cat, (0, 0, 30, 0), 'replicate')
            cos = torch.cos(pad)
            floor = torch.floor(cos)
            div = torch.div(cos, floor)
            pad_1 = torch.nn.functional.pad(cos, (0, 0, 0, 0, 0, 0), 'replicate')
            interpolate = torch.nn.functional.interpolate(floor, size=[1, 1], mode='nearest')
            return (pad_1, interpolate, div)
    class Model1(torch.nn.Module):
        def forward(self, x0, x1, x2, x3):
            cat = torch.cat((x3, x2, x1, x0), dim=0)
            pad = torch.nn.functional.pad(cat, (0, 0, 30, 0), 'replicate')
            cos = torch.cos(pad)
            floor = torch.floor(cos)
            div = torch.div(cos, floor)
            pad_1 = torch.nn.functional.pad(cos, (0, 0, 0, 0, 0, 0), 'replicate')
            interpolate = torch.nn.functional.interpolate(floor, size=[1, 1], mode='nearest')
            return (div, pad_1, interpolate)
    np.random.seed(42)
    x0 = torch.from_numpy(np.random.normal(5, 1, size=(25, 53, 1, 1)).astype(np.float16)).to('cuda')
    x1 = torch.from_numpy(np.random.normal(5, 1, size=(1, 53, 1, 1)).astype(np.float16)).to('cuda')
    x2 = torch.from_numpy(np.random.normal(5, 1, size=(1, 53, 1, 1)).astype(np.float16)).to('cuda')
    x3 = torch.from_numpy(np.random.normal(5, 1, size=(1, 53, 1, 1)).astype(np.float16)).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x0, x1, x2, x3)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x0, x1, x2, x3)
    if not torch.allclose(out0[1], out1[2], atol=1e-3):
        diff = (out0[1] - out1[2]).abs().max().item()
        return ('BUG', f'interpolate differs with output order swap, max_diff={diff:.6f}')
    return ('PASS', '')

def test_114229():
    """extra intermediate output corrupts argmax (CUDA)"""
    p0 = torch.tensor(-0.2727, device='cuda')
    np.random.seed(42)
    p1 = torch.from_numpy(np.random.normal(0, 1, size=(1, 1, 22, 54, 41)).astype(np.float32)).to('cuda')
    class Model0(torch.nn.Module):
        def forward(self, x):
            atan = torch.atan(p0)
            div = torch.div(p1, atan)
            tan = torch.tan(x)
            mul = torch.mul(div, tan)
            sub = torch.sub(mul, mul)
            argmax = sub.argmax(3)
            return (argmax,)
    class Model1(torch.nn.Module):
        def forward(self, x):
            atan = torch.atan(p0)
            div = torch.div(p1, atan)
            tan = torch.tan(x)
            mul = torch.mul(div, tan)
            sub = torch.sub(mul, mul)
            argmax = sub.argmax(3)
            return (mul, argmax)
    x = torch.from_numpy(np.random.normal(5, 1, size=(1, 1, 22, 54, 1)).astype(np.float32)).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x)
    if not torch.equal(out0[0], out1[1]):
        n_diff = (out0[0] != out1[1]).sum().item()
        return ('BUG', f'argmax differs with extra output, {n_diff} elements')
    return ('PASS', '')

def test_115160():
    """output reorder corrupts cos/min (CPU)"""
    class Model0(torch.nn.Module):
        def forward(self, x):
            interpolate = torch.nn.functional.interpolate(x, size=[1, 1], mode='bilinear')
            pad = torch.nn.functional.pad(interpolate, (48, 0, 44, 0), 'replicate')
            min_1 = torch.min(x, pad)
            cos = torch.cos(pad)
            return (min_1, cos)
    class Model1(torch.nn.Module):
        def forward(self, x):
            interpolate = torch.nn.functional.interpolate(x, size=[1, 1], mode='bilinear')
            pad = torch.nn.functional.pad(interpolate, (48, 0, 44, 0), 'replicate')
            min_1 = torch.min(x, pad)
            cos = torch.cos(pad)
            return (cos, min_1)
    np.random.seed(42)
    x = torch.from_numpy(np.random.normal(5, 1, size=(1, 18, 1, 1)).astype(np.float32))
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x)
    if not torch.allclose(out0[1], out1[0], atol=1e-5):
        diff = (out0[1] - out1[0]).abs().max().item()
        return ('BUG', f'cos differs with output order swap, max_diff={diff:.6f}')
    return ('PASS', '')

def test_131321():
    """scatter_reduce silently wrong with negative indices (CUDA)"""
    src = torch.tensor([1., 2., 3., 4., 5., 6.], device='cuda')
    index = torch.tensor([0, 1, 0, 1, 2, -1], device='cuda')
    inp = torch.tensor([1., 2., 3., 4.], device='cuda')
    eager_error = False
    try:
        eager_out = torch.scatter_reduce(inp, 0, index, src, "sum")
    except (RuntimeError, IndexError):
        eager_error = True
    torch._dynamo.reset()
    compiled_sr = torch.compile(torch.scatter_reduce)
    compiled_error = False
    try:
        compiled_out = compiled_sr(inp, 0, index, src, "sum")
    except (RuntimeError, IndexError):
        compiled_error = True
    if eager_error and not compiled_error:
        return ('BUG', f'eager raises error but compiled silently returns wrong result')
    if not eager_error and not compiled_error:
        if not torch.allclose(eager_out, compiled_out, atol=1e-5):
            return ('BUG', f'results differ')
    return ('PASS', '')


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
        # Batch C: 93xxx series
        (93283,  "stack+inplace silu [CUDA]",                test_93283),
        (93351,  "pad+gt float64 [CPU]",                     test_93351),
        (93357,  "cat+expand Parameter [CUDA]",              test_93357),
        (93365,  "cat+mul global zeros [CPU]",               test_93365),
        (93374,  "where broadcast wrong [CPU]",              test_93374),
        (93823,  "masked_fill no-op [CUDA]",                 test_93823),
        (93824,  "reciprocal_ squeezed view [CPU]",          test_93824),
        (93829,  "neg+lt uint8 [CUDA]",                      test_93829),

        # Batch D
        (95695,  "ge int vs float [CUDA]",                   test_95695),
        (98149,  "pow+cos uint8 [CUDA]",                     test_98149),
        (99840,  "ConvTranspose+gt+mul NaN [CUDA]",          test_99840),
        (100466, "x*(x>=0) NaN [CPU]",                       test_100466),
        (100794, "conv2d+interpolate grad [CPU]",             test_100794),
        (108520, "int32 overflow [CPU]",                      test_108520),

        # Batch A: output-order miscompiles
        (96064,  "reduce-overhead loop [CUDA]",              test_96064),
        (113632, "output order sub [CUDA]",                  test_113632),
        (114221, "extra cat corrupts lt [CUDA]",             test_114221),
        (114227, "extra branch corrupts sub [CUDA]",         test_114227),
        (115159, "extra transpose corrupts sin [CPU]",       test_115159),
        (115774, "stale buffer attr reset [CPU]",            test_115774),

        # Batch B
        (124357, "mutable attr as constant [CPU]",           test_124357),
        (125999, "copy_() no-op [CUDA]",                     test_125999),
        (126537, "math.trunc wrong dtype [CUDA]",            test_126537),

        # Batch E: output-order miscompiles + scatter_reduce
        (113636, "argmin output order [CUDA]",               test_113636),
        (114020, "extra cat corrupts div [CUDA]",            test_114020),
        (114222, "lt output order [CUDA]",                   test_114222),
        (114223, "mean output order [CUDA]",                 test_114223),
        (114224, "add output order [CUDA]",                  test_114224),
        (114225, "extra cat corrupts softmax [CUDA]",        test_114225),
        (114226, "interpolate output order [CUDA]",          test_114226),
        (114229, "extra output corrupts argmax [CUDA]",      test_114229),
        (115160, "cos/min output order [CPU]",               test_115160),
        (131321, "scatter_reduce neg index [CUDA]",          test_131321),
    ]

    for issue_num, desc, test_fn in tests:
        test_bug(issue_num, desc, test_fn)

    bugs = [(k, v) for k, v in RESULTS.items() if v[0] == 'BUG']
    passes = [(k, v) for k, v in RESULTS.items() if v[0] == 'PASS']
    errors = [(k, v) for k, v in RESULTS.items() if v[0] == 'ERROR']
    skips = [(k, v) for k, v in RESULTS.items() if v[0] == 'SKIP']

    print(f"{'Issue':>8s}  {'Status':>6s}  {'Description':<45s}  Detail")
    print("-" * 110)
    for issue_num in sorted(RESULTS.keys()):
        status, desc, detail = RESULTS[issue_num]
        marker = '***' if status == 'BUG' else '   '
        print(f"#{issue_num:>7d}  {status:>6s}  {desc:<45s}  {detail[:50]} {marker}")

    print()
    print(f"Total tests:         {len(RESULTS)}")
    print(f"BUG (reproducible):  {len(bugs)}")
    print(f"PASS (fixed/absent): {len(passes)}")
    print(f"ERROR (test issue):  {len(errors)}")
    print(f"SKIP:                {len(skips)}")
    if bugs:
        print()
        print("Reproducible bug issues:")
        for k, v in sorted(bugs):
            print(f"  #{k}: {v[1]}")
