#!/usr/bin/env python3
"""
Verify which bugs from the PyTorch 2.2.0 correctness bug dataset are reproducible.
Each test uses the ORIGINAL reproducer from the GitHub issue.
Returns: eager vs compiled comparison (differential oracle).

Dataset: 63 confirmed torch.compile correctness bugs from GitHub issues.
"""
import torch
import sys
import traceback
import copy
import math

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
# EXISTING 30 tests (from verify_bugs.py)
# ============================================================================

def test_93262():
    """interpolate bilinear to [1,1] wrong result (CUDA)"""
    def fn(x):
        return torch.nn.functional.interpolate(x, size=[1, 1], mode="bilinear")
    x = torch.rand([2, 8, 7, 10], device='cuda')
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_97083():
    """squeeze+mutation+if inconsistent (CPU, int64)"""
    def forward():
        a = torch.tensor([1])
        a = a[0:1]
        b = a.squeeze()
        a[0] = 0
        if a[0] < 1e5:
            pass
        a[0] = 2
        return b
    with torch.no_grad():
        eager = forward()
        torch._dynamo.reset()
        compiled = torch.compile(forward)()
    if eager.item() != compiled.item():
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_98143():
    """add+unfold+abs_ wrong results (CUDA)"""
    def fn(x):
        t1 = torch.add(x, x)
        t2 = t1.unfold(1, 3, 2)
        t3 = t2.abs_()
        return t3
    x = torch.rand([1, 5], device='cuda')
    eager = fn(x.clone())
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x.clone())
    if not torch.allclose(eager, compiled, rtol=1e-2, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_99686():
    """split+cat wrong reordering (CPU)"""
    torch.manual_seed(420)
    def forward(x):
        x = torch.split(x, [3, 2, 3], dim=1)
        x = torch.cat([x[1], x[0], x[2]], dim=1)
        return x
    x = torch.randn(1, 8)
    eager = forward(x)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(x)
    if not torch.allclose(eager, compiled):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_100775():
    """mul(big).softmax NaN (CPU)"""
    torch.manual_seed(420)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Linear(32, 1)
            self.key = torch.nn.Linear(32, 1)
            self.scale_factor = torch.nn.Parameter(torch.tensor([-1.7197e+14]))
        def forward(self, x):
            query = self.query(x)
            key = self.key(x)
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores_scaled = scores.mul(self.scale_factor)
            return scores_scaled.softmax(dim=-1)
    x = torch.randn(10, 32)
    model = Model()
    eager = model(x)
    torch._dynamo.reset()
    compiled = torch.compile(model)(x)
    eager_has_nan = eager.isnan().any().item()
    compiled_has_nan = compiled.isnan().any().item()
    if not eager_has_nan and compiled_has_nan:
        return ('BUG', f'compiled produces NaN, eager does not')
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3, equal_nan=True):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_100800():
    """where SIMD incorrect (CPU)"""
    input_tensor = torch.ones(3, 3)
    def f(x):
        return torch.where(torch.ones_like(x).to(torch.bool),
                           torch.zeros_like(x),
                           torch.ones_like(x) * 2)
    eager = f(input_tensor)
    torch._dynamo.reset()
    compiled = torch.compile(f)(input_tensor)
    if not torch.allclose(eager, compiled):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_100987():
    """AvgPool ceil_mode inconsistent (CPU)"""
    torch.manual_seed(420)
    x = torch.randn(1, 3, 6, 6)
    def forward(x):
        return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
    eager = forward(x)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(x)
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_101039():
    """softmax sum != 1 (CUDA)"""
    torch.manual_seed(420)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=-1)
        def forward(self, query, key):
            y = torch.matmul(query, key.transpose(-2, -1))
            z = y.div(1e-06)
            w = self.softmax(z)
            return w
    device = 'cuda'
    func = Model().to(device)
    query = torch.randn(1, 10, 40).to(device)
    key = torch.randn(1, 2, 40).to(device)
    eager = func(query, key)
    torch._dynamo.reset()
    compiled = torch.compile(func)(query, key)
    eager_sums = eager.sum(dim=-1)
    compiled_sums = compiled.sum(dim=-1)
    if not torch.allclose(eager_sums, compiled_sums, atol=1e-3):
        diff = (eager_sums - compiled_sums).abs().max().item()
        return ('BUG', f'sum_diff={diff:.6f}')
    return ('PASS', '')

def test_104505():
    """as_strided wrong functionalization (CPU)"""
    def foo(a):
        e = a.diagonal()
        f = e.as_strided((2,), (1,), 0)
        f.add_(1.0)
        return a
    a = torch.randn(2, 4)
    a_ref = a.clone()
    eager = foo(a_ref)
    torch._dynamo.reset()
    compiled = torch.compile(foo, dynamic=True)(a.clone())
    if not torch.allclose(eager, compiled):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_111133():
    """index_add_ stride mismatch (CPU)"""
    def transpose_inplace_index_add(x, a, index):
        y = x.permute(1, 0)
        y.index_add_(0, index, a)
        return y
    x = torch.ones(3, 5)
    to_be_added = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    index = torch.tensor([0, 4, 2])
    x_eager = x.clone()
    eager = transpose_inplace_index_add(x_eager, to_be_added, index)
    torch._dynamo.reset()
    x_compiled = x.clone()
    compiled_fn = torch.compile(transpose_inplace_index_add)
    compiled = compiled_fn(x_compiled, to_be_added, index)
    if not torch.allclose(eager, compiled):
        diff = (eager.float() - compiled.float()).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    if eager.stride() != compiled.stride():
        return ('BUG', f'stride mismatch: eager={eager.stride()} compiled={compiled.stride()}')
    return ('PASS', '')

def test_113013():
    """adaptive_avg_pool1d+argmax (CPU, float64)"""
    def forward(x):
        x = torch.nn.functional.adaptive_avg_pool1d(input=x, output_size=2)
        x = torch.argmax(input=x)
        return x
    x = torch.rand([3, 3, 3], dtype=torch.float64)
    eager = forward(x)
    torch._dynamo.reset()
    compiled = torch.compile(forward, fullgraph=True, dynamic=True)(x)
    if not torch.allclose(eager.float(), compiled.float()):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_113151():
    """torch.asinh wrong output with out= (CUDA)"""
    def forward(x, y):
        return torch.asinh(out=x, input=y)
    y = torch.tensor([487875.875, -956238.8125, 630736.0, -161079.578125, 104060.9375,
                       757224.3125, -153601.859375, -648042.5, 733955.4375, -214764.90625],
                      dtype=torch.float32, device='cuda')
    x1 = torch.rand([10], dtype=torch.float32, device='cuda')
    x2 = x1.clone()
    eager = forward(x1, y)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(forward, mode='max-autotune-no-cudagraphs', fullgraph=True)(x2, y)
    except Exception:
        compiled = torch.compile(forward, fullgraph=True)(x2, y)
    if not torch.allclose(eager.cpu(), compiled.cpu(), rtol=1e-3, atol=1e-3):
        diff = (eager.cpu() - compiled.cpu()).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_113641():
    """double inplace clamp wrong result (CPU, static shapes)"""
    def eager_fn(input, top, left, height, width):
        output = input - torch.tensor([left, top, left, top], dtype=input.dtype, device=input.device)
        output[..., 0::2].clamp_(min=0, max=width)
        output[..., 1::2].clamp_(min=0, max=height)
        return output
    input_t = torch.tensor([[0.0, 1.0, 10.0, 14.0]])
    kwargs = dict(top=7, left=3, height=3, width=5)
    eager = eager_fn(input_t.clone(), **kwargs)
    torch._dynamo.reset()
    compiled_static = torch.compile(eager_fn, dynamic=False)(input_t.clone(), **kwargs)
    if not torch.allclose(eager, compiled_static):
        diff = (eager - compiled_static).abs().max().item()
        return ('BUG', f'eager={eager.tolist()} compiled={compiled_static.tolist()} max_diff={diff}')
    return ('PASS', '')

def test_114302():
    """expand+mutation wrong result (CPU, int64)"""
    @torch.compile(backend="inductor")
    def foo(x):
        y = x.expand(2, *x.shape)
        y[0, 0] = 5
        return y
    torch._dynamo.reset()
    x = torch.arange(6, device="cpu")
    r = foo(x)
    expected_x = torch.tensor([5, 1, 2, 3, 4, 5])
    expected_r = torch.tensor([[5, 1, 2, 3, 4, 5], [5, 1, 2, 3, 4, 5]])
    if not torch.equal(x, expected_x) or not torch.equal(r, expected_r):
        return ('BUG', f'x={x.tolist()} r={r.tolist()}')
    return ('PASS', '')

def test_114464():
    """clone+reshape silently ignored (CPU, fullgraph)"""
    def eager_fn(x):
        y = x.clone().reshape(-1, 4)
        y[:, [2, 0]] = y[:, [0, 2]]
        return y
    x = torch.tensor([0, 1, 2, 3])
    x_for_compiled = x.clone()
    eager = eager_fn(x.clone())
    torch._dynamo.reset()
    compiled = torch.compile(backend="inductor", fullgraph=True)(eager_fn)(x_for_compiled)
    if not torch.equal(x, torch.tensor([0, 1, 2, 3])):
        return ('BUG', f'input x mutated to {x_for_compiled.tolist()} (clone not respected)')
    if not torch.allclose(eager.float(), compiled.float()):
        diff = (eager.float() - compiled.float()).abs().max().item()
        return ('BUG', f'max_diff={diff}')
    return ('PASS', '')

def test_116433():
    """different output strides (CUDA)"""
    def f(x):
        tmp = x + 1
        return tmp.view(-1, 1, 2)
    x = torch.arange(8, device='cuda', dtype=torch.float32)
    eager = f(x)
    torch._dynamo.reset()
    compiled = torch.compile(f)(x)
    if eager.stride() != compiled.stride():
        return ('BUG', f'stride mismatch: eager={eager.stride()} compiled={compiled.stride()}')
    return ('PASS', '')

def test_116935():
    """Compiled Transformer accuracy (CUDA)"""
    torch.manual_seed(0)
    def fn(x):
        layer = torch.nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True).cuda().eval()
        return layer(x)
    x = torch.randn(1, 8, 16, device='cuda')
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.allclose(eager, compiled, atol=1e-2):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_117019():
    """diag_embed different outputs (CPU, float64, non-default dims)"""
    def forward(x):
        return torch.diag_embed(input=x, dim1=-1, dim2=0, offset=1)
    x = torch.rand([8, 6, 8, 6, 6, 1], dtype=torch.float64)
    eager = forward(x)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(x)
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_117544():
    """sinh+relu wrong result (CPU, max-autotune)"""
    def forward(x):
        out = torch.sinh(input=x)
        out = torch.relu(input=out)
        return out
    x = torch.tensor([-89.0, -1.0, 0.0, 1.0, 89.0])
    eager = forward(x)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(forward, mode='max-autotune')(x)
    except Exception:
        compiled = torch.compile(forward)(x)
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3, equal_nan=True):
        diff_mask = ~torch.isclose(eager, compiled, rtol=1e-3, atol=1e-3, equal_nan=True)
        if diff_mask.any():
            idx = diff_mask.nonzero()[0].item()
            return ('BUG', f'eager[{idx}]={eager[idx].item()} compiled[{idx}]={compiled[idx].item()}')
    if eager.isnan().any() != compiled.isnan().any():
        return ('BUG', f'NaN mismatch')
    return ('PASS', '')

def test_118265():
    """bitwise_right_shift wrong with out= int16 (CPU)"""
    def forward(out, input, other):
        return torch.bitwise_right_shift(out=out, input=input, other=other)
    out1 = torch.randint(-32768, 32767, [7, 2, 6, 7], dtype=torch.int16)
    out2 = out1.clone()
    input_t = torch.randint(0, 100, [7, 2, 6, 7], dtype=torch.int64)
    other_t = torch.randint(0, 100, [7, 2, 6, 7], dtype=torch.int64)
    eager = forward(out1, input_t, other_t)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(forward, mode='reduce-overhead')(out2, input_t, other_t)
    except Exception:
        compiled = torch.compile(forward)(out2, input_t, other_t)
    if not torch.equal(eager, compiled):
        n_diff = (eager != compiled).sum().item()
        total = eager.numel()
        return ('BUG', f'{n_diff}/{total} elements differ')
    return ('PASS', '')

def test_118266():
    """log1p+argmax inconsistency with out= (CPU)"""
    def forward(input, out):
        return torch.argmax(input=torch.log1p(out=out, input=input))
    input_t = torch.randn([10, 9, 8], dtype=torch.float32)
    out1 = torch.rand([10, 9, 8], dtype=torch.float32)
    out2 = out1.clone()
    eager = forward(input_t, out1)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(input_t, out2)
    if not torch.equal(eager, compiled):
        diff = (eager.float() - compiled.float()).abs().max().item()
        return ('BUG', f'argmax diff={diff}')
    return ('PASS', '')

def test_121174():
    """fuse_attention inconsistent (CPU)"""
    torch.manual_seed(0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_conv = torch.nn.Conv2d(4, 4, 1)
            self.k_conv = torch.nn.Conv2d(4, 4, 1)
            self.v_conv = torch.nn.Conv2d(4, 4, 1)
        def forward(self, x):
            q = self.q_conv(x)
            k = self.k_conv(x)
            v = self.v_conv(x)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            div = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            attn_weight = torch.nn.functional.softmax(div, dim=-1)
            return torch.matmul(attn_weight, v)
    func = Model()
    x = torch.randn(1, 4, 2, 2)
    with torch.no_grad():
        eager = func(x.clone())
        torch._dynamo.reset()
        compiled = torch.compile(func)(x.clone())
    if not torch.allclose(eager, compiled, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_122260():
    """fma precision issue (CUDA)"""
    torch.set_default_device('cuda')
    scale = torch.tensor(0.180336877703666687)
    x = torch.tensor(1134139801600.000000)
    def f(x, scale):
        max_scaled = x * scale
        return torch.exp(max_scaled - x * scale)
    eager = f(x, scale)
    torch._dynamo.reset()
    compiled = torch.compile(f)(x, scale)
    torch.set_default_device('cpu')
    if eager.item() != compiled.item() and (compiled.isinf().item() or compiled.isnan().item()):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_100793():
    """pad reflect wrong backward gradient (CPU, fullgraph)"""
    torch.manual_seed(0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v9 = torch.nn.Parameter(torch.ones(size=[7, 37, 7, 2], dtype=torch.float32))
        def forward(self):
            return torch.nn.functional.pad(self.v9, pad=[-1, 0, 6, 6], mode='reflect')
    model = Model()
    model_compiled = copy.deepcopy(model)
    model_compiled.train()
    torch._dynamo.reset()
    compiled = torch.compile(model_compiled, fullgraph=True)
    ret_compiled = compiled()
    grad_compiled = torch.autograd.grad(
        outputs=ret_compiled, inputs=list(compiled.parameters()),
        grad_outputs=torch.ones_like(ret_compiled), retain_graph=True)
    model.train()
    ret_eager = model()
    grad_eager = torch.autograd.grad(
        outputs=ret_eager, inputs=list(model.parameters()),
        grad_outputs=torch.ones_like(ret_eager), retain_graph=True)
    fwd_ok = torch.allclose(ret_compiled, ret_eager)
    bwd_ok = torch.allclose(grad_compiled[0], grad_eager[0], rtol=1e-2, atol=1e-3)
    if not bwd_ok:
        diff = (grad_compiled[0] - grad_eager[0]).abs().max().item()
        return ('BUG', f'grad_diff={diff:.6f} fwd_ok={fwd_ok}')
    if not fwd_ok:
        return ('BUG', f'forward mismatch')
    return ('PASS', '')

def test_127749():
    """gumbel_softmax NaN (CUDA)"""
    import torch.nn.functional as F
    class GumbelVQ(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_proj = torch.nn.Linear(256, 32 * 320)
            self.weight_proj.weight.data.normal_(mean=0.0, std=1)
            self.weight_proj.bias.data.zero_()
        def forward(self, x):
            b, s, h = x.shape
            x = self.weight_proj(x)
            x = x.view(b * s * 32, -1)
            return F.gumbel_softmax(x.float(), tau=2, hard=True).type_as(x)
    vq = GumbelVQ().cuda()
    torch._dynamo.reset()
    vq_compiled = torch.compile(vq)
    for i in range(200):
        x = torch.randn(4, 400, 256, device='cuda')
        out = vq(x)
        out_c = vq_compiled(x)
        if out.isnan().any() or out_c.isnan().any():
            if out_c.isnan().any() and not out.isnan().any():
                return ('BUG', f'compiled NaN at iter {i}, eager ok')
            elif out.isnan().any() and out_c.isnan().any():
                return ('PASS', 'both NaN (numerical)')
            break
    return ('PASS', '')

def test_132107():
    """cumsum NaN with inf inputs (CUDA)"""
    torch._dynamo.reset()
    cfn = torch.compile(torch.cumsum)
    results = []
    for n in [100, 10, 100]:
        r = torch.sum(cfn(torch.full((n,), float('inf'), device='cuda', dtype=torch.float64), -1))
        results.append(r)
    if any(r.isnan().item() for r in results):
        bad = [i for i, r in enumerate(results) if r.isnan().item()]
        return ('BUG', f'NaN at calls {bad}')
    return ('PASS', '')

def test_138908():
    """inplace mutation 4x vs 2x mismatch (CPU, dynamic)"""
    @torch.compile(backend="inductor", dynamic=True)
    def test1_fn(x, y):
        x = x + 10
        y[x] += y[x]
    x = torch.randint(-10, -9, (1, 2), dtype=torch.int64)
    y = torch.randn((2, 32), dtype=torch.float32)
    x_clone = x.clone()
    y_eager = y.clone()
    with torch.no_grad():
        x_e = x_clone + 10
        y_eager[x_e] += y_eager[x_e]
        torch._dynamo.reset()
        test1_fn(x.clone(), y)
    if not torch.allclose(y_eager, y, atol=1e-5):
        diff = (y_eager - y).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_141538():
    """FractionalMaxPool2d inconsistent (CPU)"""
    try:
        from torch._inductor import config
        config.fallback_random = True
    except Exception:
        pass
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.FractionalMaxPool2d(kernel_size=(1, 1), output_ratio=(0.5, 0.5))
        def forward(self, x):
            return self.pool(x)
    model = Model().eval()
    torch._dynamo.reset()
    c_model = torch.compile(model)
    x = torch.randn(1, 1, 10, 10)
    torch.manual_seed(0)
    output1 = model(x)
    torch.manual_seed(0)
    output2 = model(x)
    torch.manual_seed(0)
    c_output = c_model(x)
    if not torch.allclose(output1, output2):
        return ('SKIP', 'eager not deterministic')
    if not torch.allclose(output1, c_output, atol=1e-3):
        diff = (output1 - c_output).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_143555():
    """bitwise_right_shift wrong with other==64 (CPU)"""
    def forward(input, other):
        return torch.bitwise_right_shift(input=input, other=other)
    input_t = torch.tensor(1000, dtype=torch.int64)
    other_t = torch.tensor(64, dtype=torch.int64)
    eager = forward(input_t, other_t)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(input_t, other_t)
    if eager.item() != compiled.item():
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_146416():
    """fusing transpose silent correctness bug (CUDA)"""
    def fn(x):
        return x.transpose(1, 2).contiguous() + 1
    x = torch.randn(2, 3, 4, device='cuda')
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.allclose(eager, compiled, atol=1e-5):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')


# ============================================================================
# NEW tests: In-Place Operation Handling Bugs
# ============================================================================

def test_93078():
    """dstack+reciprocal wrong on bool scalar (CUDA)"""
    def fn(input):
        v1 = torch.dstack([input, input, input])
        return torch.reciprocal(v1)
    x = torch.tensor(True)
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.allclose(eager.float(), compiled.float(), atol=1e-3):
        return ('BUG', f'eager={eager.tolist()} compiled={compiled.tolist()}')
    return ('PASS', '')

def test_93363():
    """AOTautograd in_place update on view tensor wrong (CUDA)"""
    def foo(a):
        a.add_(1.0)
        return
    a1 = torch.ones([4], requires_grad=True).add(0.)
    a1_view = a1[0::2]
    a2 = torch.ones([4], requires_grad=True).add(0.)
    a2_view = a2[0::2]
    foo(a2_view)
    torch._dynamo.reset()
    torch.compile(foo)(a1_view)
    if not torch.allclose(a1, a2, atol=1e-3):
        return ('BUG', f'compiled a1={a1.tolist()} eager a2={a2.tolist()}')
    return ('PASS', '')

def test_96604():
    """neg on uint8 wrong results (CUDA)"""
    def forward():
        x = torch.tensor(5, device='cuda', dtype=torch.uint8)
        y = torch.neg(x)
        return x < y
    eager = forward()
    torch._dynamo.reset()
    compiled = torch.compile(forward)()
    if eager.item() != compiled.item():
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_97117():
    """F.pad zero-padding alias bug (CUDA)"""
    def forward():
        a = torch.tensor([2.], device='cuda:0')
        b = torch.nn.functional.pad(a, (0, 0))
        if a.mean() >= -1e5:
            a[0] = 1
        return b
    with torch.no_grad():
        eager = forward()
        torch._dynamo.reset()
        compiled = torch.compile(forward)()
    if not torch.allclose(eager, compiled):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_97130():
    """shared storage mutation in nested func (CPU)"""
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
    eager = forward()
    torch._dynamo.reset()
    compiled = torch.compile(forward)()
    if not torch.allclose(eager, compiled):
        return ('BUG', f'eager={eager.tolist()} compiled={compiled.tolist()}')
    return ('PASS', '')

def test_100224():
    """masked in-place multiply wrong shape (CPU)"""
    torch.manual_seed(420)
    x = torch.randn(1, 3, 2, 2)
    class Model(torch.nn.Module):
        def forward(self, x):
            out = x
            mask1 = out > 0
            out[mask1] *= 0.5
            return out
    func = Model()
    res1 = func(x.clone())
    torch._dynamo.reset()
    res2 = torch.compile(func)(x.clone())
    if res1.shape != res2.shape:
        return ('BUG', f'shape: eager={res1.shape} compiled={res2.shape}')
    if not torch.allclose(res1, res2, atol=1e-5):
        diff = (res1.flatten() - res2.flatten()).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_100970():
    """batch_norm wrong output shape (CPU)"""
    import torch.nn.functional as F
    torch.manual_seed(420)
    x = torch.randn(1, 3, 2, 2)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(3, 1, 1, 1)
            self.bias = torch.randn(3)
            self.bn = torch.nn.BatchNorm2d(num_features=3, affine=False)
        def forward(self, x):
            return F.batch_norm(x, self.bn.running_mean, self.bn.running_var,
                                self.weight, self.bias)
    func = Model().eval()
    with torch.no_grad():
        res1 = func(x)
        torch._dynamo.reset()
        res2 = torch.compile(func)(x)
    if res1.shape != res2.shape:
        return ('BUG', f'shape: eager={res1.shape} compiled={res2.shape}')
    if not torch.allclose(res1, res2, atol=1e-4):
        diff = (res1 - res2).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_104020():
    """permute_linear_fusion ignores resize_ (CUDA)"""
    torch.manual_seed(420)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
        def forward(self, x1):
            v1 = x1.permute(0, 2, 1)
            v1.resize_(1, 1, 2)
            v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
            return v2
    func = Model().cuda()
    x = torch.randn(1, 2, 2).cuda()
    with torch.no_grad():
        func.eval()
        res1 = func(x)
        torch._dynamo.reset()
        res2 = torch.compile(func)(x)
    if res1.shape != res2.shape:
        return ('BUG', f'shape: eager={res1.shape} compiled={res2.shape}')
    if not torch.allclose(res1, res2, rtol=1e-3, atol=1e-3):
        diff = (res1 - res2).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_108472():
    """storage offset discarded on sliced inputs (CPU)"""
    @torch.compile(fullgraph=True)
    def fn(x):
        return x[16:32]
    with torch.no_grad():
        x = torch.randn(1024)
        r1 = fn(x[0:])
        ok1 = torch.equal(r1, x[16:][:16])
        torch._dynamo.reset()
        @torch.compile(fullgraph=True)
        def fn2(x):
            return x[16:32]
        r2 = fn2(x[128:])
        ok2 = torch.equal(r2, x[128 + 16:][:16])
    if not ok2:
        return ('BUG', f'offset 128 discarded')
    return ('PASS', '')

def test_113030():
    """.data= assignment mismatch (CPU)"""
    def func(x, y):
        x.data = y
        y.data = torch.zeros(0)
        return x
    a = torch.rand([6])
    b = torch.rand([6])
    a_ref = a.clone()
    b_ref = b.clone()
    eager = func(a_ref, b_ref)
    a2 = a.clone()
    b2 = b.clone()
    torch._dynamo.reset()
    compiled = torch.compile(func)(a2, b2)
    if eager.numel() != compiled.numel():
        return ('BUG', f'eager numel={eager.numel()} compiled numel={compiled.numel()}')
    if eager.numel() > 0 and compiled.numel() > 0 and not torch.allclose(eager, compiled):
        return ('BUG', f'values differ')
    return ('PASS', '')

def test_115071():
    """torch.add input mutation on complex64 (CPU)"""
    import numpy as np
    class Model0(torch.nn.Module):
        def forward(self, *args):
            neg = torch.neg(args[0])
            add = torch.add(args[0], args[0])
            return (neg, add)
    data = np.random.normal(5, 1, size=(41,)).astype(np.complex64)
    input_t = torch.from_numpy(data).to('cpu')
    input_orig = input_t.clone()
    torch._dynamo.reset()
    model_compiled = torch.compile(Model0(), backend='inductor')
    out1 = model_compiled(input_t)
    if not torch.allclose(input_t, input_orig, atol=1e-5):
        diff = (input_t - input_orig).abs().max().item()
        return ('BUG', f'input mutated! max_diff={diff:.6f}')
    return ('PASS', '')

def test_125284():
    """nn.Parameter aliasing lost under compile (CPU)"""
    def f(x):
        y = x + 1
        z = torch.nn.Parameter(y)
        with torch.no_grad():
            z.mul_(2)
        return y + z
    x = torch.ones(2, requires_grad=True)
    out_ref = f(x)
    torch._dynamo.reset()
    out_test = torch.compile(f, backend='aot_eager')(x)
    if not torch.allclose(out_ref.detach(), out_test.detach(), atol=1e-3):
        return ('BUG', f'eager={out_ref.detach().tolist()} compiled={out_test.detach().tolist()}')
    return ('PASS', '')

def test_127474():
    """tile+as_strided+add wrong output (CUDA)"""
    def foo(x):
        view_size = (3, 2)
        full = x.tile((3, 2))
        view = torch.as_strided(full, view_size, full.stride())
        result = view + view
        return result
    device = torch.device("cuda")
    x = torch.randn(1, 1).to(device)
    ref = foo(x)
    torch._dynamo.reset()
    compiled = torch.compile(foo)(x)
    if not torch.allclose(ref, compiled, atol=1e-5):
        return ('BUG', f'max_diff={(ref-compiled).abs().max().item():.6f}')
    return ('PASS', '')

def test_128035():
    """detach+inplace zeros gradient (CPU)"""
    def f(a):
        tmp = a.detach()
        a.mul_(2)
        return a, tmp
    inp = torch.ones(3, requires_grad=True)
    graph_inp = inp.mul(2)
    outs = f(graph_inp)
    outs[0].sum().backward()
    eager_grad = inp.grad.clone()
    inp2 = torch.ones(3, requires_grad=True)
    graph_inp2 = inp2.mul(2)
    torch._dynamo.reset()
    outs2 = torch.compile(f, backend="aot_eager")(graph_inp2)
    outs2[0].sum().backward(retain_graph=True)
    compiled_grad = inp2.grad
    if not torch.allclose(eager_grad, compiled_grad, atol=1e-3):
        return ('BUG', f'eager_grad={eager_grad.tolist()} compiled_grad={compiled_grad.tolist()}')
    return ('PASS', '')


# ============================================================================
# NEW tests: Operator Transformation Bugs
# ============================================================================

def test_108963():
    """index_add with alpha returns 0 (CPU)"""
    @torch.compile()
    def f():
        device = 'cpu'
        tensor = torch.zeros((1,), dtype=torch.double, device=device)
        index = torch.tensor([0], dtype=torch.long, device=device)
        source = torch.rand((1,), dtype=torch.double, device=device)
        out = tensor.index_add(0, index, source, alpha=2.) / 2
        return source.item(), out.item()
    torch.random.manual_seed(0)
    torch._dynamo.reset()
    src_val, out_val = f()
    if abs(out_val) < 1e-10 and abs(src_val) > 1e-5:
        return ('BUG', f'source={src_val:.6f} but index_add result={out_val:.6f}')
    return ('PASS', '')

def test_109925():
    """cumsum on bool returns wrong dtype (CUDA)"""
    class Model(torch.nn.Module):
        def forward(self):
            t1 = torch.full([2, 4], 1)
            t2 = t1.to(dtype=torch.bool)
            t3 = torch.cumsum(t2, 1)
            return t3
    eager = Model()()
    torch._dynamo.reset()
    compiled = torch.compile(Model())()
    if eager.dtype != compiled.dtype:
        return ('BUG', f'dtype: eager={eager.dtype} compiled={compiled.dtype}')
    if not torch.equal(eager, compiled):
        return ('BUG', f'values differ')
    return ('PASS', '')

def test_113167():
    """var_mean with fractional correction wrong (CUDA)"""
    def fn(x):
        return torch.var_mean(input=x, correction=4.9, dim=-1, keepdim=True)
    x = torch.rand([], dtype=torch.float32, device='cuda')
    eag = fn(x)
    torch._dynamo.reset()
    opt = torch.compile(fn)(x)
    if not torch.allclose(eag[0].cpu(), opt[0].cpu(), rtol=1e-3, atol=1e-3, equal_nan=True):
        return ('BUG', f'var: eager={eag[0].item():.6f} compiled={opt[0].item():.6f}')
    return ('PASS', '')

def test_121227():
    """merge_stack_tanh_unbind wrong values (CPU)"""
    torch.manual_seed(42)
    class Model(torch.nn.Module):
        def forward(self, x1):
            v1 = torch.split(x1, [3, 3, 3], dim=-1)
            v2 = torch.stack(v1, dim=-1)
            v3 = torch.tanh(v2)
            return v3
    func = Model()
    x = torch.randn(1, 9)
    with torch.no_grad():
        eager = func(x.clone())
        torch._dynamo.reset()
        compiled = torch.compile(func)(x.clone())
    if not torch.allclose(eager, compiled, atol=1e-5):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_122379():
    """split_cat wrong output shape (CPU)"""
    torch.manual_seed(0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indices = [i for i in range(0, 10)]
        def forward(self, x):
            split_tensors = torch.split(x, 1, 0)
            chosen_tensors = [split_tensors[i] for i in self.indices if i in range(0, 10)]
            result = torch.cat(chosen_tensors, 0)
            return result
    func = Model()
    x = torch.randn(10, 3, 64, 64)
    with torch.no_grad():
        eager = func(x.clone())
        torch._dynamo.reset()
        compiled = torch.compile(func)(x.clone())
    if eager.shape != compiled.shape:
        return ('BUG', f'shape: eager={list(eager.shape)} compiled={list(compiled.shape)}')
    if not torch.allclose(eager, compiled, atol=1e-5):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_122382():
    """conv_add optimization drops alpha (CPU)"""
    torch.manual_seed(420)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
            self.other_tensor = torch.randn(2, 1, 1)
        def forward(self, x1):
            v1 = self.conv(x1)
            v2 = torch.add(v1, self.other_tensor, alpha=0.875)
            return v2
    func = Model()
    x = torch.randn(1, 3, 2, 2)
    with torch.no_grad():
        eager = func(x.clone())
        torch._dynamo.reset()
        compiled = torch.compile(func)(x.clone())
    if not torch.allclose(eager, compiled, atol=1e-4):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_105190():
    """SDPA NaN with negative scale (CPU)"""
    torch.manual_seed(420)
    class Model(torch.nn.Module):
        def forward(self, x):
            v1 = torch.matmul(x, x.transpose(-1, -2))
            v2 = v1 / -0.0001
            v3 = v2.softmax(dim=-1)
            v4 = torch.matmul(v3, x)
            return v4
    func = Model()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        func.eval()
        res1 = func(x)
        torch._dynamo.reset()
        res2 = torch.compile(func)(x)
    if res2.isnan().any() and not res1.isnan().any():
        return ('BUG', f'compiled produces NaN, eager does not')
    if not torch.allclose(res1, res2, atol=1e-3, equal_nan=True):
        diff = (res1 - res2).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')


# ============================================================================
# NEW tests: Low-level Code Generation Bugs
# ============================================================================

def test_113010():
    """abs with transposed out= wrong (CUDA)"""
    def forward(x, y):
        return torch.abs(input=x, out=torch.t(input=y))
    x = torch.rand([9, 10], dtype=torch.float32)
    y1 = torch.rand([10, 9], dtype=torch.float32)
    y2 = y1.clone()
    eager = forward(x, y1)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(x, y2)
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_113012():
    """hinge_embedding_loss discrepancy (CUDA)"""
    def forward(x, y):
        return torch.nn.functional.hinge_embedding_loss(
            input=x, target=y, margin=0.11, reduction='sum')
    x = torch.tensor([-668044.5], dtype=torch.float32)
    y = torch.rand([1], dtype=torch.float32)
    eager = forward(x, y)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(forward, mode='reduce-overhead', fullgraph=True)(x, y)
    except Exception:
        compiled = torch.compile(forward)(x, y)
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3):
        diff = (eager - compiled).abs().item()
        return ('BUG', f'eager={eager.item():.6f} compiled={compiled.item():.6f}')
    return ('PASS', '')

def test_113018():
    """Conv2d inconsistent output (CPU)"""
    torch.manual_seed(42)
    conv = torch.nn.Conv2d(39, 1, kernel_size=(1, 17), stride=(2, 2))
    class Model0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m4 = conv
        def forward(self, x):
            relu = torch.relu(x)
            m4 = self.m4(x)
            return (relu, m4)
    x = torch.randn(1, 39, 10, 20)
    model = Model0()
    with torch.no_grad():
        model.eval()
        eager = model(x)
        torch._dynamo.reset()
        compiled_model = torch.compile(model, backend='inductor')
        compiled_out = compiled_model(x)
    if compiled_out[1].isnan().any() and not eager[1].isnan().any():
        return ('BUG', f'compiled produces NaN in conv output')
    if not torch.allclose(eager[1], compiled_out[1], atol=1e-3):
        diff = (eager[1] - compiled_out[1]).abs().max().item()
        return ('BUG', f'conv max_diff={diff:.6f}')
    return ('PASS', '')

def test_113141():
    """torch.sum with swapped mul inputs wrong (CPU)"""
    import numpy as np
    DEVICE = 'cpu'
    p0 = torch.nn.Parameter(torch.empty([40, 30], dtype=torch.uint8), requires_grad=False).to(DEVICE)
    p1 = torch.nn.Parameter(torch.empty([30], dtype=torch.uint8), requires_grad=False).to(DEVICE)
    torch.manual_seed(42)
    p0.data = torch.randint(0, 255, [40, 30], dtype=torch.uint8)
    p1.data = torch.randint(0, 255, [30], dtype=torch.uint8)
    class Model0(torch.nn.Module):
        def forward(self, x):
            mul = torch.mul(torch.add(p1.reshape(30), x), torch.neg(p0))
            return (mul.sum(0),)
    class Model1(torch.nn.Module):
        def forward(self, x):
            mul = torch.mul(torch.neg(p0), torch.add(p1.reshape(30), x))
            return (mul.sum(0),)
    x = [torch.randn(30).to(DEVICE)]
    torch._dynamo.reset()
    c0 = torch.compile(Model0(), backend='inductor')(*x)
    torch._dynamo.reset()
    c1 = torch.compile(Model1(), backend='inductor')(*x)
    if not torch.allclose(c0[0].float(), c1[0].float(), rtol=1e-3, atol=1e-3):
        diff = (c0[0].float() - c1[0].float()).abs().max().item()
        return ('BUG', f'commutativity violated, max_diff={diff:.6f}')
    return ('PASS', '')

def test_115260():
    """cat+mul+tan fusion wrong (CPU, float16)"""
    DEVICE = 'cpu'
    p0 = torch.tensor([1.0879], dtype=torch.float16).to(DEVICE)
    class Model0(torch.nn.Module):
        def forward(self, *args):
            cat = torch.cat((args[3], args[2], args[1], args[0]), dim=2)
            max_1 = torch.max(args[4], p0)
            mul = torch.mul(cat, max_1)
            tan = torch.tan(mul)
            return (tan,)
    class Model1(torch.nn.Module):
        def forward(self, *args):
            cat = torch.cat((args[3], args[2], args[1], args[0]), dim=2)
            max_1 = torch.max(args[4], p0)
            mul = torch.mul(cat, max_1)
            tan = torch.tan(mul)
            return (mul, tan)
    torch.manual_seed(42)
    inputs = [torch.randn(1, 1, i+1, dtype=torch.float16).to(DEVICE) for i in range(4)]
    inputs.append(torch.randn(1, dtype=torch.float16).to(DEVICE))
    torch._dynamo.reset()
    r0 = torch.compile(Model0(), backend='inductor')(*inputs)[0]
    torch._dynamo.reset()
    r1 = torch.compile(Model1(), backend='inductor')(*inputs)[1]
    if not torch.allclose(r0, r1, rtol=1e-2, atol=1e-2):
        diff = (r0.float() - r1.float()).abs().max().item()
        return ('BUG', f'output depends on return set, max_diff={diff:.6f}')
    return ('PASS', '')

def test_115261():
    """matmul output changes with extra outputs (CPU, uint8)"""
    import numpy as np
    DEVICE = 'cpu'
    class Model0(torch.nn.Module):
        def forward(self, *args):
            max_1 = torch.max(args[1], args[0])
            max_2 = torch.max(max_1, args[2])
            flatten = max_2.flatten()
            matmul = torch.matmul(flatten.float(), flatten.float())
            return (matmul,)
    class Model1(torch.nn.Module):
        def forward(self, *args):
            max_1 = torch.max(args[1], args[0])
            add = torch.add(max_1, args[0])
            cat = torch.cat((args[2], args[2]), dim=0)
            max_2 = torch.max(max_1, args[2])
            flatten = max_2.flatten()
            matmul = torch.matmul(flatten.float(), flatten.float())
            return (add, cat, matmul)
    np.random.seed(42)
    data_0 = np.array(6, dtype=np.uint8)
    data_1 = np.random.normal(5, 1, size=(43, 1, 32, 24)).astype(np.uint8)
    data_2 = np.random.normal(5, 1, size=(1, 1, 1)).astype(np.uint8)
    inputs = [torch.from_numpy(v).to(DEVICE) for v in [data_0, data_1, data_2]]
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), backend='inductor')(*inputs)[0]
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), backend='inductor')(*inputs)[2]
    if not torch.allclose(out0, out1, atol=1e-3):
        return ('BUG', f'matmul result depends on other outputs: {out0.item():.2f} vs {out1.item():.2f}')
    return ('PASS', '')

def test_117778():
    """roll+fill_ wrong outputs (CUDA)"""
    def roll_and_zero_first_row(x):
        x = torch.roll(x, -1, 0)
        x[0].fill_(0.0)
        return x
    size = (1024, 2)
    inp = torch.ones(size, device="cuda")
    eager = roll_and_zero_first_row(inp.clone())
    torch._dynamo.reset()
    compiled = torch.compile(roll_and_zero_first_row)(inp.clone())
    if not torch.allclose(eager, compiled, atol=1e-5):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_118267():
    """acosh returns -inf instead of NaN for invalid input (CPU)"""
    def forward(input, out):
        return torch.acosh(input=input, out=out)
    inputs = torch.tensor([[-1.0, 2.0, 3.0]], dtype=torch.float32)
    out1 = torch.zeros_like(inputs)
    out2 = torch.zeros_like(inputs)
    eager = forward(inputs, out1)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(inputs, out2)
    eager_nan = eager.isnan()
    compiled_nan = compiled.isnan()
    compiled_inf = compiled.isinf()
    if (eager_nan & compiled_inf).any():
        return ('BUG', f'eager=NaN but compiled=-inf')
    if not torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3, equal_nan=True):
        return ('BUG', f'values differ')
    return ('PASS', '')

def test_127310():
    """bitwise_right_shift signed/unsigned codegen wrong (CPU)"""
    def foo():
        a = torch.randint(-1, 0, (1, 1, 1), device="cpu", dtype=torch.int64)
        res = torch.bitwise_right_shift(a, 31)
        return a, res
    torch.manual_seed(42)
    a_e, eager = foo()
    torch._dynamo.reset()
    torch.manual_seed(42)
    a_c, compiled = torch.compile(foo)()
    if not torch.equal(eager, compiled):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')


# ============================================================================
# NEW tests: Graph Semantic Capturing / Caching Bugs
# ============================================================================

def test_102970():
    """custom autograd.Function backward silently wrong (CPU)"""
    class Foo(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, gx):
            return gx * 0.5
    def f(x):
        return Foo.apply(x)
    x = torch.randn([], requires_grad=True)
    torch._dynamo.reset()
    y = torch.compile(f)(x)
    result, = torch.autograd.grad(y, x)
    if abs(result.item() - 0.5) > 1e-3:
        return ('BUG', f'grad={result.item()}, expected 0.5')
    return ('PASS', '')

def test_107187():
    """torch.manual_seed ignored under dynamo (CPU)"""
    try:
        torch._inductor.config.fallback_random = True
    except Exception:
        pass
    @torch.compile()
    def foo():
        torch.manual_seed(3)
        return torch.rand([4])
    torch._dynamo.reset()
    r1 = foo()
    r2 = foo()
    if not torch.allclose(r1, r2):
        return ('BUG', f'manual_seed ignored: r1={r1.tolist()} r2={r2.tolist()}')
    return ('PASS', '')

def test_108898():
    """missing guard on global function (CPU)"""
    import types
    _mod = types.ModuleType("_test_108898")
    _mod.some_fn = lambda x: torch.sin(x) + 20
    def foo(x, y):
        x = _mod.some_fn(x)
        def add(x, y):
            return x + y
        return add(x, y)
    x = torch.ones(1,)
    y = torch.ones(1,)
    torch._dynamo.reset()
    opt = torch.compile(foo, fullgraph=True, backend='aot_eager')
    r1 = opt(x, y)
    expected1 = foo(x, y)
    if not torch.allclose(r1, expected1, atol=1e-3):
        return ('BUG', f'first call wrong: got {r1.item()} expected {expected1.item()}')
    _mod.some_fn = lambda x: torch.cos(x)
    r2 = opt(x, y)
    expected2 = foo(x, y)
    if not torch.allclose(r2, expected2, atol=1e-3):
        return ('BUG', f'stale cache: got {r2.item()} expected {expected2.item()}')
    return ('PASS', '')

def test_102315():
    """ModuleList empty evaluates truthy under compile (CPU)"""
    class IfModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emptylist = torch.nn.ModuleList()
        def forward(self):
            if self.emptylist:
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)
    eager = IfModule()()
    torch._dynamo.reset()
    compiled = torch.compile(IfModule(), backend="eager")()
    if eager.item() != compiled.item():
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()} (empty ModuleList truthy)')
    return ('PASS', '')

def test_103563():
    """VariableTracker nested list mutation wrong (CPU)"""
    def f(x):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        for i in range(3):
            if i == 0:
                data[0][i] = x
            else:
                data[0][i] = data[0][i - 1] + 1
        return data[0][-1]
    x = torch.rand(2)
    eager = f(x)
    torch._dynamo.reset()
    compiled = torch.compile(backend="eager")(f)(x)
    if not torch.allclose(eager, compiled, atol=1e-5):
        return ('BUG', f'eager={eager.tolist()} compiled={compiled.tolist()}')
    return ('PASS', '')

def test_104117():
    """FakeTensorMode dtype wrong with torch.full (CPU)"""
    def func():
        a = torch.full((8, 8), 1)
        return a.dtype
    eager_dtype = func()
    torch._dynamo.reset()
    @torch.compile(backend='eager')
    def compiled_func():
        a = torch.full((8, 8), 1)
        return a
    result = compiled_func()
    if result.dtype != eager_dtype:
        return ('BUG', f'dtype: eager={eager_dtype} compiled={result.dtype}')
    return ('PASS', '')

def test_108863():
    """conv_bn fusion remembers old weights (CPU)"""
    class BackboneModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(16, 16, 6)
            self.bn1 = torch.nn.BatchNorm2d(16)
        def forward(self, x):
            return self.bn1(self.conv1(x))
    torch.manual_seed(42)
    model = BackboneModel().eval()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt_model = torch.compile(model)
    a = torch.rand(4, 16, 32, 32)
    model.train()
    for i in range(5):
        optim.zero_grad()
        opt_model(a).sum().backward()
        optim.step()
    with torch.no_grad():
        model.eval()
        eager_out = model(a)
        compiled_out = opt_model(a)
    diff = (eager_out - compiled_out).abs().max().item()
    if diff > 1e-2:
        return ('BUG', f'stale weights in compiled model, max_diff={diff:.6f}')
    return ('PASS', '')

def test_111884():
    """Custom FFT implementation wrong under compile (CPU)"""
    def complex_matrix(real, imag):
        return torch.stack((
            torch.stack((real, -imag), dim=-1),
            torch.stack((imag, real), dim=-1)
        ), dim=-2)
    to_complex = lambda x: torch.complex(x[..., 0, 0], x[..., 1, 0])
    def block_dft(input):
        input_block = input.view(16, 16, -1)
        last_dim_size = input_block.shape[-1]
        tau = 2 * math.pi
        range_16 = torch.arange(16, device=input.device)
        range_large = torch.arange(16 * last_dim_size, device=input.device)
        range_small = torch.arange(last_dim_size, device=input.device)
        dft_real = torch.cos(-(range_16.unsqueeze(-1) * range_16) / 16 * tau)
        dft_imag = torch.sin(-(range_16.unsqueeze(-1) * range_16) / 16 * tau)
        dft = complex_matrix(dft_real, dft_imag)
        dft_small_real = torch.cos(-(range_small.unsqueeze(-1) * range_small) / last_dim_size * tau)
        dft_small_imag = torch.sin(-(range_small.unsqueeze(-1) * range_small) / last_dim_size * tau)
        dft_small = complex_matrix(dft_small_real, dft_small_imag)
        twid_real = torch.cos(-(range_16.unsqueeze(-1) * range_large) / (16 * 16 * last_dim_size) * tau).reshape(16, 16, -1)
        twid_imag = torch.sin(-(range_16.unsqueeze(-1) * range_large) / (16 * 16 * last_dim_size) * tau).reshape(16, 16, -1)
        twid = complex_matrix(twid_real, twid_imag)
        twid_small_real = torch.cos(-(range_16.unsqueeze(-1) * range_small) / (16 * last_dim_size) * tau)
        twid_small_imag = torch.sin(-(range_16.unsqueeze(-1) * range_small) / (16 * last_dim_size) * tau)
        twid_small = complex_matrix(twid_small_real, twid_small_imag)
        input_block = complex_matrix(input_block, torch.zeros_like(input_block))
        return torch.einsum('xyzAB,xfBC,ygDE,zhFG,fyzCD,gzEF->hgfAG',
                            input_block, dft, dft, dft_small, twid, twid_small).flatten(0, 2)
    input_t = torch.rand(1024)
    input_f = torch.fft.fft(input_t)
    eager_ok = torch.allclose(input_f, to_complex(block_dft(input_t)), rtol=1e-3, atol=1e-3)
    if not eager_ok:
        return ('SKIP', 'eager block_dft does not match fft')
    torch._dynamo.reset()
    compiled_result = torch.compile(block_dft)(input_t)
    compiled_ok = torch.allclose(input_f, to_complex(compiled_result), rtol=1e-3, atol=1e-3)
    if not compiled_ok:
        diff = (input_f - to_complex(compiled_result)).abs().max().item()
        return ('BUG', f'compiled FFT wrong, max_diff={diff:.6f}')
    return ('PASS', '')

def test_115679():
    """Rprop optimizer step not incremented (CPU)"""
    def training_loop():
        input_t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(3, 2)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.Sigmoid(),
            torch.nn.Linear(3, 1), torch.nn.Sigmoid(),
        )
        params = list(model.parameters())
        optimizer = torch.optim.Rprop(params)
        steps = []
        for i in range(6):
            optimizer.zero_grad()
            if i != 3:
                output = model(input_t)
                loss = output.sum()
                loss.backward()
            optimizer.step()
            steps.append(optimizer.state[params[0]]["step"].item()
                         if isinstance(optimizer.state[params[0]]["step"], torch.Tensor)
                         else optimizer.state[params[0]]["step"])
        return steps
    eager_steps = training_loop()
    torch._dynamo.reset()
    compiled_loop = torch._dynamo.optimize("eager")(training_loop)
    compiled_steps = compiled_loop()
    if eager_steps != compiled_steps:
        return ('BUG', f'step counts differ: eager={eager_steps} compiled={compiled_steps}')
    return ('PASS', '')

def test_125387():
    """graph caching returns stale index (CPU)"""
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = -1
            self.cache = torch.tensor([2, 3, 4, 5, 6, 7])
        def forward(self, value):
            self.value += 1
            return self.cache[self.value]
    model_e = ToyModel()
    model_c = ToyModel()
    torch._dynamo.reset()
    model_c.forward = torch.compile(model_c.forward, fullgraph=True)
    eager_vals = [model_e(v).item() for v in [6, 8, 10, 12, 13, 14]]
    compiled_vals = [model_c(v).item() for v in [6, 8, 10, 12, 13, 14]]
    if eager_vals != compiled_vals:
        return ('BUG', f'stale index: eager={eager_vals} compiled={compiled_vals}')
    return ('PASS', '')

def test_91470():
    """functorch transforms error under compile (CPU)"""
    def f(x):
        y = x * x * x * x * x * 1. * 1. * 1.
        return y
    def g(x):
        return torch.vmap(f)(x)
    x = torch.randn(5)
    eager = g(x)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(backend='aot_eager')(g)(x)
        if not torch.allclose(eager, compiled, atol=1e-5):
            return ('BUG', f'values differ')
        return ('PASS', '')
    except Exception as e:
        return ('BUG', f'compile error: {type(e).__name__}')

def test_124205():
    """NestedTensor silent specialization on _max_seqlen (CUDA)"""
    try:
        from torch.nested._internal.nested_tensor import NestedTensor
    except ImportError:
        return ('SKIP', 'NestedTensor not available')
    @torch.compile
    def sample_fun(njt):
        njt = njt.clamp(0.1, 0.5)
        njt *= njt._max_seqlen
        return njt
    def create_njt(el_per_row):
        torch.manual_seed(0)
        njt = NestedTensor(
            values=torch.randn(10 * el_per_row, device="cuda"),
            offsets=torch.arange(11, device="cuda") * el_per_row,
        )
        return njt
    with torch.inference_mode():
        torch._dynamo.reset()
        r1 = sample_fun(create_njt(el_per_row=1))
        r2 = sample_fun(create_njt(el_per_row=2))
        expected = create_njt(el_per_row=2)
        expected = expected.clamp(0.1, 0.5)
        expected_vals = expected.values() * expected._max_seqlen
        if not torch.allclose(r2.values(), expected_vals, atol=1e-3):
            return ('BUG', f'stale _max_seqlen specialization')
    return ('PASS', '')

def test_127995():
    """custom op nondeterminism under compile (CPU)"""
    import numpy as np
    try:
        lib = torch.library.Library("_test127995", "FRAGMENT")
        lib.define("numpy_sin(Tensor input, Tensor(a!) output) -> ()")
        def numpy_sin(input, output):
            np.sin(input.numpy(), out=output.numpy())
        lib.impl("numpy_sin", numpy_sin, "CPU")
    except Exception:
        return ('SKIP', 'custom op registration failed')
    @torch.compile(fullgraph=True)
    def f(x):
        out = torch.empty(3)
        torch.ops._test127995.numpy_sin(x, out)
        return out
    torch._dynamo.reset()
    x = torch.randn(3)
    y = f(x)
    if not torch.allclose(y, x.sin(), atol=1e-5):
        return ('BUG', f'wrong result')
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
        # Original 30
        (93262,  "interpolate bilinear wrong [CUDA]",       test_93262),
        (97083,  "squeeze+mutation+if [CPU]",               test_97083),
        (98143,  "add+unfold+abs_ [CUDA]",                  test_98143),
        (99686,  "split+cat wrong reorder [CPU]",           test_99686),
        (100775, "mul(big).softmax NaN [CPU]",              test_100775),
        (100800, "where SIMD incorrect [CPU]",              test_100800),
        (100987, "AvgPool ceil_mode [CPU]",                 test_100987),
        (101039, "softmax sum!=1 [CUDA]",                   test_101039),
        (104505, "as_strided functionalization [CPU]",      test_104505),
        (111133, "index_add_ stride [CPU]",                 test_111133),
        (113013, "adaptive_pool1d+argmax [CPU,f64]",        test_113013),
        (113151, "asinh out= [CUDA]",                       test_113151),
        (113641, "double clamp_ static [CPU]",              test_113641),
        (114302, "expand+mutation [CPU,int64]",             test_114302),
        (114464, "clone+reshape ignored [CPU,fullgraph]",   test_114464),
        (116433, "output stride mismatch [CUDA]",           test_116433),
        (116935, "Transformer accuracy [CUDA]",             test_116935),
        (117019, "diag_embed non-default dims [CPU,f64]",   test_117019),
        (117544, "sinh+relu NaN [CPU,max-autotune]",        test_117544),
        (118265, "bitwise_right_shift out= [CPU]",          test_118265),
        (118266, "log1p+argmax out= [CPU]",                 test_118266),
        (121174, "fuse_attention [CPU]",                    test_121174),
        (122260, "fma precision [CUDA]",                    test_122260),
        (100793, "pad reflect backward [CPU,fullgraph]",    test_100793),
        (127749, "gumbel_softmax NaN [CUDA]",               test_127749),
        (132107, "cumsum NaN with inf [CUDA]",              test_132107),
        (138908, "scatter-add 4x vs 2x [CPU,dynamic]",     test_138908),
        (141538, "FractionalMaxPool2d [CPU]",               test_141538),
        (143555, "bitwise_right_shift ==64 [CPU]",          test_143555),
        (146416, "transpose fusion [CUDA]",                 test_146416),

        # NEW: In-Place Operation Handling
        (93078,  "dstack+reciprocal bool [CUDA]",           test_93078),
        (93363,  "AOTautograd inplace view [CUDA]",         test_93363),
        (96604,  "neg uint8 [CUDA]",                        test_96604),
        (97117,  "F.pad zero-pad alias [CUDA]",             test_97117),
        (97130,  "shared storage mutation [CPU]",           test_97130),
        (100224, "masked inplace mul [CPU]",                test_100224),
        (100970, "batch_norm wrong shape [CPU]",            test_100970),
        (104020, "permute_linear_fusion resize_ [CUDA]",    test_104020),
        (108472, "storage offset discarded [CPU]",          test_108472),
        (113030, ".data= assignment [CPU]",                 test_113030),
        (115071, "add input mutation complex64 [CPU]",      test_115071),
        (125284, "nn.Parameter aliasing [CPU]",             test_125284),
        (127474, "tile+as_strided+add [CUDA]",              test_127474),
        (128035, "detach+inplace zeros grad [CPU]",         test_128035),

        # NEW: Operator Transformation
        (108963, "index_add alpha=2 [CPU]",                 test_108963),
        (109925, "cumsum bool wrong dtype [CUDA]",          test_109925),
        (113167, "var_mean fractional correction [CUDA]",   test_113167),
        (121227, "split+stack+tanh wrong [CPU]",            test_121227),
        (122379, "split_cat wrong shape [CPU]",             test_122379),
        (122382, "conv_add drops alpha [CPU]",              test_122382),
        (105190, "SDPA NaN negative scale [CPU]",           test_105190),

        # NEW: Low-level Code Generation
        (113010, "abs transposed out= [CPU]",               test_113010),
        (113012, "hinge_embedding_loss [CUDA]",             test_113012),
        (113018, "Conv2d NaN [CPU]",                        test_113018),
        (113141, "sum(mul) commutativity [CPU]",            test_113141),
        (115260, "cat+mul+tan fusion fp16 [CPU]",           test_115260),
        (115261, "matmul extra outputs [CPU,uint8]",        test_115261),
        (117778, "roll+fill_ [CUDA]",                       test_117778),
        (118267, "acosh -inf vs NaN [CPU]",                 test_118267),
        (127310, "bitwise_right_shift signed [CPU]",        test_127310),

        # NEW: Graph Semantic Capturing / Caching
        (102970, "autograd.Function backward [CPU]",        test_102970),
        (107187, "manual_seed ignored [CPU]",               test_107187),
        (108898, "missing guard global fn [CPU]",           test_108898),
        (102315, "ModuleList empty truthy [CPU]",           test_102315),
        (103563, "VariableTracker list mutation [CPU]",     test_103563),
        (104117, "FakeTensor torch.full dtype [CPU]",       test_104117),
        (108863, "conv_bn stale weights [CPU]",             test_108863),
        (111884, "custom FFT wrong [CPU]",                  test_111884),
        (115679, "Rprop step not incremented [CPU]",        test_115679),
        (125387, "graph caching stale index [CPU]",         test_125387),
        (91470,  "functorch vmap error [CPU]",              test_91470),
        (124205, "NestedTensor specialization [CUDA]",      test_124205),
        (127995, "custom op nondeterminism [CPU]",          test_127995),
    ]

    for issue_num, desc, test_fn in tests:
        test_bug(issue_num, desc, test_fn)

    # Summary
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
    print()
    if bugs:
        print("Reproducible bug issues:")
        for k, v in sorted(bugs):
            print(f"  #{k}: {v[1]}")
