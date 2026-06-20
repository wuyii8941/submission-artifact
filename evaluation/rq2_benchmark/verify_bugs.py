#!/usr/bin/env python3
"""
Verify which bugs from the correctness bug study are reproducible.
Each test uses the ORIGINAL reproducer from the GitHub issue.
Returns: eager vs compiled comparison (differential oracle).
"""
import torch
import sys
import traceback
import copy

try:
    torch._inductor.config.force_disable_caches = True
except AttributeError:
    pass

RESULTS = {}

def test_bug(issue_num, description, test_fn):
    """Run a single bug reproducer. test_fn should return (status, detail)."""
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
# Bug reproducers (faithful to original GitHub issues)
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
    """torch.asinh wrong output with out= (CUDA, max-autotune-no-cudagraphs)"""
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
        return ('BUG', f'NaN mismatch: eager_nan={eager.isnan().sum().item()} compiled_nan={compiled.isnan().sum().item()}')
    return ('PASS', '')

def test_118265():
    """bitwise_right_shift wrong with out= int16 (CPU, reduce-overhead)"""
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
    import math
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
    import copy
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
        outputs=ret_compiled,
        inputs=list(compiled.parameters()),
        grad_outputs=torch.ones_like(ret_compiled),
        retain_graph=True,
    )
    model.train()
    ret_eager = model()
    grad_eager = torch.autograd.grad(
        outputs=ret_eager,
        inputs=list(model.parameters()),
        grad_outputs=torch.ones_like(ret_eager),
        retain_graph=True,
    )
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
    y_clone = y.clone()
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
# Run all tests
# ============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    tests = [
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
    ]

    for issue_num, desc, test_fn in tests:
        test_bug(issue_num, desc, test_fn)

    # Summary
    bugs = [(k, v) for k, v in RESULTS.items() if v[0] == 'BUG']
    passes = [(k, v) for k, v in RESULTS.items() if v[0] == 'PASS']
    errors = [(k, v) for k, v in RESULTS.items() if v[0] == 'ERROR']
    skips = [(k, v) for k, v in RESULTS.items() if v[0] == 'SKIP']

    print(f"{'Issue':>8s}  {'Status':>6s}  {'Description':<45s}  Detail")
    print("-" * 100)
    for issue_num in sorted(RESULTS.keys()):
        status, desc, detail = RESULTS[issue_num]
        marker = '***' if status == 'BUG' else '   '
        print(f"#{issue_num:>7d}  {status:>6s}  {desc:<45s}  {detail[:50]} {marker}")

    print()
    print(f"BUG (reproducible):  {len(bugs)}")
    print(f"PASS (fixed/absent): {len(passes)}")
    print(f"ERROR (test issue):  {len(errors)}")
    print(f"SKIP:                {len(skips)}")
