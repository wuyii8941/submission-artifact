#!/usr/bin/env python3
"""
Additional bug reproducers from broader GitHub search (round 2).
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
# Early bugs (93xxx-102xxx)
# ============================================================================

def test_93078():
    """dstack + reciprocal wrong values"""
    def fn(input):
        v1 = torch.dstack([input, input, input])
        return torch.reciprocal(v1)
    x = torch.tensor(True)
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn)(x)
    if not torch.allclose(eager, compiled, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_96694():
    """torch.abs returns negative for typecasted tensor (CUDA)"""
    x = torch.tensor(1, device='cuda')
    def forward():
        minus_x = -x
        minus_x_float = minus_x.to(dtype=torch.float32)
        return torch.abs(minus_x_float)
    eager = forward()
    torch._dynamo.reset()
    compiled = torch.compile(forward)()
    if not torch.allclose(eager, compiled, atol=1e-5):
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    return ('PASS', '')

def test_99833():
    """recursive function with tensor slicing wrong result"""
    def recursive_identity(x):
        length = x.shape[-1]
        if length <= 1:
            return x
        x_0 = x[..., :length//2]
        x_1 = x[..., length//2:]
        y_1 = recursive_identity(x_1)
        y_0 = x_0
        y = torch.cat([y_0, y_1], dim=-1)
        return y
    a = 1 + torch.arange(4)
    eager = recursive_identity(a)
    torch._dynamo.reset()
    compiled = torch.compile(recursive_identity)(a)
    if not torch.equal(eager, compiled):
        return ('BUG', f'eager={eager.tolist()} compiled={compiled.tolist()}')
    return ('PASS', '')

def test_100224():
    """conditional mask *= produces wrong shape/values"""
    torch.manual_seed(420)
    x = torch.randn(1, 3, 2, 2)
    class Model(torch.nn.Module):
        def forward(self, x):
            out = x.clone()
            mask1 = out > 0
            out[mask1] *= 0.5
            return out
    func = Model()
    eager = func(x.clone())
    torch._dynamo.reset()
    compiled = torch.compile(func)(x.clone())
    if eager.shape != compiled.shape:
        return ('BUG', f'shape mismatch: eager={eager.shape} compiled={compiled.shape}')
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'values differ')
    return ('PASS', '')

def test_100793():
    """F.pad reflect wrong backward gradient"""
    torch.manual_seed(0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v9 = torch.nn.Parameter(torch.ones(size=[7, 37, 7, 2], dtype=torch.float32))
        def forward(self):
            return torch.nn.functional.pad(self.v9, pad=[-1, 0, 6, 6], mode='reflect')
    model = Model()
    model_tmp = copy.deepcopy(model)
    model_tmp.train()
    torch._dynamo.reset()
    compiled = torch.compile(model_tmp, fullgraph=True)
    ret_c = compiled()
    grad_c = torch.autograd.grad(ret_c, list(compiled.parameters()),
                                  grad_outputs=torch.ones_like(ret_c), retain_graph=True)
    model.train()
    ret_e = model()
    grad_e = torch.autograd.grad(ret_e, list(model.parameters()),
                                  grad_outputs=torch.ones_like(ret_e), retain_graph=True)
    if not torch.allclose(grad_c[0], grad_e[0], atol=1e-3):
        n_diff = (~torch.isclose(grad_c[0], grad_e[0], atol=1e-3)).sum().item()
        total = grad_c[0].numel()
        return ('BUG', f'grad differs: {n_diff}/{total} elements')
    return ('PASS', '')

def test_100800():
    """CPU inductor where SIMD inverted mask"""
    input_tensor = torch.ones(3, 3)
    def f(x):
        return torch.where(torch.ones_like(x).to(torch.bool), torch.zeros_like(x), torch.ones_like(x) * 2)
    eager = f(input_tensor)
    torch._dynamo.reset()
    compiled = torch.compile(f)(input_tensor)
    if not torch.allclose(eager, compiled, atol=1e-5):
        return ('BUG', f'eager={eager[0,0].item()} compiled={compiled[0,0].item()}')
    return ('PASS', '')

def test_100970():
    """batch_norm wrong output shape with non-1d weight"""
    torch.manual_seed(420)
    x = torch.randn(1, 3, 2, 2)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(3, 1, 1, 1)
            self.bias = torch.randn(3)
            self.bn = torch.nn.BatchNorm2d(num_features=3, affine=False)
        def forward(self, x):
            return torch.nn.functional.batch_norm(x, self.bn.running_mean, self.bn.running_var,
                                                   self.weight, self.bias)
    func = Model()
    with torch.no_grad():
        func.train(False)
        eager = func(x)
        torch._dynamo.reset()
        compiled = torch.compile(func)(x)
    if eager.shape != compiled.shape:
        return ('BUG', f'shape mismatch: eager={list(eager.shape)} compiled={list(compiled.shape)}')
    if not torch.allclose(eager, compiled, atol=1e-3):
        return ('BUG', f'values differ')
    return ('PASS', '')

def test_102970():
    """autograd.Function returning input gets wrong gradient"""
    class Foo(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, gx):
            return gx * 0.5
    x = torch.randn([], requires_grad=True)
    def f(x):
        return Foo.apply(x)
    torch._dynamo.reset()
    y = torch.compile(f)(x)
    result, = torch.autograd.grad(y, x)
    expected = torch.tensor(0.5)
    if not torch.allclose(result, expected, atol=1e-3):
        return ('BUG', f'grad={result.item()} expected=0.5')
    return ('PASS', '')


# ============================================================================
# Output-order / out= parameter bugs (109xxx-114xxx)
# ============================================================================

def test_109001():
    """out= parameter with complex matmul wrong (CUDA)"""
    x = torch.tensor([-8.4784-1.7658j], device='cuda')
    y = torch.tensor([-8.4784-1.7658j], device='cuda')
    torch._dynamo.reset()
    ans = torch.compile(torch.matmul)(x, y)
    out = torch.empty_like(ans)
    torch._dynamo.reset()
    torch.compile(torch.matmul)(x, y, out=out)
    if not torch.allclose(ans, out, atol=1e-3):
        return ('BUG', f'ans={ans.item()} out={out.item()}')
    return ('PASS', '')

def test_112710():
    """argmin+argsort wrong under max-autotune (CUDA)"""
    def forward(x):
        x = torch.argmin(input=x, dim=0, keepdim=True)
        x = torch.argsort(input=x, descending=True, dim=1, stable=False)
        return x
    torch.manual_seed(42)
    x = torch.rand([10, 9], dtype=torch.float32)
    eager = forward(x)
    x_cuda = x.to('cuda')
    torch._dynamo.reset()
    compiled = torch.compile(forward, mode='max-autotune')(x_cuda)
    if not torch.equal(eager, compiled.cpu()):
        return ('BUG', f'results differ')
    return ('PASS', '')

def test_113010():
    """torch.abs with transposed out= wrong"""
    def forward(x, y):
        return torch.abs(input=x, out=torch.t(input=y))
    x = torch.rand([9, 10], dtype=torch.float32)
    y = torch.rand([10, 9], dtype=torch.float32)
    eager = forward(x.clone(), y.clone())
    torch._dynamo.reset()
    compiled = torch.compile(forward)(x.clone(), y.clone())
    if not torch.allclose(eager, compiled, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_113144():
    """abs associative law int8 overflow wrong"""
    p0 = torch.tensor(5, dtype=torch.int8)
    class Model0(torch.nn.Module):
        def forward(self, x0, x1):
            add = torch.add(x1, p0)
            mul = torch.mul(add, p0)
            mul_1 = torch.mul(x0, mul)
            return torch.abs(mul_1)
    x0 = torch.tensor(5, dtype=torch.int8)
    x1 = torch.tensor(4, dtype=torch.int8)
    eager = Model0()(x0, x1)
    torch._dynamo.reset()
    compiled = torch.compile(Model0(), fullgraph=True, backend='inductor')(x0, x1)
    if eager.item() != compiled.item():
        return ('BUG', f'eager={eager.item()} compiled={compiled.item()}')
    if compiled.item() < 0:
        return ('BUG', f'abs returned negative: {compiled.item()}')
    return ('PASS', '')

def test_113151():
    """asinh with out= parameter wrong"""
    def forward(x, y):
        return torch.asinh(out=x, input=y)
    y = torch.tensor([487875.875, -956238.8125, 630736.0, -161079.578125, 104060.9375,
                       757224.3125, -153601.859375, -648042.5, 733955.4375, -214764.90625],
                      dtype=torch.float32)
    x1 = torch.rand([10], dtype=torch.float32)
    x2 = x1.clone()
    eager = forward(x1, y)
    torch._dynamo.reset()
    compiled = torch.compile(forward)(x2, y)
    if not torch.allclose(eager, compiled, atol=1e-3):
        diff = (eager - compiled).abs().max().item()
        return ('BUG', f'max_diff={diff:.6f}')
    return ('PASS', '')

def test_113629():
    """sub corrupted by extra cat output (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x0, x1):
            div = torch.div(x1, x0)
            sum_1 = div.sum(1)
            cat = torch.cat((sum_1, sum_1, sum_1), dim=0)
            sub = torch.sub(div, sum_1)
            return (cat, sub)
    class Model1(torch.nn.Module):
        def forward(self, x0, x1):
            div = torch.div(x1, x0)
            sum_1 = div.sum(1)
            sub = torch.sub(div, sum_1)
            return (sub,)
    x0 = torch.tensor(6.973, dtype=torch.float16, device='cuda')
    data = np.array([[6.81], [6.49], [6.277], [4.77], [6.793], [5.598],
                     [5.676], [5.008], [6.67], [4.08], [6.465], [3.543],
                     [4.902], [3.346], [5.875], [4.414], [3.791], [5.438],
                     [6.125], [6.08]], dtype=np.float16)
    x1 = torch.from_numpy(data).to('cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x0, x1)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x0, x1)
    if not torch.allclose(out0[1], out1[0], atol=1e-2):
        diff = (out0[1] - out1[0]).abs().max().item()
        return ('BUG', f'sub differs with/without cat, max_diff={diff:.6f}')
    return ('PASS', '')

def test_114215():
    """mul args swap produces wrong result (CUDA)"""
    class Model0(torch.nn.Module):
        def forward(self, x):
            interpolate = torch.nn.functional.interpolate(x, size=[8, 1], mode='bicubic')
            pad = torch.nn.functional.pad(interpolate, (0, 0, 6, 0), 'reflect')
            pad_1 = torch.nn.functional.pad(pad, (0, 15), 'constant', value=0.5)
            return torch.mul(pad, pad_1)
    class Model1(torch.nn.Module):
        def forward(self, x):
            interpolate = torch.nn.functional.interpolate(x, size=[8, 1], mode='bicubic')
            pad = torch.nn.functional.pad(interpolate, (0, 0, 6, 0), 'reflect')
            pad_1 = torch.nn.functional.pad(pad, (0, 15), 'constant', value=0.5)
            return torch.mul(pad_1, pad)
    torch.manual_seed(42)
    x = torch.rand(1, 46, 1, 1, device='cuda')
    torch._dynamo.reset()
    out0 = torch.compile(Model0(), fullgraph=True, backend='inductor')(x)
    torch._dynamo.reset()
    out1 = torch.compile(Model1(), fullgraph=True, backend='inductor')(x)
    if not torch.allclose(out0, out1, atol=1e-3):
        n_diff = (~torch.isclose(out0, out1, atol=1e-3)).sum().item()
        total = out0.numel()
        return ('BUG', f'mul(a,b)!=mul(b,a), {n_diff}/{total} elements differ')
    return ('PASS', '')


# ============================================================================
# Dynamo-level bugs
# ============================================================================

def test_115607():
    """dynamo caches optimizer step as no-op"""
    torch.manual_seed(42)
    input_t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(3, 2)
    def run_training(use_compile):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.Sigmoid(),
            torch.nn.Linear(3, 1), torch.nn.Sigmoid(),
        )
        torch.manual_seed(42)
        for p in model.parameters():
            p.data = torch.randn_like(p)
        optimizer = torch.optim.Adam(model.parameters())
        fn = model
        if use_compile:
            torch._dynamo.reset()
            fn = torch._dynamo.optimize("eager", save_config=False)(model)
        steps = []
        for i in range(6):
            optimizer.zero_grad()
            if i != 3:
                output = fn(input_t)
                loss = output.sum()
                loss.backward()
            optimizer.step()
            steps.append(optimizer.state[list(model.parameters())[0]]["step"].item()
                         if isinstance(optimizer.state[list(model.parameters())[0]]["step"], torch.Tensor)
                         else optimizer.state[list(model.parameters())[0]]["step"])
        return steps, [p.data.clone() for p in model.parameters()]
    eager_steps, eager_params = run_training(False)
    compiled_steps, compiled_params = run_training(True)
    for i, (ep, cp) in enumerate(zip(eager_params, compiled_params)):
        if not torch.allclose(ep, cp, atol=1e-3):
            diff = (ep - cp).abs().max().item()
            return ('BUG', f'param {i} differs, max_diff={diff:.6f}')
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
        # Early bugs
        (93078,  "dstack+reciprocal [CPU]",                  test_93078),
        (96694,  "abs negative typecast [CUDA]",             test_96694),
        (99833,  "recursive slicing [CPU]",                  test_99833),
        (100224, "conditional mask *= [CPU]",                 test_100224),
        (100793, "pad reflect grad [CPU]",                   test_100793),
        (100800, "where SIMD inverted [CPU]",                test_100800),
        (100970, "batch_norm wrong shape [CPU]",             test_100970),
        (102970, "autograd.Function grad [CPU]",             test_102970),

        # Output-order / out= bugs
        (109001, "complex matmul out= [CUDA]",               test_109001),
        (112710, "argmin+argsort autotune [CUDA]",           test_112710),
        (113010, "abs transposed out= [CPU]",                test_113010),
        (113144, "abs associative int8 [CPU]",               test_113144),
        (113151, "asinh out= [CPU]",                         test_113151),
        (113629, "sub+extra cat [CUDA]",                     test_113629),
        (114215, "mul args swap [CUDA]",                     test_114215),

        # Dynamo-level
        (115607, "optimizer step cached [CPU]",              test_115607),
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
