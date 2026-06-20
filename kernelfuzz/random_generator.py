#!/usr/bin/env python3
"""
Fair Random Program Generator (ablation baseline).

Generates random PyTorch programs WITHOUT kernel-state knowledge but with
a FAIR op pool and shape distribution:
  - Broad op pool (~60 ops covering unary, binary, reduction, scan, pool, scatter, norm, matmul, conv)
  - Log-uniform shape distribution [1, 8192] per dim (not artificially small)
  - Standard dtype distribution
  - Random composition (1-6 ops)

The key constraint: NO knowledge of which ops trigger which kernel family,
NO knowledge of codegen boundary thresholds, NO knowledge of fusion patterns.
Same oracle pipeline as targeted generator — only program construction differs.
"""

import math
import random
from typing import Tuple, Optional

# Broad op pool — includes all common categories a generic fuzzer would know about
OPS_UNARY = [
    'torch.relu', 'torch.sigmoid', 'torch.tanh', 'torch.exp',
    'torch.neg', 'torch.abs', 'torch.sin', 'torch.cos',
    'torch.reciprocal', 'torch.sqrt', 'torch.log1p',
    'torch.nn.functional.silu', 'torch.nn.functional.gelu',
    'torch.nn.functional.elu', 'torch.nn.functional.leaky_relu',
    'torch.erf', 'torch.erfc', 'torch.floor', 'torch.ceil',
    'torch.sign', 'torch.square',
]

OPS_REDUCTION = [
    ('.sum(dim=-1, keepdim=True)', 'sum'),
    ('.mean(dim=-1, keepdim=True)', 'mean'),
    ('.amax(dim=-1, keepdim=True)', 'amax'),
    ('.amin(dim=-1, keepdim=True)', 'amin'),
    ('.std(dim=-1, keepdim=True)', 'std'),
    ('.var(dim=-1, keepdim=True)', 'var'),
    ('.prod(dim=-1, keepdim=True)', 'prod'),
    ('.norm(dim=-1, keepdim=True)', 'norm'),
]

OPS_SCAN = [
    ('torch.cumsum({x}, dim=-1)', 'cumsum'),
    ('torch.cumprod({x}.sigmoid(), dim=-1)', 'cumprod'),
]

OPS_POOL = [
    ('torch.nn.functional.adaptive_avg_pool2d({x}, (1,1))', 'adaptive_avg_pool2d'),
    ('torch.nn.functional.adaptive_max_pool2d({x}, (1,1))', 'adaptive_max_pool2d'),
    ('torch.nn.functional.avg_pool2d({x}, kernel_size=2, stride=2)', 'avg_pool2d'),
    ('torch.nn.functional.max_pool2d({x}, kernel_size=2, stride=2)', 'max_pool2d'),
]

OPS_NORM = [
    ('torch.nn.functional.layer_norm({x}, [{last_dim}])', 'layer_norm'),
    ('torch.nn.functional.softmax({x}, dim=-1)', 'softmax'),
    ('torch.nn.functional.log_softmax({x}, dim=-1)', 'log_softmax'),
]

OPS_SCATTER = [
    ('torch.scatter({x}, -1, {idx}, {src})', 'scatter'),
    ('{x}.index_select(-1, {idx1d})', 'index_select'),
]

DTYPES = ['torch.float32'] * 50 + ['torch.float64'] * 20 + \
         ['torch.bfloat16'] * 15 + ['torch.float16'] * 15


def _log_uniform_int(lo: int, hi: int) -> int:
    """Sample integer from log-uniform distribution over [lo, hi]."""
    log_lo = math.log(max(lo, 1))
    log_hi = math.log(hi)
    return int(round(math.exp(random.uniform(log_lo, log_hi))))


def generate_random_program(seed: Optional[int] = None) -> Tuple[str, dict]:
    """
    Generate a random PyTorch program with no kernel-state awareness.
    Uses broad op pool, log-uniform shapes, standard dtypes.
    """
    if seed is not None:
        random.seed(seed)

    dtype = random.choice(DTYPES)
    n_ops = random.randint(1, 6)
    ndim = random.choice([2, 3, 4])

    # Log-uniform shape: each dim in [1, 8192], giving fair coverage of small AND large
    shape = tuple(_log_uniform_int(1, 8192) for _ in range(ndim))
    # Clamp to avoid OOM: total elements <= 2^24 (~16M)
    total = 1
    for s in shape:
        total *= s
    if total > 2**24:
        # Scale down proportionally
        scale = (2**24 / total) ** (1.0 / ndim)
        shape = tuple(max(1, int(s * scale)) for s in shape)

    batch = shape[0]

    meta = {
        'generator': 'random_fair',
        'dtype': dtype,
        'n_ops': n_ops,
        'shape': shape,
        'ndim': ndim,
        'seed': seed,
    }

    # Random program type — weighted by commonality in real workloads
    prog_type = random.choices(
        ['pointwise', 'reduce', 'matmul', 'scan', 'conv', 'pool', 'norm'],
        weights=[30, 20, 15, 10, 10, 8, 7],
        k=1
    )[0]

    meta['prog_type'] = prog_type

    shape_str = ', '.join(str(s) for s in shape)

    if prog_type == 'pointwise':
        lines = [
            f"_x = torch.randn({shape_str}, dtype={dtype})",
            f"_program_inputs = (_x,)",
            f"",
            f"def program(x):",
        ]
        for i in range(n_ops):
            op = random.choice(OPS_UNARY)
            lines.append(f"    x = {op}(x)")
        lines.append(f"    return x")
        code = "\n".join(lines)

    elif prog_type == 'reduce':
        lines = [
            f"_x = torch.randn({shape_str}, dtype={dtype})",
            f"_program_inputs = (_x,)",
            f"",
            f"def program(x):",
        ]
        # Optional pre-reduction pointwise ops
        for i in range(max(0, n_ops - 1)):
            op = random.choice(OPS_UNARY)
            lines.append(f"    x = {op}(x)")
        red_expr, _ = random.choice(OPS_REDUCTION)
        lines.append(f"    return x{red_expr}")
        code = "\n".join(lines)

    elif prog_type == 'matmul':
        M = _log_uniform_int(1, 2048)
        K = _log_uniform_int(1, 2048)
        N = _log_uniform_int(1, 2048)
        # Clamp to avoid OOM
        if batch * M * K > 2**24:
            M = min(M, 2**24 // (batch * K + 1))
            M = max(1, M)
        lines = [
            f"_x = torch.randn({batch}, {M}, {K}, dtype={dtype})",
            f"_w = torch.randn({K}, {N}, dtype={dtype})",
            f"_program_inputs = (_x, _w)",
            f"",
            f"def program(x, w):",
            f"    y = torch.matmul(x, w)",
        ]
        for i in range(max(0, n_ops - 1)):
            op = random.choice(OPS_UNARY)
            lines.append(f"    y = {op}(y)")
        lines.append(f"    return y")
        code = "\n".join(lines)

    elif prog_type == 'scan':
        lines = [
            f"_x = torch.randn({shape_str}, dtype={dtype})",
            f"_program_inputs = (_x,)",
            f"",
            f"def program(x):",
        ]
        for i in range(max(0, n_ops - 1)):
            op = random.choice(OPS_UNARY)
            lines.append(f"    x = {op}(x)")
        scan_expr, _ = random.choice(OPS_SCAN)
        expr = scan_expr.format(x='x')
        lines.append(f"    return {expr}")
        code = "\n".join(lines)

    elif prog_type == 'conv':
        C_in = random.randint(1, 64)
        C_out = random.randint(1, 64)
        spatial = _log_uniform_int(4, 256)
        # Clamp
        if batch * C_in * spatial * spatial > 2**24:
            spatial = int((2**24 / (batch * C_in)) ** 0.5)
            spatial = max(4, spatial)
        lines = [
            f"_x = torch.randn({batch}, {C_in}, {spatial}, {spatial}, dtype={dtype})",
            f"_w = torch.randn({C_out}, {C_in}, 3, 3, dtype={dtype})",
            f"_program_inputs = (_x, _w)",
            f"",
            f"def program(x, w):",
            f"    return torch.nn.functional.conv2d(x, w, padding=1)",
        ]
        code = "\n".join(lines)

    elif prog_type == 'pool':
        C = random.randint(1, 64)
        spatial = _log_uniform_int(4, 256)
        if batch * C * spatial * spatial > 2**24:
            spatial = int((2**24 / (batch * C)) ** 0.5)
            spatial = max(4, spatial)
        pool_expr, pool_name = random.choice(OPS_POOL)
        lines = [
            f"_x = torch.randn({batch}, {C}, {spatial}, {spatial}, dtype={dtype})",
            f"_program_inputs = (_x,)",
            f"",
            f"def program(x):",
        ]
        # Optional pre-pool pointwise
        for i in range(max(0, n_ops - 1)):
            op = random.choice(OPS_UNARY)
            lines.append(f"    x = {op}(x)")
        expr = pool_expr.format(x='x')
        lines.append(f"    return {expr}")
        meta['pool_op'] = pool_name
        code = "\n".join(lines)

    elif prog_type == 'norm':
        last_dim = shape[-1]
        norm_expr, norm_name = random.choice(OPS_NORM)
        lines = [
            f"_x = torch.randn({shape_str}, dtype={dtype})",
            f"_program_inputs = (_x,)",
            f"",
            f"def program(x):",
        ]
        for i in range(max(0, n_ops - 1)):
            op = random.choice(OPS_UNARY)
            lines.append(f"    x = {op}(x)")
        expr = norm_expr.format(x='x', last_dim=last_dim)
        lines.append(f"    return {expr}")
        meta['norm_op'] = norm_name
        code = "\n".join(lines)

    else:
        code = (f"_x = torch.randn({shape_str}, dtype={dtype})\n"
                f"_program_inputs = (_x,)\n\n"
                f"def program(x):\n    return x + 1")

    return code, meta


if __name__ == '__main__':
    for i in range(10):
        code, meta = generate_random_program(seed=i)
        print(f"=== Random program {i} ({meta['prog_type']}, shape={meta['shape']}, dtype={meta['dtype']}) ===")
        print(code)
        print()
