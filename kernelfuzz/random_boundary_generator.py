#!/usr/bin/env python3
"""
Random + BoundaryShape Generator (ablation: isolate shape contribution).

Same random op selection and composition as random_generator.py, but uses
adversarial boundary shapes from targeted_generator.py instead of log-uniform.

This isolates the contribution of BoundaryShape: if random+boundary matches
targeted on D and bugs, then OpsForFamily/ComposeGraph adds nothing. If it
doesn't, then family-aware op selection is independently valuable.
"""

import math
import random
from typing import Tuple, Optional

from targeted_generator import BOUNDARY_SHAPES

# Reuse the same op pool from random_generator
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

DTYPES = ['torch.float32'] * 50 + ['torch.float64'] * 20 + \
         ['torch.bfloat16'] * 15 + ['torch.float16'] * 15

# All boundary dimensions pooled from every family
ALL_BOUNDARY_DIMS = sorted(set(
    BOUNDARY_SHAPES['persistent_reduction']['threshold_dim'] +
    BOUNDARY_SHAPES['reduction']['threshold_dim'] +
    BOUNDARY_SHAPES['split_scan']['threshold_dim'] +
    BOUNDARY_SHAPES['scan']['threshold_dim'] +
    BOUNDARY_SHAPES['cooperative_reduction']['threshold_dim'] +
    BOUNDARY_SHAPES['tile_mask'] +
    BOUNDARY_SHAPES['fp64_persistent']
))


def _pick_boundary_dim() -> int:
    """Pick a random boundary dimension from the full pool."""
    return random.choice(ALL_BOUNDARY_DIMS)


def _boundary_shape(ndim: int) -> tuple:
    """Generate shape with at least one boundary dimension.

    Strategy: one dim is a boundary value, others are moderate log-uniform.
    """
    shape = []
    boundary_pos = random.randint(0, ndim - 1)
    for d in range(ndim):
        if d == boundary_pos:
            shape.append(_pick_boundary_dim())
        else:
            # Moderate dims for non-boundary positions
            shape.append(random.choice([2, 4, 8, 16, 32, 64, 128]))

    # Clamp total to avoid OOM
    total = 1
    for s in shape:
        total *= s
    if total > 2**24:
        scale = (2**24 / total) ** (1.0 / ndim)
        shape = [max(1, int(s * scale)) for s in shape]

    return tuple(shape)


def generate_random_boundary_program(seed: Optional[int] = None) -> Tuple[str, dict]:
    """
    Generate a random PyTorch program with boundary shapes but no
    kernel-family-aware op selection or composition patterns.

    Same random op choice as random_generator; only shapes differ.
    """
    if seed is not None:
        random.seed(seed)

    dtype = random.choice(DTYPES)
    n_ops = random.randint(1, 6)
    ndim = random.choice([2, 3, 4])
    shape = _boundary_shape(ndim)
    batch = shape[0]

    meta = {
        'generator': 'random_boundary',
        'dtype': dtype,
        'n_ops': n_ops,
        'shape': shape,
        'ndim': ndim,
        'seed': seed,
    }

    # Same weighted prog_type as random_generator
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
        for i in range(max(0, n_ops - 1)):
            op = random.choice(OPS_UNARY)
            lines.append(f"    x = {op}(x)")
        red_expr, _ = random.choice(OPS_REDUCTION)
        lines.append(f"    return x{red_expr}")
        code = "\n".join(lines)

    elif prog_type == 'matmul':
        # Use boundary dims for M, K, N
        M = _pick_boundary_dim()
        K = random.choice([32, 64, 128, 256])
        N = random.choice([32, 64, 128, 256])
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
        # Use boundary dim for scan dimension
        scan_dim = _pick_boundary_dim()
        if batch * scan_dim > 2**24:
            scan_dim = 2**24 // max(batch, 1)
        lines = [
            f"_x = torch.randn({batch}, {scan_dim}, dtype={dtype})",
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
        spatial = random.choice([15, 16, 17, 31, 32, 33, 63, 64, 65])
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
        spatial = random.choice([15, 16, 17, 31, 32, 33, 63, 64, 65])
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
        code, meta = generate_random_boundary_program(seed=i)
        print(f"=== Random+Boundary {i} ({meta['prog_type']}, shape={meta['shape']}, dtype={meta['dtype']}) ===")
        print(code)
        print()
