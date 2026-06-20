#!/usr/bin/env python3
"""
Kernel-state-targeted program generator.

Given a target kernel state (family, dtype), generates a PyTorch program
that triggers that codegen path when compiled with torch.compile.

The mapping from kernel states to program structures is derived from
Inductor source code analysis:
  - _get_heuristic() determines kernel family
  - Fusion degree depends on op composition in the FX graph
  - Persistence mode depends on reduction numel vs threshold
  - Boundary shapes correspond to codegen threshold constants

See also: generation/compose.py for the clean API.
"""

import random
from typing import Optional, Tuple, List, Dict

# Op pools: which ops trigger which kernel family.
# Derived from Inductor's _get_heuristic() + scheduler logic.

FAMILY_OPS = {
    'pointwise': {
        'unary': [
            'torch.relu', 'torch.sigmoid', 'torch.tanh', 'torch.exp',
            'torch.neg', 'torch.abs', 'torch.sin', 'torch.cos',
            'torch.reciprocal', 'torch.sqrt', 'torch.log1p',
            'torch.nn.functional.silu', 'torch.nn.functional.gelu',
        ],
        'binary': [
            ('torch.add', '{a}, {b}'),
            ('torch.mul', '{a}, {b}'),
            ('torch.sub', '{a}, {b}'),
            ('torch.div', '{a} + 1e-6, {b}.abs() + 1e-6'),
            ('torch.pow', '{a}.abs() + 1e-3, 2.0'),
            ('torch.maximum', '{a}, {b}'),
            ('torch.minimum', '{a}, {b}'),
        ],
    },
    'reduction': [
        ('sum',  '.sum(dim=-1, keepdim=True)'),
        ('mean', '.mean(dim=-1, keepdim=True)'),
        ('amax', '.amax(dim=-1, keepdim=True)'),
        ('amin', '.amin(dim=-1, keepdim=True)'),
        ('prod', '.prod(dim=-1, keepdim=True)'),
        ('std',  '.std(dim=-1, keepdim=True)'),
        ('var',  '.var(dim=-1, keepdim=True)'),
    ],
    'persistent_reduction': [
        ('sum',  '.sum(dim=-1, keepdim=True)'),
        ('mean', '.mean(dim=-1, keepdim=True)'),
        ('amax', '.amax(dim=-1, keepdim=True)'),
        ('softmax', 'torch.nn.functional.softmax({x}, dim=-1)'),
        ('log_softmax', 'torch.nn.functional.log_softmax({x}, dim=-1)'),
        ('layer_norm', 'torch.nn.functional.layer_norm({x}, [{hidden}])'),
    ],
    'split_scan': [
        ('cumsum', 'torch.cumsum({x}, dim={dim})'),
        ('cumprod', 'torch.cumprod({x}.sigmoid(), dim={dim})'),
    ],
    'scan': [
        ('cumsum', 'torch.cumsum({x}, dim={dim})'),
        ('cumprod', 'torch.cumprod({x}.sigmoid(), dim={dim})'),
    ],
    'extern_gemm': [
        ('matmul', 'torch.matmul({a}, {b})'),
        ('mm', 'torch.mm({a2d}, {b2d})'),
        ('bmm', 'torch.bmm({a3d}, {b3d})'),
        ('linear', 'torch.nn.functional.linear({a}, {w})'),
        ('addmm', 'torch.addmm({bias}, {a2d}, {b2d})'),
    ],
    'extern_conv': [
        ('conv2d', 'torch.nn.functional.conv2d({x4d}, {w_conv})'),
        ('conv1d', 'torch.nn.functional.conv1d({x3d}, {w_conv1d})'),
    ],
}

# Boundary shapes from Inductor codegen threshold analysis.
# Each boundary corresponds to a decision point in Triton codegen.

BOUNDARY_SHAPES = {
    'persistent_reduction': {
        # rnumel <= 1024 -> persistent_reduction; > 1024 -> reduction
        'threshold_dim': [511, 512, 513, 1023, 1024, 1025],
    },
    'reduction': {
        # rnumel > 1024, < 16384
        'threshold_dim': [1025, 2047, 2048, 2049, 4096, 8192],
    },
    'split_scan': {
        # cumsum/cumprod: scan_dim >= 16384 -> SplitScan
        'threshold_dim': [16383, 16384, 16385, 32768],
    },
    'scan': {
        # cumsum/cumprod: scan_dim < 16384
        'threshold_dim': [63, 64, 65, 127, 128, 129, 256, 512],
    },
    'cooperative_reduction': {
        # needs very large rnumel + sufficient SM count
        'threshold_dim': [65536, 131072, 262144],
    },
    'tile_mask': [15, 16, 17, 31, 32, 33, 63, 64, 65],
    'fp64_persistent': [511, 512, 513, 514],
}

# Composition patterns for graph construction

COMPOSITION_PATTERNS = [
    'linear_chain',      # op1 → op2 → ... → opN
    'reduce_after_pw',   # pointwise... → reduction
    'branch_merge',      # x → [f(x), g(x)] → add
    'with_backward',     # forward graph + loss.backward()
    'pool_flatten_reduce',  # pool → flatten → reduce (bug_006 pattern)
]


def _pick_boundary_shape(family: str, dtype_str: str) -> int:
    """Select adversarial shape dimension for the target family."""
    if family == 'persistent_reduction':
        if dtype_str == 'float64':
            return random.choice(BOUNDARY_SHAPES['fp64_persistent'])
        return random.choice(BOUNDARY_SHAPES['persistent_reduction']['threshold_dim'])
    elif family == 'reduction':
        return random.choice(BOUNDARY_SHAPES['reduction']['threshold_dim'])
    elif family == 'split_scan':
        return random.choice(BOUNDARY_SHAPES['split_scan']['threshold_dim'])
    elif family == 'scan':
        return random.choice(BOUNDARY_SHAPES['scan']['threshold_dim'])
    elif family == 'cooperative_reduction':
        return random.choice(BOUNDARY_SHAPES['cooperative_reduction']['threshold_dim'])
    else:
        return random.choice(BOUNDARY_SHAPES['tile_mask'])


def _gen_pointwise_chain(n_ops: int, dtype: str) -> Tuple[str, bool]:
    """Generate a chain of n pointwise ops."""
    unary_ops = FAMILY_OPS['pointwise']['unary']
    ops = [random.choice(unary_ops) for _ in range(n_ops)]
    lines = []
    lines.append(f"    v = x")
    for i, op in enumerate(ops):
        lines.append(f"    v = {op}(v)")
    lines.append(f"    return v")
    return "\n".join(lines), True


def _gen_reduction_program(family: str, dtype_str: str,
                           n_fused: int, batch: int,
                           rdim: int) -> Tuple[str, str, bool]:
    """Generate a program targeting reduction/persistent_reduction."""
    ops = FAMILY_OPS.get(family, FAMILY_OPS['reduction'])
    op_name, op_expr = random.choice(ops)

    pw_ops = FAMILY_OPS['pointwise']['unary']

    body_lines = []
    if 'softmax' in op_name or 'log_softmax' in op_name:
        # softmax-style: use the functional form
        body_lines.append(f"_x = torch.randn({batch}, {rdim}, dtype={dtype_str})")
        body_lines.append(f"_program_inputs = (_x,)")
        body_lines.append(f"")
        body_lines.append(f"def program(x):")
        # Pre-chain pointwise ops to increase fusion degree
        for i in range(max(0, n_fused - 2)):
            pw = random.choice(pw_ops)
            body_lines.append(f"    x = {pw}(x)")
        expr = op_expr.format(x='x', hidden=rdim)
        body_lines.append(f"    return {expr}")
    elif 'layer_norm' in op_name:
        body_lines.append(f"_x = torch.randn({batch}, {rdim}, dtype={dtype_str})")
        body_lines.append(f"_program_inputs = (_x,)")
        body_lines.append(f"")
        body_lines.append(f"def program(x):")
        for i in range(max(0, n_fused - 2)):
            pw = random.choice(pw_ops)
            body_lines.append(f"    x = {pw}(x)")
        expr = op_expr.format(x='x', hidden=rdim)
        body_lines.append(f"    return {expr}")
    else:
        # Generic reduction: chain pointwise + reduction
        body_lines.append(f"_x = torch.randn({batch}, {rdim}, dtype={dtype_str})")
        body_lines.append(f"_program_inputs = (_x,)")
        body_lines.append(f"")
        body_lines.append(f"def program(x):")
        for i in range(max(0, n_fused - 1)):
            pw = random.choice(pw_ops)
            body_lines.append(f"    x = {pw}(x)")
        body_lines.append(f"    return x{op_expr}")

    return "\n".join(body_lines), True


def _gen_scan_program(family: str, dtype_str: str,
                      n_fused: int, batch: int,
                      scan_dim_size: int) -> Tuple[str, str, bool]:
    """Generate a program targeting scan/split_scan."""
    ops = FAMILY_OPS[family]
    op_name, op_expr = random.choice(ops)
    pw_ops = FAMILY_OPS['pointwise']['unary']

    # scan dim is dim=1
    dim = 1
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {scan_dim_size}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    # Pre-chain pointwise for fusion
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    expr = op_expr.format(x='x', dim=dim)
    body_lines.append(f"    return {expr}")

    return "\n".join(body_lines), True


def _gen_gemm_program(dtype_str: str, n_fused: int,
                      batch: int, M: int, K: int) -> Tuple[str, str, bool]:
    """Generate a program targeting extern_gemm (matmul/linear/bmm)."""
    N = random.choice([32, 64, 128, 256])
    pw_ops = FAMILY_OPS['pointwise']['unary']

    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {M}, {K}, dtype={dtype_str})")
    body_lines.append(f"_w = torch.randn({K}, {N}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x, _w)")
    body_lines.append(f"")
    body_lines.append(f"def program(x, w):")
    body_lines.append(f"    y = torch.matmul(x, w)")
    # Epilogue fusion: pointwise after GEMM
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    y = {pw}(y)")
    body_lines.append(f"    return y")

    return "\n".join(body_lines), True


def _gen_conv_program(dtype_str: str, batch: int,
                      spatial: int) -> Tuple[str, str, bool]:
    """Generate a program targeting extern_conv."""
    C_in = random.choice([3, 16, 32, 64])
    C_out = random.choice([16, 32, 64])

    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {C_in}, {spatial}, {spatial}, dtype={dtype_str})")
    body_lines.append(f"_w = torch.randn({C_out}, {C_in}, 3, 3, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x, _w)")
    body_lines.append(f"")
    body_lines.append(f"def program(x, w):")
    body_lines.append(f"    return torch.nn.functional.conv2d(x, w, padding=1)")

    return "\n".join(body_lines), True


def _gen_pointwise_program(dtype_str: str, n_fused: int,
                           batch: int, dim: int) -> Tuple[str, str, bool]:
    """Generate a program targeting pointwise kernels."""
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")

    pw_ops = FAMILY_OPS['pointwise']['unary']
    for i in range(max(1, n_fused)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    body_lines.append(f"    return x")

    return "\n".join(body_lines), True


def _gen_pointwise_bf16_program(n_fused: int, batch: int,
                                dim: int) -> Tuple[str, str, bool]:
    """Generate pointwise program with bf16 cast chain."""
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype=torch.float32)")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    body_lines.append(f"    x = x.to(torch.bfloat16).float()")  # cast chain
    pw_ops = FAMILY_OPS['pointwise']['unary']
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    body_lines.append(f"    return x")

    return "\n".join(body_lines), True


def _gen_pointwise_fp16_program(n_fused: int, batch: int,
                                dim: int) -> Tuple[str, str, bool]:
    """Generate pointwise program with fp16 cast chain."""
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype=torch.float32)")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    body_lines.append(f"    x = x.to(torch.float16).float()")
    pw_ops = FAMILY_OPS['pointwise']['unary']
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    body_lines.append(f"    return x")

    return "\n".join(body_lines), True


def _gen_pool_flatten_reduce(dtype_str: str, batch: int) -> Tuple[str, str, bool]:
    """Pool + flatten + reduce pattern."""
    C = random.choice([33, 65, 129, 257, 513, 1025, 2049])  # not multiple of 32
    spatial = random.choice([7, 8, 9])
    pool_size = random.choice([3, 5, 7])

    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {C}, {spatial}, {spatial}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    body_lines.append(f"    y = torch.nn.functional.adaptive_avg_pool2d(x, {pool_size})")
    body_lines.append(f"    return y.flatten(1).sum(dim=-1)")

    return "\n".join(body_lines), True


def _gen_backward_program(dtype_str: str, n_fused: int,
                          batch: int, dim: int) -> Tuple[str, str, bool]:
    """Generate a program that tests backward pass."""
    pw_ops = FAMILY_OPS['pointwise']['unary']

    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype={dtype_str}, requires_grad=True)")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    for i in range(max(1, n_fused)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    body_lines.append(f"    return x.sum()")

    return "\n".join(body_lines), True


def _gen_slice_scatter_backward(dtype_str: str, batch: int,
                                dim: int) -> Tuple[str, str, bool]:
    """slice_scatter backward pattern."""
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype={dtype_str}, requires_grad=True)")
    body_lines.append(f"_src = torch.randn({batch}, {dim // 2}, dtype={dtype_str}, requires_grad=True)")
    body_lines.append(f"_program_inputs = (_x, _src)")
    body_lines.append(f"")
    body_lines.append(f"def program(x, src):")
    body_lines.append(f"    y = x.slice_scatter(src, dim=1, start=0, end={dim // 2})")
    body_lines.append(f"    return y.sum()")

    return "\n".join(body_lines), True


def _gen_inference_mode_matmul(dtype_str: str) -> Tuple[str, str, bool]:
    """inference_mode + batched matmul."""
    B = random.choice([2, 3, 4])
    M = random.choice([4, 8, 16])
    K = random.choice([8, 16, 32])

    body_lines = []
    body_lines.append(f"_x = torch.randn({B}, {M}, {K}, dtype={dtype_str})")
    body_lines.append(f"_w = torch.randn({K}, {K}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x, _w)")
    body_lines.append(f"")
    body_lines.append(f"def program(x, w):")
    body_lines.append(f"    with torch.inference_mode():")
    body_lines.append(f"        return (x @ w).sum()")

    return "\n".join(body_lines), True


# Main generator interface

def generate_targeted_program(
    family: str,
    dtype_str: str = 'torch.float32',
    n_fused_target: str = '2-3',
    pattern: str = 'auto',
    seed: Optional[int] = None,
) -> Tuple[str, dict]:
    """
    Generate a program targeting a specific kernel state.

    Args:
        family: Target kernel family (pointwise, reduction, persistent_reduction,
                split_scan, scan, extern_gemm, extern_conv)
        dtype_str: Target dtype ('torch.float32', 'torch.float64', etc.)
        n_fused_target: Target fusion degree ('1', '2-3', '4+')
        pattern: Composition pattern (auto, linear_chain, with_backward, etc.)
        seed: Random seed for reproducibility

    Returns:
        (body_code, metadata) where body_code is executable Python and
        metadata contains generation parameters.
    """
    if seed is not None:
        random.seed(seed)

    # Parse fusion degree
    if n_fused_target == '1':
        n_fused = 1
    elif n_fused_target == '2-3':
        n_fused = random.choice([2, 3])
    else:  # '4+'
        n_fused = random.choice([4, 5, 6])

    batch = random.choice([1, 2, 3, 4, 7, 8])
    rdim = _pick_boundary_shape(family, dtype_str.replace('torch.', ''))

    meta = {
        'family': family,
        'dtype': dtype_str,
        'n_fused_target': n_fused_target,
        'n_fused_actual': n_fused,
        'batch': batch,
        'boundary_dim': rdim,
        'pattern': pattern,
        'seed': seed,
    }

    # Dispatch to family-specific generator
    if family == 'pointwise':
        if 'bfloat16' in dtype_str:
            code, det = _gen_pointwise_bf16_program(n_fused, batch, rdim)
        elif 'float16' in dtype_str:
            code, det = _gen_pointwise_fp16_program(n_fused, batch, rdim)
        elif pattern == 'with_backward':
            code, det = _gen_backward_program(dtype_str, n_fused, batch, rdim)
        elif pattern == 'slice_scatter_bwd':
            code, det = _gen_slice_scatter_backward(dtype_str, batch, rdim)
        else:
            code, det = _gen_pointwise_program(dtype_str, n_fused, batch, rdim)

    elif family in ('reduction', 'persistent_reduction'):
        if pattern == 'pool_flatten_reduce':
            code, det = _gen_pool_flatten_reduce(dtype_str, batch)
        else:
            code, det = _gen_reduction_program(family, dtype_str, n_fused, batch, rdim)

    elif family in ('split_scan', 'scan'):
        code, det = _gen_scan_program(family, dtype_str, n_fused, batch, rdim)

    elif family == 'extern_gemm':
        if pattern == 'inference_mode':
            code, det = _gen_inference_mode_matmul(dtype_str)
        else:
            code, det = _gen_gemm_program(dtype_str, n_fused, batch, rdim, rdim)

    elif family == 'extern_conv':
        code, det = _gen_conv_program(dtype_str, batch, rdim)

    else:
        code, det = _gen_pointwise_program(dtype_str, n_fused, batch, rdim)

    meta['is_deterministic'] = det
    return code, meta


def enumerate_target_states() -> List[dict]:
    """Enumerate all valid (family, dtype, n_fused) tuples reachable in
    the canonical KernelStateSpace. Returns a list of dicts with keys
    `family`, `dtype`, `n_fused` suitable for generate_targeted_program.
    """
    from kernel.state_space import KernelStateSpace
    kss = KernelStateSpace()

    states = []
    for family in kss.KERNEL_FAMILIES:
        constraints = kss._FAMILY_CONSTRAINTS.get(family, {})
        dtypes = sorted(constraints.get('dtype', kss.DIMENSION_DTYPE))
        n_fused_options = sorted(constraints.get(
            'n_fused', kss.DIMENSION_N_FUSED))
        for dtype_str in dtypes:
            for n in n_fused_options:
                states.append({
                    'family': family,
                    'dtype': f'torch.{dtype_str}',
                    'n_fused': n,
                })
    return states


def generate_for_all_states(n_per_state: int = 3,
                            seed: int = 42) -> List[Tuple[str, dict]]:
    """Generate n programs per reachable kernel state."""
    states = enumerate_target_states()
    programs = []
    for i, state in enumerate(states):
        for j in range(n_per_state):
            s = seed + i * 1000 + j
            code, meta = generate_targeted_program(
                family=state['family'],
                dtype_str=state['dtype'],
                n_fused_target=state['n_fused'],
                seed=s,
            )
            meta['state'] = state
            programs.append((code, meta))
    return programs


if __name__ == '__main__':
    import sys
    # Quick smoke test: generate one program per family
    for family in ['pointwise', 'reduction', 'persistent_reduction',
                   'split_scan', 'scan', 'extern_gemm', 'extern_conv']:
        code, meta = generate_targeted_program(
            family=family, dtype_str='torch.float32',
            n_fused_target='2-3', seed=42)
        print(f"=== {family} ===")
        print(code)
        print(f"  meta: {meta}")
        print()
