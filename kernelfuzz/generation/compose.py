"""
ComposeGraph: construct computation graphs that preserve target interactions.

Two composition modes:
  1. Family-targeted: generate a program targeting a specific kernel state
     (family, dtype) by selecting ops from the corresponding pool and
     sampling shapes at codegen decision boundaries.
  2. Cross-category: chain 2-3 stages from different template categories
     (reduction, GEMM, layout, dependency, boundary) to create novel
     computation graph topologies.
"""

import random
from typing import Optional, Tuple, List

from generation.ops import FAMILY_OPS, pointwise_unary_ops
from generation.conditions import (
    pick_boundary_shape, FAMILY_BOUNDARY_SHAPES,
)


# ============================================================================
# Family-targeted program generators
# ============================================================================

def gen_pointwise_program(dtype_str: str, n_fused: int,
                          batch: int, dim: int) -> Tuple[str, bool]:
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    pw_ops = pointwise_unary_ops()
    for i in range(max(1, n_fused)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    body_lines.append(f"    return x")
    return "\n".join(body_lines), True


def gen_pointwise_bf16_program(n_fused: int, batch: int,
                               dim: int) -> Tuple[str, bool]:
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype=torch.float32)")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    body_lines.append(f"    x = x.to(torch.bfloat16).float()")
    pw_ops = pointwise_unary_ops()
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    body_lines.append(f"    return x")
    return "\n".join(body_lines), True


def gen_pointwise_fp16_program(n_fused: int, batch: int,
                               dim: int) -> Tuple[str, bool]:
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype=torch.float32)")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    body_lines.append(f"    x = x.to(torch.float16).float()")
    pw_ops = pointwise_unary_ops()
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    body_lines.append(f"    return x")
    return "\n".join(body_lines), True


def gen_reduction_program(family: str, dtype_str: str,
                          n_fused: int, batch: int,
                          rdim: int) -> Tuple[str, bool]:
    ops = FAMILY_OPS.get(family, FAMILY_OPS['reduction'])
    op_name, op_expr = random.choice(ops)
    pw_ops = pointwise_unary_ops()

    body_lines = []
    if 'softmax' in op_name or 'log_softmax' in op_name:
        body_lines.append(f"_x = torch.randn({batch}, {rdim}, dtype={dtype_str})")
        body_lines.append(f"_program_inputs = (_x,)")
        body_lines.append(f"")
        body_lines.append(f"def program(x):")
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
        body_lines.append(f"_x = torch.randn({batch}, {rdim}, dtype={dtype_str})")
        body_lines.append(f"_program_inputs = (_x,)")
        body_lines.append(f"")
        body_lines.append(f"def program(x):")
        for i in range(max(0, n_fused - 1)):
            pw = random.choice(pw_ops)
            body_lines.append(f"    x = {pw}(x)")
        body_lines.append(f"    return x{op_expr}")
    return "\n".join(body_lines), True


def gen_scan_program(family: str, dtype_str: str,
                     n_fused: int, batch: int,
                     scan_dim_size: int) -> Tuple[str, bool]:
    ops = FAMILY_OPS[family]
    op_name, op_expr = random.choice(ops)
    pw_ops = pointwise_unary_ops()
    dim = 1

    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {scan_dim_size}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x,)")
    body_lines.append(f"")
    body_lines.append(f"def program(x):")
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    x = {pw}(x)")
    expr = op_expr.format(x='x', dim=dim)
    body_lines.append(f"    return {expr}")
    return "\n".join(body_lines), True


def gen_gemm_program(dtype_str: str, n_fused: int,
                     batch: int, M: int, K: int) -> Tuple[str, bool]:
    N = random.choice([32, 64, 128, 256])
    pw_ops = pointwise_unary_ops()

    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {M}, {K}, dtype={dtype_str})")
    body_lines.append(f"_w = torch.randn({K}, {N}, dtype={dtype_str})")
    body_lines.append(f"_program_inputs = (_x, _w)")
    body_lines.append(f"")
    body_lines.append(f"def program(x, w):")
    body_lines.append(f"    y = torch.matmul(x, w)")
    for i in range(max(0, n_fused - 1)):
        pw = random.choice(pw_ops)
        body_lines.append(f"    y = {pw}(y)")
    body_lines.append(f"    return y")
    return "\n".join(body_lines), True


def gen_conv_program(dtype_str: str, batch: int,
                     spatial: int) -> Tuple[str, bool]:
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


def gen_backward_program(dtype_str: str, n_fused: int,
                         batch: int, dim: int) -> Tuple[str, bool]:
    pw_ops = pointwise_unary_ops()
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


def gen_slice_scatter_backward(dtype_str: str, batch: int,
                               dim: int) -> Tuple[str, bool]:
    body_lines = []
    body_lines.append(f"_x = torch.randn({batch}, {dim}, dtype={dtype_str}, requires_grad=True)")
    body_lines.append(f"_src = torch.randn({batch}, {dim // 2}, dtype={dtype_str}, requires_grad=True)")
    body_lines.append(f"_program_inputs = (_x, _src)")
    body_lines.append(f"")
    body_lines.append(f"def program(x, src):")
    body_lines.append(f"    y = x.slice_scatter(src, dim=1, start=0, end={dim // 2})")
    body_lines.append(f"    return y.sum()")
    return "\n".join(body_lines), True


def gen_inference_mode_matmul(dtype_str: str) -> Tuple[str, bool]:
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


def gen_pool_flatten_reduce(dtype_str: str, batch: int) -> Tuple[str, bool]:
    C = random.choice([33, 65, 129, 257, 513, 1025, 2049])
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


# ============================================================================
# Main targeted generation interface
# ============================================================================

def generate_targeted_program(
    family: str,
    dtype_str: str = 'torch.float32',
    n_fused_target: str = '2-3',
    pattern: str = 'auto',
    seed: Optional[int] = None,
) -> Tuple[str, dict]:
    """Generate a program targeting a specific kernel state.

    Args:
        family: Target kernel family (pointwise, reduction, persistent_reduction,
                split_scan, scan, extern_gemm, extern_conv)
        dtype_str: Target dtype ('torch.float32', 'torch.float64', etc.)
        n_fused_target: Target fusion degree ('1', '2-3', '4+')
        pattern: Composition pattern (auto, with_backward, etc.)
        seed: Random seed for reproducibility

    Returns:
        (body_code, metadata)
    """
    if seed is not None:
        random.seed(seed)

    if n_fused_target == '1':
        n_fused = 1
    elif n_fused_target == '2-3':
        n_fused = random.choice([2, 3])
    else:
        n_fused = random.choice([4, 5, 6])

    batch = random.choice([1, 2, 3, 4, 7, 8])
    rdim = pick_boundary_shape(family, dtype_str.replace('torch.', ''))

    meta = {
        'family': family, 'dtype': dtype_str,
        'n_fused_target': n_fused_target, 'n_fused_actual': n_fused,
        'batch': batch, 'boundary_dim': rdim,
        'pattern': pattern, 'seed': seed,
    }

    if family == 'pointwise':
        if 'bfloat16' in dtype_str:
            code, det = gen_pointwise_bf16_program(n_fused, batch, rdim)
        elif 'float16' in dtype_str:
            code, det = gen_pointwise_fp16_program(n_fused, batch, rdim)
        elif pattern == 'with_backward':
            code, det = gen_backward_program(dtype_str, n_fused, batch, rdim)
        elif pattern == 'slice_scatter_bwd':
            code, det = gen_slice_scatter_backward(dtype_str, batch, rdim)
        else:
            code, det = gen_pointwise_program(dtype_str, n_fused, batch, rdim)
    elif family in ('reduction', 'persistent_reduction'):
        if pattern == 'pool_flatten_reduce':
            code, det = gen_pool_flatten_reduce(dtype_str, batch)
        else:
            code, det = gen_reduction_program(family, dtype_str, n_fused, batch, rdim)
    elif family in ('split_scan', 'scan'):
        code, det = gen_scan_program(family, dtype_str, n_fused, batch, rdim)
    elif family == 'extern_gemm':
        if pattern == 'inference_mode':
            code, det = gen_inference_mode_matmul(dtype_str)
        else:
            code, det = gen_gemm_program(dtype_str, n_fused, batch, rdim, rdim)
    elif family == 'extern_conv':
        code, det = gen_conv_program(dtype_str, batch, rdim)
    else:
        code, det = gen_pointwise_program(dtype_str, n_fused, batch, rdim)

    meta['is_deterministic'] = det
    return code, meta


def enumerate_target_states() -> List[dict]:
    """Enumerate all valid (family, dtype, n_fused) tuples reachable in
    the canonical KernelStateSpace."""
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


# ============================================================================
# Cross-category composition (TemplateCompositor)
# ============================================================================

class TemplateCompositor:
    """Creates novel computation graphs by chaining stages from
    different template categories. This forces structural diversity
    that cannot be achieved by mutating within a single template."""

    _STAGES = [
        ("reduce_norm", "D1", lambda b, s, h, d: (
            f"    _mean = x.mean(dim=-1, keepdim=True)\n"
            f"    _var = ((x - _mean) ** 2).mean(dim=-1, keepdim=True)\n"
            f"    x = (x - _mean) / torch.sqrt(_var + 1e-5)\n"
        )),
        ("reduce_softmax", "D1", lambda b, s, h, d: (
            f"    _xmax = x.amax(dim=-1, keepdim=True)\n"
            f"    x = torch.exp(x - _xmax)\n"
            f"    x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)\n"
        )),
        ("reduce_scale", "D1", lambda b, s, h, d: (
            f"    _scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)\n"
            f"    x = x / _scale\n"
        )),
        ("reduce_broadcast_cross", "D1", lambda b, s, h, d: (
            f"    _xm = x.mean(dim=-1, keepdim=True)\n"
            f"    x = (x - _xm) * x.std(dim=-1, keepdim=True).clamp(min=1e-5)\n"
            f"    x = x.sum(dim=1, keepdim=True).expand_as(x) + x\n"
        )),
        ("gemm_project", "D2", None),
        ("gemm_bilinear", "D2", None),
        ("layout_permute", "D3", lambda b, s, h, d: (
            f"    x = x.transpose(-1, -2).contiguous().transpose(-1, -2)\n"
        )),
        ("layout_chunk_cat", "D3", lambda b, s, h, d: (
            f"    _chunks = x.chunk(2, dim=-1)\n"
            f"    x = torch.cat([_chunks[1], _chunks[0]], dim=-1)\n"
        )),
        ("layout_permute_contig", "D3", lambda b, s, h, d: (
            f"    x = x.transpose(-1, -2).contiguous()\n"
            f"    x = x.transpose(-1, -2)\n"
        )),
        ("layout_slice_stride", "D3", lambda b, s, h, d: (
            f"    _half = {h} // 2 + {h} % 2\n"
            f"    x = x[..., ::2]\n"
            f"    x = F.pad(x, (0, {h} - _half), value=0.0)\n"
        )),
        ("dep_gating", "D4", lambda b, s, h, d: (
            f"    _gate = torch.sigmoid(x.mean(dim=-1, keepdim=True))\n"
            f"    x = x * _gate\n"
        )),
        ("dep_residual_scale", "D4", lambda b, s, h, d: (
            f"    _res = x.clone()\n"
            f"    _s = x.std(dim=-1, keepdim=True).clamp(min=1e-5)\n"
            f"    x = _res + x / _s\n"
        )),
        ("bound_dtype_cast", "D5", lambda b, s, h, d: (
            f"    x = x.to(torch.float64).to({d})\n"
        )),
        ("bound_narrow_expand", "D5", lambda b, s, h, d: (
            f"    _n = min(x.shape[-1], max(1, x.shape[-1] - 1))\n"
            f"    x = x.narrow(-1, 0, _n)\n"
            f"    x = F.pad(x, (0, {h} - _n), value=0.0)\n"
        )),
        ("bound_half_roundtrip", "D5", lambda b, s, h, d: (
            f"    x = x.half().float()\n"
            f"    x = torch.relu(x)\n"
        )),
        ("bound_logit_eps", "D5", lambda b, s, h, d: (
            f"    x = x.sigmoid()\n"
            f"    x = torch.logit(x, eps=0.9)\n"
        )),
    ]

    _GEMM_GENERATORS = {
        "gemm_project": lambda b, s, h, d: {
            "input_code": (
                f"_w_proj = torch.randn({h}, {h}, dtype={d})\n"
                f"if torch.cuda.is_available(): _w_proj = _w_proj.cuda()\n"
            ),
            "input_names": ["_w_proj"],
            "body_code": f"    x = torch.matmul(x, _w_proj)\n",
        },
        "gemm_bilinear": lambda b, s, h, d: {
            "input_code": (
                f"_w_bil = torch.randn({h}, {h}, dtype={d})\n"
                f"_b_bil = torch.randn({h}, dtype={d})\n"
                f"if torch.cuda.is_available(): _w_bil = _w_bil.cuda(); _b_bil = _b_bil.cuda()\n"
            ),
            "input_names": ["_w_bil", "_b_bil"],
            "body_code": (
                f"    x = torch.addmm(_b_bil, x.reshape(-1, {h}), _w_bil).reshape(x.shape)\n"
            ),
        },
    }

    @staticmethod
    def compose(batch, seq, hidden, dtype):
        """Generate a novel program by chaining 2-3 random stages
        from different categories."""
        n_stages = random.choice([2, 2, 3])
        available = list(TemplateCompositor._STAGES)
        random.shuffle(available)

        chosen = []
        used_cats = set()
        for stage_name, cat, gen_fn in available:
            if cat not in used_cats or len(chosen) < 2:
                chosen.append((stage_name, cat, gen_fn))
                used_cats.add(cat)
            if len(chosen) >= n_stages:
                break

        stage_code = ""
        stage_names = []
        extra_input_code = ""
        extra_input_names = []
        for stage_name, cat, gen_fn in chosen:
            if gen_fn is None and stage_name in TemplateCompositor._GEMM_GENERATORS:
                gemm = TemplateCompositor._GEMM_GENERATORS[stage_name](
                    batch, seq, hidden, dtype)
                extra_input_code += gemm["input_code"]
                extra_input_names.extend(gemm["input_names"])
                stage_code += gemm["body_code"]
            elif gen_fn is not None:
                stage_code += gen_fn(batch, seq, hidden, dtype)
            stage_names.append(f"{cat}:{stage_name}")

        name = "composed_" + "_".join(s[0] for s in chosen)
        all_params = ["x"] + extra_input_names
        func_sig = ", ".join(all_params)
        all_inputs = ["_x"] + extra_input_names

        body = (
            f"import torch.nn.functional as F\n"
            f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
            f"{extra_input_code}"
            f"_program_inputs = ({', '.join(all_inputs)},)\n\n"
            f"def program({func_sig}):\n"
            f"{stage_code}"
            f"    return x\n"
        )
        return name, body, True
