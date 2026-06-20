#!/usr/bin/env python3
"""
Template registry for kernel-space fuzzing (paper §3.2).

Template families target specific codegen decisions:
  D1: Pointwise-Reduction Fusion
  D2: Epilogue Fusion into GEMM/Conv
  D3: Layout Propagation
  D4: Cross-op Data Dependency
  D5: Shape/Dtype Boundary + Alias/Autograd
  D6+: Extended coverage (scatter, linalg, sparse, etc.)
  CF: Composition Fusion patterns
  DYN: Dynamic shape patterns

Each template is a function (batch, seq, hidden, dtype) -> (body_code, setup_code).
PROGRAM_TEMPLATES is the weighted registry used by the fuzzer.
"""

import math
import random

import torch

class GraphPoisonInjector:
    """
    Inject edge-case values into tensor inputs to stress codegen paths.

    Follows NNSmith principle: avoid FP exceptional values that cause false
    positives. NaN/Inf injection probability is kept very low (5%); primarily
    injects extreme finite values. Also supports 'zero-mean' poisoning that
    subtracts tensor.mean(), causing x/x.mean() or x/x.std() to produce
    inf/nan and activate nan_to_num / where / clamp code paths.
    """
    @staticmethod
    def inject(body_code, dtype):
        roll = random.random()
        if   roll < 0.03: poison_expr = "float('inf')"
        elif roll < 0.06: poison_expr = "float('nan')"
        elif roll < 0.20: poison_expr = random.choice(["1e15", "-1e15", "1e-15"])
        elif roll < 0.30: poison_expr = "ZERO_MEAN"  # new: zero-mean shift
        else: return body_code

        lines, new_lines, injected = body_code.split("\n"), [], False
        for line in lines:
            new_lines.append(line)
            if (not injected and "torch.randn" in line and "=" in line
                    and "_program_inputs" not in line):
                var = line.split("=")[0].strip()
                if var.startswith("_") and not var.startswith("_program"):
                    if poison_expr == "ZERO_MEAN":
                        # Make mean ≈ 0 → x/mean or x/std produces inf
                        new_lines.append(
                            f"if {var}.numel() > 0: "
                            f"{var}.sub_({var}.mean())")
                    else:
                        new_lines.append(
                            f"if {var}.numel() > 0: {var}.reshape(-1)[0] = {poison_expr}")
                    injected = True
        return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Input Value Diversification — replace torch.randn() with edge-case
# values to trigger codegen paths that normal Gaussian never reaches.
# ---------------------------------------------------------------------------
_INPUT_VALUE_STRATEGIES = [
    # (name, code_template) — {var} is the tensor variable, {shape} its shape expr
    ('zeros',    '{var} = torch.zeros({shape}, dtype={dtype})'),
    ('ones',     '{var} = torch.ones({shape}, dtype={dtype})'),
    ('large',    '{var} = torch.randn({shape}, dtype={dtype}) * 1e6'),
    ('small',    '{var} = torch.randn({shape}, dtype={dtype}) * 1e-7'),
    ('integer',  '{var} = torch.randint(-10, 10, {shape}, dtype={dtype}).to({dtype})'),
    ('sparse',   '{var} = torch.randn({shape}, dtype={dtype}); {var}[{var}.abs() < 1.5] = 0'),
    ('uniform',  '{var} = torch.rand({shape}, dtype={dtype})'),
    ('constant', '{var} = torch.full({shape}, {const}, dtype={dtype})'),
]

def _apply_value_diversification(body_code, dtype):
    """20% chance to replace randn inputs with edge-case values.
    Only modifies the first input tensor to keep the program valid."""
    if random.random() >= 0.20:
        return body_code, None

    lines = body_code.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if ("torch.randn" in stripped and "=" in stripped
                and "_program_inputs" not in stripped
                and stripped.split("=")[0].strip().startswith("_")):
            var = stripped.split("=")[0].strip()
            # Extract shape args from torch.randn(...)
            import re as _re
            m = _re.search(r'torch\.randn\(([^)]+)\)', stripped)
            if not m:
                continue
            args_str = m.group(1)
            # Split on 'dtype=' to separate shape from kwargs
            if 'dtype=' in args_str:
                shape_part = args_str[:args_str.index('dtype=')].rstrip(', ')
            else:
                shape_part = args_str
            # For strategies that need tuple shape (randint, full, etc.)
            shape_tuple = f"({shape_part},)" if ',' not in shape_part else f"({shape_part})"
            shape_raw = shape_part.strip()

            strategy_name, tpl = random.choice(_INPUT_VALUE_STRATEGIES)
            const = random.choice([0.5, -1.0, 3.14, 100.0])
            indent = " " * (len(line) - len(line.lstrip()))
            new_line = indent + tpl.format(
                var=var, shape=shape_raw, dtype=dtype, const=const)
            # integer strategy: randint needs tuple shape
            if strategy_name == 'integer':
                new_line = (f"{indent}{var} = torch.randint(-10, 10, "
                            f"{shape_tuple}, dtype={dtype}).to({dtype})")
            elif strategy_name == 'constant':
                new_line = (f"{indent}{var} = torch.full("
                            f"{shape_tuple}, {const}, dtype={dtype})")
            lines[i] = new_line
            return "\n".join(lines), strategy_name
    return body_code, None


# =============================================================================
# D1: Pointwise-Reduction Fusion Templates
# Inductor fuses pointwise epilogues into reduction kernels, changing
# accumulation order and buffer allocation decisions.
# =============================================================================

def _tpl_d1_softmax_fused(batch, seq, hidden, dtype):
    """
    Scaled softmax: scale -> exp -> sum -> div.
    Inductor fuses exp and div into the sum reduction kernel.
    Multi-head variants trigger Triton tile size decisions.
    """
    n_heads = random.choice([1, 2, 4])
    head_dim = max(8, hidden // n_heads)
    scale = float(head_dim) ** -0.5
    return (
        f"_x = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x * {scale:.8f}\n"
        f"    x_max = x.amax(dim=-1, keepdim=True)\n"
        f"    x = x - x_max\n"
        f"    x_exp = torch.exp(x)\n"
        f"    x_sum = x_exp.sum(dim=-1, keepdim=True)\n"
        f"    return x_exp / x_sum\n"
    ), True


def _tpl_d1_layernorm_manual(batch, seq, hidden, dtype):
    """
    Manual layernorm: mean -> variance -> normalize.
    Inductor tries to fuse two reductions (mean, variance) plus the final
    elementwise op, which may diverge from torch.nn.functional.layer_norm.
    """
    return (
        f"_x   = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w   = torch.ones({hidden}, dtype={dtype})\n"
        f"_b   = torch.zeros({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    mean = x.mean(dim=-1, keepdim=True)\n"
        f"    var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)\n"
        f"    x_n  = (x - mean) / torch.sqrt(var + 1e-5)\n"
        f"    return x_n * w + b\n"
    ), True


def _tpl_d1_reduce_cascade(batch, seq, hidden, dtype):
    """
    Cascaded reduction: multi-step reduce + elementwise interleaving.
    Inductor's buffer allocation between multiple reductions is error-prone.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    # reductions along two dims interact\n"
        f"    m0 = x.mean(dim=0, keepdim=True)   # [1, seq, hidden]\n"
        f"    m1 = x.mean(dim=1, keepdim=True)   # [batch, 1, hidden]\n"
        f"    s  = x.std(dim=-1, keepdim=True)   # [batch, seq, 1]\n"
        f"    x  = (x - m0) * (x - m1) / (s + 1e-5)\n"
        f"    return x.mean(dim=(0, 1))\n"
    ), True


def _tpl_d1_online_softmax(batch, seq, hidden, dtype):
    """
    Online softmax via log-sum-exp stabilization.
    log(sum(exp(x - max))) reduce -> pointwise chain.
    Graph-level variant of the torch.logsumexp pattern.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    lse = torch.logsumexp(x, dim=-1, keepdim=True)\n"
        f"    attn = torch.exp(x - lse)\n"
        f"    return torch.matmul(attn, w)\n"
    ), True


def _tpl_d1_multi_axis_reduce(batch, seq, hidden, dtype):
    """
    Multi-axis simultaneous reduction: x.sum(dim=(0,2)).
    Inductor needs nested loops or multiple reductions; Triton tile
    boundary handling is complex for multi-axis cases.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    r1 = x.sum(dim=(0, 2))              # [{seq}]\n"
        f"    r2 = x.mean(dim=(1,), keepdim=True)  # [{batch}, 1, {hidden}]\n"
        f"    return x - r2 + r1.unsqueeze(0).unsqueeze(-1)\n"
    ), True


def _tpl_d1_large_reduce_float(batch, seq, hidden, dtype):
    """
    Float reduction on large tensors (>2^18 elements).
    Inspired by PyTorch #113976 (Triton codegen loop variable type error
    on large tensors). Generalized to float sum for differential oracle.
    """
    large_n = max(640, seq * 10)
    large_m = max(512, hidden * 8)
    return (
        f"_x = torch.randn({large_n}, {large_m}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    neg_count = (x < 0.0).float().sum()        # large reduction\n"
        f"    scale = neg_count / x.numel()               # scalar\n"
        f"    return x * scale + x.mean(dim=0)\n"
    ), True


def _tpl_d1_chained_norm(batch, seq, hidden, dtype):
    """
    Chained layernorm -> rmsnorm: multiple reduce + pointwise fusion.
    When consecutive norm reductions are fused, intermediate buffer
    allocation may conflict.
    """
    return (
        f"_x  = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w1 = torch.ones({hidden}, dtype={dtype})\n"
        f"_w2 = torch.ones({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w1, _w2)\n\n"
        f"def program(x, w1, w2):\n"
        f"    # layernorm\n"
        f"    mean = x.mean(dim=-1, keepdim=True)\n"
        f"    var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)\n"
        f"    x = (x - mean) / torch.sqrt(var + 1e-5) * w1\n"
        f"    # rmsnorm (no mean subtraction)\n"
        f"    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-5)\n"
        f"    x = x / rms * w2\n"
        f"    return x\n"
    ), True


def _tpl_d1_softmax_large_dim(batch, seq, hidden, dtype):
    """
    Softmax on large hidden dim (>1024).
    Triton's online reduction must switch strategies when the dimension
    exceeds the tile size.
    """
    large_hidden = max(1024, hidden * 16)
    return (
        f"_x = torch.randn({batch}, {seq}, {large_hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.softmax(x, dim=-1)\n"
        f"    return x.sum(dim=1)\n"
    ), True


def _tpl_d1_var_mean_fused(batch, seq, hidden, dtype):
    """
    torch.var_mean returns a tuple (var, mean).
    Tests Inductor's var+mean fusion handling of tuple outputs.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    var, mean = torch.var_mean(x, dim=-1, keepdim=True)\n"
        f"    x = (x - mean) / torch.sqrt(var + 1e-5)\n"
        f"    return torch.matmul(x, w)\n"
    ), True


def _tpl_d1_cross_entropy_manual(batch, seq, hidden, dtype):
    """
    Manual cross-entropy: log_softmax -> nll -> reduce.
    The deep fusion chain max -> sub -> exp -> sum -> log -> gather -> mean
    is a frequent error path in Inductor's log_softmax fusion.
    """
    n_classes = max(16, hidden)
    return (
        f"_logits = torch.randn({batch}, {n_classes}, dtype={dtype})\n"
        f"_labels = torch.randint(0, {n_classes}, ({batch},))\n"
        f"_program_inputs = (_logits, _labels)\n\n"
        f"def program(logits, labels):\n"
        f"    max_val = logits.amax(dim=-1, keepdim=True)\n"
        f"    shifted = logits - max_val\n"
        f"    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))\n"
        f"    log_probs = shifted - log_sum_exp\n"
        f"    nll = -log_probs.gather(1, labels.unsqueeze(1))\n"
        f"    return nll.mean()\n"
    ), True


def _tpl_d1_welford_online_var(batch, seq, hidden, dtype):
    """
    Welford online variance algorithm: incremental mean/var updates.
    Each step depends on the previous running mean, preventing arbitrary
    parallelization by Inductor.
    """
    n_chunks = random.choice([3, 4, 5])
    chunk_size = max(4, seq // n_chunks)
    total_seq = chunk_size * n_chunks
    return (
        f"_x = torch.randn({batch}, {total_seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    chunks = x.split({chunk_size}, dim=1)\n"
        f"    mean = torch.zeros({batch}, 1, {hidden}, dtype=x.dtype, device=x.device)\n"
        f"    m2 = torch.zeros({batch}, 1, {hidden}, dtype=x.dtype, device=x.device)\n"
        f"    count = 0\n"
        f"    for chunk in chunks:\n"
        f"        chunk_mean = chunk.mean(dim=1, keepdim=True)\n"
        f"        n = chunk.shape[1]\n"
        f"        delta = chunk_mean - mean\n"
        f"        mean = mean + delta * n / (count + n)\n"
        f"        m2 = m2 + ((chunk - chunk_mean) ** 2).sum(dim=1, keepdim=True) + delta ** 2 * count * n / (count + n)\n"
        f"        count += n\n"
        f"    return mean.squeeze(1) + torch.sqrt(m2.squeeze(1) / max(count, 1) + 1e-8)\n"
    ), True


def _tpl_d1_kl_divergence_manual(batch, seq, hidden, dtype):
    """
    KL divergence: p * log(p/q) reduction.
    Long fusion chain of log + div + mul + sum; Inductor has a special
    pass for log(softmax_output).
    """
    n_classes = max(16, hidden)
    return (
        f"_p_logits = torch.randn({batch}, {n_classes}, dtype={dtype})\n"
        f"_q_logits = torch.randn({batch}, {n_classes}, dtype={dtype})\n"
        f"_program_inputs = (_p_logits, _q_logits)\n\n"
        f"def program(p_logits, q_logits):\n"
        f"    p = torch.nn.functional.softmax(p_logits, dim=-1)\n"
        f"    log_p = torch.nn.functional.log_softmax(p_logits, dim=-1)\n"
        f"    log_q = torch.nn.functional.log_softmax(q_logits, dim=-1)\n"
        f"    kl = (p * (log_p - log_q)).sum(dim=-1)\n"
        f"    return kl\n"
    ), True


def _tpl_d1_prod_reduce(batch, seq, hidden, dtype):
    """
    torch.prod reduction -- less common than sum, possibly under-tested
    in Inductor. The prod + log + exp numerical chain is error-prone
    in Triton codegen.
    """
    small_hidden = max(4, min(16, hidden // 4))
    return (
        f"_x = torch.randn({batch}, {seq}, {small_hidden}, dtype={dtype}).clamp(-2, 2)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.sigmoid(x)\n"
        f"    prod_val = x.prod(dim=-1)\n"
        f"    log_prod = torch.log(prod_val + 1e-10)\n"
        f"    return log_prod.sum(dim=-1)\n"
    ), True


# =============================================================================
# D2: Epilogue Fusion into GEMM Templates
# Inductor fuses post-matmul bias/activation into GEMM Triton kernels
# as epilogues, which is a major source of codegen bugs.
# =============================================================================

def _tpl_d2_linear_bias_act(batch, seq, hidden, dtype):
    """
    Linear + bias + activation epilogue.
    Inductor's Triton codegen fuses bias_add + gelu into the matmul kernel.
    Epilogue fusion has known bugs at tile boundaries.
    """
    mid = max(8, hidden * 2)
    act = random.choice(["torch.relu", "torch.nn.functional.gelu", "torch.sigmoid"])
    return (
        f"_x  = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w1 = torch.randn({hidden}, {mid}, dtype={dtype})\n"
        f"_b1 = torch.randn({mid}, dtype={dtype})\n"
        f"_w2 = torch.randn({mid}, {hidden}, dtype={dtype})\n"
        f"_b2 = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w1, _b1, _w2, _b2)\n\n"
        f"def program(x, w1, b1, w2, b2):\n"
        f"    h = {act}(torch.matmul(x, w1) + b1)\n"
        f"    return {act}(torch.matmul(h, w2) + b2)\n"
    ), True


def _tpl_d2_matmul_scale_add(batch, seq, hidden, dtype):
    """
    matmul -> scale -> add (residual connection epilogue).
    Inductor fuses scale + residual add into GEMM epilogue, which can
    produce incorrect results when strides are misaligned.
    """
    return (
        f"_x   = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w   = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_res = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _res)\n\n"
        f"def program(x, w, res):\n"
        f"    h = torch.matmul(x, w) * 0.125\n"
        f"    h = h + res\n"
        f"    return torch.nn.functional.layer_norm(h, [{hidden}])\n"
    ), True


def _tpl_d2_conv_bn_act(batch, seq, hidden, dtype):
    """
    Conv -> BatchNorm -> activation.
    After conv2d + bn fusion, Inductor's C++ codegen may produce
    duplicate buffer declarations (Issue #159245).
    """
    c = max(4, hidden // 8)
    hw = max(4, seq // 4)
    return (
        f"_x = torch.randn({batch}, {c}, {hw}, {hw}, dtype={dtype})\n"
        f"_w = torch.randn({c}, {c}, 3, 3, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = torch.nn.functional.conv2d(x, w, padding=1)\n"
        f"    x = torch.nn.functional.batch_norm(\n"
        f"        x,\n"
        f"        running_mean=torch.zeros({c}, dtype=x.dtype, device=x.device),\n"
        f"        running_var=torch.ones({c}, dtype=x.dtype, device=x.device),\n"
        f"        training=False,\n"
        f"    )\n"
        f"    return torch.relu(x)\n"
    ), True


def _tpl_d2_attention_epilogue(batch, seq, hidden, dtype):
    """
    Q*K^T/sqrt(d) -> softmax -> *V: attention score epilogue.
    Inductor tries to fuse scale + softmax reduction into GEMM epilogue.
    This is one of the highest bug-density subgraphs in Transformer models.
    """
    n_heads = random.choice([1, 2, 4])
    head_dim = max(8, hidden // n_heads)
    scale = float(head_dim) ** -0.5
    return (
        f"_q = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_v = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    scores = torch.matmul(q, k.transpose(-2, -1)) * {scale:.8f}\n"
        f"    attn = torch.nn.functional.softmax(scores, dim=-1)\n"
        f"    return torch.matmul(attn, v)\n"
    ), True


def _tpl_d2_interpolate_chain(batch, seq, hidden, dtype):
    """
    interpolate(bicubic) -> neg -> linear -> tan.
    Inspired by PyTorch #145264 (100% element mismatch, abs diff up to ~4132).
    Bicubic interpolation output through multi-step epilogue triggers
    completely incorrect Inductor fusion results.
    """
    h_in = max(4, seq // 4)
    h_out = max(16, seq)
    out_features = max(8, hidden // 4)
    return (
        f"_x = torch.randn({batch}, 1, {h_in}, 1, dtype={dtype})\n"
        f"_w = torch.randn({out_features}, {h_out}, dtype={dtype})\n"
        f"_b = torch.randn({out_features}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    x = torch.nn.functional.interpolate(x, size=[{h_out}, 1], mode='bicubic', align_corners=False)\n"
        f"    x = torch.neg(x)\n"
        f"    x = x.view(x.size(0), -1)\n"
        f"    x = torch.matmul(x, w.t()) + b\n"
        f"    return torch.tanh(x)\n"
    ), True


def _tpl_d2_pool_int_dtype(batch, seq, hidden, dtype):
    """
    argsort(int64) -> float cast -> avg_pool2d.
    Inspired by PyTorch #143738 (avg_pool2d mishandles int64 input).
    Tests Inductor lowering correctness on dtype cast + pool chains.
    """
    c = max(1, batch)
    h = max(4, seq // 2)
    w = max(4, hidden // 4)
    return (
        f"_x = torch.randn({c}, 1, {h}, {w}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    idx = torch.argsort(x, dim=3)        # int64 output\n"
        f"    x_sorted = torch.gather(x, 3, idx)   # reorder by argsort\n"
        f"    x_sorted = torch.nn.functional.avg_pool2d(x_sorted, kernel_size=2, stride=2)\n"
        f"    return x_sorted\n"
    ), True


def _tpl_d2_multi_output_epilogue(batch, seq, hidden, dtype):
    """
    GEMM result consumed by multiple branches (branch_a = y.sum, branch_b = y*y).
    Epilogue fusion with multiple consumers may only be correct for one branch.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_b = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    y = torch.relu(torch.matmul(x, w) + b)\n"
        f"    branch_a = y.sum(dim=-1)           # reduce consumer\n"
        f"    branch_b = y * y                   # pointwise consumer\n"
        f"    return branch_a.unsqueeze(-1) + branch_b\n"
    ), True


def _tpl_d2_conv_groupnorm_skip(batch, seq, hidden, dtype):
    """
    conv2d + GroupNorm + skip connection.
    GroupNorm's per-group reduction makes conv epilogue fusion more complex
    than BatchNorm.
    """
    c = max(8, hidden // 4)
    # num_groups must evenly divide channels
    num_groups = random.choice([g for g in [1, 2, 4, 8] if c % g == 0])
    hw = max(4, seq // 4)
    return (
        f"_x = torch.randn({batch}, {c}, {hw}, {hw}, dtype={dtype})\n"
        f"_w = torch.randn({c}, {c}, 3, 3, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    residual = x\n"
        f"    x = torch.nn.functional.conv2d(x, w, padding=1)\n"
        f"    x = torch.nn.functional.group_norm(x, {num_groups})\n"
        f"    x = torch.relu(x + residual)\n"
        f"    return x\n"
    ), True


def _tpl_d2_reduction_transpose_fusion(batch, seq, hidden, dtype):
    """
    amax -> scale -> transpose.contiguous fusion.
    Inspired by PyTorch #146416 (Inductor uses ReinterpretView instead of
    a real transpose, 99.5% element mismatch).
    """
    m = max(8, seq)
    n = max(8, hidden)
    return (
        f"_x = torch.randn({m}, {n}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    scale = x.abs().amax(dim=-1, keepdim=True)\n"
        f"    x = x / (scale + 1e-8)\n"
        f"    x = x.t().contiguous()\n"
        f"    return torch.matmul(x, x.t())\n"
    ), True


def _tpl_d2_addmm_pattern(batch, seq, hidden, dtype):
    """
    torch.addmm(bias, x, w): directly triggers Inductor's addmm_pattern
    optimization pass (registered in pattern_matcher.py).
    """
    out_dim = max(8, hidden * 2)
    return (
        f"_x   = torch.randn({batch * seq}, {hidden}, dtype={dtype})\n"
        f"_w   = torch.randn({hidden}, {out_dim}, dtype={dtype})\n"
        f"_b   = torch.randn({out_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    y = torch.addmm(b, x, w)\n"
        f"    y = torch.relu(y)\n"
        f"    y2 = torch.addmm(b[:y.shape[-1]], y, w[:y.shape[-1], :y.shape[-1]])\n"
        f"    return y2.mean(dim=0)\n"
    ), True


def _tpl_d2_einsum_multi_contract(batch, seq, hidden, dtype):
    """
    torch.einsum with multi-dimensional contraction.
    Inspired by PyTorch #162722 (einsum Inductor lowering issues in
    Transformer subgraphs).
    """
    n_heads = random.choice([1, 2, 4])
    head_dim = max(8, hidden // n_heads)
    return (
        f"_q = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_v = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    scores = torch.einsum('bhsd,bhtd->bhst', q, k)\n"
        f"    attn = torch.nn.functional.softmax(scores, dim=-1)\n"
        f"    out = torch.einsum('bhst,bhtd->bhsd', attn, v)\n"
        f"    return out.mean(dim=1)\n"
    ), True


def _tpl_d2_bmm_epilogue_chain(batch, seq, hidden, dtype):
    """
    torch.bmm + multi-step epilogue: softmax -> dropout_mask -> bmm.
    bmm uses a different Triton lowering path than matmul (extern kernel
    vs triton template). Epilogue chains > 2 steps are more likely to
    trigger fusion boundary errors.
    """
    n_heads = random.choice([2, 4, 8])
    head_dim = max(8, hidden // n_heads)
    scale = float(head_dim) ** -0.5
    return (
        f"_q = torch.randn({batch * n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({batch * n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_v = torch.randn({batch * n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    scores = torch.bmm(q, k.transpose(1, 2)) * {scale:.8f}\n"
        f"    attn = torch.nn.functional.softmax(scores, dim=-1)\n"
        f"    attn = attn * (attn > 0.01).float()  # pseudo-dropout mask\n"
        f"    out = torch.bmm(attn, v)\n"
        f"    return out.sum(dim=1)\n"
    ), True


def _tpl_d2_conv_transpose_epilogue(batch, seq, hidden, dtype):
    """
    ConvTranspose2d + bias + activation epilogue.
    Transposed convolution uses a different Triton kernel generation path
    than conv2d, making epilogue fusion error-prone.
    """
    c = max(4, hidden // 8)
    hw = max(4, seq // 4)
    return (
        f"_x = torch.randn({batch}, {c}, {hw}, {hw}, dtype={dtype})\n"
        f"_w = torch.randn({c}, {c}, 3, 3, dtype={dtype})\n"
        f"_b = torch.randn({c}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    x = torch.nn.functional.conv_transpose2d(x, w, bias=b, padding=1)\n"
        f"    x = torch.nn.functional.silu(x)\n"
        f"    return x.mean(dim=(2, 3))\n"
    ), True


def _tpl_d2_baddbmm_pattern(batch, seq, hidden, dtype):
    """
    torch.baddbmm: batched addmm with dedicated Inductor pattern_matcher
    optimization. Non-1.0 beta/alpha scaling epilogues are additional
    error-prone points.
    """
    n_heads = random.choice([2, 4])
    head_dim = max(8, hidden // n_heads)
    beta = random.choice([0.0, 0.5, 1.0])
    alpha = random.choice([0.5, 1.0, 2.0])
    return (
        f"_bias = torch.randn({batch * n_heads}, {seq}, {seq}, dtype={dtype})\n"
        f"_q = torch.randn({batch * n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({batch * n_heads}, {head_dim}, {seq}, dtype={dtype})\n"
        f"_program_inputs = (_bias, _q, _k)\n\n"
        f"def program(bias, q, k):\n"
        f"    out = torch.baddbmm(bias, q, k, beta={beta}, alpha={alpha})\n"
        f"    return torch.nn.functional.softmax(out, dim=-1).mean(dim=0)\n"
    ), True


def _tpl_d2_linear_residual_norm(batch, seq, hidden, dtype):
    """
    Linear -> residual add -> layernorm -> Linear: classic Transformer FFN
    post-processing. Residual + norm fusion is a core Inductor optimization,
    but add + layernorm epilogue can have subtle differences across tile
    partition strategies.
    """
    ffn_dim = hidden * 4
    return (
        f"_x    = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w_up = torch.randn({hidden}, {ffn_dim}, dtype={dtype})\n"
        f"_w_dn = torch.randn({ffn_dim}, {hidden}, dtype={dtype})\n"
        f"_w_o  = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w_up, _w_dn, _w_o)\n\n"
        f"def program(x, w_up, w_dn, w_o):\n"
        f"    residual = x\n"
        f"    h = torch.nn.functional.silu(torch.matmul(x, w_up))\n"
        f"    h = torch.matmul(h, w_dn)\n"
        f"    x = residual + h\n"
        f"    mean = x.mean(dim=-1, keepdim=True)\n"
        f"    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)\n"
        f"    x = (x - mean) / torch.sqrt(var + 1e-5)\n"
        f"    return torch.matmul(x, w_o)\n"
    ), True


def _tpl_d2_depthwise_conv_epilogue(batch, seq, hidden, dtype):
    """
    Depthwise conv2d (groups=channels) + pointwise conv (1x1).
    MobileNet-style depthwise separable conv; Inductor handles groups>1
    differently from standard convolutions.
    """
    c = max(8, hidden // 4)
    hw = max(8, seq // 2)
    return (
        f"_x  = torch.randn({batch}, {c}, {hw}, {hw}, dtype={dtype})\n"
        f"_dw = torch.randn({c}, 1, 3, 3, dtype={dtype})\n"
        f"_pw = torch.randn({c}, {c}, 1, 1, dtype={dtype})\n"
        f"_program_inputs = (_x, _dw, _pw)\n\n"
        f"def program(x, dw, pw):\n"
        f"    x = torch.nn.functional.conv2d(x, dw, padding=1, groups={c})\n"
        f"    x = torch.nn.functional.silu(x)\n"
        f"    x = torch.nn.functional.conv2d(x, pw)\n"
        f"    return x.mean(dim=(2, 3))\n"
    ), True


# =============================================================================
# D3: Layout Propagation Templates
# Layout decisions (whether to insert contiguous) can propagate incorrectly
# across op boundaries, causing stride and memory format errors.
# =============================================================================

def _tpl_d3_permute_matmul(batch, seq, hidden, dtype):
    """
    permute -> matmul: Inductor's layout propagation can mis-infer strides
    after multiple ReinterpretViews. Graph-level variant of Issue #159462.
    """
    return (
        f"_x  = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w  = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x.permute(1, 0, 2)          # [seq, batch, hidden]\n"
        f"    x = torch.relu(x)\n"
        f"    x = x.permute(1, 0, 2)          # back to [batch, seq, hidden]\n"
        f"    return torch.matmul(x, w)\n"
    ), True


def _tpl_d3_reshape_chain(batch, seq, hidden, dtype):
    """
    reshape chain -> GEMM: after multiple reshapes Inductor's memory layout
    assumptions may be incorrect.
    """
    return (
        f"_x  = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w  = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x.reshape({batch * seq}, {hidden})\n"
        f"    x = torch.relu(x)\n"
        f"    x = x.reshape({batch}, {seq}, {hidden})\n"
        f"    x = x.transpose(1, 2)            # non-contiguous\n"
        f"    x = x.reshape({batch}, {hidden}, {seq})\n"
        f"    return torch.matmul(x, w).mean(dim=-1)\n"
    ), True


def _tpl_d3_noncontiguous_reduce(batch, seq, hidden, dtype):
    """
    Non-contiguous tensor -> reduction: Inductor has bugs in tile boundary
    handling when reducing over non-contiguous strided tensors.
    """
    sub = max(2, seq // 2)
    step = random.choice([2, 3])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x[:, :{sub}, :]           # slice → non-contiguous\n"
        f"    x = x[..., ::{step}]           # stride slice\n"
        f"    m = x.mean(dim=1, keepdim=True)\n"
        f"    x = x - m\n"
        f"    return torch.matmul(x, w[:x.shape[-1], :])\n"
    ), True


def _tpl_d3_channels_last_chain(batch, seq, hidden, dtype):
    """
    channels_last -> conv -> pointwise.
    Inductor's layout tag may not propagate correctly to epilogue ops after
    channels_last operations (Issue #159134).
    """
    c = max(4, hidden // 8)
    hw = max(4, seq // 4)
    return (
        f"_x  = torch.randn({batch}, {c}, {hw}, {hw}, dtype={dtype})\n"
        f"_x  = _x.to(memory_format=torch.channels_last)\n"
        f"_w1 = torch.randn({c}, {c}, 3, 3, dtype={dtype})\n"
        f"_w2 = torch.randn({c * hw * hw}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w1, _w2)\n\n"
        f"def program(x, w1, w2):\n"
        f"    x = torch.nn.functional.conv2d(x, w1, padding=1)\n"
        f"    x = torch.relu(x).contiguous()\n"
        f"    x = x.view({batch}, -1)\n"
        f"    return torch.matmul(x, w2)\n"
    ), True


def _tpl_d3_pad_reshape_chunk(batch, seq, hidden, dtype):
    """
    ReflectionPad -> view -> chunk -> distance.
    Inspired by PyTorch #151198 (memory stride after pad mismatches view
    assumptions; layout inference after chunk is incorrect).
    Affects both Triton and C++ backends.
    """
    c = max(2, batch)
    h = max(4, seq // 4)
    w = max(4, hidden // 8)
    return (
        f"_x = torch.randn({c}, 3, {h}, {h}, {w}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.pad(x, [1,1,1,1,1,1], mode='reflect')\n"
        f"    x = x.view(x.size(0), -1)\n"
        f"    a, b = torch.chunk(x, 2, dim=1)\n"
        f"    return torch.sqrt(((a - b) ** 2).sum(dim=1) + 1e-8)\n"
    ), True


def _tpl_d3_alias_cat_expand(batch, seq, hidden, dtype):
    """
    cat(x, x) same-buffer concatenation then expand + compute.
    Inspired by PyTorch #93357: Inductor codegen uses __restrict__ on cat
    output, but when both inputs are the same tensor the pointers alias.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.cat([x, x], dim=1)\n"
        f"    y = y.unsqueeze(0).expand(2, -1, -1, -1)\n"
        f"    return y.sum(dim=0) + x\n"
    ), True


def _tpl_d3_strided_overlap_reduce(batch, seq, hidden, dtype):
    """
    unfold creates overlapping views then performs reduction.
    Inductor's tile parallelization assumes non-overlapping tiles;
    unfold violates this assumption.
    """
    window = random.choice([4, 8, 16])
    step = random.choice([s for s in [1, 2, 3, 4] if s < window])
    flat = max(window * 4, seq * hidden)
    return (
        f"_x = torch.randn({flat}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    windows = x.unfold(0, {window}, {step})\n"
        f"    return windows.sum(dim=-1)\n"
    ), True


def _tpl_d3_inplace_view_mutation(batch, seq, hidden, dtype):
    """
    Clone then mutate a strided view, testing the functionalization pass's
    alias tracking. Inductor must correctly track the alias relationship
    between views and the original tensor.
    """
    seq_half = max(1, seq // 2)
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_y = torch.randn({batch}, {seq_half}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    x = x.clone()\n"
        f"    x[:, ::2, :] = x[:, ::2, :] + y[:, :{seq_half}, :]\n"
        f"    return x.mean(dim=1)\n"
    ), True


def _tpl_d3_contiguous_elision(batch, seq, hidden, dtype):
    """
    transpose -> contiguous -> matmul.
    Inspired by PyTorch #132102: Inductor may treat .contiguous() as a
    no-op and optimize it away incorrectly.
    """
    return (
        f"_x = torch.randn({batch}, {hidden}, {seq}, dtype={dtype})\n"
        f"_w = torch.randn({seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x.permute(0, 2, 1)           # [{batch}, {seq}, {hidden}] non-contiguous\n"
        f"    x = x.contiguous()               # must materialize\n"
        f"    x = torch.relu(x)\n"
        f"    return torch.matmul(x, w)\n"
    ), True


def _tpl_d3_triu_tril_stride(batch, seq, hidden, dtype):
    """
    triu/tril produces triangular mask then combines with matmul.
    Triangular matrix ops may produce non-standard strides or mask patterns;
    subsequent GEMM codegen assumes standard layout.
    """
    size = max(8, min(seq, hidden))
    return (
        f"_x = torch.randn({size}, {size}, dtype={dtype})\n"
        f"_w = torch.randn({size}, {size}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    mask = torch.triu(torch.ones_like(x), diagonal=1)\n"
        f"    x = x * mask\n"
        f"    x = torch.relu(x)\n"
        f"    return torch.matmul(x, w).sum()\n"
    ), True


def _tpl_d3_expand_stride_zero(batch, seq, hidden, dtype):
    """
    expand produces stride=0 dimensions then performs reduction.
    Inductor's tile parallelization on stride=0 dimensions may read
    data incorrectly.
    """
    return (
        f"_x = torch.randn(1, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x.expand({batch}, -1, -1)    # stride=0 on dim 0\n"
        f"    x = torch.relu(x)\n"
        f"    x = x.sum(dim=0)                 # reduce over stride-0 dim\n"
        f"    return torch.matmul(x, w)\n"
    ), True


# =============================================================================
# D4: Cross-op Data Dependency Templates
# op1 output flows into op2 with semantic constraints; the optimizer may
# produce incorrect results during propagation across these dependencies.
# =============================================================================

def _tpl_d4_rope_pattern(batch, seq, hidden, dtype):
    """
    RoPE: chunk + neg + cat forms a data dependency chain.
    Issue #159239 (Qwen3 Inductor inconsistency): the chunk->neg->cat
    combination in rotate_half produces incorrect results after Inductor
    fusion. Confirmed in PyTorch production.
    """
    head_dim = max(16, hidden // 4)
    n_heads  = max(1, hidden // head_dim)
    scale    = float(head_dim) ** -0.5
    return (
        f"_q   = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k   = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_cos = torch.randn({seq}, {head_dim}, dtype={dtype})\n"
        f"_sin = torch.randn({seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_q, _k, _cos, _sin)\n\n"
        f"def rotate_half(x):\n"
        f"    x1, x2 = x.chunk(2, dim=-1)    # split → dependency\n"
        f"    return torch.cat((-x2, x1), dim=-1)\n\n"
        f"def program(q, k, cos, sin):\n"
        f"    cos = cos.unsqueeze(0).unsqueeze(0)\n"
        f"    sin = sin.unsqueeze(0).unsqueeze(0)\n"
        f"    q = q * cos + rotate_half(q) * sin\n"
        f"    k = k * cos + rotate_half(k) * sin\n"
        f"    return torch.nn.functional.softmax(\n"
        f"        torch.matmul(q, k.transpose(-2, -1)) * {scale:.8f}, dim=-1)\n"
    ), True


def _tpl_d4_reduce_as_scale(batch, seq, hidden, dtype):
    """
    Reduction result used as a scale factor.
    op1 (reduction) output is the coefficient for op2 (pointwise scale),
    then flows into op3 (another reduction). Tests deep dependency chains.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    # reduce -> scale -> reduce dependency chain\n"
        f"    norm = x.norm(dim=-1, keepdim=True)      # [batch, seq, 1]\n"
        f"    x_scaled = x / (norm + 1e-8)             # dependency: norm → scale\n"
        f"    gate = torch.sigmoid(x_scaled.sum(dim=1)) # dependency: scaled → gate\n"
        f"    h = torch.matmul(x_scaled, w)             # dependency: scaled → gemm\n"
        f"    return h * gate.unsqueeze(1)              # dependency: gate → mul\n"
    ), True


def _tpl_d4_gather_scatter(batch, seq, hidden, dtype):
    """
    gather -> compute -> scatter dependency.
    Issue #134535 (gather prologue fusion): gather's loop-invariant load
    has bugs in Triton codegen fusion.
    """
    vocab = max(16, hidden)
    return (
        f"_src = torch.randn({batch}, {vocab}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {vocab}, ({batch}, {seq}))\n"
        f"_w   = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_src, _idx, _w)\n\n"
        f"def program(src, idx, w):\n"
        f"    idx_exp = idx.unsqueeze(-1).expand({batch}, {seq}, {hidden})\n"
        f"    x = torch.gather(src, 1, idx_exp)    # gather creates dependency\n"
        f"    x = torch.relu(x)                    # pointwise after gather\n"
        f"    out = torch.matmul(x, w)             # GEMM with gather output\n"
        f"    # scatter back: dependency chain gather → compute → scatter\n"
        f"    result = torch.zeros_like(src)\n"
        f"    result.scatter_(1, idx_exp, out)\n"
        f"    return result.mean(dim=1)\n"
    ), True


def _tpl_d4_cumulative_dependency(batch, seq, hidden, dtype):
    """
    Cumulative loop dependency: each step's output is the next step's input.
    Deep dependency chain where Inductor's buffer reuse is error-prone.
    Layer count randomly chosen from 3-5 to increase dependency depth.
    """
    n_layers = random.choice([3, 4, 5])
    layer_code = ""
    for i in range(n_layers):
        layer_code += (f"    x = torch.relu(torch.matmul(x, w) + b)\n"
                       if i < n_layers - 1
                       else f"    x = torch.matmul(x, w)\n")
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_b = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"{layer_code}"
        f"    return x\n"
    ), True


def _tpl_d4_scatter_reduce_dup(batch, seq, hidden, dtype):
    """
    scatter_add_ with duplicate indices.
    Inductor may parallelize scatter without atomics; duplicate indices
    cause logic errors. Deterministic scatter logic errors (not races)
    can still be detected by differential oracle.
    """
    small_vocab = max(4, seq // 4)
    return (
        f"_src = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {small_vocab}, ({batch}, {seq}))\n"
        f"_program_inputs = (_src, _idx)\n\n"
        f"def program(src, idx):\n"
        f"    out = torch.zeros({batch}, {small_vocab}, {hidden}, dtype=src.dtype, device=src.device)\n"
        f"    idx_exp = idx.unsqueeze(-1).expand_as(src)\n"
        f"    out.scatter_add_(1, idx_exp, src)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d4_where_branch_matmul(batch, seq, hidden, dtype):
    """
    torch.where conditional branch followed by matmul.
    Tests Inductor's scalar promotion and codegen for torch.where.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    mask = x > 0\n"
        f"    x = torch.where(mask, x, torch.zeros_like(x))\n"
        f"    x = torch.matmul(x, w)\n"
        f"    x = torch.where(x.abs() > 1.0, x * 0.1, x)\n"
        f"    return x\n"
    ), True


def _tpl_d4_index_put_accumulate(batch, seq, hidden, dtype):
    """
    index_put_ with accumulate=True and duplicate indices.
    accumulate=True requires atomic add; Inductor may generate a
    non-atomic version.
    """
    small_vocab = max(8, hidden // 2)
    num_updates = max(16, seq * 2)
    return (
        f"_x   = torch.zeros({small_vocab}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {small_vocab}, ({num_updates},))\n"
        f"_vals = torch.randn({num_updates}, {hidden}, dtype={dtype})\n"
        f"_w   = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _idx, _vals, _w)\n\n"
        f"def program(x, idx, vals, w):\n"
        f"    x = x.clone()\n"
        f"    x.index_put_((idx,), vals, accumulate=True)\n"
        f"    return torch.matmul(x, w).sum(dim=0)\n"
    ), True


def _tpl_d4_cumsum_pointwise(batch, seq, hidden, dtype):
    """
    cumsum + pointwise interleaving.
    cumsum is a sequential scan; Inductor's parallelization strategy may
    break data dependencies.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = torch.cumsum(x, dim=1)\n"
        f"    x = torch.relu(x)\n"
        f"    x = x * torch.sigmoid(x)\n"
        f"    return torch.matmul(x, w).mean(dim=1)\n"
    ), True


def _tpl_d4_pointwise_then_cumsum(batch, seq, hidden, dtype):
    """
    Pointwise → cumsum fusion pattern (bug_011 gap fix).
    Bug_011 (#180221) needs pointwise ops BEFORE cumsum so that Inductor
    fuses them into a SplitScan kernel. The existing d4_cumsum_pointwise
    has cumsum first. This template reverses the order.
    Uses adversarial seq dimensions (129, 257, 513) to hit SplitScan threshold.
    """
    scan_dim = random.choice([129, 130, 257, 513, 1025])
    return (
        f"_x = torch.randn({batch}, {scan_dim}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, dtype={dtype})\n"
        f"_b = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    x = torch.sigmoid(x * w + b)\n"
        f"    x = torch.cumsum(x, dim=1)\n"
        f"    return x.mean(dim=1)\n"
    ), True


def _tpl_d4_cumprod_fused(batch, seq, hidden, dtype):
    """
    Pointwise → cumprod fusion. Extends bug_011 pattern to cumprod
    (also triggers SplitScan in Inductor).
    """
    scan_dim = random.choice([129, 130, 257, 513])
    return (
        f"_x = torch.randn({batch}, {scan_dim}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.sigmoid(x)\n"
        f"    x = torch.cumprod(x, dim=1)\n"
        f"    return x.sum(dim=-1)\n"
    ), True


def _tpl_d4_chunk_neg_cat(batch, seq, hidden, dtype):
    """
    chunk -> neg -> cat (core pattern of RoPE rotate_half, isolated).
    Inspired by PyTorch #159239: chunk->neg->cat produces incorrect results
    after Inductor fusion. More general variant extracted from d4_rope_pattern.
    """
    head_dim = max(16, hidden // 2)
    return (
        f"_x = torch.randn({batch}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_cos = torch.randn({seq}, {head_dim}, dtype={dtype})\n"
        f"_sin = torch.randn({seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x, _cos, _sin)\n\n"
        f"def program(x, cos, sin):\n"
        f"    x1, x2 = x.chunk(2, dim=-1)\n"
        f"    rotated = torch.cat((-x2, x1), dim=-1)\n"
        f"    cos = cos.unsqueeze(0)\n"
        f"    sin = sin.unsqueeze(0)\n"
        f"    return x * cos + rotated * sin\n"
    ), True


def _tpl_d4_transformer_subgraph(batch, seq, hidden, dtype):
    """
    Full Transformer subgraph: attention + layernorm + FFN.
    Inspired by PyTorch #162722 (multiple optimization passes interact to
    produce severe numerical inconsistency, 99.9% element mismatch).
    """
    n_heads = random.choice([1, 2, 4])
    head_dim = max(8, hidden // n_heads)
    ffn_dim = hidden * 2
    scale = float(head_dim) ** -0.5
    return (
        f"_x   = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_wq  = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_wk  = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_wv  = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_wo  = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_w1  = torch.randn({hidden}, {ffn_dim}, dtype={dtype})\n"
        f"_w2  = torch.randn({ffn_dim}, {hidden}, dtype={dtype})\n"
        f"_ln_w = torch.ones({hidden}, dtype={dtype})\n"
        f"_ln_b = torch.zeros({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _wq, _wk, _wv, _wo, _w1, _w2, _ln_w, _ln_b)\n\n"
        f"def program(x, wq, wk, wv, wo, w1, w2, ln_w, ln_b):\n"
        f"    q = torch.matmul(x, wq)\n"
        f"    k = torch.matmul(x, wk)\n"
        f"    v = torch.matmul(x, wv)\n"
        f"    scores = torch.matmul(q, k.transpose(-2, -1)) * {scale:.8f}\n"
        f"    attn = torch.nn.functional.softmax(scores, dim=-1)\n"
        f"    attn_out = torch.matmul(attn, v)\n"
        f"    attn_out = torch.matmul(attn_out, wo)\n"
        f"    x = x + attn_out\n"
        f"    mean = x.mean(dim=-1, keepdim=True)\n"
        f"    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)\n"
        f"    x = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b\n"
        f"    h = torch.relu(torch.matmul(x, w1))\n"
        f"    return torch.matmul(h, w2)\n"
    ), True


def _tpl_d4_topk_mask_matmul(batch, seq, hidden, dtype):
    """
    topk -> mask -> matmul dependency chain.
    topk returns (values, indices); indices are used to construct a mask.
    Inductor's topk lowering + subsequent scatter using the indices
    dependency is error-prone.
    """
    k = random.choice([4, 8, 16])
    k = min(k, hidden)
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    vals, idx = torch.topk(x, k={k}, dim=-1)\n"
        f"    mask = torch.zeros_like(x)\n"
        f"    mask.scatter_(-1, idx, 1.0)\n"
        f"    x = x * mask\n"
        f"    return torch.matmul(x, w).mean(dim=1)\n"
    ), True


def _tpl_d4_sort_dependent_ops(batch, seq, hidden, dtype):
    """
    sort -> diff -> cumsum dependency: sort then compute differences and
    cumulative sum. sort's Inductor lowering uses an extern kernel, but
    subsequent pointwise ops may be incorrectly fused.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    sorted_x, _ = torch.sort(x, dim=1)\n"
        f"    diff = sorted_x[:, 1:, :] - sorted_x[:, :-1, :]\n"
        f"    cs = torch.cumsum(diff.abs(), dim=1)\n"
        f"    return cs.mean(dim=(1, 2))\n"
    ), True


def _tpl_d4_masked_fill_softmax(batch, seq, hidden, dtype):
    """
    masked_fill -> softmax: classic causal attention mask pattern.
    Inductor has a dedicated pattern for masked_fill(-inf) + softmax fusion,
    but it is error-prone when mask shape does not fully align with input.
    """
    n_heads = random.choice([1, 2, 4])
    head_dim = max(8, hidden // n_heads)
    scale = float(head_dim) ** -0.5
    return (
        f"_q = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_v = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    scores = torch.matmul(q, k.transpose(-2, -1)) * {scale:.8f}\n"
        f"    mask = torch.triu(torch.ones({seq}, {seq}, device=scores.device, dtype=torch.bool), diagonal=1)\n"
        f"    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))\n"
        f"    attn = torch.nn.functional.softmax(scores, dim=-1)\n"
        f"    return torch.matmul(attn, v)\n"
    ), True


def _tpl_d4_residual_gating(batch, seq, hidden, dtype):
    """
    gate = sigmoid(linear(x)) -> x * gate + (1-gate) * linear(x).
    GLU (Gated Linear Unit) variant: two branches share input; gate is
    a data dependency. Inductor's multi-consumer optimization of sigmoid
    output is error-prone.
    """
    return (
        f"_x   = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w_g = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_w_v = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w_g, _w_v)\n\n"
        f"def program(x, w_g, w_v):\n"
        f"    gate = torch.sigmoid(torch.matmul(x, w_g))\n"
        f"    val = torch.nn.functional.silu(torch.matmul(x, w_v))\n"
        f"    return gate * val + (1.0 - gate) * x\n"
    ), True


def _tpl_d4_swiglu(batch, seq, hidden, dtype):
    """
    SwiGLU: x1 * silu(x2) where [x1, x2] = linear(x).chunk(2).
    Core component in LLaMA/Mistral; chunk + different activation
    dependencies are a hot path in Inductor.
    """
    ffn_dim = hidden * 2
    return (
        f"_x  = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w1 = torch.randn({hidden}, {ffn_dim * 2}, dtype={dtype})\n"
        f"_w2 = torch.randn({ffn_dim}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w1, _w2)\n\n"
        f"def program(x, w1, w2):\n"
        f"    h = torch.matmul(x, w1)\n"
        f"    gate, val = h.chunk(2, dim=-1)\n"
        f"    h = torch.nn.functional.silu(gate) * val\n"
        f"    return torch.matmul(h, w2)\n"
    ), True


def _tpl_d4_multi_head_split_merge(batch, seq, hidden, dtype):
    """
    reshape -> per-head compute -> reshape back: multi-head split/merge
    dependency chain. reshape does not change data but changes stride;
    layout inference after per-head parallel ops and merge is error-prone.
    """
    n_heads = random.choice([2, 4])
    head_dim = max(8, hidden // n_heads)
    actual_hidden = n_heads * head_dim
    return (
        f"_x = torch.randn({batch}, {seq}, {actual_hidden}, dtype={dtype})\n"
        f"_w = torch.randn({actual_hidden}, {actual_hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    b, s, d = x.shape\n"
        f"    x = x.view(b, s, {n_heads}, {head_dim}).transpose(1, 2)\n"
        f"    x = x * torch.sigmoid(x.mean(dim=-1, keepdim=True))\n"
        f"    x = x.transpose(1, 2).contiguous().view(b, s, d)\n"
        f"    return torch.matmul(x, w)\n"
    ), True


# =============================================================================
# D5: Shape / Dtype Boundary Conditions
# Triton tile masking, dtype cast, and extreme dimensions create boundary
# conditions that stress codegen edge cases.
# =============================================================================

def _tpl_d5_unaligned_matmul(batch, seq, hidden, dtype):
    """
    Non-power-of-2 dimensions (7, 13, 33, 65) for matmul.
    Triton tile sizes are typically multiples of 16/32/64; unaligned
    dimensions require masking logic that is prone to off-by-one errors.
    """
    ua_seq = random.choice([7, 13, 33, 65, 127])
    ua_hidden = random.choice([7, 13, 33, 65, 127])
    ua_out = random.choice([7, 13, 33, 65, 127])
    return (
        f"_x = torch.randn({batch}, {ua_seq}, {ua_hidden}, dtype={dtype})\n"
        f"_w = torch.randn({ua_hidden}, {ua_out}, dtype={dtype})\n"
        f"_b = torch.randn({ua_out}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    y = torch.matmul(x, w) + b\n"
        f"    return torch.relu(y).mean(dim=1)\n"
    ), True


def _tpl_d5_dtype_cast_chain(batch, seq, hidden, dtype):
    """
    Multiple dtype casts mid-graph (f32->f64->f32).
    Inductor C++ codegen's type promotion may have bugs in cast ordering.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype=torch.float32)\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype=torch.float32)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x.to(torch.float64)\n"
        f"    x = torch.matmul(x, w.to(torch.float64))\n"
        f"    x = torch.relu(x)\n"
        f"    x = x.to(torch.float32)\n"
        f"    return x.mean(dim=-1)\n"
    ), True


def _tpl_d5_scalar_tensor_interaction(batch, seq, hidden, dtype):
    """
    0-dim tensor (scalar) interaction with batched tensors.
    Inductor's Triton codegen handles 0-dim tensors differently from >0-dim;
    scalar broadcast + reduce combinations easily trigger tile masking
    boundary issues.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    scale = x.norm()  # scalar (0-dim)\n"
        f"    x = x / (scale + 1e-8)\n"
        f"    y = torch.matmul(x, w)\n"
        f"    offset = y.mean()  # another scalar\n"
        f"    return y - offset\n"
    ), True


def _tpl_d5_extreme_batch1(batch, seq, hidden, dtype):
    """
    batch=1 + large seq/hidden: extreme shape.
    With batch=1, some Triton kernels use a different tile strategy
    (no batch loop).
    """
    large_seq = random.choice([255, 511, 1023])
    large_hidden = random.choice([127, 255, 511])
    return (
        f"_x = torch.randn(1, {large_seq}, {large_hidden}, dtype={dtype})\n"
        f"_w = torch.randn({large_hidden}, {large_hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = torch.nn.functional.layer_norm(x, [{large_hidden}])\n"
        f"    x = torch.matmul(x, w)\n"
        f"    return torch.nn.functional.softmax(x, dim=-1).sum(dim=1)\n"
    ), True


def _tpl_d5_empty_dim_ops(batch, seq, hidden, dtype):
    """
    Operations with size=0 or size=1 dimensions.
    Inductor handles empty dimensions (numel=0) and singleton dimensions
    differently at boundary cases.
    """
    return (
        f"_x = torch.randn({batch}, 1, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x.expand({batch}, {seq}, {hidden})\n"
        f"    x = torch.nn.functional.layer_norm(x, [{hidden}])\n"
        f"    x = torch.matmul(x, w)\n"
        f"    return torch.relu(x).mean(dim=1)\n"
    ), True


def _tpl_d5_mixed_precision_matmul(batch, seq, hidden, dtype):
    """
    Mixed-precision matmul: input and weight have different dtypes.
    Inductor's implicit type promotion cast before matmul may cause
    inconsistent tile strategies across precision combinations in
    Triton codegen.
    """
    other_dtype = "torch.float64" if dtype == "torch.float32" else "torch.float32"
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={other_dtype})\n"
        f"_b = torch.randn({hidden}, dtype={other_dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    y = torch.matmul(x.to(w.dtype), w) + b\n"
        f"    y = torch.relu(y)\n"
        f"    return y.to(x.dtype).mean(dim=1)\n"
    ), True


def _tpl_d5_reduction_keepdim_false(batch, seq, hidden, dtype):
    """
    keepdim=False reduction followed by ops that require aligned dimensions.
    When keepdim=False, output rank decreases; subsequent broadcast stride
    computation is a common error point.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    r = x.sum(dim=1)  # [{batch}, {hidden}], keepdim=False\n"
        f"    r = r * w  # broadcast with 1-dim weight\n"
        f"    x_scaled = x * r.unsqueeze(1)  # re-expand for broadcast\n"
        f"    return x_scaled.mean(dim=(1, 2))\n"
    ), True


# =============================================================================
# Additional templates targeting known open Inductor bugs
# =============================================================================

def _tpl_d5_logit_eps(batch, seq, hidden, dtype):
    """
    torch.logit with eps > input → Inductor decomposition flips sign.
    Known bug: pytorch/pytorch#145379 (open on 2.6.0).
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype}).sigmoid()\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.logit(x, eps=0.9)\n"
        f"    return x.mean(dim=-1)\n"
    ), True


def _tpl_d5_logit_chain(batch, seq, hidden, dtype):
    """
    logit with varying eps in a computation chain.
    Probes Inductor's decomposition of logit clamp logic.
    """
    eps = random.choice([0.3, 0.5, 0.7, 0.9, 0.99])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype}).sigmoid()\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.logit(x, eps={eps})\n"
        f"    x = torch.relu(x)\n"
        f"    return x.sum(dim=1)\n"
    ), True


def _tpl_d5_dtype_cast_half_round(batch, seq, hidden, dtype):
    """
    fp32 → fp16 → fp32 round-trip inside a fusion chain.
    Probes fusion boundary dtype handling — compiler must decide
    whether to cast inside or outside the fused kernel.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype=torch.float32)\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype=torch.float32)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = torch.matmul(x, w)\n"
        f"    x = x.half().float()\n"
        f"    x = torch.relu(x)\n"
        f"    return x.mean(dim=-1)\n"
    ), True


def _tpl_d5_dtype_cast_bf16_round(batch, seq, hidden, dtype):
    """
    fp32 → bf16 → fp32 round-trip. BFloat16 has different mantissa bits
    than fp16, testing a different truncation path.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype=torch.float32)\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype=torch.float32)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = torch.matmul(x, w)\n"
        f"    x = x.to(torch.bfloat16).float()\n"
        f"    x = x * torch.sigmoid(x)\n"
        f"    return x.sum(dim=1)\n"
    ), True


def _tpl_d1_broadcast_reduce_cross(batch, seq, hidden, dtype):
    """
    reduction → broadcast → reduction cross pattern.
    Forces Inductor to do complex tile and loop scheduling.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_bias = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _bias)\n\n"
        f"def program(x, bias):\n"
        f"    x_mean = x.mean(dim=-1, keepdim=True)\n"
        f"    x = x - x_mean\n"
        f"    x = x * bias\n"
        f"    x = x.sum(dim=1)\n"
        f"    return x\n"
    ), True


def _tpl_d1_reduce_chain_mixed_dim(batch, seq, hidden, dtype):
    """
    Reductions along different dims interleaved with elementwise ops.
    Stresses Triton's loop scheduling for multi-dim reductions.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x - x.mean(dim=0, keepdim=True)\n"
        f"    x = torch.relu(x)\n"
        f"    x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)\n"
        f"    return x.mean(dim=1)\n"
    ), True


def _tpl_d4_where_logit_clamp(batch, seq, hidden, dtype):
    """
    torch.where + logit + clamp chain.
    Combines conditional branching with ops known to have Inductor issues.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mask = x > 0\n"
        f"    x = torch.where(mask, x.sigmoid(), torch.zeros_like(x) + 0.5)\n"
        f"    x = torch.logit(x, eps=0.8)\n"
        f"    x = torch.clamp(x, -5.0, 5.0)\n"
        f"    return x.mean(dim=-1)\n"
    ), True


# =============================================================================
# Expanded op coverage templates (high-value uncovered ops)
# =============================================================================

def _tpl_d2_bmm_epilogue(batch, seq, hidden, dtype):
    """bmm (batched matmul) + epilogue fusion. Distinct lowering path from matmul."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_y = torch.randn({batch}, {hidden}, {seq}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    x = torch.bmm(x, y)\n"
        f"    x = torch.relu(x)\n"
        f"    return x.mean(dim=-1)\n"
    ), True


def _tpl_d2_baddbmm(batch, seq, hidden, dtype):
    """baddbmm: batched addmm with alpha/beta. Complex lowering with scaling params."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_y = torch.randn({batch}, {hidden}, {seq}, dtype={dtype})\n"
        f"_bias = torch.randn({batch}, {seq}, {seq}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y, _bias)\n\n"
        f"def program(x, y, bias):\n"
        f"    out = torch.baddbmm(bias, x, y, beta=0.5, alpha=0.3)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d4_embedding_lookup(batch, seq, hidden, dtype):
    """embedding lookup — discrete indexing, completely different from tensor math."""
    vocab = max(hidden, 64)
    return (
        f"_idx = torch.randint(0, {vocab}, ({batch}, {seq}))\n"
        f"_weight = torch.randn({vocab}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_idx, _weight)\n\n"
        f"def program(idx, weight):\n"
        f"    x = torch.nn.functional.embedding(idx, weight)\n"
        f"    return x.sum(dim=1)\n"
    ), True


def _tpl_d3_unfold_ops(batch, seq, hidden, dtype):
    """unfold — creates a view with overlapping windows, complex stride math."""
    kernel = min(4, hidden)
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.unfold(-1, {kernel}, 1)\n"
        f"    return x.sum(dim=-1).mean(dim=-1)\n"
    ), True


def _tpl_d3_roll_shift(batch, seq, hidden, dtype):
    """roll — circular shift, tests index wrapping in Triton."""
    shift = random.choice([1, -1, 3, -3, hidden // 2])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.roll(x, shifts={shift}, dims=-1)\n"
        f"    return x.relu().sum(dim=1)\n"
    ), True


def _tpl_d4_scatter_reduce_sum(batch, seq, hidden, dtype):
    """scatter_reduce — indexed accumulation, tricky parallel lowering."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {max(seq//2, 1)}, ({batch}, {seq}, {hidden}))\n"
        f"_src = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _idx, _src)\n\n"
        f"def program(x, idx, src):\n"
        f"    x = x.scatter_reduce(1, idx, src, reduce='sum')\n"
        f"    return x.mean(dim=(1, 2))\n"
    ), True


def _tpl_d4_index_select_gather(batch, seq, hidden, dtype):
    """index_select + gather — indexed reads from different dims."""
    n_idx = min(seq, 16)
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {seq}, ({n_idx},))\n"
        f"_program_inputs = (_x, _idx)\n\n"
        f"def program(x, idx):\n"
        f"    x = torch.index_select(x, 1, idx)\n"
        f"    return x.relu().mean(dim=-1)\n"
    ), True


def _tpl_d2_upsample_nearest(batch, seq, hidden, dtype):
    """upsample_nearest2d — resizing op with integer scale factor."""
    h_2d = max(int(hidden ** 0.5), 4)
    return (
        f"_x = torch.randn({batch}, 3, {h_2d}, {h_2d}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')\n"
        f"    return x.mean(dim=(2, 3))\n"
    ), True


def _tpl_d2_adaptive_avgpool(batch, seq, hidden, dtype):
    """adaptive_avg_pool2d — variable-size pooling, complex output shape calculation."""
    h_2d = max(int(hidden ** 0.5), 4)
    out_size = random.choice([1, 3, 7])
    return (
        f"_x = torch.randn({batch}, {seq}, {h_2d}, {h_2d}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.adaptive_avg_pool2d(x, {out_size})\n"
        f"    return x.flatten(1).sum(dim=-1)\n"
    ), True


def _tpl_d1_logcumsumexp(batch, seq, hidden, dtype):
    """logcumsumexp — log-domain cumulative sum, numerically tricky."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.logcumsumexp(x, dim=1)\n"
        f"    return x.mean(dim=-1)\n"
    ), True


def _tpl_d5_lerp_addcdiv(batch, seq, hidden, dtype):
    """lerp + addcdiv — compound elementwise with division, tests fusion of complex expressions."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_y = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    z = torch.lerp(x, y, 0.3)\n"
        f"    z = torch.addcdiv(z, x, y.clamp(min=1e-5), value=0.1)\n"
        f"    return z.sum(dim=1)\n"
    ), True


def _tpl_d4_searchsorted_bucketize(batch, seq, hidden, dtype):
    """searchsorted — binary search lowering, very different from typical tensor ops."""
    n_bounds = min(hidden, 32)
    return (
        f"_x = torch.randn({batch}, {seq}, dtype={dtype})\n"
        f"_bounds = torch.sort(torch.randn({n_bounds}, dtype={dtype}))[0]\n"
        f"_program_inputs = (_x, _bounds)\n\n"
        f"def program(x, bounds):\n"
        f"    idx = torch.searchsorted(bounds, x)\n"
        f"    return idx.float().mean(dim=-1)\n"
    ), True


def _tpl_d2_group_norm_conv(batch, seq, hidden, dtype):
    """group_norm — different normalization path from layernorm/batchnorm."""
    channels = max(hidden, 4)
    groups = random.choice([g for g in [1, 2, 4, 8] if channels % g == 0])
    h_2d = max(int(seq ** 0.5), 4)
    return (
        f"_x = torch.randn({batch}, {channels}, {h_2d}, {h_2d}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.group_norm(x, {groups})\n"
        f"    return x.mean(dim=(2, 3))\n"
    ), True


# =============================================================================
# Phase 2: Decomposition-targeting templates (D6)
# =============================================================================
# These target ops with complex decomposition logic in torch/_decomp/decompositions.py.
# Bug hypothesis: decomposition functions are hand-written, often have edge cases
# in dtype promotion, conditional branching, view/reshape, and clamp/where logic.

def _tpl_d6_softplus_threshold(batch, seq, hidden, dtype):
    """softplus: decomposition has dtype promotion + threshold conditional + exp/log."""
    beta = random.choice([0.5, 1.0, 2.0, 5.0])
    thresh = random.choice([10.0, 15.0, 20.0, 50.0])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.softplus(x, beta={beta}, threshold={thresh})\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_d6_elu_variants(batch, seq, hidden, dtype):
    """elu/celu: decomposition has dtype promotion + where + alpha/scale params."""
    act = random.choice(['elu', 'celu'])
    alpha = random.choice([0.5, 1.0, 1.5, 2.0])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.{act}(x, alpha={alpha})\n"
        f"    return x.mean(dim=-1)\n"
    ), True

def _tpl_d6_masked_fill_chain(batch, seq, hidden, dtype):
    """masked_fill: decomposition has dtype promotion + where. Chain with softmax."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_mask = torch.randint(0, 2, ({batch}, 1, {hidden}), dtype=torch.bool)\n"
        f"_program_inputs = (_x, _mask)\n\n"
        f"def program(x, mask):\n"
        f"    x = x.masked_fill(mask, float('-inf'))\n"
        f"    x = torch.softmax(x, dim=-1)\n"
        f"    return x.sum(dim=1)\n"
    ), True

def _tpl_d6_huber_loss(batch, seq, hidden, dtype):
    """huber_loss: decomposition has conditional where + delta threshold."""
    delta = random.choice([0.1, 0.5, 1.0, 2.0])
    reduction = random.choice(['none', 'mean', 'sum'])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_target = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _target)\n\n"
        f"def program(x, target):\n"
        f"    return torch.nn.functional.huber_loss(x, target, delta={delta}, reduction='{reduction}')\n"
    ), True

def _tpl_d6_logaddexp(batch, seq, hidden, dtype):
    """logaddexp: decomposition has dtype promotion + where + numerics-sensitive log/exp."""
    return (
        f"_a = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_b = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_a, _b)\n\n"
        f"def program(a, b):\n"
        f"    x = torch.logaddexp(a, b)\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_d6_xlogy(batch, seq, hidden, dtype):
    """xlogy: decomposition has special case x=0 via where + dtype promotion."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype}).abs()\n"
        f"_y = torch.randn({batch}, {seq}, {hidden}, dtype={dtype}).abs() + 1e-6\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    # Some zeros to trigger the x=0 special case in decomposition\n"
        f"    x = x * (x > 0.3).float()\n"
        f"    return torch.xlogy(x, y).mean(dim=-1)\n"
    ), True

def _tpl_d6_nan_to_num(batch, seq, hidden, dtype):
    """nan_to_num: decomposition has dtype promotion + where for nan/inf replacement."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    # Create NaN/Inf values through division\n"
        f"    denom = x.mean(dim=-1, keepdim=True)\n"
        f"    x = x / denom\n"
        f"    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_d6_threshold_chain(batch, seq, hidden, dtype):
    """threshold: decomposition has where + conditional. Chain with matmul."""
    thresh = random.choice([0.0, 0.1, 0.5])
    value = random.choice([-1.0, 0.0, -0.5])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = torch.nn.functional.threshold(x, {thresh}, {value})\n"
        f"    return torch.matmul(x, w).mean(dim=1)\n"
    ), True

def _tpl_d6_renorm(batch, seq, hidden, dtype):
    """renorm: decomposition has norm + clamp + where + dtype promotion."""
    p = random.choice([1.0, 2.0])
    maxnorm = random.choice([0.5, 1.0, 2.0])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.renorm(x, {p}, 0, {maxnorm})\n"
        f"    return x.sum(dim=(1, 2))\n"
    ), True

def _tpl_d6_select_scatter(batch, seq, hidden, dtype):
    """select_scatter: decomposition has view + where, short but tricky index logic."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_src = torch.randn({batch}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _src)\n\n"
        f"def program(x, src):\n"
        f"    x = x.select_scatter(src, 1, 0)\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_d6_diag_embed(batch, seq, hidden, dtype):
    """diag_embed: decomposition has complex view + index + dtype conditional."""
    diag_size = min(hidden, 16)
    return (
        f"_x = torch.randn({batch}, {seq}, {diag_size}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    d = torch.diag_embed(x)\n"
        f"    return d.sum(dim=(-1, -2))\n"
    ), True

def _tpl_d6_multi_margin_loss(batch, seq, hidden, dtype):
    """multi_margin_loss: decomposition has clamp + where + conditional branching."""
    n_classes = min(hidden, 32)
    p = random.choice([1, 2])
    margin = random.choice([0.5, 1.0, 2.0])
    return (
        f"_x = torch.randn({batch * seq}, {n_classes}, dtype={dtype})\n"
        f"_target = torch.randint(0, {n_classes}, ({batch * seq},))\n"
        f"_program_inputs = (_x, _target)\n\n"
        f"def program(x, target):\n"
        f"    return torch.nn.functional.multi_margin_loss(x, target, p={p}, margin={margin})\n"
    ), True

def _tpl_d6_channel_shuffle(batch, seq, hidden, dtype):
    """channel_shuffle: decomposition has view + permute, layout-sensitive."""
    channels = random.choice([g for g in [6, 12, 24, 48] if g <= hidden])
    groups = random.choice([2, 3, 6])
    while channels % groups != 0:
        groups = random.choice([2, 3])
    h_2d = max(int(seq ** 0.5), 4)
    return (
        f"_x = torch.randn({batch}, {channels}, {h_2d}, {h_2d}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.channel_shuffle(x, {groups})\n"
        f"    return x.mean(dim=(2, 3))\n"
    ), True

def _tpl_d6_addcmul_chain(batch, seq, hidden, dtype):
    """addcmul/addcdiv: decomposition has dtype promotion + conditional."""
    op = random.choice(['addcmul', 'addcdiv'])
    value = random.choice([0.1, 0.5, 1.0, 2.0])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    t1 = x.roll(1, -1)\n"
        f"    t2 = x.roll(-1, -1).abs() + 1e-6\n"
        f"    x = torch.{op}(x, t1, t2, value={value})\n"
        f"    return x.mean(dim=-1)\n"
    ), True

def _tpl_d6_linspace_index(batch, seq, hidden, dtype):
    """linspace: decomposition has dtype promotion + where + conditional branching."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    idx = torch.linspace(0, 1, {hidden}, device=x.device, dtype=x.dtype)\n"
        f"    x = x * idx.unsqueeze(0).unsqueeze(0)\n"
        f"    return x.sum(dim=-1)\n"
    ), True


# =============================================================================
# D7: PyTorch 2.7+ new code path templates
# =============================================================================
# Target new features introduced in 2.7: prologue fusion, online softmax,
# emulate_precision_casts, cooperative reductions, new decompositions.

def _tpl_d7_prologue_pointwise_gemm(batch, seq, hidden, dtype):
    """Prologue fusion: pointwise ops before matmul should be fused into GEMM prologue."""
    op = random.choice(['relu', 'sigmoid', 'tanh', 'silu'])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x * 2.0 + 1.0\n"
        f"    x = torch.nn.functional.{op}(x)\n"
        f"    return torch.matmul(x, w).sum(dim=1)\n"
    ), True

def _tpl_d7_prologue_norm_gemm(batch, seq, hidden, dtype):
    """Prologue fusion: layernorm/rmsnorm before matmul — new fusion candidate in 2.7."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)\n"
        f"    return torch.matmul(x, w).mean(dim=1)\n"
    ), True

def _tpl_d7_online_softmax_large(batch, seq, hidden, dtype):
    """Online softmax: 2.7 uses single-pass algorithm. Test with large dim to stress it."""
    dim = random.choice([512, 1024, 2048, 4096])
    return (
        f"_x = torch.randn({batch}, {dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.softmax(x, dim=-1)\n"
    ), True

def _tpl_d7_online_softmax_odd(batch, seq, hidden, dtype):
    """Online softmax with odd dims — previously triggered bug_005 on Triton 3.2."""
    dim = random.choice([513, 515, 517, 1023, 1025, 2047])
    return (
        f"_x = torch.randn({batch}, {dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.softmax(x, dim=-1)\n"
    ), True

def _tpl_d7_precision_cast_emulate(batch, seq, hidden, dtype):
    """Precision cast with emulate_precision_casts — 2.7 new config path."""
    cast_dtype = random.choice(['torch.bfloat16', 'torch.float16'])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.to({cast_dtype}).float()\n"
        f"    x = x.sigmoid()\n"
        f"    x = x.to({cast_dtype}).float()\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_d7_addmv_decomp(batch, seq, hidden, dtype):
    """addmv: new decomposition added in 2.7 (#143792)."""
    return (
        f"_x = torch.randn({hidden}, dtype={dtype})\n"
        f"_M = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_v = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _M, _v)\n\n"
        f"def program(x, M, v):\n"
        f"    return torch.addmv(x, M, v)\n"
    ), True

def _tpl_d7_split_cat_pattern(batch, seq, hidden, dtype):
    """split+cat pattern: 2.7 added new split_cat_aten_pass and select_cat_aten_pass."""
    n_splits = random.choice([2, 3, 4])
    split_dim = hidden - (hidden % n_splits)
    return (
        f"_x = torch.randn({batch}, {seq}, {split_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    parts = torch.split(x, {split_dim // n_splits}, dim=-1)\n"
        f"    parts = [p * (i + 1) for i, p in enumerate(parts)]\n"
        f"    x = torch.cat(parts, dim=-1)\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_d7_cooperative_reduction(batch, seq, hidden, dtype):
    """Large reduction that may trigger cooperative/split reduction paths in 2.7."""
    large_dim = random.choice([4096, 8192, 16384])
    return (
        f"_x = torch.randn({batch}, {large_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_d7_max_pool_as_reduction(batch, seq, hidden, dtype):
    """max_pool2d with large kernel: 2.7 treats this as reduction (#147876)."""
    h_2d = random.choice([14, 16, 28, 32])
    k = random.choice([7, 14])
    channels = min(hidden, 32)
    return (
        f"_x = torch.randn({batch}, {channels}, {h_2d}, {h_2d}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.nn.functional.max_pool2d(x, kernel_size={k}, stride=1, padding={k//2}).mean(dim=(2,3))\n"
    ), True

def _tpl_d7_groupnorm_parallel(batch, seq, hidden, dtype):
    """GroupNorm parallel reduction: new optimization in 2.7 (#144020)."""
    channels = random.choice([g for g in [32, 64, 128, 256] if g <= max(hidden, 32)])
    groups = random.choice([g for g in [4, 8, 16, 32] if channels % g == 0])
    h_2d = max(int(seq ** 0.5), 8)
    return (
        f"_x = torch.randn({batch}, {channels}, {h_2d}, {h_2d}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.group_norm(x, {groups})\n"
        f"    return x.relu().mean(dim=(2, 3))\n"
    ), True


# =============================================================================
# D8: Indexing ops (target ModularIndexing-related code paths)
# =============================================================================
# Rationale: bug_001 (ReflectionPad) lives in inductor's ModularIndexing path.
# Scatter / gather / index_put use the same indexing infrastructure, so they
# are the highest-likelihood neighborhood for finding similar bugs.

def _tpl_d8_scatter_along_dim(batch, seq, hidden, dtype):
    """scatter on dim 1 with UNIQUE indices per (batch, h) slice.

    PyTorch's scatter with non-unique indices on the scatter dim has
    undefined "which writer wins" behavior, so eager vs inductor can
    legitimately disagree. We force unique indices via per-slice argsort
    so any diff is a real bug, not template UB.
    """
    n_idx = min(max(seq // 2, 1), seq)
    return (
        f"_x = torch.zeros({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_src = torch.randn({batch}, {n_idx}, {hidden}, dtype={dtype})\n"
        # argsort of a uniform random tensor gives a per-(batch, h) permutation
        # of [0, seq). Slicing the first n_idx rows yields unique indices along
        # dim 1 within each (batch, h) channel.
        f"_idx = torch.argsort(torch.rand({batch}, {seq}, {hidden}), dim=1)[:, :{n_idx}, :]\n"
        f"_program_inputs = (_x, _src, _idx)\n\n"
        f"def program(x, src, idx):\n"
        f"    out = x.scatter(1, idx, src)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d8_scatter_add_overlap(batch, seq, hidden, dtype):
    """scatter_add with duplicate indices — exercises atomic accumulation paths."""
    n_idx = max(seq, 4)
    return (
        f"_x = torch.zeros({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_src = torch.randn({batch}, {n_idx}, {hidden}, dtype={dtype})\n"
        # half the indices duplicate to force scatter_add atomics
        f"_idx = torch.randint(0, max({seq} // 2, 1), ({batch}, {n_idx}, {hidden}))\n"
        f"_program_inputs = (_x, _src, _idx)\n\n"
        f"def program(x, src, idx):\n"
        f"    out = x.scatter_add(1, idx, src)\n"
        f"    return out.amax(dim=-1)\n"
    ), False  # non-deterministic on some backends due to atomic order


def _tpl_d8_index_put_accumulate_3d(batch, seq, hidden, dtype):
    """index_put with accumulate=True on a 3D tensor — duplicate indices."""
    n_idx = max(min(seq, 16), 4)
    return (
        f"_x = torch.zeros({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_v = torch.randn({n_idx}, {hidden}, dtype={dtype})\n"
        f"_b_idx = torch.randint(0, {batch}, ({n_idx},))\n"
        f"_s_idx = torch.randint(0, {seq}, ({n_idx},))\n"
        f"_program_inputs = (_x, _v, _b_idx, _s_idx)\n\n"
        f"def program(x, v, b_idx, s_idx):\n"
        f"    x = x.clone()\n"
        f"    x.index_put_((b_idx, s_idx), v, accumulate=True)\n"
        f"    return x.sum(dim=1)\n"
    ), False


def _tpl_d8_gather_neg_dim(batch, seq, hidden, dtype):
    """gather with negative dim — index logic less exercised."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {hidden}, ({batch}, {seq}, max({hidden} // 2, 1)))\n"
        f"_program_inputs = (_x, _idx)\n\n"
        f"def program(x, idx):\n"
        f"    out = torch.gather(x, -1, idx)\n"
        f"    return (out * 2.0 + 1.0).mean(dim=-1)\n"
    ), True


def _tpl_d8_masked_scatter_chain(batch, seq, hidden, dtype):
    """masked_scatter — value packing into masked positions."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_src = torch.randn({batch} * {seq} * {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _src)\n\n"
        f"def program(x, src):\n"
        f"    mask = x > 0\n"
        f"    out = x.masked_scatter(mask, src)\n"
        f"    return out.sum(dim=1)\n"
    ), True


def _tpl_d8_advanced_index_view(batch, seq, hidden, dtype):
    """Advanced indexing x[idx1, idx2] then view — combines two indexing paths."""
    n = max(min(seq, hidden, 16), 2)
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_i1 = torch.randint(0, {seq}, ({n},))\n"
        f"_i2 = torch.randint(0, {hidden}, ({n},))\n"
        f"_program_inputs = (_x, _i1, _i2)\n\n"
        f"def program(x, i1, i2):\n"
        f"    out = x[:, i1, i2]\n"
        f"    out = out.contiguous().view({batch}, -1)\n"
        f"    return (out + 1.0).mean(dim=-1)\n"
    ), True


def _tpl_d8_take_indexing(batch, seq, hidden, dtype):
    """torch.take — flat indexing into ravel buffer."""
    n_idx = max(min(seq * hidden // 4, 64), 4)
    total = batch * seq * hidden
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {total}, ({n_idx},))\n"
        f"_program_inputs = (_x, _idx)\n\n"
        f"def program(x, idx):\n"
        f"    out = torch.take(x, idx)\n"
        f"    return out.relu().sum()\n"
    ), True


def _tpl_d8_index_select_strided_reduce(batch, seq, hidden, dtype):
    """index_select on a transposed (strided) view, then reduce."""
    n_idx = max(min(hidden, 16), 2)
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {seq}, ({n_idx},))\n"
        f"_program_inputs = (_x, _idx)\n\n"
        f"def program(x, idx):\n"
        f"    x = x.transpose(1, 2)  # non-contiguous\n"
        f"    out = torch.index_select(x, 2, idx)\n"
        f"    return out.sum(dim=(1, 2))\n"
    ), True


# =============================================================================
# D9: Complex dtype (cfloat / cdouble) — undertested codegen path
# =============================================================================
# Rationale: torch.compile / Inductor support for complex64/128 has matured
# only since 2.6, and the lowerings are far less exercised than fp32/fp64.
# These templates force complex execution paths.

def _tpl_d9_complex_matmul(batch, seq, hidden, dtype):
    """complex64 matmul — minimal complex GEMM path."""
    # ignore the input dtype, force complex
    return (
        f"_xr = torch.randn({batch}, {seq}, {hidden})\n"
        f"_xi = torch.randn({batch}, {seq}, {hidden})\n"
        f"_x = torch.complex(_xr, _xi)\n"
        f"_wr = torch.randn({hidden}, {hidden})\n"
        f"_wi = torch.randn({hidden}, {hidden})\n"
        f"_w = torch.complex(_wr, _wi)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    out = torch.matmul(x, w)\n"
        f"    return out.abs().sum(dim=-1)\n"
    ), True


def _tpl_d9_rfft_pointwise_irfft(batch, seq, hidden, dtype):
    """rfft → pointwise on complex → irfft round-trip."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    X = torch.fft.rfft(x, dim=-1)\n"
        f"    X = X * torch.complex(torch.ones_like(X.real), torch.zeros_like(X.imag) + 0.1)\n"
        f"    out = torch.fft.irfft(X, n={hidden}, dim=-1)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d9_complex_pointwise_chain(batch, seq, hidden, dtype):
    """complex elementwise: mul, exp, conj — pointwise complex codegen."""
    return (
        f"_xr = torch.randn({batch}, {seq}, {hidden})\n"
        f"_xi = torch.randn({batch}, {seq}, {hidden})\n"
        f"_x = torch.complex(_xr, _xi)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x * x.conj()\n"
        f"    y = torch.exp(y * 0.01)\n"
        f"    return y.real.mean(dim=-1)\n"
    ), True


def _tpl_d9_view_as_complex_roundtrip(batch, seq, hidden, dtype):
    """view_as_complex / view_as_real round trip — alias-sensitive path."""
    even_h = (hidden // 2) * 2
    if even_h == 0:
        even_h = 2
    return (
        f"_x = torch.randn({batch}, {seq}, {even_h}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.contiguous()\n"
        f"    c = torch.view_as_complex(x.reshape({batch}, {seq}, {even_h // 2}, 2))\n"
        f"    c = c * 2.0 + 1.0\n"
        f"    r = torch.view_as_real(c)\n"
        f"    return r.reshape({batch}, {seq}, {even_h}).sum(dim=-1)\n"
    ), True


def _tpl_d9_complex_norm_reduce(batch, seq, hidden, dtype):
    """complex .abs() / .angle() into reduction — tests complex reductions."""
    return (
        f"_xr = torch.randn({batch}, {seq}, {hidden})\n"
        f"_xi = torch.randn({batch}, {seq}, {hidden})\n"
        f"_x = torch.complex(_xr, _xi)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mag = x.abs()\n"
        f"    ang = x.angle()\n"
        f"    return (mag * torch.cos(ang)).sum(dim=-1)\n"
    ), True


def _tpl_d9_stft_magnitude(batch, seq, hidden, dtype):
    """STFT magnitude — short-time fft path; uses 1D input."""
    n_fft = min(max(hidden, 16), 64)
    sig_len = max(seq * hidden, n_fft * 4)
    return (
        f"_x = torch.randn({batch}, {sig_len}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    X = torch.stft(x, n_fft={n_fft}, hop_length={n_fft // 2},\n"
        f"                   return_complex=True, center=True)\n"
        f"    return X.abs().mean(dim=(-2, -1))\n"
    ), True


# =============================================================================
# D10: PyTorch 2.8 - 2.11 new code paths (highest priority — newest code)
# =============================================================================
# Each template targets a specific change introduced after 2.7.0:
#   - 2.8: DecomposeK GEMM, min_num_split, max_fusion_buffer_group_pairwise_attempts
#   - 2.9: 3D online softmax fix, slice_scatter dtype fix, sort NaN fix,
#          persistent reduction more aggressive, fused RMS/LN backward (2.10)
#   - 2.10: mix-order reduction fusion, assume_32bit_indexing, matmul out_dtype,
#           qconv lowerings, deterministic mode
#   - 2.11: avg_pool2d rewritten as reduction, ldexp lowering, NVGEMM,
#           combo_kernels_pointwise_only, FP8 BlockWise scaling

def _tpl_d10_decompose_k_matmul(batch, seq, hidden, dtype):
    """Large-K matmul to trigger DecomposeK path (2.8)."""
    K = random.choice([2048, 4096, 8192])  # K must be large for DecomposeK
    M = max(batch * seq, 8)
    N = max(hidden, 8)
    return (
        f"_x = torch.randn({M}, {K}, dtype={dtype})\n"
        f"_w = torch.randn({K}, {N}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return torch.matmul(x, w).sum(dim=-1)\n"
    ), True


def _tpl_d10_3d_online_softmax(batch, seq, hidden, dtype):
    """3D softmax — recently fixed in 2.9 (#162341), neighborhood likely buggy."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    # 3D softmax along middle dim — exercises 3d tiled online softmax path\n"
        f"    return torch.softmax(x, dim=1).sum(dim=-1)\n"
    ), True


def _tpl_d10_avg_pool_as_reduction(batch, seq, hidden, dtype):
    """avg_pool2d rewritten as reduction in 2.11 (#167228) — NEW lowering."""
    h_2d = max(int(seq ** 0.5), 8)
    channels = min(hidden, 32)
    k = random.choice([3, 5, 7, 9])
    return (
        f"_x = torch.randn({batch}, {channels}, {h_2d * 2}, {h_2d * 2}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.nn.functional.avg_pool2d(x, kernel_size={k}, stride=2, padding={k // 2})\n"
        f"    return x.mean(dim=(2, 3))\n"
    ), True


def _tpl_d10_fused_rmsnorm_backward(batch, seq, hidden, dtype):
    """RMSNorm followed by op + sum — 2.10 added fused backward (#165370)."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()\n"
        f"    x = x * rms * w\n"
        f"    return x.relu().sum(dim=-1)\n"
    ), True


def _tpl_d10_mix_order_reduction(batch, seq, hidden, dtype):
    """Mix-order reduction along multiple axes — 2.10 fusion changes (#168209).

    Two reductions in different orders should now fuse into a single kernel
    via the mix-order reduction pass. The output is a scalar to keep the
    shape math trivial.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    inner = x.sum(dim=-1)        # reduce hidden  → [batch, seq]\n"
        f"    outer = x.mean(dim=0)        # reduce batch   → [seq, hidden]\n"
        f"    a = inner.sum(dim=-1)        # → [batch]\n"
        f"    b = outer.sum(dim=-1)        # → [seq]\n"
        f"    return a.sum() + b.sum()\n"
    ), True


def _tpl_d10_ldexp_chain(batch, seq, hidden, dtype):
    """torch.ldexp(x, e) — 2.11 added native lowering (#171721)."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        # ldexp expects integer exponent tensor
        f"_e = torch.randint(-5, 5, ({batch}, {seq}, {hidden}))\n"
        f"_program_inputs = (_x, _e)\n\n"
        f"def program(x, e):\n"
        f"    out = torch.ldexp(x, e)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d10_matmul_out_dtype(batch, seq, hidden, dtype):
    """matmul with explicit out_dtype — 2.10 added support (#163393)."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    # explicit out_dtype on matmul exercises a fresh codegen branch\n"
        f"    out = torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d10_slice_scatter_dtype(batch, seq, hidden, dtype):
    """slice_scatter — recently fixed dtype issue 2.9 (#160851)."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_y = torch.randn({batch}, {seq // 2}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    out = torch.slice_scatter(x, y, dim=1, start=0, end={seq // 2})\n"
        f"    return out.sum(dim=1)\n"
    ), True


def _tpl_d10_sort_with_nan(batch, seq, hidden, dtype):
    """sort with NaN values — 2.9 fixed CUDA NaN ordering (#159308).

    The NaN mask is precomputed and passed in as an input so eager and
    inductor receive identical inputs (otherwise rand_like at runtime would
    desynchronize them).
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_mask = torch.rand({batch}, {seq}, {hidden}) < 0.05\n"
        f"_program_inputs = (_x, _mask)\n\n"
        f"def program(x, mask):\n"
        f"    nan_x = torch.where(mask, torch.full_like(x, float('nan')), x)\n"
        f"    sorted_x, _ = torch.sort(nan_x, dim=-1)\n"
        f"    return torch.nan_to_num(sorted_x, nan=0.0).sum(dim=-1)\n"
    ), True


def _tpl_d10_constant_pad_nd_dynamic(batch, seq, hidden, dtype):
    """constant_pad_nd — meta function fixed 2.9 (#159878)."""
    pad = random.choice([(0, 1), (1, 1), (2, 3), (1, 2, 1, 2)])
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    out = torch.nn.functional.pad(x, {list(pad)}, mode='constant', value=0.5)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d10_persistent_reduction_aggressive(batch, seq, hidden, dtype):
    """Pattern that 2.9's more aggressive persistent reduction (#161055) should hit."""
    # smaller dims so persistent reduction is preferred over split reduction
    h = min(hidden, 256)
    return (
        f"_x = torch.randn({batch}, {seq}, {h}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    # multi-step reduction that should fold into one persistent kernel\n"
        f"    m = x.mean(dim=-1, keepdim=True)\n"
        f"    v = (x - m).pow(2).mean(dim=-1, keepdim=True)\n"
        f"    return ((x - m) / (v + 1e-6).sqrt()).sum(dim=-1)\n"
    ), True


# =============================================================================
# D11: 2.10+ stable APIs that earlier templates don't exercise
# =============================================================================
# Targets:
#   - scaled_dot_product_attention (causal/custom mask paths)
#   - flex_attention + create_block_mask (the public block-sparse API)
#   - _foreach_* ops (used by optimizers, fused codegen path)
#   - vmap + compile composition
#   - cummax / cummin / unique_consecutive (under-tested cumulative ops)
#
# These hit code paths that became stable in 2.10/2.11 and are mostly missed
# by D1-D10. Each template returns a reduced scalar/tensor that compares
# cleanly between eager and inductor.

def _tpl_d11_sdpa_causal(batch, seq, hidden, dtype):
    """SDPA with causal mask. Hits the FlashAttention/efficient backend lowering."""
    head_dim = max(min(hidden, 64), 8)
    n_heads = max(min(hidden // head_dim, 8), 1)
    return (
        f"_q = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_v = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    out = torch.nn.functional.scaled_dot_product_attention(\n"
        f"        q, k, v, is_causal=True)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d11_sdpa_attn_mask(batch, seq, hidden, dtype):
    """SDPA with explicit attn_mask — different lowering than is_causal."""
    head_dim = max(min(hidden, 64), 8)
    n_heads = max(min(hidden // head_dim, 8), 1)
    return (
        f"_q = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        f"_v = torch.randn({batch}, {n_heads}, {seq}, {head_dim}, dtype={dtype})\n"
        # Random additive mask in [-1, 0]
        f"_m = (torch.rand({batch}, {n_heads}, {seq}, {seq}) - 1.0).to({dtype})\n"
        f"_program_inputs = (_q, _k, _v, _m)\n\n"
        f"def program(q, k, v, m):\n"
        f"    out = torch.nn.functional.scaled_dot_product_attention(\n"
        f"        q, k, v, attn_mask=m)\n"
        f"    return out.mean(dim=-1)\n"
    ), True


def _tpl_d11_flex_attention_block_mask(batch, seq, hidden, dtype):
    """flex_attention with a block_mask — exercises the new public API path.

    flex_attention requires seq to be a multiple of 128 for the block_mask
    to work cleanly, AND head_dim >= 16 (NYI for smaller). Round both up.

    NOTE: force fp32 regardless of requested dtype. fp64 flex_attention on
    T4 (sm_75) triggers Triton shared-memory OOM which becomes an
    uncatchable C10 abort (see bug_008). Using fp64 here corrupts CUDA
    state for the rest of the fuzzer run. fp32 keeps the template
    exercisable without the crash risk.
    """
    head_dim = max(min(hidden, 64), 16)
    n_heads = max(min(max(hidden, head_dim) // head_dim, 8), 1)
    # flex_attention requires seq aligned to BLOCK (128 by default)
    s = ((max(seq, 128) + 127) // 128) * 128
    safe_dtype = "torch.float32"  # always fp32 to avoid fp64 C10 abort
    return (
        f"from torch.nn.attention.flex_attention import flex_attention, create_block_mask\n"
        f"_q = torch.randn({batch}, {n_heads}, {s}, {head_dim}, dtype={safe_dtype})\n"
        f"_k = torch.randn({batch}, {n_heads}, {s}, {head_dim}, dtype={safe_dtype})\n"
        f"_v = torch.randn({batch}, {n_heads}, {s}, {head_dim}, dtype={safe_dtype})\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def _causal(b, h, q_idx, kv_idx):\n"
        f"    return q_idx >= kv_idx\n"
        f"_block_mask = create_block_mask(_causal, B=None, H=None, Q_LEN={s}, KV_LEN={s})\n\n"
        f"def program(q, k, v):\n"
        f"    out = flex_attention(q, k, v, block_mask=_block_mask)\n"
        f"    return out.sum(dim=-1)\n"
    ), True


def _tpl_d11_foreach_add_chain(batch, seq, hidden, dtype):
    """torch._foreach_add over a list of tensors → single fused kernel.

    Optimizers (Adam/SGD) hit this path heavily. Inductor 2.10 added a
    foreach combo kernel that bundles all elementwise ops on a list into
    one kernel — that fusion logic is mostly untested.
    """
    n = random.choice([3, 5, 7])
    return (
        f"_xs = [torch.randn({batch}, {seq}, {hidden}, dtype={dtype}) for _ in range({n})]\n"
        f"_ys = [torch.randn({batch}, {seq}, {hidden}, dtype={dtype}) for _ in range({n})]\n"
        f"_program_inputs = (_xs, _ys)\n\n"
        f"def program(xs, ys):\n"
        f"    zs = torch._foreach_add(xs, ys, alpha=0.5)\n"
        f"    zs = torch._foreach_mul(zs, 1.5)\n"
        f"    # Reduce each then stack — gives a clean (n,) shaped output\n"
        f"    return torch.stack([z.sum() for z in zs])\n"
    ), True


def _tpl_d11_foreach_norm_chain(batch, seq, hidden, dtype):
    """_foreach_norm + reduction — variant of optimizer fusion path."""
    n = random.choice([3, 5])
    return (
        f"_xs = [torch.randn({batch}, {seq}, {hidden}, dtype={dtype}) for _ in range({n})]\n"
        f"_program_inputs = (_xs,)\n\n"
        f"def program(xs):\n"
        f"    # Use addcmul (also part of foreach in optim) and then reduce\n"
        f"    a = torch._foreach_addcmul(xs, xs, xs, value=0.1)\n"
        f"    return torch.stack([t.mean() for t in a])\n"
    ), True


def _tpl_d11_vmap_compile(batch, seq, hidden, dtype):
    """torch.func.vmap of a small function then compile.

    Hits the FunctionalTensor path which has different lowering semantics
    than direct compile. vmap + compile is a 2.9+ stable composition.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_w = torch.randn({hidden}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def _row_op(row, w):\n"
        f"    return torch.nn.functional.silu(row @ w).sum(dim=-1)\n"
        f"def program(x, w):\n"
        f"    # vmap over batch then over seq → 2-level batched call\n"
        f"    return torch.vmap(torch.vmap(_row_op, in_dims=(0, None)), in_dims=(0, None))(x, w)\n"
    ), True


def _tpl_d11_cummax_noncontig(batch, seq, hidden, dtype):
    """torch.cummax along a non-contiguous dim — under-tested cumulative path."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    # cummax along middle dim (non-innermost) — different codegen\n"
        f"    vals, _idx = torch.cummax(x, dim=1)\n"
        f"    return vals.sum(dim=-1)\n"
    ), True


def _tpl_d11_cummin_with_neg(batch, seq, hidden, dtype):
    """torch.cummin paired with neg — exercises sign-tracking in cumulative."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    vals, _idx = torch.cummin(-x, dim=-1)\n"
        f"    return (-vals).mean(dim=-1)\n"
    ), True


def _tpl_d11_unique_consecutive(batch, seq, hidden, dtype):
    """torch.unique_consecutive — rarely tested in compile path.

    Note: output is data-dependent (variable length), so we wrap in
    return_counts=True to get a fixed-shape (count_per_run) tensor that we
    then reduce. Marked is_deterministic=True since the OP itself is.
    """
    return (
        f"_x = torch.randint(0, 4, ({batch}, {seq}, {hidden})).to({dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    # unique_consecutive on the innermost dim — data-dependent shape\n"
        f"    _, counts = torch.unique_consecutive(x, return_counts=True, dim=-1)\n"
        f"    return counts.float().sum()\n"
    ), True  # deterministic op; backward skipped via _NO_BACKWARD_TEMPLATES


# =============================================================================
# D12: data-dependent shape ops (bug_008 neighbors)
# =============================================================================
# bug_008a (unique_consecutive + sum + fullgraph → silent zero) pointed at a
# codegen pattern: "data-dependent-shape tensor followed by a reduction over
# the dynamic dim". These templates sweep the op siblings that produce
# dynamic-length outputs: nonzero, unique, masked_select, bincount, boolean
# indexing, etc. Each is composed with a reduction to reproduce exactly the
# trigger pattern of bug_008a.

def _tpl_d12_nonzero_reduce(batch, seq, hidden, dtype):
    """torch.nonzero → [N, ndim] where N is data-dependent, then reduce.

    Forward generates a float input, thresholds it, then counts nonzero
    positions. The output of nonzero has a dynamic first dim; reducing it
    is the exact pattern that bug_008a triggered.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mask = x > 0\n"
        f"    idx = torch.nonzero(mask)          # [N, 3] dynamic N\n"
        f"    return idx.float().sum()\n"
    ), True


def _tpl_d12_masked_select_sum(batch, seq, hidden, dtype):
    """torch.masked_select returns a 1D dynamic-length tensor."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_mask = torch.rand({batch}, {seq}, {hidden}) > 0.5\n"
        f"_program_inputs = (_x, _mask)\n\n"
        f"def program(x, mask):\n"
        f"    selected = torch.masked_select(x, mask)  # 1D dynamic N\n"
        f"    return selected.sum()\n"
    ), True


def _tpl_d12_masked_select_compose(batch, seq, hidden, dtype):
    """masked_select piped through a pointwise chain before reducing.

    Stresses codegen for `dynamic_len + pointwise + reduce` fusion.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_mask = torch.rand({batch}, {seq}, {hidden}) > 0.7\n"
        f"_program_inputs = (_x, _mask)\n\n"
        f"def program(x, mask):\n"
        f"    s = torch.masked_select(x, mask)\n"
        f"    return (s.abs() + 1.0).log().sum()\n"
    ), True


def _tpl_d12_bool_index_reduce(batch, seq, hidden, dtype):
    """`x[x > 0].sum()` — boolean-indexing form of masked_select.

    Boolean indexing is a different lowering path than masked_select even
    though the semantics overlap — exercise both.
    """
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    selected = x[x > 0]\n"
        f"    return selected.sum()\n"
    ), True


def _tpl_d12_unique_sum(batch, seq, hidden, dtype):
    """torch.unique (non-consecutive, sorted) — deduplication returning
    a dynamic-length 1D tensor."""
    return (
        f"_x = torch.randint(0, 8, ({batch}, {seq}, {hidden})).to({dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    vals = torch.unique(x)             # dynamic N\n"
        f"    return vals.float().sum()\n"
    ), True


def _tpl_d12_unique_inverse_compose(batch, seq, hidden, dtype):
    """torch.unique(return_inverse=True) — vals has dynamic length, inverse
    has fixed shape of input. Mix the two in the output."""
    return (
        f"_x = torch.randint(0, 8, ({batch}, {seq}, {hidden})).to({dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    vals, inv = torch.unique(x, return_inverse=True)\n"
        f"    return vals.float().sum() + inv.float().mean()\n"
    ), True


def _tpl_d12_bincount_reduce(batch, seq, hidden, dtype):
    """torch.bincount — output size = (max_input + 1), data-dependent.

    bincount is int-only; we produce an int tensor and return float
    reductions. The output dtype of bincount is int64.
    """
    n = max(batch * seq, 16)
    return (
        f"_x = torch.randint(0, {max(hidden, 4)}, ({n},), dtype=torch.int64)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    counts = torch.bincount(x)          # dynamic length = x.max()+1\n"
        f"    return counts.float().sum()\n"
    ), True


def _tpl_d12_unique_return_counts(batch, seq, hidden, dtype):
    """torch.unique(return_counts=True) — both vals and counts are dynamic-length.
    Direct analog of bug_008a on the non-consecutive variant."""
    return (
        f"_x = torch.randint(0, 6, ({batch}, {seq}, {hidden})).to({dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    vals, counts = torch.unique(x, return_counts=True)\n"
        f"    return counts.float().sum()\n"  # bug_008a's exact reduction pattern
    ), True


def _tpl_d12_sort_threshold_reduce(batch, seq, hidden, dtype):
    """Sort then boolean-index with a threshold — fused sort + dynamic slice."""
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    s, _ = torch.sort(x.flatten())\n"
        f"    return s[s > 0].sum()\n"
    ), True


# =============================================================================
# D13: Memory Alias & In-place Mutation
# =============================================================================
# Targets Inductor's functionalization pass and alias analysis.
# In-place ops on aliased views (view, expand, as_strided, narrow, etc.)
# require the compiler to correctly track buffer sharing. Bugs here are
# silent correctness issues — no crash, just wrong output.

def _tpl_d13_view_inplace_sum(batch, seq, hidden, dtype):
    """view(-1) + strided slice + in-place add: tests alias tracking on flat view."""
    total = batch * seq * hidden
    return (
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.clone()\n"
        f"    y = x.view(-1)\n"
        f"    z = y[::2]\n"
        f"    z.add_(1.0)\n"
        f"    return x.sum()\n"
    ), True


def _tpl_d13_expand_inplace_alias(batch, seq, hidden, dtype):
    """expand creates stride-0 dim; in-place on slice tests whether expansion is
    materialized before mutation. If not, all expanded slices get mutated."""
    return (
        f"_x = torch.randn(1, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.clone()\n"
        f"    y = x.expand({batch}, -1, -1)\n"
        f"    out = y.clone()\n"
        f"    out[0].mul_(2.0)\n"
        f"    return out.sum(dim=0)\n"
    ), True


def _tpl_d13_as_strided_overlap(batch, seq, hidden, dtype):
    """as_strided with overlapping elements — reduction over overlapping view.
    Compiler must not assume non-overlapping writes/reads."""
    flat = max(64, seq * 4)
    window = min(16, flat // 4)
    stride = max(1, window // 2)
    n_windows = (flat - window) // stride + 1
    return (
        f"_x = torch.randn({flat}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.as_strided(({n_windows}, {window}), ({stride}, 1))\n"
        f"    return y.sum(dim=-1)\n"
    ), True


def _tpl_d13_slice_inplace_backward(batch, seq, hidden, dtype):
    """Slice + in-place + backward: gradient must accumulate correctly through
    the aliased region. Functionalization bugs lose gradient contributions."""
    seq2 = max(4, seq)
    half = seq2 // 2
    return (
        f"_x = torch.randn({batch}, {seq2}, {hidden}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    y[:, :{half}, :].add_(1.0)\n"
        f"    return y.pow(2).sum()\n"
    ), True


def _tpl_d13_view_chain_mutation(batch, seq, hidden, dtype):
    """Chain of views (reshape → transpose → reshape) then mutation on leaf view.
    Tests multi-hop alias propagation."""
    h2 = max(2, hidden // 2) * 2
    return (
        f"_x = torch.randn({batch}, {seq}, {h2}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.clone()\n"
        f"    a = x.view({batch}, {seq}, 2, {h2 // 2})\n"
        f"    b = a.transpose(2, 3)\n"
        f"    c = b.reshape({batch}, {seq}, -1)\n"
        f"    c[:, 0, :].fill_(0.0)\n"
        f"    return x.mean()\n"
    ), True


def _tpl_d13_transpose_inplace_matmul(batch, seq, hidden, dtype):
    """transpose creates non-contiguous view → in-place on original → matmul on view.
    Tests whether inductor correctly sequences mutation before matmul read."""
    h = max(8, hidden)
    return (
        f"_x = torch.randn({batch}, {h}, {h}, dtype={dtype})\n"
        f"_w = torch.randn({h}, {h}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    x = x.clone()\n"
        f"    view_t = x.transpose(1, 2)\n"
        f"    x[:, 0, :].add_(1.0)\n"
        f"    return torch.matmul(view_t, w).sum()\n"
    ), True


def _tpl_d13_unfold_scatter_reduce(batch, seq, hidden, dtype):
    """unfold creates overlapping sliding window → scatter_add on unfolded view.
    Tests overlap-aware buffer materialization."""
    flat = max(32, seq * 2)
    window = 8
    step = 4
    n_win = (flat - window) // step + 1
    return (
        f"_x = torch.randn({flat}, dtype={dtype})\n"
        f"_src = torch.ones({n_win}, {window}, dtype={dtype})\n"
        f"_program_inputs = (_x, _src)\n\n"
        f"def program(x, src):\n"
        f"    y = x.unfold(0, {window}, {step})\n"
        f"    return (y * src).sum()\n"
    ), True


def _tpl_d13_diagonal_inplace(batch, seq, hidden, dtype):
    """diagonal() returns a view sharing storage with original.
    In-place on diagonal must be visible in original."""
    n = min(seq, hidden, 32)
    return (
        f"_x = torch.randn({batch}, {n}, {n}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.clone()\n"
        f"    d = x.diagonal(dim1=1, dim2=2)\n"
        f"    d.fill_(0.0)\n"
        f"    return x.sum()\n"
    ), True


def _tpl_d13_narrow_copy_on_write(batch, seq, hidden, dtype):
    """narrow() is a view — mutation on narrow slice must trigger copy-on-write
    or correct buffer tracking. Tests functionalization of narrow + mul_."""
    seq2 = max(8, seq)
    narrow_len = seq2 // 2
    return (
        f"_x = torch.randn({batch}, {seq2}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.clone()\n"
        f"    part = x.narrow(1, 0, {narrow_len})\n"
        f"    part.mul_(0.5)\n"
        f"    rest = x.narrow(1, {narrow_len}, {seq2 - narrow_len})\n"
        f"    return part.sum() + rest.sum()\n"
    ), True


def _tpl_d13_select_scatter_alias(batch, seq, hidden, dtype):
    """select_scatter produces new tensor from source + scattered slice.
    When source is a view, tests whether scatter respects alias."""
    seq2 = max(4, seq)
    return (
        f"_x = torch.randn({batch}, {seq2}, {hidden}, dtype={dtype})\n"
        f"_fill = torch.ones({batch}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x, _fill)\n\n"
        f"def program(x, fill):\n"
        f"    y = x.select_scatter(fill, dim=1, index=0)\n"
        f"    return y.sum() + x.sum()\n"
    ), True


def _tpl_d13_chunk_inplace_cat(batch, seq, hidden, dtype):
    """chunk splits tensor into views — in-place on one chunk affects original.
    Tests chunk alias tracking + cat reconstruction."""
    h2 = max(4, hidden // 2) * 2
    return (
        f"_x = torch.randn({batch}, {seq}, {h2}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.clone()\n"
        f"    a, b = x.chunk(2, dim=-1)\n"
        f"    a.relu_()\n"
        f"    return torch.cat([a, b], dim=-1).sum()\n"
    ), True


def _tpl_d13_storage_offset_mutation(batch, seq, hidden, dtype):
    """Slicing creates views with non-zero storage offset.
    In-place mutation on overlapping slices tests offset-aware scheduling."""
    seq2 = max(8, seq)
    return (
        f"_x = torch.randn({batch}, {seq2}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.clone()\n"
        f"    left = x[:, :({seq2}//2 + 1), :]\n"
        f"    right = x[:, ({seq2}//2 - 1):, :]\n"
        f"    left.add_(1.0)\n"
        f"    right.mul_(2.0)\n"
        f"    return x.sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D14: Graph Break & Partial Compilation Templates
# These force torch.compile to handle graph breaks: control flow that causes
# Dynamo to split the graph into multiple compiled segments + eager fallback.
# Bug surface: state transfer between segments, view/alias info loss at breaks.
# ---------------------------------------------------------------------------

def _tpl_d14_graph_break_data_dep(batch, seq, hidden, dtype):
    """Data-dependent branch forces graph break. Tests Dynamo's handling of
    conditional control flow where the condition depends on tensor values."""
    return (
        f"import torch\n"
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.sin()\n"
        f"    s = y.sum().item()\n"
        f"    if s > 0:\n"
        f"        z = y * 2.0 + x.cos()\n"
        f"    else:\n"
        f"        z = y * 0.5 - x.cos()\n"
        f"    return z.sum()\n"
    ), True

def _tpl_d14_graph_break_attr_mutate(batch, seq, hidden, dtype):
    """Module attribute mutation between forward calls. Dynamo must detect that
    the captured attribute changed and recompile — if guard misses it, silent
    wrong result."""
    return (
        f"import torch\nimport torch.nn as nn\n"
        f"class _M(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n"
        f"        self.scale = 1.0\n"
        f"    def forward(self, x):\n"
        f"        return x * self.scale + x.mean(dim=-1, keepdim=True)\n"
        f"_m = _M()\n"
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    _m.scale = 2.0\n"
        f"    out1 = _m(x)\n"
        f"    _m.scale = 0.5\n"
        f"    out2 = _m(x)\n"
        f"    return out1 + out2\n"
    ), True

def _tpl_d14_graph_break_loop(batch, seq, hidden, dtype):
    """Loop with tensor-dependent trip count. Dynamo unrolls or breaks the
    graph. Tests whether the loop produces correct accumulated result."""
    n_iters = min(seq, 8)
    return (
        f"import torch\n"
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    acc = torch.zeros_like(x)\n"
        f"    for i in range({n_iters}):\n"
        f"        acc = acc + x.roll(i, dims=1) * (0.9 ** i)\n"
        f"    return acc.mean()\n"
    ), True

def _tpl_d14_graph_break_cond(batch, seq, hidden, dtype):
    """torch.cond (higher-order op) for branchless control flow in compiled
    mode. Tests the cond decomposition / tracing path in Dynamo."""
    return (
        f"import torch\n"
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    def true_fn(x):\n"
        f"        return x.sin() + x.cos()\n"
        f"    def false_fn(x):\n"
        f"        return x.tanh() - x.sigmoid()\n"
        f"    pred = x.sum() > 0\n"
        f"    return torch.cond(pred, true_fn, false_fn, (x,))\n"
    ), True

def _tpl_d14_graph_break_print(batch, seq, hidden, dtype):
    """print() inside compiled function forces graph break. Tests state
    transfer: the value computed before print must be correctly passed to
    the post-break compiled segment."""
    return (
        f"import torch\n"
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x @ x.transpose(-2, -1) if x.shape[-2] == x.shape[-1] else x * x\n"
        f"    y = y / (y.abs().max() + 1e-6)\n"
        f"    _unused = y.sum().item()\n"
        f"    z = torch.nn.functional.layer_norm(y, y.shape[-1:])\n"
        f"    return z.sum()\n"
    ), True

def _tpl_d14_graph_break_multi_segment(batch, seq, hidden, dtype):
    """Multiple graph breaks create 3+ compiled segments. Tests that
    tensor metadata (shape, stride, dtype) is correctly passed across
    all segment boundaries."""
    return (
        f"import torch\n"
        f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    # Segment 1: pointwise\n"
        f"    y = x.sin() + x.cos()\n"
        f"    s1 = y.sum().item()  # graph break 1\n"
        f"    # Segment 2: reduction\n"
        f"    z = y * y - y.mean(dim=-1, keepdim=True)\n"
        f"    s2 = z.mean().item()  # graph break 2\n"
        f"    # Segment 3: matmul-ish\n"
        f"    w = z / (z.norm(dim=-1, keepdim=True) + 1e-6)\n"
        f"    return w.sum()\n"
    ), True


# =============================================================================
# Template Registry
# =============================================================================
# Categorized by D1-D7, with weight design:
#   D1 (pointwise-reduction fusion): high weight
#   D2 (epilogue into GEMM): highest weight
#   D3 (layout propagation): low weight
#   D4 (cross-op dependency): high weight
#   D5 (shape/dtype boundary): high weight
#   D6 (decomposition coverage): high weight
#   D7 (new code paths): highest weight

PROGRAM_TEMPLATES = [
    # D1: Pointwise-Reduction Fusion
    ("d1_softmax_fused",        _tpl_d1_softmax_fused,        3),
    ("d1_layernorm_manual",     _tpl_d1_layernorm_manual,     3),
    ("d1_reduce_cascade",       _tpl_d1_reduce_cascade,       2),
    ("d1_online_softmax",       _tpl_d1_online_softmax,       3),
    ("d1_multi_axis_reduce",    _tpl_d1_multi_axis_reduce,    3),
    ("d1_large_reduce_float",   _tpl_d1_large_reduce_float,   3),
    ("d1_chained_norm",         _tpl_d1_chained_norm,         3),
    ("d1_softmax_large_dim",    _tpl_d1_softmax_large_dim,    3),
    ("d1_var_mean_fused",       _tpl_d1_var_mean_fused,       3),
    ("d1_cross_entropy_manual", _tpl_d1_cross_entropy_manual, 4),
    ("d1_welford_online_var",   _tpl_d1_welford_online_var,   3),
    ("d1_kl_divergence_manual", _tpl_d1_kl_divergence_manual, 4),
    ("d1_prod_reduce",          _tpl_d1_prod_reduce,          3),

    # D2: Epilogue Fusion into GEMM
    ("d2_linear_bias_act",            _tpl_d2_linear_bias_act,            4),
    ("d2_matmul_scale_add",           _tpl_d2_matmul_scale_add,           4),
    ("d2_conv_bn_act",                _tpl_d2_conv_bn_act,                3),
    ("d2_attention_epilogue",         _tpl_d2_attention_epilogue,         4),
    ("d2_interpolate_chain",          _tpl_d2_interpolate_chain,          4),
    ("d2_pool_int_dtype",             _tpl_d2_pool_int_dtype,             3),
    ("d2_multi_output_epilogue",      _tpl_d2_multi_output_epilogue,      4),
    ("d2_conv_groupnorm_skip",        _tpl_d2_conv_groupnorm_skip,        3),
    ("d2_reduction_transpose_fusion", _tpl_d2_reduction_transpose_fusion, 4),
    ("d2_addmm_pattern",             _tpl_d2_addmm_pattern,              3),
    ("d2_einsum_multi_contract",     _tpl_d2_einsum_multi_contract,      4),
    ("d2_bmm_epilogue_chain",        _tpl_d2_bmm_epilogue_chain,         4),
    ("d2_conv_transpose_epilogue",   _tpl_d2_conv_transpose_epilogue,    4),
    ("d2_baddbmm_pattern",           _tpl_d2_baddbmm_pattern,            4),
    ("d2_linear_residual_norm",      _tpl_d2_linear_residual_norm,       4),
    ("d2_depthwise_conv_epilogue",   _tpl_d2_depthwise_conv_epilogue,    4),

    # D3: Layout Propagation (lower weight -- mostly hits known bugs)
    ("d3_permute_matmul",        _tpl_d3_permute_matmul,        2),
    ("d3_reshape_chain",         _tpl_d3_reshape_chain,         1),
    ("d3_noncontiguous_reduce",  _tpl_d3_noncontiguous_reduce,  2),
    ("d3_channels_last_chain",   _tpl_d3_channels_last_chain,   1),
    ("d3_pad_reshape_chunk",     _tpl_d3_pad_reshape_chunk,     2),
    ("d3_alias_cat_expand",      _tpl_d3_alias_cat_expand,      2),
    ("d3_strided_overlap_reduce",_tpl_d3_strided_overlap_reduce,1),
    ("d3_inplace_view_mutation", _tpl_d3_inplace_view_mutation, 2),
    ("d3_contiguous_elision",    _tpl_d3_contiguous_elision,    2),
    ("d3_triu_tril_stride",      _tpl_d3_triu_tril_stride,      1),
    ("d3_expand_stride_zero",    _tpl_d3_expand_stride_zero,    1),

    # D4: Cross-op Data Dependency
    ("d4_rope_pattern",           _tpl_d4_rope_pattern,           4),
    ("d4_reduce_as_scale",        _tpl_d4_reduce_as_scale,        3),
    ("d4_gather_scatter",         _tpl_d4_gather_scatter,         3),
    ("d4_cumulative_dep",         _tpl_d4_cumulative_dependency,  2),
    ("d4_scatter_reduce_dup",     _tpl_d4_scatter_reduce_dup,     4),
    ("d4_where_branch_matmul",    _tpl_d4_where_branch_matmul,    3),
    ("d4_index_put_accumulate",   _tpl_d4_index_put_accumulate,   4),
    ("d4_cumsum_pointwise",       _tpl_d4_cumsum_pointwise,       3),
    ("d4_pointwise_then_cumsum",  _tpl_d4_pointwise_then_cumsum,  5),  # bug_011 gap fix
    ("d4_cumprod_fused",          _tpl_d4_cumprod_fused,          4),  # SplitScan variant
    ("d4_chunk_neg_cat",          _tpl_d4_chunk_neg_cat,          4),
    ("d4_transformer_subgraph",   _tpl_d4_transformer_subgraph,   4),
    ("d4_topk_mask_matmul",       _tpl_d4_topk_mask_matmul,       4),
    ("d4_sort_dependent_ops",     _tpl_d4_sort_dependent_ops,     3),
    ("d4_masked_fill_softmax",    _tpl_d4_masked_fill_softmax,    4),
    ("d4_residual_gating",        _tpl_d4_residual_gating,        4),
    ("d4_swiglu",                 _tpl_d4_swiglu,                 5),
    ("d4_multi_head_split_merge", _tpl_d4_multi_head_split_merge, 4),

    # D5: Shape / Dtype Boundary Conditions
    ("d5_unaligned_matmul",         _tpl_d5_unaligned_matmul,         4),
    ("d5_dtype_cast_chain",         _tpl_d5_dtype_cast_chain,         3),
    ("d5_scalar_tensor_interaction",_tpl_d5_scalar_tensor_interaction, 4),
    ("d5_extreme_batch1",           _tpl_d5_extreme_batch1,           4),
    ("d5_empty_dim_ops",            _tpl_d5_empty_dim_ops,            3),
    ("d5_mixed_precision_matmul",   _tpl_d5_mixed_precision_matmul,   4),
    ("d5_reduction_keepdim_false",  _tpl_d5_reduction_keepdim_false,  3),

    # Targeting known open Inductor bugs
    ("d5_logit_eps",                _tpl_d5_logit_eps,                5),  # #145379
    ("d5_logit_chain",              _tpl_d5_logit_chain,              4),  # #145379 variants
    # bug_004 family — known unfixed in 2.11 but already reported.
    # Keep them in the registry (rare seed for mutation diversity) but
    # at very low weight so they don't dominate.
    ("d5_dtype_cast_half_round",    _tpl_d5_dtype_cast_half_round,    1),
    ("d5_dtype_cast_bf16_round",    _tpl_d5_dtype_cast_bf16_round,    1),
    ("d1_broadcast_reduce_cross",   _tpl_d1_broadcast_reduce_cross,   4),  # tile scheduling stress
    ("d1_reduce_chain_mixed_dim",   _tpl_d1_reduce_chain_mixed_dim,   4),  # multi-dim reduction
    ("d4_where_logit_clamp",        _tpl_d4_where_logit_clamp,        4),  # conditional + logit

    # Expanded op coverage
    ("d2_bmm_epilogue",             _tpl_d2_bmm_epilogue,             3),
    ("d2_baddbmm",                  _tpl_d2_baddbmm,                  3),
    ("d4_embedding_lookup",         _tpl_d4_embedding_lookup,         3),
    ("d3_unfold_ops",               _tpl_d3_unfold_ops,               3),
    ("d3_roll_shift",               _tpl_d3_roll_shift,               3),
    ("d4_scatter_reduce_sum",       _tpl_d4_scatter_reduce_sum,       3),
    ("d4_index_select_gather",      _tpl_d4_index_select_gather,      3),
    ("d2_upsample_nearest",         _tpl_d2_upsample_nearest,         3),
    ("d2_adaptive_avgpool",         _tpl_d2_adaptive_avgpool,         3),
    ("d1_logcumsumexp",             _tpl_d1_logcumsumexp,             3),
    ("d5_lerp_addcdiv",             _tpl_d5_lerp_addcdiv,             3),
    # d4_searchsorted_bucketize removed: Inductor raises NotImplementedError (StarDep)
    # ("d4_searchsorted_bucketize",   _tpl_d4_searchsorted_bucketize,   3),
    ("d2_group_norm_conv",          _tpl_d2_group_norm_conv,          3),

    # Phase 2: Decomposition-targeting templates (D6)
    # Target ops with complex decomposition logic in torch/_decomp/decompositions.py
    ("d6_softplus_threshold",       _tpl_d6_softplus_threshold,       4),
    ("d6_elu_variants",             _tpl_d6_elu_variants,             4),
    ("d6_masked_fill_chain",        _tpl_d6_masked_fill_chain,        4),
    ("d6_huber_loss",               _tpl_d6_huber_loss,               4),
    ("d6_logaddexp",                _tpl_d6_logaddexp,                4),
    ("d6_xlogy",                    _tpl_d6_xlogy,                    4),
    ("d6_nan_to_num",               _tpl_d6_nan_to_num,               4),
    ("d6_threshold_chain",          _tpl_d6_threshold_chain,          4),
    ("d6_renorm",                   _tpl_d6_renorm,                   4),
    ("d6_select_scatter",           _tpl_d6_select_scatter,           4),
    ("d6_diag_embed",               _tpl_d6_diag_embed,               3),
    ("d6_multi_margin_loss",        _tpl_d6_multi_margin_loss,        4),
    ("d6_channel_shuffle",          _tpl_d6_channel_shuffle,          3),
    ("d6_addcmul_chain",            _tpl_d6_addcmul_chain,            4),
    ("d6_linspace_index",           _tpl_d6_linspace_index,           3),

    # D7: PyTorch 2.7+ new code paths (high weight — untested code)
    ("d7_prologue_pointwise_gemm",  _tpl_d7_prologue_pointwise_gemm,  5),
    ("d7_prologue_norm_gemm",       _tpl_d7_prologue_norm_gemm,       5),
    ("d7_online_softmax_large",     _tpl_d7_online_softmax_large,     4),
    ("d7_online_softmax_odd",       _tpl_d7_online_softmax_odd,       4),
    # D7 precision_cast also matches bug_004 in 2.11 — lower its weight.
    ("d7_precision_cast_emulate",   _tpl_d7_precision_cast_emulate,   1),
    ("d7_addmv_decomp",            _tpl_d7_addmv_decomp,             4),
    ("d7_split_cat_pattern",        _tpl_d7_split_cat_pattern,        4),
    ("d7_cooperative_reduction",    _tpl_d7_cooperative_reduction,    3),
    ("d7_max_pool_as_reduction",    _tpl_d7_max_pool_as_reduction,    4),
    ("d7_groupnorm_parallel",       _tpl_d7_groupnorm_parallel,       4),

    # D8: Indexing ops — target ModularIndexing-related code paths (bug_001 family)
    ("d8_scatter_along_dim",        _tpl_d8_scatter_along_dim,        5),
    ("d8_scatter_add_overlap",      _tpl_d8_scatter_add_overlap,      4),
    ("d8_index_put_accumulate_3d",  _tpl_d8_index_put_accumulate_3d,  4),
    ("d8_gather_neg_dim",           _tpl_d8_gather_neg_dim,           5),
    ("d8_masked_scatter_chain",     _tpl_d8_masked_scatter_chain,     4),
    ("d8_advanced_index_view",      _tpl_d8_advanced_index_view,      5),
    ("d8_take_indexing",            _tpl_d8_take_indexing,            4),
    ("d8_index_select_strided_reduce", _tpl_d8_index_select_strided_reduce, 5),

    # D9: Complex dtype (cfloat / cdouble) — undertested codegen path
    # NOTE: lowered weight in 2.11 since complex codegen still falls back to eager
    ("d9_complex_matmul",           _tpl_d9_complex_matmul,           2),
    ("d9_rfft_pointwise_irfft",     _tpl_d9_rfft_pointwise_irfft,     3),
    ("d9_complex_pointwise_chain",  _tpl_d9_complex_pointwise_chain,  2),
    ("d9_view_as_complex_roundtrip",_tpl_d9_view_as_complex_roundtrip,3),
    ("d9_complex_norm_reduce",      _tpl_d9_complex_norm_reduce,      2),
    ("d9_stft_magnitude",           _tpl_d9_stft_magnitude,           3),

    # D10: PyTorch 2.8-2.11 new code paths (HIGHEST priority — newest code)
    ("d10_decompose_k_matmul",      _tpl_d10_decompose_k_matmul,            6),
    ("d10_3d_online_softmax",       _tpl_d10_3d_online_softmax,             6),
    ("d10_avg_pool_as_reduction",   _tpl_d10_avg_pool_as_reduction,         6),
    ("d10_fused_rmsnorm_backward",  _tpl_d10_fused_rmsnorm_backward,        6),
    ("d10_mix_order_reduction",     _tpl_d10_mix_order_reduction,           6),
    ("d10_ldexp_chain",             _tpl_d10_ldexp_chain,                   5),
    ("d10_matmul_out_dtype",        _tpl_d10_matmul_out_dtype,              5),
    ("d10_slice_scatter_dtype",     _tpl_d10_slice_scatter_dtype,           5),
    ("d10_sort_with_nan",           _tpl_d10_sort_with_nan,                 5),
    ("d10_constant_pad_nd_dynamic", _tpl_d10_constant_pad_nd_dynamic,       5),
    ("d10_persistent_reduction_aggressive",
                                    _tpl_d10_persistent_reduction_aggressive, 5),

    # D11: PyTorch 2.10/2.11 stable APIs (SDPA, FlexAttention, foreach, vmap, cum*)
    ("d11_sdpa_causal",             _tpl_d11_sdpa_causal,                   6),
    ("d11_sdpa_attn_mask",          _tpl_d11_sdpa_attn_mask,                6),
    ("d11_flex_attention_block_mask", _tpl_d11_flex_attention_block_mask,   5),
    ("d11_foreach_add_chain",       _tpl_d11_foreach_add_chain,             5),
    ("d11_foreach_norm_chain",      _tpl_d11_foreach_norm_chain,            5),
    ("d11_vmap_compile",            _tpl_d11_vmap_compile,                  4),
    ("d11_cummax_noncontig",        _tpl_d11_cummax_noncontig,              5),
    ("d11_cummin_with_neg",         _tpl_d11_cummin_with_neg,               4),
    ("d11_unique_consecutive",      _tpl_d11_unique_consecutive,            4),

    # D12: data-dependent shape ops (targeting bug_008 neighbors)
    # High weight — this category just found bug_008 and is under-explored.
    ("d12_nonzero_reduce",          _tpl_d12_nonzero_reduce,                6),
    ("d12_masked_select_sum",       _tpl_d12_masked_select_sum,             6),
    ("d12_masked_select_compose",   _tpl_d12_masked_select_compose,         5),
    ("d12_bool_index_reduce",       _tpl_d12_bool_index_reduce,             6),
    ("d12_unique_sum",              _tpl_d12_unique_sum,                    6),
    ("d12_unique_inverse_compose",  _tpl_d12_unique_inverse_compose,        5),
    ("d12_bincount_reduce",         _tpl_d12_bincount_reduce,               5),
    ("d12_unique_return_counts",    _tpl_d12_unique_return_counts,          6),
    ("d12_sort_threshold_reduce",   _tpl_d12_sort_threshold_reduce,         5),

    # D13: Memory Alias & In-place Mutation — targets functionalization pass
    # and alias analysis in Inductor. Bugs here are silent correctness issues
    # where in-place ops on aliased views produce wrong results after compilation.
    ("d13_view_inplace_sum",        _tpl_d13_view_inplace_sum,              7),
    ("d13_expand_inplace_alias",    _tpl_d13_expand_inplace_alias,          7),
    ("d13_as_strided_overlap",      _tpl_d13_as_strided_overlap,            6),
    ("d13_slice_inplace_backward",  _tpl_d13_slice_inplace_backward,        7),
    ("d13_view_chain_mutation",     _tpl_d13_view_chain_mutation,           6),
    ("d13_transpose_inplace_matmul", _tpl_d13_transpose_inplace_matmul,    6),
    ("d13_unfold_scatter_reduce",   _tpl_d13_unfold_scatter_reduce,         6),
    ("d13_diagonal_inplace",        _tpl_d13_diagonal_inplace,              5),
    ("d13_narrow_copy_on_write",    _tpl_d13_narrow_copy_on_write,          7),
    ("d13_select_scatter_alias",    _tpl_d13_select_scatter_alias,          6),
    ("d13_chunk_inplace_cat",       _tpl_d13_chunk_inplace_cat,             6),
    ("d13_storage_offset_mutation", _tpl_d13_storage_offset_mutation,       5),

    # D14: Graph Break & Partial Compilation — targets Dynamo graph break handling,
    # state transfer between compiled segments, and control-flow compilation.
    ("d14_graph_break_data_dep",    _tpl_d14_graph_break_data_dep,          7),
    ("d14_graph_break_attr_mutate", _tpl_d14_graph_break_attr_mutate,       7),
    ("d14_graph_break_loop",        _tpl_d14_graph_break_loop,              7),
    ("d14_graph_break_cond",        _tpl_d14_graph_break_cond,              7),
    ("d14_graph_break_print",       _tpl_d14_graph_break_print,             6),
    ("d14_graph_break_multi_segment", _tpl_d14_graph_break_multi_segment,   7),
]

# ---------------------------------------------------------------------------
# Indexing-heavy templates — non-regular access patterns that stress
# Triton kernel indexing, masking, and broadcast logic.
# ---------------------------------------------------------------------------

def _tpl_idx_gather_noncontiguous(B, S, H, dtype):
    """gather on permuted tensor → reduction: non-contiguous index + stride"""
    code = f"""
_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')
_x = _x.permute(0, 2, 1).contiguous().permute(0, 2, 1)
_idx = torch.randint(0, {H}, ({B}, {S}, min(16, {H})), device='cuda')
x = torch.gather(_x, dim=-1, index=_idx)
x = x.sum(dim=-1)
_program_inputs = (_x,)
"""
    return code.strip(), False

def _tpl_idx_scatter_broadcast(B, S, H, dtype):
    """scatter_add with broadcast source → stride-0 write pattern"""
    code = f"""
_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')
_src = _x.mean(dim=-1, keepdim=True).expand_as(_x)
_idx = torch.randint(0, {H}, ({B}, {S}, {H}), device='cuda')
x = torch.zeros_like(_x).scatter_add(dim=-1, index=_idx, src=_src)
x = x.sum(dim=-1)
_program_inputs = (_x,)
"""
    return code.strip(), False

def _tpl_idx_bool_mask_reduce(B, S, H, dtype):
    """boolean mask + broadcast assign + reduction"""
    code = f"""
_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')
_mask = _x > 0
_fill = _x.mean(dim=-1, keepdim=True).expand_as(_x)
x = torch.where(_mask, _x, _fill)
x = x.sum(dim=-1)
_program_inputs = (_x,)
"""
    return code.strip(), True

def _tpl_idx_advanced_index_chain(B, S, H, dtype):
    """multi-level advanced indexing with reduction"""
    code = f"""
_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')
_idx1 = torch.randint(0, {S}, (min(16, {S}),), device='cuda')
_idx2 = torch.randint(0, {H}, (min(8, {H}),), device='cuda')
x = _x[:, _idx1][:, :, _idx2]
x = x.sum(dim=-1).sum(dim=-1)
_program_inputs = (_x,)
"""
    return code.strip(), False

def _tpl_idx_scatter_view_chain(B, S, H, dtype):
    """view → scatter → contiguous check: aliasing + write pattern"""
    code = f"""
_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')
_flat = _x.reshape({B}, -1)
_idx = torch.randint(0, _flat.shape[-1], ({B}, min(32, _flat.shape[-1])), device='cuda')
_src = torch.randn({B}, min(32, _flat.shape[-1]), dtype={dtype}, device='cuda')
x = _flat.scatter(dim=-1, index=_idx, src=_src)
x = x.reshape({B}, {S}, {H}).sum(dim=(1,2))
_program_inputs = (_x,)
"""
    return code.strip(), False

def _tpl_idx_expand_inplace_pattern(B, S, H, dtype):
    """expand (stride-0) → clone → in-place mul: tests stride-0 handling"""
    code = f"""
_x = torch.randn({B}, 1, {H}, dtype={dtype}, device='cuda')
_y = _x.expand({B}, {S}, {H}).clone()
_scale = torch.randn({B}, {S}, 1, dtype={dtype}, device='cuda')
x = _y * _scale.expand_as(_y)
x = x.sum(dim=(1,2))
_program_inputs = (_x,)
"""
    return code.strip(), True

PROGRAM_TEMPLATES.extend([
    ("idx_gather_noncontiguous",    _tpl_idx_gather_noncontiguous,    7),
    ("idx_scatter_broadcast",       _tpl_idx_scatter_broadcast,       7),
    ("idx_bool_mask_reduce",        _tpl_idx_bool_mask_reduce,        7),
    ("idx_advanced_index_chain",    _tpl_idx_advanced_index_chain,    7),
    ("idx_scatter_view_chain",      _tpl_idx_scatter_view_chain,      7),
    ("idx_expand_inplace_pattern",  _tpl_idx_expand_inplace_pattern,  7),
])

# ---------------------------------------------------------------------------
# Diamond/Branch graph templates — fan-out from shared input, then merge.
# Stresses fusion scheduling at the merge point: inductor must decide how to
# handle a node consumed by multiple downstream paths.
# ---------------------------------------------------------------------------

def _tpl_d4_diamond_reduce_merge(B, S, H, dtype):
    """Diamond: x splits into sum-path and mean-path, merged by add.
    Fusion must handle the shared input correctly."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    path_a = x.sum(dim=-1, keepdim=True)\n"
        f"    path_b = x.mean(dim=-1, keepdim=True)\n"
        f"    return (x + path_a * path_b).sum()\n"
    ), True

def _tpl_d4_diamond_matmul_act(B, S, H, dtype):
    """Diamond: shared x goes through two different linear projections,
    then element-wise gating merge. Classic GLU/SwiGLU structure."""
    h2 = max(8, H // 2)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_w1 = torch.randn({H}, {h2}, dtype={dtype}) * {H ** -0.5:.6f}\n"
        f"_w2 = torch.randn({H}, {h2}, dtype={dtype}) * {H ** -0.5:.6f}\n"
        f"_program_inputs = (_x, _w1, _w2)\n\n"
        f"def program(x, w1, w2):\n"
        f"    gate = torch.sigmoid(x @ w1)\n"
        f"    value = x @ w2\n"
        f"    return (gate * value).sum(dim=-1)\n"
    ), True

def _tpl_d4_diamond_norm_residual(B, S, H, dtype):
    """Diamond: x used as both residual and layernorm input, then merged.
    The residual add after norm is a critical fusion pattern."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_w = torch.ones({H}, dtype={dtype})\n"
        f"_b = torch.zeros({H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    normed = torch.nn.functional.layer_norm(x, [{H}], w, b)\n"
        f"    return (x + normed).sum(dim=-1)\n"
    ), True

def _tpl_d4_diamond_multi_reduce(B, S, H, dtype):
    """Three-way fan-out from one tensor: different reductions, then combine.
    Tests fusion with 3 consumers of a single buffer."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    a = x.amax(dim=-1, keepdim=True)\n"
        f"    b = x.amin(dim=-1, keepdim=True)\n"
        f"    c = x.mean(dim=-1, keepdim=True)\n"
        f"    scale = (a - b).clamp(min=1e-6)\n"
        f"    return ((x - c) / scale).sum()\n"
    ), True

def _tpl_d4_diamond_view_split(B, S, H, dtype):
    """Split tensor along hidden dim, process each half differently, concat back.
    Tests fusion across chunk + cat with different intermediate ops."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    a, b = x.chunk(2, dim=-1)\n"
        f"    a = torch.relu(a)\n"
        f"    b = torch.sigmoid(b)\n"
        f"    return torch.cat([a, b], dim=-1).sum(dim=-1)\n"
    ), True

def _tpl_d4_diamond_inplace_alias(B, S, H, dtype):
    """Diamond with view aliasing: x and x[:, ::2] share storage.
    Inductor must handle buffer aliasing at the merge."""
    S2 = max(4, S // 2) * 2  # ensure even
    return (
        f"_x = torch.randn({B}, {S2}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    full_sum = x.sum(dim=-1)\n"
        f"    strided = x[:, ::2, :].mean(dim=-1)\n"
        f"    return full_sum + strided\n"
    ), True

PROGRAM_TEMPLATES.extend([
    ("d4_diamond_reduce_merge",    _tpl_d4_diamond_reduce_merge,    6),
    ("d4_diamond_matmul_act",      _tpl_d4_diamond_matmul_act,      6),
    ("d4_diamond_norm_residual",   _tpl_d4_diamond_norm_residual,   6),
    ("d4_diamond_multi_reduce",    _tpl_d4_diamond_multi_reduce,    6),
    ("d4_diamond_view_split",      _tpl_d4_diamond_view_split,      6),
    ("d4_diamond_inplace_alias",   _tpl_d4_diamond_inplace_alias,   6),
])

# ---------------------------------------------------------------------------
# Backward-focused micro-templates
# One differentiable op + .sum() — systematic backward correctness testing.
# These target ops with complex backward decompositions (index manipulation,
# shape changes, in-place semantics) where inductor is most likely to have bugs.
# ---------------------------------------------------------------------------

def _tpl_bwd_slice_scatter(B, S, H, dtype):
    s_src = max(1, S // 3)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_y = torch.randn({B}, {s_src}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y)\n"
        f"def program(x, y):\n"
        f"    return torch.slice_scatter(x, y, dim=1, start=0, end={s_src}).sum()\n"
    ), True

def _tpl_bwd_select_scatter(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_src = torch.randn({B}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _src)\n"
        f"def program(x, src):\n"
        f"    return x.select_scatter(src, 1, 0).sum()\n"
    ), True

def _tpl_bwd_index_select(B, S, H, dtype):
    n_idx = max(1, S // 4)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {S}, ({n_idx},))\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return torch.index_select(x, 1, _idx.to(x.device)).sum()\n"
    ), True

def _tpl_bwd_gather(B, S, H, dtype):
    n_idx = max(1, S // 2)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {S}, ({B}, {n_idx}, {H}))\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return torch.gather(x, 1, _idx.to(x.device)).sum()\n"
    ), True

def _tpl_bwd_scatter_add(B, S, H, dtype):
    n_src = max(1, S // 3)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_src = torch.randn({B}, {n_src}, {H}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {S}, ({B}, {n_src}, {H}))\n"
        f"_program_inputs = (_x, _src)\n"
        f"def program(x, src):\n"
        f"    return x.scatter_add(1, _idx.to(x.device), src).sum()\n"
    ), True

def _tpl_bwd_unfold(B, S, H, dtype):
    window = min(4, S)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.unfold(1, {window}, 1).sum()\n"
    ), True

def _tpl_bwd_repeat(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.repeat(1, 2, 1).sum()\n"
    ), True

def _tpl_bwd_expand(B, S, H, dtype):
    return (
        f"_x = torch.randn(1, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.expand({B}, -1, -1).sum()\n"
    ), True

def _tpl_bwd_chunk_cat(B, S, H, dtype):
    n_chunks = min(3, S)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    parts = x.chunk({n_chunks}, dim=1)\n"
        f"    return torch.cat([p * (i+1) for i, p in enumerate(parts)], dim=1).sum()\n"
    ), True

def _tpl_bwd_diagonal(B, S, H, dtype):
    dim = min(S, H)
    return (
        f"_x = torch.randn({B}, {dim}, {dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.diagonal(dim1=1, dim2=2).sum()\n"
    ), True

def _tpl_bwd_flip(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return torch.flip(x, [1]).sum()\n"
    ), True

def _tpl_bwd_roll(B, S, H, dtype):
    shift = max(1, S // 4)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return torch.roll(x, {shift}, 1).sum()\n"
    ), True

def _tpl_bwd_narrow(B, S, H, dtype):
    length = max(1, S // 2)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.narrow(1, 0, {length}).sum()\n"
    ), True

def _tpl_bwd_where(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_y = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y)\n"
        f"def program(x, y):\n"
        f"    return torch.where(x > 0, x, y).sum()\n"
    ), True

def _tpl_bwd_masked_fill(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    mask = x > 0\n"
        f"    return x.masked_fill(mask, 0.0).sum()\n"
    ), True

def _tpl_bwd_index_copy(B, S, H, dtype):
    n_idx = max(1, S // 4)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_src = torch.randn({B}, {n_idx}, {H}, dtype={dtype})\n"
        f"_idx = torch.randperm({S})[:{n_idx}]\n"
        f"_program_inputs = (_x, _src)\n"
        f"def program(x, src):\n"
        f"    return x.index_copy(1, _idx.to(x.device), src).sum()\n"
    ), True

def _tpl_bwd_scatter_reduce(B, S, H, dtype):
    n_src = max(1, S // 3)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_src = torch.randn({B}, {n_src}, {H}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {S}, ({B}, {n_src}, {H}))\n"
        f"_program_inputs = (_x, _src)\n"
        f"def program(x, src):\n"
        f"    return x.scatter_reduce(1, _idx.to(x.device), src, reduce='sum').sum()\n"
    ), True

def _tpl_bwd_grid_sample(B, S, H, dtype):
    C = min(16, H)
    return (
        f"_x = torch.randn({B}, {C}, {S}, {S}, dtype={dtype})\n"
        f"_grid = torch.randn({B}, {S}, {S}, 2, dtype={dtype}) * 0.5\n"
        f"_program_inputs = (_x, _grid)\n"
        f"def program(x, grid):\n"
        f"    return torch.nn.functional.grid_sample(x, grid, align_corners=False).sum()\n"
    ), True

# ---------------------------------------------------------------------------
# D5A: Alias / view / stride backward templates
# These target inductor's aliasing/view/stride handling in backward passes —
# overlapping writes, stride-0 patterns, shared-storage grad accumulation.
# ---------------------------------------------------------------------------

def _tpl_d5a_view_sum_backward(B, S, H, dtype):
    """View reshapes storage layout; backward must respect original strides."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.view({B}, {H}, {S}).sin().sum()\n"
    ), True

def _tpl_d5a_overlapping_slice_grad(B, S, H, dtype):
    """Two overlapping slices share storage — backward must accumulate correctly."""
    mid = max(1, S // 2)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    a = x[:, :{mid+1}, :]\n"
        f"    b = x[:, {mid-1}:, :]\n"
        f"    return (a.sum() + b.sum())\n"
    ), True

def _tpl_d5a_expand_grad_accum(B, S, H, dtype):
    """Expand creates stride-0 dim; backward must reduce-scatter correctly."""
    return (
        f"_x = torch.randn({B}, 1, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.expand({B}, {S}, {H}).sin().sum()\n"
    ), True

def _tpl_d5a_as_strided_backward(B, S, H, dtype):
    """as_strided with non-contiguous strides — backward indexing stress test."""
    numel = S * H
    return (
        f"_x = torch.randn({B}, {numel}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return torch.as_strided(x, ({B}, {S}, {H}), (x.stride(0), {H}, 1)).cos().sum()\n"
    ), True

def _tpl_d5a_unflatten_flatten_roundtrip(B, S, H, dtype):
    """unflatten then flatten — backward must invert the reshape correctly."""
    return (
        f"_x = torch.randn({B}, {S * H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    y = x.unflatten(1, ({S}, {H}))\n"
        f"    return y.flatten(1).sin().sum()\n"
    ), True

def _tpl_d5a_slice_scatter_backward(B, S, H, dtype):
    """slice_scatter in forward — backward goes through scatter_backward."""
    s_src = max(1, S // 3)
    return (
        f"_x = torch.zeros({B}, {S}, {H}, dtype={dtype})\n"
        f"_src = torch.randn({B}, {s_src}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _src)\n"
        f"def program(x, src):\n"
        f"    return x.slice_scatter(src, dim=1, start=0, end={s_src}).sum()\n"
    ), True

def _tpl_d5a_select_scatter_grad(B, S, H, dtype):
    """select_scatter — backward must route grad to the selected slice."""
    return (
        f"_x = torch.zeros({B}, {S}, {H}, dtype={dtype})\n"
        f"_src = torch.randn({B}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _src)\n"
        f"def program(x, src):\n"
        f"    return x.select_scatter(src, dim=1, index=0).sum()\n"
    ), True

def _tpl_d5a_transpose_chain_backward(B, S, H, dtype):
    """Chain of transposes — backward must compose strides correctly."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.transpose(0, 1).transpose(1, 2).contiguous().sin().sum()\n"
    ), True

def _tpl_d5a_diagonal_scatter_backward(B, S, H, dtype):
    """diagonal view — backward must scatter grad to the diagonal positions."""
    dim = min(S, H)
    return (
        f"_x = torch.randn({B}, {dim}, {dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.diagonal(dim1=1, dim2=2).sin().sum()\n"
    ), True

def _tpl_d5a_narrow_expand_backward(B, S, H, dtype):
    """narrow then expand — backward must handle the combined view."""
    narrow_len = max(1, S // 2)
    return (
        f"_x = torch.randn({B}, {S}, 1, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.narrow(1, 0, {narrow_len}).expand({B}, {narrow_len}, {H}).sum()\n"
    ), True

def _tpl_d5a_permute_noncontig_matmul_bwd(B, S, H, dtype):
    """permute creates non-contiguous input to matmul — backward stress."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_y = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y)\n"
        f"def program(x, y):\n"
        f"    return torch.bmm(x.permute(0, 2, 1), y).sum()\n"
    ), True

def _tpl_d5a_split_interleave_backward(B, S, H, dtype):
    """split + cat interleave — backward must route grads to correct chunks."""
    half = max(1, H // 2)
    return (
        f"_x = torch.randn({B}, {S}, {half * 2}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    a, b = x.split({half}, dim=2)\n"
        f"    return torch.cat([b, a], dim=2).sin().sum()\n"
    ), True

def _tpl_d5a_repeat_interleave_backward(B, S, H, dtype):
    """repeat_interleave — backward must handle the expanded dim correctly."""
    return (
        f"_x = torch.randn({B}, {S}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.repeat_interleave(3, dim=1).sin().sum()\n"
    ), True

def _tpl_d5a_unfold_overlap_backward(B, S, H, dtype):
    """unfold creates overlapping windows — backward must accumulate overlaps."""
    kernel_size = min(3, S)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    return x.unfold(1, {kernel_size}, 1).sum()\n"
    ), True

def _tpl_d5a_multi_view_grad_accum(B, S, H, dtype):
    """Multiple views of same tensor used in forward — grad must accumulate from all."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n"
        f"def program(x):\n"
        f"    a = x[:, :, :].sin()\n"
        f"    b = x.reshape({B}, {S * H}).cos()\n"
        f"    return a.sum() + b.sum()\n"
    ), True


# Weight=10: higher than normal to ensure backward coverage
PROGRAM_TEMPLATES.extend([
    ("bwd_slice_scatter",   _tpl_bwd_slice_scatter,    10),
    ("bwd_select_scatter",  _tpl_bwd_select_scatter,   10),
    ("bwd_index_select",    _tpl_bwd_index_select,     10),
    ("bwd_gather",          _tpl_bwd_gather,           10),
    ("bwd_scatter_add",     _tpl_bwd_scatter_add,      10),
    ("bwd_unfold",          _tpl_bwd_unfold,           10),
    ("bwd_repeat",          _tpl_bwd_repeat,           10),
    ("bwd_expand",          _tpl_bwd_expand,           10),
    ("bwd_chunk_cat",       _tpl_bwd_chunk_cat,        10),
    ("bwd_diagonal",        _tpl_bwd_diagonal,         10),
    ("bwd_flip",            _tpl_bwd_flip,             10),
    ("bwd_roll",            _tpl_bwd_roll,             10),
    ("bwd_narrow",          _tpl_bwd_narrow,           10),
    ("bwd_where",           _tpl_bwd_where,            10),
    ("bwd_masked_fill",     _tpl_bwd_masked_fill,      10),
    ("bwd_index_copy",      _tpl_bwd_index_copy,       10),
    ("bwd_scatter_reduce",  _tpl_bwd_scatter_reduce,   10),
    ("bwd_grid_sample",     _tpl_bwd_grid_sample,      10),
])

# D5A: Alias / view / stride backward templates (weight=10)
PROGRAM_TEMPLATES.extend([
    ("d5a_view_sum_backward",           _tpl_d5a_view_sum_backward,           10),
    ("d5a_overlapping_slice_grad",      _tpl_d5a_overlapping_slice_grad,      10),
    ("d5a_expand_grad_accum",           _tpl_d5a_expand_grad_accum,           10),
    ("d5a_as_strided_backward",         _tpl_d5a_as_strided_backward,         10),
    ("d5a_unflatten_flatten_roundtrip", _tpl_d5a_unflatten_flatten_roundtrip, 10),
    ("d5a_slice_scatter_backward",      _tpl_d5a_slice_scatter_backward,      10),
    ("d5a_select_scatter_grad",         _tpl_d5a_select_scatter_grad,         10),
    ("d5a_transpose_chain_backward",    _tpl_d5a_transpose_chain_backward,    10),
    ("d5a_diagonal_scatter_backward",   _tpl_d5a_diagonal_scatter_backward,   10),
    ("d5a_narrow_expand_backward",      _tpl_d5a_narrow_expand_backward,      10),
    ("d5a_permute_noncontig_matmul_bwd",_tpl_d5a_permute_noncontig_matmul_bwd,10),
    ("d5a_split_interleave_backward",   _tpl_d5a_split_interleave_backward,   10),
    ("d5a_repeat_interleave_backward",  _tpl_d5a_repeat_interleave_backward,  10),
    ("d5a_unfold_overlap_backward",     _tpl_d5a_unfold_overlap_backward,     10),
    ("d5a_multi_view_grad_accum",       _tpl_d5a_multi_view_grad_accum,       10),
])

# ---------------------------------------------------------------------------
# D15: PyTorch 2.12-2.13 new decompositions and codegen changes
# ---------------------------------------------------------------------------

def _tpl_d15_associative_scan(B, S, H, dtype):
    """torch.associative_scan — new higher-order op in 2.12+"""
    dim = random.choice([0, 1])
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    def combine_fn(a, b):\n"
        f"        return a + b\n"
        f"    return torch._higher_order_ops.associative_scan(combine_fn, x, dim={dim})\n"
    ), True

def _tpl_d15_scan_mul(B, S, H, dtype):
    """associative_scan with multiplicative combine — tests different accumulation"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}).clamp(-2, 2)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    def combine_fn(a, b):\n"
        f"        return a * b\n"
        f"    return torch._higher_order_ops.associative_scan(combine_fn, x, dim=1)\n"
    ), True

def _tpl_d15_while_loop(B, S, H, dtype):
    """torch.while_loop — control flow higher-order op"""
    max_iter = random.choice([3, 5, 8])
    return (
        f"_x = torch.randn({B}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    def cond_fn(i, x):\n"
        f"        return i < {max_iter}\n"
        f"    def body_fn(i, x):\n"
        f"        return i + 1, x * 0.9 + 0.1\n"
        f"    init_i = torch.tensor(0, device=x.device)\n"
        f"    _, result = torch.while_loop(cond_fn, body_fn, (init_i, x))\n"
        f"    return result.sum()\n"
    ), True

def _tpl_d15_cond_nested(B, S, H, dtype):
    """torch.cond with nested computation — more complex than d14 version"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_flag = torch.tensor(True, device='cuda')\n"
        f"_program_inputs = (_x, _flag)\n\n"
        f"def program(x, flag):\n"
        f"    def true_fn(x):\n"
        f"        return torch.softmax(x, dim=-1).sum(dim=1)\n"
        f"    def false_fn(x):\n"
        f"        return torch.log_softmax(x, dim=-1).mean(dim=1)\n"
        f"    return torch.cond(flag, true_fn, false_fn, (x,))\n"
    ), True

def _tpl_d15_cooperative_reduce_large(B, S, H, dtype):
    """Large reduction exceeding single-CTA threshold — cooperative/split reduction path"""
    large_dim = random.choice([8192, 16384, 32768])
    return (
        f"_x = torch.randn({B}, {large_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return x.softmax(dim=-1).sum(dim=0)\n"
    ), True

def _tpl_d15_scan_cummax_chain(B, S, H, dtype):
    """cummax followed by operations — scan kernel + pointwise fusion"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    vals, idxs = torch.cummax(x, dim=1)\n"
        f"    return (vals * x).sum() + idxs.float().mean()\n"
    ), True

def _tpl_d15_flex_attention_score_mod(B, S, H, dtype):
    """flex_attention with custom score_mod — recent higher-order attention API"""
    n_heads = random.choice([1, 2, 4])
    head_dim = max(16, H // n_heads)
    return (
        f"_q = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype={dtype})\n"
        f"_k = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype={dtype})\n"
        f"_v = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    def score_mod(score, b, h, q_idx, kv_idx):\n"
        f"        return score * 0.5\n"
        f"    return torch.nn.attention.flex_attention(q, k, v, score_mod=score_mod)\n"
    ), True


# ---------------------------------------------------------------------------
# D16: Autocast / AMP path — dtype promotion under autocast context
# ---------------------------------------------------------------------------

def _tpl_d16_autocast_matmul(B, S, H, dtype):
    """matmul under autocast — Inductor must respect autocast dtype decisions"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_w = torch.randn({H}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    with torch.autocast('cuda', dtype=torch.float16):\n"
        f"        y = x @ w\n"
        f"        y = torch.nn.functional.relu(y)\n"
        f"    return y.float().sum()\n"
    ), True

def _tpl_d16_autocast_layernorm(B, S, H, dtype):
    """layernorm under autocast — norm ops have special autocast behavior"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_w = torch.ones({H}, dtype=torch.float32, device='cuda')\n"
        f"_b = torch.zeros({H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    with torch.autocast('cuda', dtype=torch.bfloat16):\n"
        f"        y = torch.nn.functional.layer_norm(x, [{H}], w, b)\n"
        f"        y = y * 2.0 + 1.0\n"
        f"    return y.float().sum()\n"
    ), True

def _tpl_d16_autocast_softmax_matmul(B, S, H, dtype):
    """attention pattern under autocast — mixed precision accumulation"""
    n_heads = random.choice([1, 2, 4])
    head_dim = max(16, H // n_heads)
    return (
        f"_q = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype=torch.float32, device='cuda')\n"
        f"_k = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype=torch.float32, device='cuda')\n"
        f"_v = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    with torch.autocast('cuda', dtype=torch.float16):\n"
        f"        attn = (q @ k.transpose(-2, -1)) / {head_dim ** 0.5:.4f}\n"
        f"        attn = torch.softmax(attn, dim=-1)\n"
        f"        out = attn @ v\n"
        f"    return out.float().sum()\n"
    ), True

def _tpl_d16_autocast_conv_bn(B, S, H, dtype):
    """conv+batchnorm under autocast — BN stays fp32 while conv goes fp16"""
    C_in = random.choice([3, 16, 33])
    C_out = random.choice([16, 32, 64])
    return (
        f"_x = torch.randn({B}, {C_in}, 32, 32, dtype=torch.float32, device='cuda')\n"
        f"_conv_w = torch.randn({C_out}, {C_in}, 3, 3, dtype=torch.float32, device='cuda')\n"
        f"_bn_w = torch.ones({C_out}, dtype=torch.float32, device='cuda')\n"
        f"_bn_b = torch.zeros({C_out}, dtype=torch.float32, device='cuda')\n"
        f"_bn_rm = torch.zeros({C_out}, dtype=torch.float32, device='cuda')\n"
        f"_bn_rv = torch.ones({C_out}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _conv_w, _bn_w, _bn_b, _bn_rm, _bn_rv)\n\n"
        f"def program(x, conv_w, bn_w, bn_b, bn_rm, bn_rv):\n"
        f"    with torch.autocast('cuda', dtype=torch.float16):\n"
        f"        y = torch.nn.functional.conv2d(x, conv_w, padding=1)\n"
        f"        y = torch.nn.functional.batch_norm(y, bn_rm, bn_rv, bn_w, bn_b, training=True)\n"
        f"        y = torch.nn.functional.relu(y)\n"
        f"    return y.float().flatten(1).sum(-1)\n"
    ), True

def _tpl_d16_autocast_mixed_exit(B, S, H, dtype):
    """computation starts in autocast, result used outside — dtype transition"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_w = torch.randn({H}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    with torch.autocast('cuda', dtype=torch.bfloat16):\n"
        f"        y = x @ w\n"
        f"    z = y.float() + x\n"
        f"    return z.sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D17: Compiled autograd — backward through Inductor
# ---------------------------------------------------------------------------

def _tpl_d17_compiled_autograd_linear(B, S, H, dtype):
    """simple linear backward under compiled autograd"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    y = x @ w\n"
        f"    loss = y.sum()\n"
        f"    loss.backward()\n"
        f"    return x.grad.clone().detach(), w.grad.clone().detach()\n"
    ), False

def _tpl_d17_compiled_autograd_norm_chain(B, S, H, dtype):
    """layernorm + matmul backward under compiled autograd"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_gamma = torch.ones({H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x, _w, _gamma)\n\n"
        f"def program(x, w, gamma):\n"
        f"    y = torch.nn.functional.layer_norm(x, [{H}])\n"
        f"    y = y * gamma\n"
        f"    y = y @ w\n"
        f"    loss = y.sum()\n"
        f"    loss.backward()\n"
        f"    return x.grad.clone().detach()\n"
    ), False

def _tpl_d17_compiled_autograd_attention(B, S, H, dtype):
    """attention backward under compiled autograd — complex backward graph"""
    head_dim = max(16, H // 2)
    return (
        f"_q = torch.randn({B}, {S}, {head_dim}, dtype={dtype}, requires_grad=True)\n"
        f"_k = torch.randn({B}, {S}, {head_dim}, dtype={dtype}, requires_grad=True)\n"
        f"_v = torch.randn({B}, {S}, {head_dim}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    attn = (q @ k.transpose(-2, -1)) / {head_dim ** 0.5:.4f}\n"
        f"    attn = torch.softmax(attn, dim=-1)\n"
        f"    out = attn @ v\n"
        f"    loss = out.sum()\n"
        f"    loss.backward()\n"
        f"    return q.grad.clone().detach()\n"
    ), False

def _tpl_d17_compiled_autograd_inplace(B, S, H, dtype):
    """in-place op in forward, backward under compiled autograd"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    y[:, :, :] = torch.relu(y)\n"
        f"    loss = y.sum()\n"
        f"    loss.backward()\n"
        f"    return x.grad.clone().detach()\n"
    ), False


# ---------------------------------------------------------------------------
# D18: Nested tensor — new Inductor codegen path
# ---------------------------------------------------------------------------

def _tpl_d18_nested_linear(B, S, H, dtype):
    """nested tensor through linear — jagged layout codegen"""
    lengths = [random.randint(4, S) for _ in range(B)]
    lengths_str = str(lengths)
    return (
        f"_tensors = [torch.randn(l, {H}, dtype={dtype}, device='cuda') for l in {lengths_str}]\n"
        f"_x = torch.nested.nested_tensor(_tensors, layout=torch.jagged)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cuda')\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    y = x @ w\n"
        f"    return y.values().sum()\n"
    ), True

def _tpl_d18_nested_softmax(B, S, H, dtype):
    """nested tensor softmax — reduction on jagged dim"""
    lengths = [random.randint(4, S) for _ in range(B)]
    lengths_str = str(lengths)
    return (
        f"_tensors = [torch.randn(l, {H}, dtype={dtype}, device='cuda') for l in {lengths_str}]\n"
        f"_x = torch.nested.nested_tensor(_tensors, layout=torch.jagged)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.softmax(x, dim=-1)\n"
        f"    return y.values().sum()\n"
    ), True

def _tpl_d18_nested_norm(B, S, H, dtype):
    """nested tensor with layernorm — norm on jagged tensor"""
    lengths = [random.randint(4, S) for _ in range(B)]
    lengths_str = str(lengths)
    return (
        f"_tensors = [torch.randn(l, {H}, dtype={dtype}, device='cuda') for l in {lengths_str}]\n"
        f"_x = torch.nested.nested_tensor(_tensors, layout=torch.jagged)\n"
        f"_w = torch.ones({H}, dtype={dtype}, device='cuda')\n"
        f"_b = torch.zeros({H}, dtype={dtype}, device='cuda')\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    y = torch.nn.functional.layer_norm(x, [{H}], w, b)\n"
        f"    return y.values().sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D19: Sparse operations — sparse tensor codegen path
# ---------------------------------------------------------------------------

def _tpl_d19_sparse_mm(B, S, H, dtype):
    """sparse @ dense matmul — sparse codegen path in Inductor"""
    nnz = min(S * H // 4, 1024)
    return (
        f"import torch\n"
        f"_indices = torch.stack([\n"
        f"    torch.randint(0, {S}, ({nnz},)),\n"
        f"    torch.randint(0, {H}, ({nnz},))\n"
        f"]).to('cuda')\n"
        f"_values = torch.randn({nnz}, dtype={dtype}, device='cuda')\n"
        f"_sparse = torch.sparse_coo_tensor(_indices, _values, ({S}, {H})).coalesce()\n"
        f"_dense = torch.randn({H}, {H}, dtype={dtype}, device='cuda')\n"
        f"_program_inputs = (_sparse, _dense)\n\n"
        f"def program(s, d):\n"
        f"    return torch.sparse.mm(s, d).sum()\n"
    ), False

def _tpl_d19_sparse_softmax(B, S, H, dtype):
    """sparse softmax — specialized sparse reduction kernel"""
    nnz = min(S * H // 4, 1024)
    return (
        f"import torch\n"
        f"_indices = torch.stack([\n"
        f"    torch.randint(0, {S}, ({nnz},)),\n"
        f"    torch.randint(0, {H}, ({nnz},))\n"
        f"]).to('cuda')\n"
        f"_values = torch.randn({nnz}, dtype={dtype}, device='cuda')\n"
        f"_sparse = torch.sparse_coo_tensor(_indices, _values, ({S}, {H})).coalesce()\n"
        f"_program_inputs = (_sparse,)\n\n"
        f"def program(s):\n"
        f"    return torch.sparse.softmax(s, dim=1).to_dense().sum()\n"
    ), False

def _tpl_d19_sparse_csr_mv(B, S, H, dtype):
    """CSR format sparse matmul — different from COO path"""
    return (
        f"import torch\n"
        f"_dense = torch.randn({S}, {H}, dtype={dtype}, device='cuda')\n"
        f"_mask = torch.rand({S}, {H}, device='cuda') < 0.3\n"
        f"_dense[~_mask] = 0\n"
        f"_csr = _dense.to_sparse_csr()\n"
        f"_vec = torch.randn({H}, 1, dtype={dtype}, device='cuda')\n"
        f"_program_inputs = (_csr, _vec)\n\n"
        f"def program(s, v):\n"
        f"    return (s @ v).to_dense().sum()\n"
    ), False


# ---------------------------------------------------------------------------
# D20: In-graph random ops — determinism under compile
# ---------------------------------------------------------------------------

def _tpl_d20_dropout_train(B, S, H, dtype):
    """dropout in training mode — random mask generation under compile"""
    p = random.choice([0.1, 0.3, 0.5])
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.nn.functional.dropout(x, p={p}, training=True)\n"
        f"    return y.sum()\n"
    ), False

def _tpl_d20_bernoulli_mask(B, S, H, dtype):
    """bernoulli-based masking — random op fused with compute"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mask = torch.bernoulli(torch.full_like(x, 0.5))\n"
        f"    return (x * mask).sum()\n"
    ), False

def _tpl_d20_multinomial_sample(B, S, H, dtype):
    """multinomial sampling — data-dependent random op"""
    return (
        f"_probs = torch.rand({B}, {H}, dtype={dtype}).softmax(dim=-1)\n"
        f"_program_inputs = (_probs,)\n\n"
        f"def program(probs):\n"
        f"    idx = torch.multinomial(probs, num_samples={min(H, 16)}, replacement=True)\n"
        f"    return idx.float().sum()\n"
    ), False


# ---------------------------------------------------------------------------
# D21: Multi-output ops — ops returning tuples/named tuples
# ---------------------------------------------------------------------------

def _tpl_d21_linalg_eigh(B, S, H, dtype):
    """torch.linalg.eigh — symmetric eigendecomposition, multi-output"""
    dim = min(H, 64)
    return (
        f"_x = torch.randn({B}, {dim}, {dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    sym = x + x.transpose(-2, -1)\n"
        f"    eigenvalues, eigenvectors = torch.linalg.eigh(sym)\n"
        f"    return eigenvalues.sum() + eigenvectors.sum()\n"
    ), True

def _tpl_d21_linalg_svd(B, S, H, dtype):
    """torch.linalg.svd — singular value decomposition"""
    m = min(S, 32)
    n = min(H, 32)
    return (
        f"_x = torch.randn({B}, {m}, {n}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    U, S, Vh = torch.linalg.svd(x, full_matrices=False)\n"
        f"    return U.sum() + S.sum() + Vh.sum()\n"
    ), True

def _tpl_d21_linalg_qr(B, S, H, dtype):
    """torch.linalg.qr — QR factorization"""
    m = min(S, 32)
    n = min(H, 32)
    return (
        f"_x = torch.randn({B}, {m}, {n}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    Q, R = torch.linalg.qr(x)\n"
        f"    return Q.sum() + R.abs().sum()\n"
    ), True

def _tpl_d21_lstsq(B, S, H, dtype):
    """torch.linalg.lstsq — least squares, multi-output"""
    m = max(min(S, 32), 8)
    n = min(H, m - 1)
    return (
        f"_A = torch.randn({B}, {m}, {n}, dtype={dtype})\n"
        f"_b = torch.randn({B}, {m}, 1, dtype={dtype})\n"
        f"_program_inputs = (_A, _b)\n\n"
        f"def program(A, b):\n"
        f"    result = torch.linalg.lstsq(A, b)\n"
        f"    return result.solution.sum()\n"
    ), True

def _tpl_d21_topk_values_indices(B, S, H, dtype):
    """topk returning both values and indices — index tensor codegen"""
    k = random.choice([3, 5, 8])
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    values, indices = torch.topk(x, k={min(k, H)}, dim=-1)\n"
        f"    gathered = torch.gather(x, -1, indices)\n"
        f"    return (values * gathered).sum()\n"
    ), True

def _tpl_d21_max_with_indices(B, S, H, dtype):
    """torch.max with dim — returns (values, indices) tuple"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    vals, idxs = torch.max(x, dim=-1)\n"
        f"    return vals.sum() + idxs.float().mean()\n"
    ), True


# ---------------------------------------------------------------------------
# D22: Large tensor / int32 overflow edge cases
# ---------------------------------------------------------------------------

def _tpl_d22_large_numel_pointwise(B, S, H, dtype):
    """tensor with numel near 2^31 — int32 indexing overflow in Triton"""
    large_s = random.choice([4096, 8192])
    large_h = random.choice([4096, 8192])
    return (
        f"_x = torch.randn(1, {large_s}, {large_h}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return (x * 2.0 + 1.0).sum()\n"
    ), True

def _tpl_d22_large_reduction(B, S, H, dtype):
    """very large reduction dim — tests reduction splitting"""
    large_dim = random.choice([65536, 131072])
    return (
        f"_x = torch.randn(4, {large_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return x.sum(dim=-1)\n"
    ), True


# ---------------------------------------------------------------------------
# D23: Convolution algorithm edge cases
# ---------------------------------------------------------------------------

def _tpl_d23_conv_odd_groups(B, S, H, dtype):
    """grouped conv2d with odd group count — non-standard channel split"""
    groups = random.choice([3, 5, 7])
    C_in = groups * random.choice([1, 3, 5])
    C_out = groups * random.choice([2, 4])
    C_per_group = C_in // groups
    return (
        f"_x = torch.randn({B}, {C_in}, 16, 16, dtype={dtype})\n"
        f"_w = torch.randn({C_out}, {C_per_group}, 3, 3, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return torch.nn.functional.conv2d(x, w, padding=1, groups={groups}).sum()\n"
    ), True

def _tpl_d23_conv_asymmetric_kernel(B, S, H, dtype):
    """conv2d with asymmetric kernel size — different codegen from square kernels"""
    C_in = random.choice([3, 16, 33])
    C_out = random.choice([16, 32])
    kh, kw = random.choice([(1, 7), (7, 1), (3, 5), (5, 3)])
    return (
        f"_x = torch.randn({B}, {C_in}, 32, 32, dtype={dtype})\n"
        f"_w = torch.randn({C_out}, {C_in}, {kh}, {kw}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return torch.nn.functional.conv2d(x, w, padding=({kh//2}, {kw//2})).flatten(1).sum(-1)\n"
    ), True

def _tpl_d23_conv3d_small(B, S, H, dtype):
    """3D convolution — less tested codegen path"""
    C_in = random.choice([3, 8, 16])
    C_out = random.choice([8, 16])
    return (
        f"_x = torch.randn({min(B, 2)}, {C_in}, 8, 8, 8, dtype={dtype})\n"
        f"_w = torch.randn({C_out}, {C_in}, 3, 3, 3, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return torch.nn.functional.conv3d(x, w, padding=1).sum()\n"
    ), True

def _tpl_d23_conv_dilated(B, S, H, dtype):
    """dilated conv2d — expanded receptive field with dilation"""
    C_in = random.choice([3, 16, 33])
    C_out = random.choice([16, 32])
    dilation = random.choice([2, 3, 4])
    return (
        f"_x = torch.randn({B}, {C_in}, 32, 32, dtype={dtype})\n"
        f"_w = torch.randn({C_out}, {C_in}, 3, 3, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return torch.nn.functional.conv2d(x, w, padding={dilation}, dilation={dilation}).sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D24: Special math functions — precision edge cases
# ---------------------------------------------------------------------------

def _tpl_d24_bessel_chain(B, S, H, dtype):
    """Bessel functions — special function precision"""
    return (
        f"_x = torch.rand({B}, {S}, {H}, dtype={dtype}) * 10 + 0.1\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.special.i0(x) + torch.special.i1(x)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d24_lgamma_digamma(B, S, H, dtype):
    """lgamma + digamma chain — gamma function precision"""
    return (
        f"_x = torch.rand({B}, {S}, {H}, dtype={dtype}) * 5 + 0.5\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return (torch.lgamma(x) + torch.digamma(x)).sum()\n"
    ), True

def _tpl_d24_erfinv_chain(B, S, H, dtype):
    """erfinv — inverse error function, numerically sensitive"""
    return (
        f"_x = torch.rand({B}, {S}, {H}, dtype={dtype}) * 1.8 - 0.9\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.erfinv(x).sum()\n"
    ), True

def _tpl_d24_sinc_chain(B, S, H, dtype):
    """sinc + sinc derivative — oscillating function precision"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return (torch.sinc(x) * torch.cos(x * 3.14159)).sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D25: torch.export path — different lowering pipeline
# ---------------------------------------------------------------------------

def _tpl_d25_export_simple(B, S, H, dtype):
    """exported program re-compiled — export + compile pipeline"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.nn.functional.gelu(x)\n"
        f"    y = y + x\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d25_export_dynamic_shape(B, S, H, dtype):
    """exported program with dynamic shapes — symbolic shape handling"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.nn.functional.layer_norm(x, [x.shape[-1]]).sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D26: Reduction + broadcast interaction patterns
# ---------------------------------------------------------------------------

def _tpl_d26_reduce_broadcast_mismatch(B, S, H, dtype):
    """reduce then broadcast back with arithmetic — shape mismatch codegen"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    m = x.mean(dim=-1, keepdim=True)\n"
        f"    s = x.std(dim=-1, keepdim=True)\n"
        f"    normed = (x - m) / (s + 1e-5)\n"
        f"    scale = normed.mean(dim=1, keepdim=True)\n"
        f"    return (normed * scale).sum()\n"
    ), True

def _tpl_d26_multi_reduce_reuse(B, S, H, dtype):
    """multiple reductions reusing the same intermediate buffer"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.relu(x)\n"
        f"    a = y.sum(dim=-1)\n"
        f"    b = y.max(dim=-1).values\n"
        f"    c = y.min(dim=-1).values\n"
        f"    return a + b - c\n"
    ), True


# ---------------------------------------------------------------------------
# D27: Comparison and boolean ops under compile
# ---------------------------------------------------------------------------

def _tpl_d27_comparison_chain(B, S, H, dtype):
    """chained comparisons producing boolean tensors fused with compute"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_threshold = torch.randn({H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _threshold)\n\n"
        f"def program(x, threshold):\n"
        f"    mask1 = x > threshold\n"
        f"    mask2 = x < -threshold\n"
        f"    mask = mask1 | mask2\n"
        f"    return torch.where(mask, x.abs(), torch.zeros_like(x)).sum()\n"
    ), True

def _tpl_d27_bool_indexing_compile(B, S, H, dtype):
    """boolean indexing under compile — data-dependent shape"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mask = x.sum(dim=-1) > 0\n"
        f"    selected = x[mask]\n"
        f"    return selected.sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D28: Einsum patterns not covered by D2
# ---------------------------------------------------------------------------

def _tpl_d28_einsum_batch_outer(B, S, H, dtype):
    """batched outer product via einsum — non-standard contraction"""
    return (
        f"_a = torch.randn({B}, {S}, dtype={dtype})\n"
        f"_b = torch.randn({B}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_a, _b)\n\n"
        f"def program(a, b):\n"
        f"    return torch.einsum('bi,bj->bij', a, b).sum()\n"
    ), True

def _tpl_d28_einsum_trace(B, S, H, dtype):
    """batched trace via einsum — diagonal contraction"""
    dim = min(S, H, 32)
    return (
        f"_x = torch.randn({B}, {dim}, {dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.einsum('bii->b', x).sum()\n"
    ), True

def _tpl_d28_einsum_4d(B, S, H, dtype):
    """4D einsum — high-rank contraction"""
    return (
        f"_a = torch.randn({B}, 4, {S}, {min(H, 32)}, dtype={dtype})\n"
        f"_b = torch.randn({B}, 4, {min(H, 32)}, {S}, dtype={dtype})\n"
        f"_program_inputs = (_a, _b)\n\n"
        f"def program(a, b):\n"
        f"    return torch.einsum('bhij,bhjk->bhik', a, b).sum()\n"
    ), True


# ---------------------------------------------------------------------------
# Register new templates
# ---------------------------------------------------------------------------

PROGRAM_TEMPLATES.extend([
    # D15: PyTorch 2.12-2.13 new features (weight=6 — new code, likely buggy)
    ("d15_associative_scan",           _tpl_d15_associative_scan,           6),
    ("d15_scan_mul",                   _tpl_d15_scan_mul,                   6),
    ("d15_while_loop",                 _tpl_d15_while_loop,                 6),
    ("d15_cond_nested",                _tpl_d15_cond_nested,                6),
    ("d15_cooperative_reduce_large",   _tpl_d15_cooperative_reduce_large,   6),
    ("d15_scan_cummax_chain",          _tpl_d15_scan_cummax_chain,          6),
    ("d15_flex_attention_score_mod",   _tpl_d15_flex_attention_score_mod,   6),

    # D16: Autocast / AMP (weight=6)
    ("d16_autocast_matmul",            _tpl_d16_autocast_matmul,            6),
    ("d16_autocast_layernorm",         _tpl_d16_autocast_layernorm,         6),
    ("d16_autocast_softmax_matmul",    _tpl_d16_autocast_softmax_matmul,    6),
    ("d16_autocast_conv_bn",           _tpl_d16_autocast_conv_bn,           6),
    ("d16_autocast_mixed_exit",        _tpl_d16_autocast_mixed_exit,        6),

    # D17: Compiled autograd (weight=8 — very new, high bug potential)
    ("d17_compiled_autograd_linear",   _tpl_d17_compiled_autograd_linear,   8),
    ("d17_compiled_autograd_norm_chain", _tpl_d17_compiled_autograd_norm_chain, 8),
    ("d17_compiled_autograd_attention", _tpl_d17_compiled_autograd_attention, 8),
    ("d17_compiled_autograd_inplace",  _tpl_d17_compiled_autograd_inplace,  8),

    # D18: Nested tensor (weight=6)
    ("d18_nested_linear",              _tpl_d18_nested_linear,              6),
    ("d18_nested_softmax",             _tpl_d18_nested_softmax,             6),
    ("d18_nested_norm",                _tpl_d18_nested_norm,                6),

    # D19: Sparse operations (weight=4 — may not compile)
    ("d19_sparse_mm",                  _tpl_d19_sparse_mm,                  4),
    ("d19_sparse_softmax",             _tpl_d19_sparse_softmax,             4),
    ("d19_sparse_csr_mv",             _tpl_d19_sparse_csr_mv,              4),

    # D20: In-graph random ops (weight=4)
    ("d20_dropout_train",              _tpl_d20_dropout_train,              4),
    ("d20_bernoulli_mask",             _tpl_d20_bernoulli_mask,             4),
    ("d20_multinomial_sample",         _tpl_d20_multinomial_sample,         4),

    # D21: Multi-output ops (weight=5)
    ("d21_linalg_eigh",                _tpl_d21_linalg_eigh,                5),
    ("d21_linalg_svd",                 _tpl_d21_linalg_svd,                 5),
    ("d21_linalg_qr",                  _tpl_d21_linalg_qr,                  5),
    ("d21_lstsq",                      _tpl_d21_lstsq,                      5),
    ("d21_topk_values_indices",        _tpl_d21_topk_values_indices,        5),
    ("d21_max_with_indices",           _tpl_d21_max_with_indices,           5),

    # D22: Large tensor edge cases (weight=4 — memory intensive)
    ("d22_large_numel_pointwise",      _tpl_d22_large_numel_pointwise,      4),
    ("d22_large_reduction",            _tpl_d22_large_reduction,            4),

    # D23: Convolution edge cases (weight=5)
    ("d23_conv_odd_groups",            _tpl_d23_conv_odd_groups,            5),
    ("d23_conv_asymmetric_kernel",     _tpl_d23_conv_asymmetric_kernel,     5),
    ("d23_conv3d_small",               _tpl_d23_conv3d_small,               5),
    ("d23_conv_dilated",               _tpl_d23_conv_dilated,               5),

    # D24: Special math functions (weight=5)
    ("d24_bessel_chain",               _tpl_d24_bessel_chain,               5),
    ("d24_lgamma_digamma",             _tpl_d24_lgamma_digamma,             5),
    ("d24_erfinv_chain",               _tpl_d24_erfinv_chain,               5),
    ("d24_sinc_chain",                 _tpl_d24_sinc_chain,                 5),

    # D25: torch.export path (weight=5)
    ("d25_export_simple",              _tpl_d25_export_simple,              5),
    ("d25_export_dynamic_shape",       _tpl_d25_export_dynamic_shape,       5),

    # D26: Reduction + broadcast (weight=4)
    ("d26_reduce_broadcast_mismatch",  _tpl_d26_reduce_broadcast_mismatch, 4),
    ("d26_multi_reduce_reuse",         _tpl_d26_multi_reduce_reuse,         4),

    # D27: Comparison / boolean (weight=4)
    ("d27_comparison_chain",           _tpl_d27_comparison_chain,           4),
    ("d27_bool_indexing_compile",      _tpl_d27_bool_indexing_compile,      4),

    # D28: Einsum advanced patterns (weight=4)
    ("d28_einsum_batch_outer",         _tpl_d28_einsum_batch_outer,         4),
    ("d28_einsum_trace",               _tpl_d28_einsum_trace,               4),
    ("d28_einsum_4d",                  _tpl_d28_einsum_4d,                  4),
])


# ---------------------------------------------------------------------------
# D29: CPU backend — C++/OpenMP codegen path (completely different from Triton)
# ---------------------------------------------------------------------------

def _tpl_d29_cpu_matmul_relu(B, S, H, dtype):
    """matmul + relu on CPU — C++ codegen path"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cpu')\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cpu')\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return torch.relu(x @ w).sum()\n"
    ), True

def _tpl_d29_cpu_layernorm(B, S, H, dtype):
    """layernorm on CPU — vectorized reduction codegen"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cpu')\n"
        f"_w = torch.ones({H}, dtype={dtype}, device='cpu')\n"
        f"_b = torch.zeros({H}, dtype={dtype}, device='cpu')\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    return torch.nn.functional.layer_norm(x, [{H}], w, b).sum()\n"
    ), True

def _tpl_d29_cpu_softmax_chain(B, S, H, dtype):
    """softmax + pointwise on CPU — reduction + elementwise fusion"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cpu')\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.softmax(x, dim=-1)\n"
        f"    return (y * x).sum()\n"
    ), True

def _tpl_d29_cpu_conv_bn(B, S, H, dtype):
    """conv2d + batchnorm on CPU — different GEMM backend from CUDA"""
    C_in = random.choice([3, 16])
    C_out = random.choice([16, 32])
    return (
        f"_x = torch.randn({B}, {C_in}, 16, 16, dtype={dtype}, device='cpu')\n"
        f"_conv_w = torch.randn({C_out}, {C_in}, 3, 3, dtype={dtype}, device='cpu')\n"
        f"_bn_w = torch.ones({C_out}, dtype={dtype}, device='cpu')\n"
        f"_bn_b = torch.zeros({C_out}, dtype={dtype}, device='cpu')\n"
        f"_bn_rm = torch.zeros({C_out}, dtype={dtype}, device='cpu')\n"
        f"_bn_rv = torch.ones({C_out}, dtype={dtype}, device='cpu')\n"
        f"_program_inputs = (_x, _conv_w, _bn_w, _bn_b, _bn_rm, _bn_rv)\n\n"
        f"def program(x, conv_w, bn_w, bn_b, bn_rm, bn_rv):\n"
        f"    y = torch.nn.functional.conv2d(x, conv_w, padding=1)\n"
        f"    y = torch.nn.functional.batch_norm(y, bn_rm, bn_rv, bn_w, bn_b, training=True)\n"
        f"    return torch.relu(y).sum()\n"
    ), True

def _tpl_d29_cpu_scatter_reduce(B, S, H, dtype):
    """scatter_reduce on CPU — atomic accumulation without CUDA atomics"""
    n_idx = min(S * 2, 64)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cpu')\n"
        f"_idx = torch.randint(0, {S}, ({B}, {n_idx}), device='cpu')\n"
        f"_src = torch.randn({B}, {n_idx}, {H}, dtype={dtype}, device='cpu')\n"
        f"_program_inputs = (_x, _idx, _src)\n\n"
        f"def program(x, idx, src):\n"
        f"    out = x.clone()\n"
        f"    out.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, {H}), src)\n"
        f"    return out.sum()\n"
    ), True

def _tpl_d29_cpu_cumsum(B, S, H, dtype):
    """cumsum on CPU — scan kernel in C++ codegen"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cpu')\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.cumsum(x, dim=1).sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D30: Integer and bitwise operations
# ---------------------------------------------------------------------------

def _tpl_d30_bitwise_and_or(B, S, H, dtype):
    """bitwise AND/OR chain on integer tensors"""
    return (
        f"_a = torch.randint(0, 255, ({B}, {S}, {H}), dtype=torch.int32, device='cuda')\n"
        f"_b = torch.randint(0, 255, ({B}, {S}, {H}), dtype=torch.int32, device='cuda')\n"
        f"_program_inputs = (_a, _b)\n\n"
        f"def program(a, b):\n"
        f"    c = a & b\n"
        f"    d = a | b\n"
        f"    e = c ^ d\n"
        f"    return e.float().sum()\n"
    ), True

def _tpl_d30_shift_ops(B, S, H, dtype):
    """bit shift operations — integer-only codegen"""
    return (
        f"_x = torch.randint(1, 1024, ({B}, {S}, {H}), dtype=torch.int32, device='cuda')\n"
        f"_shift = torch.randint(0, 8, ({B}, {S}, {H}), dtype=torch.int32, device='cuda')\n"
        f"_program_inputs = (_x, _shift)\n\n"
        f"def program(x, shift):\n"
        f"    left = x << shift\n"
        f"    right = x >> shift\n"
        f"    return (left + right).float().sum()\n"
    ), True

def _tpl_d30_int_div_mod(B, S, H, dtype):
    """integer division and modulo — different from float division codegen"""
    return (
        f"_x = torch.randint(1, 1000, ({B}, {S}, {H}), dtype=torch.int64, device='cuda')\n"
        f"_d = torch.randint(1, 50, ({B}, {S}, {H}), dtype=torch.int64, device='cuda')\n"
        f"_program_inputs = (_x, _d)\n\n"
        f"def program(x, d):\n"
        f"    q = torch.div(x, d, rounding_mode='trunc')\n"
        f"    r = x % d\n"
        f"    return (q + r).float().sum()\n"
    ), True

def _tpl_d30_int_comparison_reduce(B, S, H, dtype):
    """integer comparisons producing bool → reduction"""
    return (
        f"_x = torch.randint(-100, 100, ({B}, {S}, {H}), dtype=torch.int32, device='cuda')\n"
        f"_y = torch.randint(-100, 100, ({B}, {S}, {H}), dtype=torch.int32, device='cuda')\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    gt = (x > y).int()\n"
        f"    eq = (x == y).int()\n"
        f"    lt = (x < y).int()\n"
        f"    return (gt + eq * 2 + lt * 3).float().sum()\n"
    ), True

def _tpl_d30_int64_large_values(B, S, H, dtype):
    """int64 with values near 2^53 — precision boundary"""
    return (
        f"_x = torch.randint(2**50, 2**53, ({B}, {S}), dtype=torch.int64, device='cuda')\n"
        f"_y = torch.randint(1, 1000, ({B}, {S}), dtype=torch.int64, device='cuda')\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    return (x + y - x).float().sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D31: Zero-element tensors — classic codegen edge case
# ---------------------------------------------------------------------------

def _tpl_d31_zero_dim_reduce(B, S, H, dtype):
    """tensor with 0-size dim → reduction"""
    zero_dim = random.choice([0, 1, 2])
    shape = [B, S, H]
    shape[zero_dim] = 0
    return (
        f"_x = torch.randn({shape[0]}, {shape[1]}, {shape[2]}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return x.sum()\n"
    ), True

def _tpl_d31_zero_cat(B, S, H, dtype):
    """cat with a zero-size tensor — codegen must handle empty input"""
    return (
        f"_a = torch.randn({B}, 0, {H}, dtype={dtype})\n"
        f"_b = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_a, _b)\n\n"
        f"def program(a, b):\n"
        f"    return torch.cat([a, b], dim=1).sum()\n"
    ), True

def _tpl_d31_zero_matmul(B, S, H, dtype):
    """matmul with zero inner dim — edge case for GEMM kernels"""
    return (
        f"_a = torch.randn({B}, {S}, 0, dtype={dtype})\n"
        f"_b = torch.randn({B}, 0, {H}, dtype={dtype})\n"
        f"_program_inputs = (_a, _b)\n\n"
        f"def program(a, b):\n"
        f"    return (a @ b).sum()\n"
    ), True

def _tpl_d31_zero_conv(B, S, H, dtype):
    """conv2d on zero-batch tensor"""
    C_in, C_out = 3, 16
    return (
        f"_x = torch.randn(0, {C_in}, 16, 16, dtype={dtype})\n"
        f"_w = torch.randn({C_out}, {C_in}, 3, 3, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return torch.nn.functional.conv2d(x, w, padding=1).sum()\n"
    ), True

def _tpl_d31_zero_softmax(B, S, H, dtype):
    """softmax on zero-length dim — NaN/empty handling"""
    return (
        f"_x = torch.randn({B}, 0, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.softmax(x, dim=1).sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D32: Activation checkpointing — recomputation under compile
# ---------------------------------------------------------------------------

def _tpl_d32_checkpoint_linear(B, S, H, dtype):
    """checkpoint wrapping linear layers — recomputation + compile interaction"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_w1 = torch.randn({H}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_w2 = torch.randn({H}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x, _w1, _w2)\n\n"
        f"def program(x, w1, w2):\n"
        f"    def block(x, w):\n"
        f"        return torch.relu(x @ w)\n"
        f"    y = torch.utils.checkpoint.checkpoint(block, x, w1, use_reentrant=False)\n"
        f"    y = torch.utils.checkpoint.checkpoint(block, y, w2, use_reentrant=False)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d32_checkpoint_attention(B, S, H, dtype):
    """checkpoint wrapping attention — complex recomputation graph"""
    head_dim = max(16, H // 2)
    return (
        f"_q = torch.randn({B}, {S}, {head_dim}, dtype={dtype}, requires_grad=True)\n"
        f"_k = torch.randn({B}, {S}, {head_dim}, dtype={dtype}, requires_grad=True)\n"
        f"_v = torch.randn({B}, {S}, {head_dim}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    def attn_fn(q, k, v):\n"
        f"        scores = (q @ k.transpose(-2, -1)) / {head_dim ** 0.5:.4f}\n"
        f"        weights = torch.softmax(scores, dim=-1)\n"
        f"        return weights @ v\n"
        f"    out = torch.utils.checkpoint.checkpoint(attn_fn, q, k, v, use_reentrant=False)\n"
        f"    return out.sum()\n"
    ), True

def _tpl_d32_checkpoint_norm_ffn(B, S, H, dtype):
    """checkpoint wrapping norm + FFN — typical transformer block pattern"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_w1 = torch.randn({H}, {H * 4}, dtype={dtype}, requires_grad=True)\n"
        f"_w2 = torch.randn({H * 4}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x, _w1, _w2)\n\n"
        f"def program(x, w1, w2):\n"
        f"    def ffn_block(x, w1, w2):\n"
        f"        y = torch.nn.functional.layer_norm(x, [{H}])\n"
        f"        y = torch.nn.functional.gelu(y @ w1)\n"
        f"        y = y @ w2\n"
        f"        return x + y\n"
        f"    return torch.utils.checkpoint.checkpoint(ffn_block, x, w1, w2, use_reentrant=False).sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D33: Autograd context interaction — no_grad / inference_mode under compile
# ---------------------------------------------------------------------------

def _tpl_d33_no_grad_partial(B, S, H, dtype):
    """part of computation in no_grad context"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    with torch.no_grad():\n"
        f"        frozen = torch.softmax(x, dim=-1)\n"
        f"    y = frozen.detach() @ w\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d33_inference_mode_compile(B, S, H, dtype):
    """inference_mode context within compiled function"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    with torch.inference_mode():\n"
        f"        y = x @ w\n"
        f"        y = torch.relu(y)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d33_grad_enabled_toggle(B, S, H, dtype):
    """toggle grad enabled/disabled mid-computation"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x * 2\n"
        f"    with torch.no_grad():\n"
        f"        scale = y.abs().mean()\n"
        f"    z = y / (scale.detach() + 1e-5)\n"
        f"    return z.sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D37: Context manager × compile stress — systematic combinations
# ---------------------------------------------------------------------------

def _tpl_d37_autocast_reduction(B, S, H, dtype):
    """autocast wrapping reduction — tests precision of fused reduction under AMP"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    with torch.autocast('cuda', dtype=torch.float16):\n"
        f"        x = torch.softmax(x, dim=-1)\n"
        f"        x = x.sum(dim=-1)\n"
        f"    return x.float().sum()\n"
    ), True

def _tpl_d37_autocast_layernorm_matmul(B, S, H, dtype):
    """autocast layernorm → matmul — epilogue fusion under mixed precision"""
    mid = max(8, H)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_w = torch.randn({H}, {mid}, dtype=torch.float32, device='cuda')\n"
        f"_ln_w = torch.ones({H}, dtype=torch.float32, device='cuda')\n"
        f"_ln_b = torch.zeros({H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _w, _ln_w, _ln_b)\n\n"
        f"def program(x, w, ln_w, ln_b):\n"
        f"    with torch.autocast('cuda', dtype=torch.bfloat16):\n"
        f"        x = torch.nn.functional.layer_norm(x, [{H}], ln_w, ln_b)\n"
        f"        x = x @ w\n"
        f"        x = torch.nn.functional.gelu(x)\n"
        f"    return x.float().sum()\n"
    ), True

def _tpl_d37_nested_no_grad_enable_grad(B, S, H, dtype):
    """no_grad wrapping enable_grad — gradient tracking toggle under compile"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    with torch.no_grad():\n"
        f"        scale = x.abs().mean()\n"
        f"        with torch.enable_grad():\n"
        f"            y = (x / (scale.detach() + 1e-5)) @ w\n"
        f"            y = torch.relu(y)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d37_inference_mode_conv(B, S, H, dtype):
    """inference_mode + conv2d — tests non-matmul paths under inference mode"""
    C = max(4, H // 8)
    return (
        f"_x = torch.randn({B}, {C}, {S}, {S}, dtype=torch.float32, device='cuda')\n"
        f"_w = torch.randn({C}, {C}, 3, 3, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    with torch.inference_mode():\n"
        f"        y = torch.nn.functional.conv2d(x, w, padding=1)\n"
        f"        y = torch.relu(y)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d37_inference_mode_sdpa(B, S, H, dtype):
    """inference_mode + scaled_dot_product_attention"""
    n_heads = random.choice([1, 2, 4])
    head_dim = max(8, H // n_heads)
    return (
        f"_q = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype=torch.float32, device='cuda')\n"
        f"_k = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype=torch.float32, device='cuda')\n"
        f"_v = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_q, _k, _v)\n\n"
        f"def program(q, k, v):\n"
        f"    with torch.inference_mode():\n"
        f"        return torch.nn.functional.scaled_dot_product_attention(q, k, v).sum()\n"
    ), True

def _tpl_d37_autocast_backward(B, S, H, dtype):
    """autocast in forward, backward outside — precision mismatch surface"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype=torch.float32, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    with torch.autocast('cuda', dtype=torch.float16):\n"
        f"        y = x @ w\n"
        f"        y = torch.nn.functional.gelu(y)\n"
        f"    return y.float().sum()\n"
    ), True

def _tpl_d37_autograd_grad_create_graph(B, S, H, dtype):
    """torch.autograd.grad with create_graph=True under compile"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = (x ** 2).sum()\n"
        f"    (grad,) = torch.autograd.grad(y, x, create_graph=True)\n"
        f"    return (grad ** 2).sum()\n"
    ), True

def _tpl_d37_set_grad_enabled_branch(B, S, H, dtype):
    """torch.set_grad_enabled toggling — dynamo must trace this correctly"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cuda')\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    torch.set_grad_enabled(False)\n"
        f"    frozen = torch.softmax(x, dim=-1)\n"
        f"    torch.set_grad_enabled(True)\n"
        f"    y = frozen.detach() @ w\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d37_autocast_cpu_cuda_mix(B, S, H, dtype):
    """autocast with explicit cpu tensor → cuda transfer inside"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_bias = torch.randn({H}, dtype=torch.float32)\n"
        f"_program_inputs = (_x, _bias)\n\n"
        f"def program(x, bias):\n"
        f"    b = bias.to(x.device)\n"
        f"    with torch.autocast('cuda', dtype=torch.float16):\n"
        f"        y = x + b\n"
        f"        y = torch.nn.functional.relu(y)\n"
        f"    return y.float().sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D34: Type promotion edge cases — cross-dtype operations
# ---------------------------------------------------------------------------

def _tpl_d34_int_float_add(B, S, H, dtype):
    """int32 + float32 addition — implicit type promotion"""
    return (
        f"_x = torch.randint(-100, 100, ({B}, {S}, {H}), dtype=torch.int32, device='cuda')\n"
        f"_y = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    return (x + y).sum()\n"
    ), True

def _tpl_d34_bool_float_mul(B, S, H, dtype):
    """bool * float — bool-to-float promotion in pointwise"""
    return (
        f"_mask = torch.rand({B}, {S}, {H}, device='cuda') > 0.5\n"
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_mask, _x)\n\n"
        f"def program(mask, x):\n"
        f"    return (mask * x + mask.float()).sum()\n"
    ), True

def _tpl_d34_bf16_fp16_interaction(B, S, H, dtype):
    """bf16 and fp16 tensors interacting — non-standard promotion"""
    return (
        f"_a = torch.randn({B}, {S}, {H}, dtype=torch.bfloat16, device='cuda')\n"
        f"_b = torch.randn({B}, {S}, {H}, dtype=torch.float16, device='cuda')\n"
        f"_program_inputs = (_a, _b)\n\n"
        f"def program(a, b):\n"
        f"    return (a.float() + b.float()).sum()\n"
    ), True

def _tpl_d34_int64_float64_matmul(B, S, H, dtype):
    """int64 cast to float64 then matmul — promotion chain"""
    dim = min(H, 32)
    return (
        f"_x = torch.randint(-10, 10, ({B}, {S}, {dim}), dtype=torch.int64, device='cuda')\n"
        f"_w = torch.randn({dim}, {dim}, dtype=torch.float64, device='cuda')\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    return (x.double() @ w).sum()\n"
    ), True

def _tpl_d34_complex_real_mul(B, S, H, dtype):
    """complex * real multiplication — complex promotion"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.complex64, device='cuda')\n"
        f"_scale = torch.randn({H}, dtype=torch.float32, device='cuda')\n"
        f"_program_inputs = (_x, _scale)\n\n"
        f"def program(x, scale):\n"
        f"    return (x * scale).abs().sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D35: Out= parameter — pre-allocated output buffers
# ---------------------------------------------------------------------------

def _tpl_d35_add_out(B, S, H, dtype):
    """torch.add with out= parameter"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_y = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_out = torch.empty({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _y, _out)\n\n"
        f"def program(x, y, out):\n"
        f"    torch.add(x, y, out=out)\n"
        f"    return out.sum()\n"
    ), True

def _tpl_d35_mm_out(B, S, H, dtype):
    """torch.mm with out= parameter — GEMM with pre-allocated output"""
    return (
        f"_x = torch.randn({S}, {H}, dtype={dtype})\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype})\n"
        f"_out = torch.empty({S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _w, _out)\n\n"
        f"def program(x, w, out):\n"
        f"    torch.mm(x, w, out=out)\n"
        f"    return out.sum()\n"
    ), True

def _tpl_d35_out_noncontiguous(B, S, H, dtype):
    """out= pointing to non-contiguous storage"""
    return (
        f"_buf = torch.empty({B}, {H}, {S}, dtype={dtype})\n"
        f"_out = _buf.permute(0, 2, 1)\n"
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _out)\n\n"
        f"def program(x, out):\n"
        f"    torch.relu(x, out=out)\n"
        f"    return out.sum()\n"
    ), True

def _tpl_d35_reduce_out(B, S, H, dtype):
    """reduction with out= parameter"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_out = torch.empty({B}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x, _out)\n\n"
        f"def program(x, out):\n"
        f"    torch.sum(x, dim=1, out=out)\n"
        f"    return out.sum()\n"
    ), True


# ---------------------------------------------------------------------------
# D36: GradScaler / mixed-precision training patterns
# ---------------------------------------------------------------------------

def _tpl_d36_gradscaler_step(B, S, H, dtype):
    """GradScaler unscale + inf check pattern under compile"""
    return (
        f"_grad = torch.randn({B}, {H}, dtype=torch.float32, device='cuda') * 1024\n"
        f"_scale = torch.tensor(1024.0, device='cuda')\n"
        f"_program_inputs = (_grad, _scale)\n\n"
        f"def program(grad, scale):\n"
        f"    unscaled = grad / scale\n"
        f"    has_inf = torch.isinf(unscaled).any()\n"
        f"    has_nan = torch.isnan(unscaled).any()\n"
        f"    ok = ~(has_inf | has_nan)\n"
        f"    return torch.where(ok, unscaled.sum(), torch.tensor(0.0, device=grad.device))\n"
    ), True

def _tpl_d36_amp_forward_backward(B, S, H, dtype):
    """autocast forward + manual backward scale — typical AMP training loop"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype=torch.float32, device='cuda', requires_grad=True)\n"
        f"_scale = torch.tensor(256.0, device='cuda')\n"
        f"_program_inputs = (_x, _w, _scale)\n\n"
        f"def program(x, w, scale):\n"
        f"    with torch.autocast('cuda', dtype=torch.float16):\n"
        f"        y = x @ w\n"
        f"        loss = y.float().sum()\n"
        f"    scaled_loss = loss * scale\n"
        f"    scaled_loss.backward()\n"
        f"    grad = x.grad.clone().detach() / scale\n"
        f"    return grad\n"
    ), False


# ---------------------------------------------------------------------------
# Register D29-D36 templates
# ---------------------------------------------------------------------------

PROGRAM_TEMPLATES.extend([
    # D29: CPU backend (weight=5 — completely untested codegen path)
    ("d29_cpu_matmul_relu",        _tpl_d29_cpu_matmul_relu,        5),
    ("d29_cpu_layernorm",          _tpl_d29_cpu_layernorm,          5),
    ("d29_cpu_softmax_chain",      _tpl_d29_cpu_softmax_chain,      5),
    ("d29_cpu_conv_bn",            _tpl_d29_cpu_conv_bn,            5),
    ("d29_cpu_scatter_reduce",     _tpl_d29_cpu_scatter_reduce,     5),
    ("d29_cpu_cumsum",             _tpl_d29_cpu_cumsum,             5),

    # D30: Integer / bitwise (weight=5)
    ("d30_bitwise_and_or",         _tpl_d30_bitwise_and_or,         5),
    ("d30_shift_ops",              _tpl_d30_shift_ops,              5),
    ("d30_int_div_mod",            _tpl_d30_int_div_mod,            5),
    ("d30_int_comparison_reduce",  _tpl_d30_int_comparison_reduce,  5),
    ("d30_int64_large_values",     _tpl_d30_int64_large_values,     5),

    # D31: Zero-element tensors (weight=6 — classic edge case)
    ("d31_zero_dim_reduce",        _tpl_d31_zero_dim_reduce,        6),
    ("d31_zero_cat",               _tpl_d31_zero_cat,               6),
    ("d31_zero_matmul",            _tpl_d31_zero_matmul,            6),
    ("d31_zero_conv",              _tpl_d31_zero_conv,              6),
    ("d31_zero_softmax",           _tpl_d31_zero_softmax,           6),

    # D32: Activation checkpointing (weight=6)
    ("d32_checkpoint_linear",      _tpl_d32_checkpoint_linear,      6),
    ("d32_checkpoint_attention",   _tpl_d32_checkpoint_attention,    6),
    ("d32_checkpoint_norm_ffn",    _tpl_d32_checkpoint_norm_ffn,     6),

    # D33: Autograd context (weight=5)
    ("d33_no_grad_partial",        _tpl_d33_no_grad_partial,        5),
    ("d33_inference_mode_compile", _tpl_d33_inference_mode_compile,  5),
    ("d33_grad_enabled_toggle",    _tpl_d33_grad_enabled_toggle,     5),

    # D37: Context manager × compile stress (weight=6)
    ("d37_autocast_reduction",          _tpl_d37_autocast_reduction,          6),
    ("d37_autocast_layernorm_matmul",   _tpl_d37_autocast_layernorm_matmul,   6),
    ("d37_nested_no_grad_enable_grad",  _tpl_d37_nested_no_grad_enable_grad,  6),
    ("d37_inference_mode_conv",         _tpl_d37_inference_mode_conv,          6),
    ("d37_inference_mode_sdpa",         _tpl_d37_inference_mode_sdpa,          6),
    ("d37_autocast_backward",           _tpl_d37_autocast_backward,           6),
    ("d37_autograd_grad_create_graph",  _tpl_d37_autograd_grad_create_graph,  6),
    ("d37_set_grad_enabled_branch",     _tpl_d37_set_grad_enabled_branch,     6),
    ("d37_autocast_cpu_cuda_mix",       _tpl_d37_autocast_cpu_cuda_mix,       6),

    # D34: Type promotion (weight=5)
    ("d34_int_float_add",          _tpl_d34_int_float_add,          5),
    ("d34_bool_float_mul",         _tpl_d34_bool_float_mul,         5),
    ("d34_bf16_fp16_interaction",  _tpl_d34_bf16_fp16_interaction,  5),
    ("d34_int64_float64_matmul",   _tpl_d34_int64_float64_matmul,  5),
    ("d34_complex_real_mul",       _tpl_d34_complex_real_mul,       5),

    # D35: Out= parameter (weight=5)
    ("d35_add_out",                _tpl_d35_add_out,                5),
    ("d35_mm_out",                 _tpl_d35_mm_out,                 5),
    ("d35_out_noncontiguous",      _tpl_d35_out_noncontiguous,      5),
    ("d35_reduce_out",             _tpl_d35_reduce_out,             5),

    # D36: GradScaler / AMP training (weight=5)
    ("d36_gradscaler_step",        _tpl_d36_gradscaler_step,        5),
    ("d36_amp_forward_backward",   _tpl_d36_amp_forward_backward,   5),
])

# ---------------------------------------------------------------------------
# P6 Oracle-targeted templates: designed for cross-fusion and dynamic-reuse oracles
# ---------------------------------------------------------------------------

# --- CF (Cross-Fusion) templates: multi-op chains where fusion vs no-fusion
# produces different intermediate precision or accumulation order ---

def _tpl_cf_long_pointwise_reduce(B, S, H, dtype):
    """5-op pointwise chain → reduction. Fused: all in registers. Unfused: 5 kernel launches
    with memory round-trips that may truncate intermediates."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x * 0.7\n"
        f"    x = x.sin()\n"
        f"    x = x + 0.3\n"
        f"    x = x.cos()\n"
        f"    x = x.exp()\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_cf_mixed_precision_chain(B, S, H, dtype):
    """bf16 input → multiple fp32 intermediate ops → reduction.
    Fused kernel keeps fp32 intermediates in registers; unfused materializes
    each step back to bf16 (the input dtype), losing precision."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.bfloat16)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = x.float()\n"
        f"    x = x * 1.4142135\n"
        f"    x = x.sin()\n"
        f"    x = x * x\n"
        f"    x = x + x.mean(dim=-1, keepdim=True)\n"
        f"    return x.sum()\n"
    ), True

def _tpl_cf_reduction_sandwich(B, S, H, dtype):
    """Reduction → pointwise → reduction: the intermediate materialization
    between two reductions may differ when fused vs unfused."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    s = x.sum(dim=-1, keepdim=True)\n"
        f"    x = x / (s.abs() + 1e-6)\n"
        f"    return x.mean(dim=1)\n"
    ), True

def _tpl_cf_diamond_fuse_merge(B, S, H, dtype):
    """Diamond: shared x → two pointwise paths → merge + reduction.
    Fusion decision at the merge point changes register pressure and may
    alter which path gets fused with the reduction."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    a = x.sin().exp()\n"
        f"    b = x.cos().log1p()\n"
        f"    return (a * b).sum(dim=-1)\n"
    ), True

def _tpl_cf_softmax_epilogue_chain(B, S, H, dtype):
    """softmax (3-op reduction) + 3 pointwise epilogue ops.
    With fusion: epilogue gets folded into softmax kernel.
    Without fusion: separate kernel for each epilogue op."""
    n_heads = random.choice([1, 2, 4])
    head_dim = max(8, H // n_heads)
    return (
        f"_x = torch.randn({B}, {n_heads}, {S}, {head_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    x = torch.softmax(x, dim=-1)\n"
        f"    x = x * 2.0\n"
        f"    x = x.clamp(min=0.01)\n"
        f"    x = x.log()\n"
        f"    return x.sum()\n"
    ), True

def _tpl_cf_matmul_multi_epilogue(B, S, H, dtype):
    """matmul + 4 epilogue ops. With fusion: all epilogue fused into GEMM output.
    Without: separate kernels for each. Accumulation order changes."""
    K = random.choice([16, 32, 64])
    return (
        f"_x = torch.randn({B}, {S}, {K}, dtype={dtype})\n"
        f"_w = torch.randn({K}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x @ _w\n"
        f"    y = y + 0.5\n"
        f"    y = torch.relu(y)\n"
        f"    y = y * torch.sigmoid(y)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_cf_cross_reduce_dependency(B, S, H, dtype):
    """Two reductions on different dims with cross-dependency.
    Fusion may merge them into one kernel with different accumulation."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    row_sum = x.sum(dim=-1, keepdim=True)\n"
        f"    col_sum = x.sum(dim=1, keepdim=True)\n"
        f"    return (x * row_sum * col_sum).sum()\n"
    ), True

def _tpl_cf_where_transcendental_chain(B, S, H, dtype):
    """Conditional + transcendental: where(mask, exp, log). Fusion merges
    the branch with the transcendentals into one kernel."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mask = x > 0\n"
        f"    pos = x.exp()\n"
        f"    neg = (-x).exp()\n"
        f"    y = torch.where(mask, pos, neg)\n"
        f"    return y.mean(dim=-1).sum()\n"
    ), True

# --- DYN (Dynamic-reuse) templates: programs with shapes near critical
# inductor thresholds so that shape perturbation crosses boundaries ---

def _tpl_dyn_reduce_near_persistent_thresh(B, S, H, dtype):
    """Reduction with dim near 2048 (persistent reduction threshold).
    Shape A may use persistent reduction, shape B (±2) may not."""
    red_dim = random.choice([2046, 2047, 2048, 2049, 2050])
    return (
        f"_x = torch.randn({B}, {red_dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return x.softmax(dim=-1).sum()\n"
    ), True

def _tpl_dyn_norm_shape_dependent(B, S, H, dtype):
    """LayerNorm where normalized_shape depends on input dim. Shape change
    affects the normalization denominator through symbolic shapes."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mean = x.mean(dim=-1, keepdim=True)\n"
        f"    var = x.var(dim=-1, keepdim=True, unbiased=False)\n"
        f"    x = (x - mean) / (var + 1e-5).sqrt()\n"
        f"    return x.sum()\n"
    ), True

def _tpl_dyn_tile_boundary_reduce(B, S, H, dtype):
    """Reduction at Triton BLOCK_SIZE boundary. Shape A and shape B on
    different sides of 128/256/512 boundary → different tile masking."""
    boundary = random.choice([128, 256, 512])
    dim = boundary + random.choice([-1, 0, 1])
    return (
        f"_x = torch.randn({B}, {dim}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return x.sum(dim=-1)\n"
    ), True

def _tpl_dyn_reshape_chain(B, S, H, dtype):
    """Reshape depends on input shape — symbolic shape expressions
    must stay correct across different concrete values."""
    total = S * H
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.reshape(x.shape[0], -1)\n"
        f"    y = y.sin()\n"
        f"    y = y.reshape(x.shape[0], x.shape[1], x.shape[2])\n"
        f"    return y.sum()\n"
    ), True

def _tpl_dyn_pad_conv_shape(B, S, H, dtype):
    """Conv1d where input length changes → different padding in Triton.
    Symbolic shape must correctly propagate through conv math."""
    C = random.choice([4, 8, 16])
    K = random.choice([3, 5, 7])
    pad = K // 2
    return (
        f"_x = torch.randn({B}, {C}, {S}, dtype={dtype})\n"
        f"_w = torch.randn({C}, {C}, {K}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.nn.functional.conv1d(x, _w, padding={pad}).sum()\n"
    ), True

def _tpl_dyn_pooling_shape_dependent(B, S, H, dtype):
    """Adaptive avg pool — output size is fixed but input varies.
    Inductor's pooling kernel calculates stride from input/output ratio."""
    out_size = random.choice([4, 8, 16])
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return torch.nn.functional.adaptive_avg_pool1d(\n"
        f"        x.permute(0, 2, 1), {out_size}).sum()\n"
    ), True

def _tpl_dyn_split_cat_variable(B, S, H, dtype):
    """Split into chunks based on dim, cat back. Chunk count changes
    when dim changes — symbolic shape must track."""
    chunks = min(4, S)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    parts = torch.chunk(x, min(4, x.shape[1]), dim=1)\n"
        f"    out = torch.cat([p.sin() for p in parts], dim=1)\n"
        f"    return out.sum()\n"
    ), True

def _tpl_dyn_arange_shape_dependent(B, S, H, dtype):
    """Uses arange(input.shape[1]) — symbolic shape flows into arange length,
    which then participates in indexing."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    idx = torch.arange(x.shape[1], device=x.device)\n"
        f"    weights = idx.float() / (x.shape[1] - 1 + 1e-6)\n"
        f"    return (x * weights.unsqueeze(0).unsqueeze(-1)).sum()\n"
    ), True


# =============================================================================
# D38: Dynamic Shape Stress — programs that break when shapes change
# =============================================================================

def _tpl_d38_dynamic_matmul_chain(B, S, H, dtype):
    """Matmul chain under dynamic=True. Symbolic shape must propagate
    through multiple matmuls where inner dims depend on input shape."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w1 = torch.randn({H}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w2 = torch.randn({H}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w1, _w2)\n\n"
        f"def program(x, w1, w2):\n"
        f"    y = x @ w1\n"
        f"    y = torch.relu(y)\n"
        f"    y = y @ w2\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d38_dynamic_split_reduce(B, S, H, dtype):
    """Split+reduce with shape-dependent chunk count under dynamic shapes.
    Guard must track that chunk count changes when seq dim changes."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    n = min(4, x.shape[1])\n"
        f"    parts = x.split(x.shape[1] // n, dim=1)\n"
        f"    return sum(p.sum() for p in parts)\n"
    ), True

def _tpl_d38_dynamic_arange_mask(B, S, H, dtype):
    """Arange+mask with shape-dependent length. Symbolic shape flows
    into arange, then into boolean indexing — tricky for guards."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)\n"
        f"    mask = (pos.unsqueeze(-1) < x.shape[2] // 2).float()\n"
        f"    return (x * mask.unsqueeze(0)).sum()\n"
    ), True

def _tpl_d38_dynamic_pad_to_multiple(B, S, H, dtype):
    """Pad tensor to next multiple of 8. Padding amount is shape-dependent
    — symbolic shape arithmetic must handle modular ops correctly."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    rem = x.shape[-1] % 8\n"
        f"    if rem != 0:\n"
        f"        x = torch.nn.functional.pad(x, (0, 8 - rem))\n"
        f"    return x.sum(dim=-1).sum()\n"
    ), True

def _tpl_d38_dynamic_reshape_matmul(B, S, H, dtype):
    """Reshape that merges batch+seq dims, matmul, reshape back.
    Symbolic product s0*s1 must track through reshape → matmul → reshape."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    b, s, h = x.shape\n"
        f"    y = x.reshape(b * s, h) @ w\n"
        f"    y = y.reshape(b, s, h)\n"
        f"    return y.relu().sum()\n"
    ), True

def _tpl_d38_dynamic_softmax_causal(B, S, H, dtype):
    """Causal attention mask with shape-dependent triu. Mask shape
    tracks symbolic seq dim — different shapes hit different tile configs."""
    S_use = min(S, 64)
    return (
        f"_x = torch.randn({B}, {S_use}, {S_use}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    s = x.shape[1]\n"
        f"    mask = torch.triu(torch.ones(s, s, device=x.device, dtype=x.dtype), diagonal=1) * -1e9\n"
        f"    return torch.softmax(x + mask, dim=-1).sum()\n"
    ), True

# =============================================================================
# D39: Mutation + Functionalization Stress — in-place ops through views
# =============================================================================

def _tpl_d39_view_mutation_chain(B, S, H, dtype):
    """Chain: view → mutation → view → mutation. Each mutation must be
    properly functionalized with correct copy-on-write semantics."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    y[:, :y.shape[1]//2, :].mul_(2.0)\n"
        f"    y[:, y.shape[1]//2:, :].add_(1.0)\n"
        f"    z = y.reshape(-1, {H})\n"
        f"    z[:z.shape[0]//2].mul_(0.5)\n"
        f"    return z.sum()\n"
    ), True

def _tpl_d39_transpose_inplace_reduce(B, S, H, dtype):
    """Transpose creates non-contiguous view, then in-place on it.
    Functionalization must materialize the view before mutation."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    yt = y.transpose(1, 2)\n"
        f"    yt.relu_()\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d39_expand_inplace_overlap(B, S, H, dtype):
    """Expand creates aliased memory, then in-place mutation.
    If not properly functionalized, all expanded copies get mutated."""
    return (
        f"_x = torch.randn({B}, 1, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    expanded = y.expand({B}, {S}, {H})\n"
        f"    result = expanded.clone()\n"
        f"    result[:, 0, :].mul_(0.0)\n"
        f"    return result.sum()\n"
    ), True

def _tpl_d39_slice_scatter_inplace(B, S, H, dtype):
    """slice_scatter followed by in-place — tests whether functionalization
    tracks the slice_scatter output as a new tensor or an alias."""
    half_s = max(1, S // 2)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_y = torch.randn({B}, {half_s}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _y)\n\n"
        f"def program(x, y):\n"
        f"    out = torch.slice_scatter(x, y, dim=1, start=0, end={half_s})\n"
        f"    out = out.clone()\n"
        f"    out[:, {half_s}:, :].relu_()\n"
        f"    return out.sum()\n"
    ), True

def _tpl_d39_narrow_mutation_backward(B, S, H, dtype):
    """Narrow view mutation with backward — gradient through in-place op
    must correctly route to original tensor positions."""
    n = max(1, S // 3)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    part = y.narrow(1, 0, {n})\n"
        f"    part.sigmoid_()\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d39_diagonal_mutation(B, S, H, dtype):
    """Diagonal view + mutation. Diagonal creates a non-contiguous view
    with non-trivial strides — mutation must respect the layout."""
    D = min(S, H)
    return (
        f"_x = torch.randn({B}, {D}, {D}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    diag = torch.diagonal(y, dim1=1, dim2=2)\n"
        f"    diag.fill_(1.0)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d39_multi_output_inplace(B, S, H, dtype):
    """Multiple outputs from same tensor, in-place on one — tests
    whether inductor correctly sequences the reads and writes."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.clone()\n"
        f"    a = y[:, :y.shape[1]//2, :].sum()\n"
        f"    y[:, :y.shape[1]//2, :].mul_(0.0)\n"
        f"    b = y.sum()\n"
        f"    return a + b\n"
    ), True

# =============================================================================
# D40: Compiled Autograd + Higher-Order Gradients
# =============================================================================

def _tpl_d40_double_backward_matmul(B, S, H, dtype):
    """Double backward (create_graph=True) through matmul.
    Tests compiled autograd's handling of second-order derivatives."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    y = (x @ w).sum()\n"
        f"    g, = torch.autograd.grad(y, x, create_graph=True)\n"
        f"    return (g ** 2).sum()\n"
    ), True

def _tpl_d40_double_backward_norm(B, S, H, dtype):
    """Double backward through layer normalization — complex reduction
    backward with create_graph=True."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.ones({H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_b = torch.zeros({H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w, _b)\n\n"
        f"def program(x, w, b):\n"
        f"    y = torch.nn.functional.layer_norm(x, [{H}], w, b).sum()\n"
        f"    gx, gw = torch.autograd.grad(y, (x, w), create_graph=True)\n"
        f"    return (gx ** 2).sum() + (gw ** 2).sum()\n"
    ), True

def _tpl_d40_grad_of_grad_softmax(B, S, H, dtype):
    """Gradient of gradient through softmax — Hessian-vector product style.
    Softmax backward is complex; second derivative adds another layer."""
    S_use = min(S, 64)
    return (
        f"_x = torch.randn({B}, {S_use}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = torch.softmax(x, dim=-1).sum()\n"
        f"    g, = torch.autograd.grad(y, x, create_graph=True)\n"
        f"    return g.sum()\n"
    ), True

def _tpl_d40_vjp_jvp_matmul(B, S, H, dtype):
    """Forward-mode AD (jvp) combined with reverse-mode — tests
    torch.func.jvp under compile."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    def f(x_):\n"
        f"        return (x_ @ w).sum()\n"
        f"    g, = torch.autograd.grad(f(x), x, create_graph=True)\n"
        f"    return (g * x).sum()\n"
    ), True

def _tpl_d40_backward_through_where(B, S, H, dtype):
    """Backward through conditional (where) — tests that autograd correctly
    handles the discontinuous gradient."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    mask = x.detach() > 0\n"
        f"    y = torch.where(mask, x * 2, x * 0.1)\n"
        f"    loss = y.sum()\n"
        f"    g, = torch.autograd.grad(loss, x, create_graph=True)\n"
        f"    return (g ** 2).sum()\n"
    ), True

def _tpl_d40_checkpoint_backward(B, S, H, dtype):
    """Gradient checkpointing (recomputation) under compile.
    Tests that checkpoint + inductor correctly recomputes activations."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    def block(x_):\n"
        f"        return torch.relu(x_ @ w)\n"
        f"    y = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d40_backward_retain_graph(B, S, H, dtype):
    """Multiple backward passes with retain_graph — tests that
    compiled graph handles re-traversal correctly."""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({H}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    y = (x @ w).relu().sum()\n"
        f"    g1, = torch.autograd.grad(y, x, retain_graph=True)\n"
        f"    g2, = torch.autograd.grad(y, w, retain_graph=False)\n"
        f"    return (g1.sum() + g2.sum())\n"
    ), True


PROGRAM_TEMPLATES.extend([
    # CF: Cross-Fusion oracle targets (weight=8 — high priority for new oracle)
    ("cf_long_pointwise_reduce",     _tpl_cf_long_pointwise_reduce,     8),
    ("cf_mixed_precision_chain",     _tpl_cf_mixed_precision_chain,     8),
    ("cf_reduction_sandwich",        _tpl_cf_reduction_sandwich,        8),
    ("cf_diamond_fuse_merge",        _tpl_cf_diamond_fuse_merge,        8),
    ("cf_softmax_epilogue_chain",    _tpl_cf_softmax_epilogue_chain,    8),
    ("cf_matmul_multi_epilogue",     _tpl_cf_matmul_multi_epilogue,     8),
    ("cf_cross_reduce_dependency",   _tpl_cf_cross_reduce_dependency,   8),
    ("cf_where_transcendental_chain",_tpl_cf_where_transcendental_chain,8),

    # DYN: Dynamic-reuse oracle targets (weight=8)
    ("dyn_reduce_near_persistent_thresh", _tpl_dyn_reduce_near_persistent_thresh, 8),
    ("dyn_norm_shape_dependent",     _tpl_dyn_norm_shape_dependent,     8),
    ("dyn_tile_boundary_reduce",     _tpl_dyn_tile_boundary_reduce,     8),
    ("dyn_reshape_chain",            _tpl_dyn_reshape_chain,            8),
    ("dyn_pad_conv_shape",           _tpl_dyn_pad_conv_shape,           8),
    ("dyn_pooling_shape_dependent",  _tpl_dyn_pooling_shape_dependent,  8),
    ("dyn_split_cat_variable",       _tpl_dyn_split_cat_variable,       8),
    ("dyn_arange_shape_dependent",   _tpl_dyn_arange_shape_dependent,   8),

    # D38: Dynamic Shape Stress (weight=8 — force dynamic=True in oracle)
    ("d38_dynamic_matmul_chain",     _tpl_d38_dynamic_matmul_chain,     8),
    ("d38_dynamic_split_reduce",     _tpl_d38_dynamic_split_reduce,     8),
    ("d38_dynamic_arange_mask",      _tpl_d38_dynamic_arange_mask,      8),
    ("d38_dynamic_pad_to_multiple",  _tpl_d38_dynamic_pad_to_multiple,  8),
    ("d38_dynamic_reshape_matmul",   _tpl_d38_dynamic_reshape_matmul,   8),
    ("d38_dynamic_softmax_causal",   _tpl_d38_dynamic_softmax_causal,   8),

    # D39: Mutation + Functionalization Stress (weight=8)
    ("d39_view_mutation_chain",      _tpl_d39_view_mutation_chain,      8),
    ("d39_transpose_inplace_reduce", _tpl_d39_transpose_inplace_reduce, 8),
    ("d39_expand_inplace_overlap",   _tpl_d39_expand_inplace_overlap,   8),
    ("d39_slice_scatter_inplace",    _tpl_d39_slice_scatter_inplace,    8),
    ("d39_narrow_mutation_backward", _tpl_d39_narrow_mutation_backward, 8),
    ("d39_diagonal_mutation",        _tpl_d39_diagonal_mutation,        8),
    ("d39_multi_output_inplace",     _tpl_d39_multi_output_inplace,     8),

    # D40: Compiled Autograd + Higher-Order Gradients (weight=8)
    ("d40_double_backward_matmul",   _tpl_d40_double_backward_matmul,   8),
    ("d40_double_backward_norm",     _tpl_d40_double_backward_norm,     8),
    ("d40_grad_of_grad_softmax",     _tpl_d40_grad_of_grad_softmax,     8),
    ("d40_vjp_jvp_matmul",           _tpl_d40_vjp_jvp_matmul,          8),
    ("d40_backward_through_where",   _tpl_d40_backward_through_where,   8),
    ("d40_checkpoint_backward",      _tpl_d40_checkpoint_backward,      8),
    ("d40_backward_retain_graph",    _tpl_d40_backward_retain_graph,    8),
])

# ---------------------------------------------------------------------------
# D41: Stride / Layout-Sensitive Templates
# ---------------------------------------------------------------------------

def _tpl_d41_transpose_matmul(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_w = torch.randn({S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x, _w)\n\n"
        f"def program(x, w):\n"
        f"    xt = x.transpose(1, 2)\n"
        f"    return torch.matmul(xt, w).sum()\n"
    ), True

def _tpl_d41_permute_reduce(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.permute(0, 2, 1)\n"
        f"    y = y * torch.sigmoid(y)\n"
        f"    return y.sum(dim=-1).sum()\n"
    ), True

def _tpl_d41_expand_reduction(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, 1, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.expand({B}, {S}, {H})\n"
        f"    return (y * y).sum(dim=1).sum()\n"
    ), True

def _tpl_d41_slice_noncontig_reduce(B, S, H, dtype):
    S2 = max(S * 2, 4)
    return (
        f"_x = torch.randn({B}, {S2}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x[:, ::2, :]\n"
        f"    return torch.nn.functional.gelu(y).sum()\n"
    ), True

def _tpl_d41_permute_softmax(B, S, H, dtype):
    nheads = max(1, min(H // 8, 8))
    head_dim = H // nheads if nheads > 0 else H
    H_use = nheads * head_dim
    return (
        f"_x = torch.randn({B}, {S}, {H_use}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    B, S, _ = x.shape\n"
        f"    y = x.view(B, S, {nheads}, {head_dim}).permute(0, 2, 1, 3)\n"
        f"    y = torch.nn.functional.softmax(y, dim=-1)\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d41_strided_chain(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda', requires_grad=True)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.transpose(1, 2)\n"
        f"    y = torch.relu(y)\n"
        f"    y = y.permute(0, 2, 1)\n"
        f"    y = torch.nn.functional.layer_norm(y, [{H}])\n"
        f"    y = y.contiguous()\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d41_as_strided_pointwise(B, S, H, dtype):
    win = min(S - 1, 4) if S > 1 else 1
    out_s = S - win
    if out_s < 1:
        out_s = 1
        win = S - 1 if S > 1 else 0
    return (
        f"_base = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')\n"
        f"_x = torch.as_strided(_base, ({B}, {out_s}, {win + 1}, {H}), "
        f"({S * H}, {H}, {H}, 1))\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    return (x.sin() + x.cos()).sum()\n"
    ), True

PROGRAM_TEMPLATES.extend([
    # D41: Stride / Layout-Sensitive (weight=10)
    ("d41_transpose_matmul",      _tpl_d41_transpose_matmul,      10),
    ("d41_permute_reduce",        _tpl_d41_permute_reduce,        10),
    ("d41_expand_reduction",      _tpl_d41_expand_reduction,      10),
    ("d41_slice_noncontig_reduce",_tpl_d41_slice_noncontig_reduce,10),
    ("d41_permute_softmax",       _tpl_d41_permute_softmax,       10),
    ("d41_strided_chain",         _tpl_d41_strided_chain,         10),
    ("d41_as_strided_pointwise",  _tpl_d41_as_strided_pointwise,  10),
])

# ---------------------------------------------------------------------------
# D42: Complex Indexing + Edge Shapes
# ---------------------------------------------------------------------------

def _tpl_d42_gather_prime_dims(B, S, H, dtype):
    S_p = [p for p in [3, 5, 7, 11, 13, 17] if p <= S][-1] if S >= 3 else S
    H_p = [p for p in [3, 5, 7, 11, 13, 17] if p <= H][-1] if H >= 3 else H
    return (
        f"_x = torch.randn({B}, {S_p}, {H_p}, dtype={dtype}, device='cuda')\n"
        f"_idx = torch.randint(0, {S_p}, ({B}, {S_p}, {H_p}), device='cuda')\n"
        f"_program_inputs = (_x, _idx)\n\n"
        f"def program(x, idx):\n"
        f"    return torch.gather(x, 1, idx).sum()\n"
    ), True

def _tpl_d42_scatter_reduce_dim1(B, S, H, dtype):
    tgt_s = max(S // 2, 1)
    return (
        f"_src = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')\n"
        f"_idx = torch.randint(0, {tgt_s}, ({B}, {S}, {H}), device='cuda')\n"
        f"_program_inputs = (_src, _idx)\n\n"
        f"def program(src, idx):\n"
        f"    tgt = torch.zeros({B}, {tgt_s}, {H}, dtype=src.dtype, device=src.device)\n"
        f"    return tgt.scatter_reduce(1, idx, src, reduce='sum').sum()\n"
    ), True

def _tpl_d42_index_put_bool_mask(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')\n"
        f"_val = torch.tensor(99.0, dtype={dtype}, device='cuda')\n"
        f"_program_inputs = (_x, _val)\n\n"
        f"def program(x, val):\n"
        f"    mask = x > 0\n"
        f"    y = x.clone()\n"
        f"    y[mask] = val\n"
        f"    return y.sum()\n"
    ), True

def _tpl_d42_advanced_int_index_chain(B, S, H, dtype):
    n_idx = min(S, 7)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')\n"
        f"_idx1 = torch.randint(0, {S}, ({n_idx},), device='cuda')\n"
        f"_idx2 = torch.randint(0, {H}, ({min(H, 5)},), device='cuda')\n"
        f"_program_inputs = (_x, _idx1, _idx2)\n\n"
        f"def program(x, idx1, idx2):\n"
        f"    y = x[:, idx1, :]\n"
        f"    z = y[:, :, idx2]\n"
        f"    return z.sum()\n"
    ), True

def _tpl_d42_gather_noncontig(B, S, H, dtype):
    return (
        f"_x = torch.randn({B}, {H}, {S}, dtype={dtype}, device='cuda').transpose(1, 2)\n"
        f"_idx = torch.randint(0, {S}, ({B}, {S}, {H}), device='cuda')\n"
        f"_program_inputs = (_x, _idx)\n\n"
        f"def program(x, idx):\n"
        f"    return torch.gather(x, 1, idx).sum()\n"
    ), True

def _tpl_d42_scatter_add_broadcast(B, S, H, dtype):
    return (
        f"_src = torch.randn(1, {S}, {H}, dtype={dtype}, device='cuda').expand({B}, {S}, {H})\n"
        f"_idx = torch.randint(0, {S}, ({B}, {S}, {H}), device='cuda')\n"
        f"_program_inputs = (_src, _idx)\n\n"
        f"def program(src, idx):\n"
        f"    tgt = torch.zeros({B}, {S}, {H}, dtype=src.dtype, device=src.device)\n"
        f"    return tgt.scatter_add(1, idx, src).sum()\n"
    ), True

def _tpl_d42_index_put_accumulate(B, S, H, dtype):
    n_idx = min(S, 10)
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype}, device='cuda')\n"
        f"_idx = torch.arange({n_idx}, device='cuda')\n"
        f"_vals = torch.randn({B}, {n_idx}, {H}, dtype={dtype}, device='cuda')\n"
        f"_program_inputs = (_x, _idx, _vals)\n\n"
        f"def program(x, idx, vals):\n"
        f"    y = x.clone()\n"
        f"    y[:, idx, :] += vals\n"
        f"    return y.sum()\n"
    ), True

PROGRAM_TEMPLATES.extend([
    # D42: Complex Indexing + Edge Shapes (weight=10)
    ("d42_gather_prime_dims",        _tpl_d42_gather_prime_dims,        10),
    ("d42_scatter_reduce_dim1",      _tpl_d42_scatter_reduce_dim1,      10),
    ("d42_index_put_bool_mask",      _tpl_d42_index_put_bool_mask,      10),
    ("d42_advanced_int_index_chain", _tpl_d42_advanced_int_index_chain, 10),
    ("d42_gather_noncontig",         _tpl_d42_gather_noncontig,         10),
    ("d42_scatter_add_broadcast",    _tpl_d42_scatter_add_broadcast,    10),
    ("d42_index_put_accumulate",     _tpl_d42_index_put_accumulate,     10),
])

# ---------------------------------------------------------------------------
# Real-model subgraph templates (parameterized, shape-diverse)
# ---------------------------------------------------------------------------
try:
    from real_model_templates import REAL_MODEL_TEMPLATES, REAL_MODEL_NO_BACKWARD
    PROGRAM_TEMPLATES.extend(REAL_MODEL_TEMPLATES)
    print(f"  Loaded {len(REAL_MODEL_TEMPLATES)} real-model subgraph templates")
except ImportError:
    REAL_MODEL_NO_BACKWARD = set()

# ---------------------------------------------------------------------------
# Bug-specific templates — ops that triggered confirmed bugs but had no
# template in the registry.  Adding them ensures full bug reachability.
# ---------------------------------------------------------------------------

def _tpl_bs_fractional_max_pool(B, S, H, dtype):
    """fractional_max_pool2d with internal random samples — bug #183249"""
    C = min(16, H)
    hw = max(8, int(S ** 0.5))
    return (
        f"_x = torch.randn({B}, {C}, {hw}, {hw}, dtype={dtype})\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    out, _ = torch.nn.functional.fractional_max_pool2d(\n"
        f"        x, kernel_size=3, output_ratio=(0.5, 0.5), return_indices=True)\n"
        f"    return out.sum()\n"
    ), True

def _tpl_bs_index_add_expanded(B, S, H, dtype):
    """index_add on expanded (stride-0) tensor — bug #183986"""
    n_idx = max(1, B // 2)
    return (
        f"_base = torch.zeros(1, {H}, dtype={dtype})\n"
        f"_src = torch.randn({n_idx}, {H}, dtype={dtype})\n"
        f"_idx = torch.randint(0, {B}, ({n_idx},))\n"
        f"_program_inputs = (_base, _src)\n\n"
        f"def program(base, src):\n"
        f"    x = base.expand({B}, -1).clone()\n"
        f"    return x.index_add(0, _idx.to(x.device), src).sum()\n"
    ), True

def _tpl_bs_float8_cast(B, S, H, dtype):
    """float8 cast — bug #184081 (crashes on compute capability < 8.9)"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype=torch.float32)\n"
        f"_program_inputs = (_x,)\n\n"
        f"def program(x):\n"
        f"    y = x.to(torch.float8_e4m3fn)\n"
        f"    return y.to(torch.float32).sum()\n"
    ), False

def _tpl_bs_div_trunc_float(B, S, H, dtype):
    """float div with rounding_mode='trunc' — bug #184408"""
    return (
        f"_x = torch.randn({B}, {S}, {H}, dtype={dtype})\n"
        f"_d = torch.randn({B}, {S}, {H}, dtype={dtype}).clamp(min=0.1)\n"
        f"_program_inputs = (_x, _d)\n\n"
        f"def program(x, d):\n"
        f"    return torch.div(x, d, rounding_mode='trunc').sum()\n"
    ), True

PROGRAM_TEMPLATES.extend([
    ("bs_fractional_max_pool",  _tpl_bs_fractional_max_pool,  5),
    ("bs_index_add_expanded",   _tpl_bs_index_add_expanded,   5),
    ("bs_float8_cast",          _tpl_bs_float8_cast,          3),
    ("bs_div_trunc_float",      _tpl_bs_div_trunc_float,      5),
])

_TPL_MAP     = {t[0]: t[1] for t in PROGRAM_TEMPLATES}
_TPL_NAMES   = [t[0] for t in PROGRAM_TEMPLATES]
_TPL_WEIGHTS = [t[2] for t in PROGRAM_TEMPLATES]

# Enrichment cluster boost — derived from retrospective analysis.
# Templates in these clusters have 3-5× higher bug rate than baseline (20.3%).
# Cluster A: scatter + reduction on same dim (bug_007 family)
# Cluster B: dtype cast chain (bug_004 family)
# Cluster C: data-dependent shape ops (bug_009 family)
ENRICHED_CLUSTER_BOOST = {
    # Cluster A — scatter + reduction
    "d10_slice_scatter_dtype": 3.0,
    "d8_scatter_along_dim": 2.5,
    # Cluster B — dtype cast chain
    "d5_dtype_cast_bf16_round": 3.0,
    "d5_dtype_cast_half_round": 3.0,
    "d7_precision_cast_emulate": 3.0,
    # Cluster C — data-dependent shape
    "d11_unique_consecutive": 2.5,
    "d12_unique_inverse_compose": 2.5,
    "d12_unique_return_counts": 2.5,
}

# Enriched cluster membership for cross-cluster composition
_ENRICHED_CLUSTERS = {
    "A": {"d10_slice_scatter_dtype", "d8_scatter_along_dim"},
    "B": {"d5_dtype_cast_bf16_round", "d5_dtype_cast_half_round", "d7_precision_cast_emulate"},
    "C": {"d11_unique_consecutive", "d12_unique_inverse_compose", "d12_unique_return_counts"},
}

def _get_enriched_cluster(template_name):
    """Return cluster label ('A','B','C') if template is in an enriched cluster, else None."""
    for label, members in _ENRICHED_CLUSTERS.items():
        if template_name in members:
            return label
    return None

# Group by category for same-category mutation preference
# Note: order matters — d10_*/d11_*/d12_* must come BEFORE d1_* string match
def _tpl_category(name):
    if name.startswith("d42"): return "D42"
    if name.startswith("d41"): return "D41"
    if name.startswith("d40"): return "D40"
    if name.startswith("d39"): return "D39"
    if name.startswith("d38"): return "D38"
    if name.startswith("d37"): return "D37"
    if name.startswith("d36"): return "D36"
    if name.startswith("d35"): return "D35"
    if name.startswith("d34"): return "D34"
    if name.startswith("d33"): return "D33"
    if name.startswith("d32"): return "D32"
    if name.startswith("d31"): return "D31"
    if name.startswith("d30"): return "D30"
    if name.startswith("d29"): return "D29"
    if name.startswith("d28"): return "D28"
    if name.startswith("d27"): return "D27"
    if name.startswith("d26"): return "D26"
    if name.startswith("d25"): return "D25"
    if name.startswith("d24"): return "D24"
    if name.startswith("d23"): return "D23"
    if name.startswith("d22"): return "D22"
    if name.startswith("d21"): return "D21"
    if name.startswith("d20"): return "D20"
    if name.startswith("d19"): return "D19"
    if name.startswith("d18"): return "D18"
    if name.startswith("d17"): return "D17"
    if name.startswith("d16"): return "D16"
    if name.startswith("d15"): return "D15"
    if name.startswith("d13"): return "D13"
    if name.startswith("d10"): return "D10"
    if name.startswith("d11"): return "D11"
    if name.startswith("d12"): return "D12"
    if name.startswith("idx_"): return "IDX"
    if name.startswith("cf_"): return "CF"
    if name.startswith("dyn_"): return "DYN"
    for d in ("d1","d2","d3","d4","d5","d6","d7","d8","d9"):
        if name.startswith(d):
            return d.upper()
    return "UNK"

_TPL_CATEGORIES = {
    cat: [t[0] for t in PROGRAM_TEMPLATES if _tpl_category(t[0]) == cat]
    for cat in [
        "D1","D2","D3","D4","D5","D6","D7","D8","D9",
        "D10","D11","D12","D13","D14","D15","D16","D17","D18",
        "D19","D20","D21","D22","D23","D24","D25","D26","D27","D28",
        "D29","D30","D31","D32","D33","D34","D35","D36",
        "D37","D38","D39","D40","D41","D42",
        "IDX","CF","DYN",
    ]
}
_TPL_CATEGORIES = {k: v for k, v in _TPL_CATEGORIES.items() if v}

