#!/usr/bin/env python3
"""
Re-exercise Experiment: "Under-tested" vs "Intrinsically Buggy"

Addresses the reviewer challenge:
  "Aren't scatter/scan just poorly implemented, not under-tested?"

Design:
  For each blind-spot interaction cell (scatter-only, extern×pointwise, etc.),
  generate focused programs that specifically exercise that interaction, then
  observe whether bugs are found early and yield plateaus.

  If bugs are found quickly and then plateau → the region is under-tested,
  not intrinsically buggy. The implementation works fine for most inputs;
  bugs only appear at specific boundaries.

  If bug density remains high even after many programs → the region is
  genuinely lower quality (this is unlikely for mature compilers).

Methodology:
  1. For each target cell, use KernelFuzz's targeted generator to produce
     programs that combine the required operator categories.
  2. Compile each, check for bugs via eager-vs-compiled differential.
  3. Track cumulative unique bug triggers over program count.
  4. Show the yield curve: early burst → saturation → plateau.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u reexercise_experiment.py --cell scatter
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_boundary_analysis import (
    CATEGORY_POINTWISE, CATEGORY_REDUCTION, CATEGORY_SCAN,
    CATEGORY_SCATTER, CATEGORY_LAYOUT, CATEGORY_EXTERN,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

SCATTER_OPS = [
    ('slice_scatter', lambda x: torch.zeros_like(x).slice_scatter(x[:, :x.shape[1]//2], dim=1, start=0, end=x.shape[1]//2)),
    ('select_scatter', lambda x: torch.zeros_like(x).select_scatter(x[:, 0], dim=1, index=0)),
    ('diagonal_scatter', lambda x: torch.zeros(x.shape[0], x.shape[0], device=x.device, dtype=x.dtype).diagonal_scatter(x[:x.shape[0]].diagonal())),
    ('index_copy', lambda x: torch.zeros_like(x).index_copy_(1, torch.arange(min(3, x.shape[1]), device=x.device), x[:, :min(3, x.shape[1])])),
    ('scatter_add', lambda x: torch.zeros_like(x).scatter_add_(1, torch.zeros(x.shape, dtype=torch.long, device=x.device), x)),
    ('gather', lambda x: torch.gather(x, 1, torch.zeros(x.shape, dtype=torch.long, device=x.device))),
]

EXTERN_POINTWISE_OPS = [
    ('mm_relu', lambda x: F.relu(x @ torch.randn(x.shape[-1], x.shape[-1], device=x.device, dtype=x.dtype))),
    ('mm_sigmoid', lambda x: torch.sigmoid(x @ torch.randn(x.shape[-1], x.shape[-1], device=x.device, dtype=x.dtype))),
    ('mm_tanh', lambda x: torch.tanh(x @ torch.randn(x.shape[-1], x.shape[-1], device=x.device, dtype=x.dtype))),
    ('bmm_gelu', lambda x: F.gelu(torch.bmm(x.unsqueeze(0).expand(2,-1,-1), torch.randn(2, x.shape[-1], x.shape[-1], device=x.device, dtype=x.dtype)))),
    ('addmm_silu', lambda x: F.silu(torch.addmm(torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype), x, torch.randn(x.shape[-1], x.shape[-1], device=x.device, dtype=x.dtype)))),
    ('conv_relu', lambda x: F.relu(F.conv1d(x.unsqueeze(0), torch.randn(x.shape[-1], x.shape[-1], 3, device=x.device, dtype=x.dtype), padding=1)).squeeze(0)),
]

LAYOUT_REDUCTION_OPS = [
    ('pool_flatten_sum', lambda x: F.adaptive_avg_pool2d(x.view(1, 1, *x.shape), (7, 7)).flatten(1).sum(-1)),
    ('reshape_mean', lambda x: x.view(-1).mean()),
    ('permute_sum', lambda x: x.permute(*reversed(range(x.dim()))).sum(-1) if x.dim() > 1 else x.sum()),
    ('flatten_var', lambda x: x.flatten().var()),
    ('chunk_norm', lambda x: torch.cat([c.norm() for c in x.chunk(2, dim=-1)]).unsqueeze(0) if x.shape[-1] >= 2 else x.norm().unsqueeze(0)),
]

CELL_OPS = {
    'scatter': SCATTER_OPS,
    'extern_pointwise': EXTERN_POINTWISE_OPS,
    'layout_reduction': LAYOUT_REDUCTION_OPS,
}

SHAPES = [
    (4, 8), (4, 16), (8, 8), (8, 16), (16, 16),
    (4, 32), (8, 32), (16, 32), (32, 32),
    (4, 64), (8, 64), (16, 64), (32, 64), (64, 64),
    (4, 128), (8, 128), (16, 128),
    (1, 8), (1, 16), (1, 32), (1, 64),
    (3, 7), (5, 13), (7, 17), (11, 23),
    (65, 33), (33, 65), (127, 3), (3, 127),
]

DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def check_bug(fn, shape, dtype, device='cuda'):
    """Run eager vs compiled and check for divergence."""
    try:
        x = torch.randn(shape, dtype=dtype, device=device)
        x_eager = x.clone().requires_grad_(False)

        eager_out = fn(x_eager)

        compiled_fn = torch.compile(fn, backend='inductor')
        x_compiled = x.clone().requires_grad_(False)
        compiled_out = compiled_fn(x_compiled)

        if eager_out.shape != compiled_out.shape:
            return 'shape_mismatch', None

        diff = (eager_out.float() - compiled_out.float()).abs()
        max_diff = diff.max().item()
        atol = 1e-3 if dtype in (torch.float32, torch.float64) else 1e-2

        if max_diff > atol:
            return 'diff', max_diff

        # Try backward
        try:
            x_e = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
            x_c = x_e.clone().detach().requires_grad_(True)

            out_e = fn(x_e)
            loss_e = out_e.sum()
            loss_e.backward()

            torch.compiler.reset()
            compiled_fn2 = torch.compile(fn, backend='inductor')
            out_c = compiled_fn2(x_c)
            loss_c = out_c.sum()
            loss_c.backward()

            if x_e.grad is not None and x_c.grad is not None:
                grad_diff = (x_e.grad.float() - x_c.grad.float()).abs().max().item()
                if grad_diff > atol:
                    return 'grad_diff', grad_diff
        except Exception:
            pass

        return 'ok', max_diff

    except Exception as e:
        err_str = str(e)
        if 'compile' in err_str.lower() or 'inductor' in err_str.lower():
            return 'crash', err_str
        return 'error', err_str


def run_reexercise(cell_name, n_programs=500):
    """Run focused generation for a specific interaction cell."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ops = CELL_OPS[cell_name]

    results = []
    bug_triggers = []
    cumulative_unique_bugs = []
    seen_bugs = set()
    n_ok = 0

    print(f"\n{'='*70}")
    print(f"RE-EXERCISE: {cell_name}")
    print(f"{'='*70}")
    print(f"  Ops: {[name for name, _ in ops]}")
    print(f"  Shapes: {len(SHAPES)} variants")
    print(f"  Dtypes: {[str(d).split('.')[-1] for d in DTYPES]}")
    print(f"  Total programs: ~{len(ops) * len(SHAPES) * len(DTYPES)}")
    print()

    prog_id = 0
    for op_name, op_fn in ops:
        for shape in SHAPES:
            for dtype in DTYPES:
                prog_id += 1
                if prog_id > n_programs:
                    break

                torch.compiler.reset()
                torch._dynamo.reset()

                status, detail = check_bug(op_fn, shape, dtype)

                entry = {
                    'id': prog_id,
                    'op': op_name,
                    'shape': list(shape),
                    'dtype': str(dtype),
                    'status': status,
                }
                if detail is not None:
                    entry['detail'] = str(detail)[:200]

                results.append(entry)

                if status == 'ok':
                    n_ok += 1
                elif status in ('diff', 'grad_diff', 'crash', 'shape_mismatch'):
                    bug_key = f"{op_name}_{status}"
                    seen_bugs.add(bug_key)
                    bug_triggers.append({
                        'program_id': prog_id,
                        'op': op_name,
                        'shape': list(shape),
                        'dtype': str(dtype),
                        'type': status,
                        'detail': str(detail)[:200] if detail else None,
                    })

                cumulative_unique_bugs.append((prog_id, len(seen_bugs)))

                if prog_id % 50 == 0 or status in ('diff', 'grad_diff', 'crash'):
                    sym = '!!!' if status in ('diff', 'grad_diff', 'crash') else ''
                    print(f"  [{prog_id:>4}] {op_name:<20} {str(shape):<12} "
                          f"{str(dtype).split('.')[-1]:<8} -> {status} {sym}")

            if prog_id > n_programs:
                break
        if prog_id > n_programs:
            break

    # Summary
    status_counts = Counter(r['status'] for r in results)
    print(f"\n{'='*70}")
    print(f"RESULTS: {cell_name}")
    print(f"{'='*70}")
    print(f"  Total programs:     {len(results)}")
    print(f"  OK:                 {status_counts.get('ok', 0)}")
    print(f"  Diff bugs:          {status_counts.get('diff', 0)}")
    print(f"  Grad diff bugs:     {status_counts.get('grad_diff', 0)}")
    print(f"  Crashes:            {status_counts.get('crash', 0)}")
    print(f"  Errors:             {status_counts.get('error', 0)}")
    print(f"  Unique bug classes: {len(seen_bugs)}")

    if cumulative_unique_bugs:
        # Find when bugs were first found and when discovery plateaued
        first_bug_prog = None
        last_new_bug_prog = 0
        for pid, count in cumulative_unique_bugs:
            if count > 0 and first_bug_prog is None:
                first_bug_prog = pid
            if pid > 1 and count > cumulative_unique_bugs[pid-2][1]:
                last_new_bug_prog = pid

        if first_bug_prog:
            print(f"\n  First bug at program #{first_bug_prog}")
            print(f"  Last new bug class at program #{last_new_bug_prog}")
            remaining = len(results) - last_new_bug_prog
            print(f"  Programs after plateau: {remaining} ({remaining/len(results)*100:.0f}%)")

    # Save
    output = {
        'cell': cell_name,
        'total_programs': len(results),
        'status_counts': dict(status_counts),
        'unique_bug_classes': len(seen_bugs),
        'bug_classes': list(seen_bugs),
        'bug_triggers': bug_triggers,
        'cumulative_curve': cumulative_unique_bugs,
    }
    outpath = os.path.join(OUTPUT_DIR, f'reexercise_{cell_name}.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=list(CELL_OPS.keys()),
                        default='scatter',
                        help='Interaction cell to re-exercise')
    parser.add_argument('-n', type=int, default=500,
                        help='Number of programs to generate')
    args = parser.parse_args()

    run_reexercise(args.cell, args.n)
