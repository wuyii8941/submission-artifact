#!/usr/bin/env python3
"""
Large-scale random baseline for fusion frequency distribution.
Generates 1000 random programs, compiles each, and records fusion patterns.
"""

import hashlib
import json
import os
import re
import sys
import glob
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

import torch
import torch._dynamo
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from random_generator import generate_random_program
from targeted_generator import generate_targeted_program, enumerate_target_states

from torch._inductor.codecache import cache_dir
CACHE_DIR = cache_dir()

PREFIX_TO_FAMILY = {
    "poi": "pointwise",
    "per": "persistent_reduction",
    "red": "reduction",
    "hel": "cooperative_reduction",
}


def _get_cache_py_files():
    return set(glob.glob(os.path.join(CACHE_DIR, '**/*.py'), recursive=True))


def _parse_triton_kernel_name(name):
    m = re.match(r'triton_(\w{3})_fused_(.+?)_(\d+)$', name)
    if m:
        prefix = m.group(1)
        ops = set()
        for op in re.split(r'(?<=[a-z0-9])_(?=[a-z])', m.group(2)):
            op = op.strip('_')
            if op:
                ops.add(op)
        return prefix, frozenset(ops)
    return None, None


def extract_trace(program_code):
    before_files = _get_cache_py_files()
    result = {'kernel_names': [], 'extern_calls': [], 'families': [],
              'fusion_patterns': [], 'fusion_ops': [], 'error': None}

    try:
        torch._dynamo.reset()
        local_ns = {'torch': torch}
        exec(program_code, local_ns)
        program_fn = local_ns['program']
        program_inputs = local_ns['_program_inputs']

        program_inputs = tuple(
            t.cuda() if isinstance(t, torch.Tensor) else t
            for t in program_inputs
        )

        compiled_fn = torch.compile(program_fn, backend='inductor')
        inp = tuple(
            t.clone().detach() if isinstance(t, torch.Tensor) else t
            for t in program_inputs
        )
        _ = compiled_fn(*inp)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        after_files = _get_cache_py_files()
        new_files = after_files - before_files

        kernel_names = set()
        extern_calls = set()
        for f in sorted(new_files):
            try:
                with open(f) as fh:
                    code = fh.read()
            except Exception:
                continue
            for km in re.finditer(r'(triton_(?:poi|per|red|hel)_fused_\w+)', code):
                kernel_names.add(km.group(1))
            for em in re.finditer(r'extern_kernels\.(\w+)', code):
                extern_calls.add(em.group(1))

        families = set()
        fusion_patterns = set()
        all_ops = set()
        for kn in kernel_names:
            prefix, fused_ops = _parse_triton_kernel_name(kn)
            if fused_ops:
                fusion_patterns.add(fused_ops)
                all_ops.update(fused_ops)
            if prefix and prefix in PREFIX_TO_FAMILY:
                families.add(PREFIX_TO_FAMILY[prefix])

        for ec in extern_calls:
            if ec in ('mm', 'bmm', 'addmm'):
                families.add('extern_gemm')
            elif ec in ('convolution',):
                families.add('extern_conv')

        result['kernel_names'] = sorted(kernel_names)
        result['extern_calls'] = sorted(extern_calls)
        result['families'] = sorted(families)
        result['fusion_patterns'] = sorted(str(fp) for fp in fusion_patterns)
        result['fusion_ops'] = sorted(all_ops)

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        torch._dynamo.reset()

    return result


def run(n_programs, method, output_dir, seed_base=0):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Method: {method}, N={n_programs}", flush=True)

    traces = []
    n_ok = 0
    n_err = 0
    op_counter = Counter()
    fusion_counter = Counter()

    if method == 'targeted':
        states = enumerate_target_states()

    for i in range(n_programs):
        if method == 'random':
            code, meta = generate_random_program(seed=seed_base + i)
        elif method == 'targeted':
            state = states[i % len(states)]
            code, meta = generate_targeted_program(
                family=state['family'],
                dtype_str=state['dtype'],
                n_fused_target=state['n_fused'],
                seed=seed_base + i,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        trace = extract_trace(code)

        if trace.get('error'):
            n_err += 1
        else:
            n_ok += 1
            for op in trace.get('fusion_ops', []):
                op_counter[op] += 1
            for fp in trace.get('fusion_patterns', []):
                fusion_counter[fp] += 1

        traces.append({
            'idx': i,
            'trace': {k: v for k, v in trace.items() if k != 'fusion_ops'},
            'fusion_ops': trace.get('fusion_ops', []),
        })

        if (i + 1) % 50 == 0 or (i + 1) == n_programs:
            print(f"  [{i+1}/{n_programs}] ok={n_ok} err={n_err} "
                  f"unique_fusions={len(fusion_counter)} unique_ops={len(op_counter)}",
                  flush=True)
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save
    with open(os.path.join(output_dir, f'traces_{method}_1k.json'), 'w') as f:
        json.dump(traces, f, indent=2, default=str)

    summary = {
        'method': method,
        'n_programs': n_programs,
        'n_compiled': n_ok,
        'n_errors': n_err,
        'n_unique_fusions': len(fusion_counter),
        'n_unique_ops': len(op_counter),
        'op_frequency': dict(op_counter.most_common()),
        'top_fusions': dict(fusion_counter.most_common(50)),
    }
    with open(os.path.join(output_dir, f'summary_{method}_1k.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}")
    print(f"  Compiled: {n_ok}/{n_programs}")
    print(f"  Unique fusions: {len(fusion_counter)}")
    print(f"  Unique ops in fusions: {len(op_counter)}")
    print(f"  Top ops: {op_counter.most_common(10)}")

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1000)
    parser.add_argument('-m', '--method', default='random')
    parser.add_argument('-o', '--output', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traces'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args.n, args.method, args.output, args.seed)
