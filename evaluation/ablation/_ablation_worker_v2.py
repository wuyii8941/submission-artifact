#!/usr/bin/env python3
"""
Ablation V2 worker: runs one (mode, seed) combination with full coverage tracking.

Tracks the paper's formalized metrics:
  L1 = KSS coverage: |hit states| / |S|, S=46, s=(f,d,p,n)
  L2 = unique fusion patterns (frozenset of fused ops per Triton kernel)
  L3 = unique kernel hashes (SHA-256 of kernel name)

Outputs JSON with per-iteration timeseries and final statistics.
Saturation: stops when KSS unchanged for `patience` consecutive programs
            (after `min_budget` programs).
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from targeted_generator import generate_targeted_program, enumerate_target_states
from random_generator import generate_random_program
from kernel.state_space import KernelStateSpace

# ---------------------------------------------------------------------------
# Kernel tracing: capture heuristic types + kernel names during compilation
# ---------------------------------------------------------------------------
_heuristics = []
_kernel_names = []
_tracing_installed = False


def _install_tracing():
    global _tracing_installed
    if _tracing_installed:
        return

    # 1. Heuristic tracing (TritonKernel)
    from torch._inductor.codegen.triton import TritonKernel
    _orig_heur = TritonKernel._get_heuristic

    def _traced_heur(self_):
        h = _orig_heur(self_)
        _heuristics.append(h)
        return h
    TritonKernel._get_heuristic = _traced_heur

    # 2. SplitScan heuristic tracing
    try:
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel
        _orig_ss = TritonSplitScanKernel._get_heuristic

        def _traced_ss(self_):
            h = _orig_ss(self_)
            _heuristics.append(h)
            return h
        TritonSplitScanKernel._get_heuristic = _traced_ss
    except Exception:
        pass

    # 3. Kernel name tracing (via wrapper codegen)
    try:
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        _orig_gen = PythonWrapperCodegen.generate_kernel_call

        def _traced_gen(self_, kernel_name, *args, **kwargs):
            _kernel_names.append(str(kernel_name))
            return _orig_gen(self_, kernel_name, *args, **kwargs)
        PythonWrapperCodegen.generate_kernel_call = _traced_gen

        _orig_extern_out = PythonWrapperCodegen.generate_extern_kernel_out
        def _traced_extern_out(self_, node, *args, **kwargs):
            _kernel_names.append(f"extern_out:{str(node)[:200]}")
            return _orig_extern_out(self_, node, *args, **kwargs)
        PythonWrapperCodegen.generate_extern_kernel_out = _traced_extern_out

        _orig_extern_alloc = PythonWrapperCodegen.generate_extern_kernel_alloc
        def _traced_extern_alloc(self_, node, *args, **kwargs):
            _kernel_names.append(f"extern_alloc:{str(node)[:200]}")
            return _orig_extern_alloc(self_, node, *args, **kwargs)
        PythonWrapperCodegen.generate_extern_kernel_alloc = _traced_extern_alloc
    except Exception as e:
        print(f"  [WARN] kernel name tracing unavailable: {e}", file=sys.stderr)

    _tracing_installed = True


def _reset_tracing():
    _heuristics.clear()
    _kernel_names.clear()


def _infer_schedule_families(heuristics, kernel_names):
    """Infer the set of schedule families from runtime tracing data."""
    families = set()
    for h in heuristics:
        families.add(h)
    for name in kernel_names:
        if name.startswith('extern_'):
            kn = name.lower()
            if any(k in kn for k in ['mm', 'gemm', 'addmm', 'bmm', 'linear']):
                families.add('extern_gemm')
            elif 'conv' in kn:
                families.add('extern_conv')
    return families


# ---------------------------------------------------------------------------
# L2 fusion pattern parsing
# ---------------------------------------------------------------------------
def _parse_fusion_pattern(kernel_name):
    """Extract frozenset of fused ops from Triton kernel name.
    e.g. 'triton_poi_fused_relu_sigmoid_0' -> frozenset({'relu', 'sigmoid'})
    """
    m = re.match(r'triton_(\w{3})_fused_(.+?)_(\d+)$', kernel_name)
    if m:
        ops_str = m.group(2)
        ops = set()
        for op in re.split(r'(?<=[a-z0-9])_(?=[a-z])', ops_str):
            op = op.strip('_')
            if op:
                ops.add(op)
        return frozenset(ops)
    return None


# ---------------------------------------------------------------------------
# Differential oracle (self-contained, no dependency on run_targeted_experiment)
# ---------------------------------------------------------------------------
_ENV_ERROR_PATTERNS = [
    "does not support bfloat16 compilation natively",
    "does not support float16 compilation natively",
    "cuda out of memory", "no space left on device",
    "too many blocks in cooperative launch",
]


def _is_trivial_crash(error_msg):
    if not error_msg:
        return False
    msg = error_msg.lower()
    return any(p in msg for p in _ENV_ERROR_PATTERNS)


def execute_program(code, meta, device='cuda'):
    """Execute program through differential oracle. Returns result dict."""
    _reset_tracing()
    result = {
        'eager_ok': False, 'compiled_ok': False,
        'max_diff': 0.0, 'bug_type': None, 'error': None,
        'trivial_crash': False,
    }

    try:
        ns = {'torch': torch}
        exec(code, ns)
        fn = ns['program']
        inputs = ns['_program_inputs']
        inputs = tuple(
            t.to(device) if isinstance(t, torch.Tensor) else t
            for t in inputs
        )

        # Eager
        eager_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        eager_out = fn(*eager_in)
        result['eager_ok'] = True

        # Compiled
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, backend='inductor')
        comp_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        compiled_out = compiled_fn(*comp_in)
        result['compiled_ok'] = True

        # Differential comparison
        if isinstance(eager_out, torch.Tensor) and isinstance(compiled_out, torch.Tensor):
            if eager_out.shape != compiled_out.shape:
                result['bug_type'] = 'shape_mismatch'
                result['max_diff'] = float('inf')
            else:
                diff = (eager_out.float() - compiled_out.float()).abs()
                max_diff = diff.max().item()
                result['max_diff'] = max_diff
                dtype = meta.get('dtype', 'torch.float32')
                atol = 1e-6 if 'float64' in dtype else 1e-3
                if max_diff > 10.0:
                    result['bug_type'] = 'CATASTROPHIC_diff'
                elif max_diff > atol * 10:
                    result['bug_type'] = 'SUSPICIOUS_diff'

    except torch._dynamo.exc.BackendCompilerFailed as e:
        err = str(e)[:300]
        result['error'] = err
        result['bug_type'] = 'compile_crash'
        if _is_trivial_crash(err):
            result['trivial_crash'] = True
    except RuntimeError as e:
        err = str(e)[:300]
        result['error'] = err
        if result['eager_ok'] and not result['compiled_ok']:
            result['bug_type'] = 'compile_crash'
            if _is_trivial_crash(err):
                result['trivial_crash'] = True
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {str(e)[:200]}'

    torch._dynamo.reset()
    return result


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------
def run_mode(mode, seed, max_budget, min_budget, patience, device):
    _install_tracing()

    kss = KernelStateSpace()
    l2_patterns = set()
    l3_hashes = set()

    states = enumerate_target_states()
    bugs = []
    trivial_crashes = 0
    families_seen = set()
    timeseries = []
    total_ok = 0

    last_kss_hits = 0
    last_kss_change_iter = 0
    t0 = time.time()

    final_iter = max_budget
    for i in range(max_budget):
        # Generate program
        if mode == 'targeted':
            state = states[i % len(states)]
            code, meta = generate_targeted_program(
                family=state['family'],
                dtype_str=state['dtype'],
                n_fused_target=state['n_fused'],
                seed=seed + i,
            )
        else:
            code, meta = generate_random_program(seed=seed + i)

        # Execute through oracle
        result = execute_program(code, meta, device)

        if result['eager_ok'] and result['compiled_ok']:
            total_ok += 1

        # --- Record coverage from captured kernel info ---
        dtype = meta.get('dtype', 'torch.float32').replace('torch.', '')

        # L1: KSS via record_from_kernels (uses kernel names + heuristics)
        if _kernel_names:
            kss.record_from_kernels(_kernel_names, dtype, _heuristics)
        else:
            # Fallback: record from heuristics only (n_fused defaults to 1)
            for h in _heuristics:
                kss.record_hit(f'triton_{h[:3]}_fused_op_0', dtype, heuristic=h)

        # Track families (coarse L1) — includes extern families
        families_seen.update(_infer_schedule_families(_heuristics, _kernel_names))

        # L2: fusion patterns
        for name in _kernel_names:
            fp = _parse_fusion_pattern(name)
            if fp:
                l2_patterns.add(fp)

        # L3: kernel hashes
        for name in _kernel_names:
            l3_hashes.add(hashlib.sha256(name.encode()).hexdigest()[:32])

        # Bugs
        if result.get('bug_type'):
            if result.get('trivial_crash'):
                trivial_crashes += 1
            else:
                bugs.append({
                    'idx': i,
                    'bug_type': result['bug_type'],
                    'max_diff': result['max_diff'],
                    'error': (result.get('error') or '')[:100],
                    'meta': {k: str(v) for k, v in meta.items()
                             if k not in ('state',)},
                })

        # Saturation tracking
        current_kss = kss.get_total_hits()
        if current_kss > last_kss_hits:
            last_kss_hits = current_kss
            last_kss_change_iter = i

        # Timeseries (every 10 iterations to reduce output size)
        if i % 10 == 0 or i == 0:
            timeseries.append({
                'iter': i,
                'kss': current_kss,
                'kss_total': kss.get_total_valid(),
                'l2': len(l2_patterns),
                'l3': len(l3_hashes),
                'bugs': len(bugs),
                'trivial': trivial_crashes,
                'families': len(families_seen),
            })

        # Progress
        if (i + 1) % 200 == 0 or (result.get('bug_type') and not result.get('trivial_crash')):
            elapsed = time.time() - t0
            bug_marker = f" BUG:{result['bug_type']}" if (
                result.get('bug_type') and not result.get('trivial_crash')) else ""
            print(f"  [{i+1:5d}/{max_budget}] KSS={current_kss}/{kss.get_total_valid()} "
                  f"L2={len(l2_patterns)} L3={len(l3_hashes)} "
                  f"bugs={len(bugs)} triv={trivial_crashes} "
                  f"t={elapsed:.0f}s{bug_marker}", file=sys.stderr)

        # Saturation check
        if i >= min_budget and (i - last_kss_change_iter) >= patience:
            print(f"  SATURATED at iter {i}: KSS={current_kss} unchanged "
                  f"for {patience} iterations", file=sys.stderr)
            final_iter = i + 1
            # Record final timeseries point
            timeseries.append({
                'iter': i,
                'kss': current_kss,
                'kss_total': kss.get_total_valid(),
                'l2': len(l2_patterns),
                'l3': len(l3_hashes),
                'bugs': len(bugs),
                'trivial': trivial_crashes,
                'families': len(families_seen),
            })
            break

    elapsed = time.time() - t0

    # Final summary
    kss_summary = kss.get_summary()

    output = {
        'mode': mode,
        'seed': seed,
        'budget_used': final_iter,
        'elapsed_sec': round(elapsed, 1),
        'total_ok': total_ok,
        'kss_hits': kss.get_total_hits(),
        'kss_total': kss.get_total_valid(),
        'kss_coverage': round(kss.get_coverage(), 4),
        'l2_count': len(l2_patterns),
        'l3_count': len(l3_hashes),
        'families_seen': sorted(families_seen),
        'n_families': len(families_seen),
        'bugs_found': len(bugs),
        'trivial_crashes': trivial_crashes,
        'bugs': bugs,
        'kss_per_family': {
            fam: {'hits': info['hits'], 'total': info['total']}
            for fam, info in kss_summary['per_family'].items()
        },
        'saturated_at': last_kss_change_iter,
        'timeseries': timeseries,
    }

    # Output JSON to stdout (orchestrator reads this)
    print(json.dumps(output, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['targeted', 'random'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-budget', type=int, default=5000)
    parser.add_argument('--min-budget', type=int, default=500)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_mode(args.mode, args.seed, args.max_budget,
             args.min_budget, args.patience, args.device)
