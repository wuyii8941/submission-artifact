#!/usr/bin/env python3
"""
NNSmith baseline worker: runs NNSmith model generation + torch.compile/Inductor
with our differential oracle and KSS tracking.

Matches the ablation V2 protocol:
  - Same oracle (eager vs compiled differential)
  - Same KSS tracking (L1/L2/L3)
  - Same saturation detection (patience-based)
  - Same kernel tracing (heuristic + kernel name)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import random as pyrandom
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
from kernel.state_space import KernelStateSpace

# ---------------------------------------------------------------------------
# Kernel tracing (same as ablation worker)
# ---------------------------------------------------------------------------
_heuristics = []
_kernel_names = []
_tracing_installed = False


def _install_tracing():
    global _tracing_installed
    if _tracing_installed:
        return

    from torch._inductor.codegen.triton import TritonKernel
    _orig_heur = TritonKernel._get_heuristic

    def _traced_heur(self_):
        h = _orig_heur(self_)
        _heuristics.append(h)
        return h
    TritonKernel._get_heuristic = _traced_heur

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

    try:
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        _orig_gen = PythonWrapperCodegen.generate_kernel_call

        def _traced_gen(self_, kernel_name, *args, **kwargs):
            _kernel_names.append(str(kernel_name))
            return _orig_gen(self_, kernel_name, *args, **kwargs)
        PythonWrapperCodegen.generate_kernel_call = _traced_gen
    except Exception as e:
        print(f"  [WARN] kernel name tracing unavailable: {e}", file=sys.stderr)

    _tracing_installed = True


def _reset_tracing():
    _heuristics.clear()
    _kernel_names.clear()


# ---------------------------------------------------------------------------
# L2 fusion pattern parsing
# ---------------------------------------------------------------------------
def _parse_fusion_pattern(kernel_name):
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
# Trivial crash filter
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


# ---------------------------------------------------------------------------
# NNSmith model generation + execution
# ---------------------------------------------------------------------------
def _init_nnsmith(seed):
    """Initialize NNSmith model generation infrastructure."""
    from nnsmith.materialize.torch import TorchModelCUDA
    from nnsmith.narrow_spec import auto_opset
    from nnsmith.backends.torchjit import TorchJITFactory

    factory = TorchJITFactory(target='cuda')
    opset = auto_opset(TorchModelCUDA, factory)

    return {
        'ModelType': TorchModelCUDA,
        'opset': opset,
    }


def _generate_and_execute(nnsmith_ctx, seed_i, device='cuda'):
    """Generate one NNSmith model, run eager vs torch.compile, return result."""
    from nnsmith.graph_gen import model_gen
    from nnsmith.util import set_seed

    _reset_tracing()
    result = {
        'eager_ok': False, 'compiled_ok': False,
        'max_diff': 0.0, 'bug_type': None, 'error': None,
        'trivial_crash': False, 'dtype': 'float32',
    }

    try:
        set_seed(seed_i)
        max_nodes = pyrandom.choice([3, 5, 8, 10, 15])

        gen = model_gen(
            opset=nnsmith_ctx['opset'],
            seed=seed_i,
            max_nodes=max_nodes,
        )
        ir = gen.make_concrete()
        model = nnsmith_ctx['ModelType'].from_gir(ir)
        model.refine_weights()

        net = model.torch_model.eval()

        if model.sat_inputs is not None:
            inputs = {k: v.to(device) for k, v in model.sat_inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in net.get_random_inps().items()}

        # Infer dtype from inputs
        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                result['dtype'] = str(v.dtype).replace('torch.', '')
                break

        # Eager execution
        with torch.no_grad():
            eager_out = net(**inputs)
        result['eager_ok'] = True

        # Compiled execution
        torch._dynamo.reset()
        compiled_net = torch.compile(net, backend='inductor')
        with torch.no_grad():
            compiled_out = compiled_net(**inputs)
        result['compiled_ok'] = True

        # Differential comparison
        if isinstance(eager_out, (tuple, list)):
            eager_list = list(eager_out)
            compiled_list = list(compiled_out)
        else:
            eager_list = [eager_out]
            compiled_list = [compiled_out]

        max_diff = 0.0
        shape_mismatch = False
        for eo, co in zip(eager_list, compiled_list):
            if isinstance(eo, torch.Tensor) and isinstance(co, torch.Tensor):
                if eo.shape != co.shape:
                    shape_mismatch = True
                    break
                d = (eo.float() - co.float()).abs().max().item()
                max_diff = max(max_diff, d)

        result['max_diff'] = max_diff
        if shape_mismatch:
            result['bug_type'] = 'shape_mismatch'
            result['max_diff'] = float('inf')
        elif max_diff > 10.0:
            result['bug_type'] = 'CATASTROPHIC_diff'
        elif max_diff > 1e-3:
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
# Main loop
# ---------------------------------------------------------------------------
def run_nnsmith(seed, max_budget, min_budget, patience, device):
    _install_tracing()

    print(f"  Initializing NNSmith (seed={seed})...", file=sys.stderr)
    nnsmith_ctx = _init_nnsmith(seed)
    print(f"  NNSmith ready: {len(nnsmith_ctx['opset'])} ops", file=sys.stderr)

    kss = KernelStateSpace()
    l2_patterns = set()
    l3_hashes = set()

    bugs = []
    trivial_crashes = 0
    families_seen = set()
    timeseries = []
    total_ok = 0
    gen_failures = 0

    last_kss_hits = 0
    last_kss_change_iter = 0
    t0 = time.time()

    final_iter = max_budget
    for i in range(max_budget):
        seed_i = seed * 100000 + i

        result = _generate_and_execute(nnsmith_ctx, seed_i, device)

        if result.get('error') and not result.get('eager_ok'):
            gen_failures += 1
            _reset_tracing()
            continue

        if result['eager_ok'] and result['compiled_ok']:
            total_ok += 1

        dtype = result.get('dtype', 'float32')

        # L1: KSS
        if _kernel_names:
            kss.record_from_kernels(_kernel_names, dtype, _heuristics)

        for h in _heuristics:
            families_seen.add(h)

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
                    'dtype': dtype,
                })

        # Saturation tracking
        current_kss = kss.get_total_hits()
        if current_kss > last_kss_hits:
            last_kss_hits = current_kss
            last_kss_change_iter = i

        # Timeseries
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
                'gen_failures': gen_failures,
            })

        # Progress
        if (i + 1) % 200 == 0 or (result.get('bug_type') and not result.get('trivial_crash')):
            elapsed = time.time() - t0
            bug_marker = f" BUG:{result['bug_type']}" if (
                result.get('bug_type') and not result.get('trivial_crash')) else ""
            print(f"  [{i+1:5d}/{max_budget}] KSS={current_kss}/{kss.get_total_valid()} "
                  f"L2={len(l2_patterns)} L3={len(l3_hashes)} "
                  f"bugs={len(bugs)} triv={trivial_crashes} fail={gen_failures} "
                  f"t={elapsed:.0f}s{bug_marker}", file=sys.stderr)

        # Saturation check
        if i >= min_budget and (i - last_kss_change_iter) >= patience:
            print(f"  SATURATED at iter {i}: KSS={current_kss} unchanged "
                  f"for {patience} iterations", file=sys.stderr)
            final_iter = i + 1
            timeseries.append({
                'iter': i,
                'kss': current_kss,
                'kss_total': kss.get_total_valid(),
                'l2': len(l2_patterns),
                'l3': len(l3_hashes),
                'bugs': len(bugs),
                'trivial': trivial_crashes,
                'families': len(families_seen),
                'gen_failures': gen_failures,
            })
            break

    elapsed = time.time() - t0
    kss_summary = kss.get_summary()

    output = {
        'mode': 'nnsmith',
        'seed': seed,
        'budget_used': final_iter,
        'elapsed_sec': round(elapsed, 1),
        'total_ok': total_ok,
        'gen_failures': gen_failures,
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

    print(json.dumps(output, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-budget', type=int, default=5000)
    parser.add_argument('--min-budget', type=int, default=500)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_nnsmith(args.seed, args.max_budget, args.min_budget,
                args.patience, args.device)
