#!/usr/bin/env python3
"""
Coverage-based efficiency worker: runs one (mode, seed) combination
with Python coverage.py instrumentation on torch._inductor.

Measures: line coverage and branch coverage of the Inductor compiler
over wall-clock time. This is the standard efficiency metric used by
NNSmith, Tzer, NeuRI, and Fuzz4All.

Outputs JSON to stdout with coverage timeseries.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import random as pyrandom
from collections import defaultdict

import coverage

import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELFUZZ_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'src')
sys.path.insert(0, KERNELFUZZ_PATH)

from targeted_generator import generate_targeted_program, enumerate_target_states
from random_generator import generate_random_program
from random_boundary_generator import generate_random_boundary_program

INDUCTOR_PATH = os.path.dirname(torch._inductor.__file__)

# ---------------------------------------------------------------------------
# Coverage measurement
# ---------------------------------------------------------------------------

def create_coverage_tracker():
    """Create a coverage.Coverage instance scoped to torch._inductor."""
    cov = coverage.Coverage(
        source=[INDUCTOR_PATH],
        branch=True,
        data_file=None,  # in-memory only
    )
    return cov


def sample_coverage(cov):
    """Extract current cumulative coverage stats without resetting."""
    data = cov.get_data()
    files = data.measured_files()
    n_lines = 0
    n_branches = 0
    for f in files:
        lines = data.lines(f)
        if lines:
            n_lines += len(lines)
        arcs = data.arcs(f)
        if arcs:
            n_branches += len(arcs)
    return {
        'lines_covered': n_lines,
        'branches_covered': n_branches,
        'files_touched': len(files),
    }


# ---------------------------------------------------------------------------
# Kernel tracing (same as _time_worker.py)
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

        _orig_extern_out = PythonWrapperCodegen.generate_extern_kernel_out
        def _traced_extern_out(self_, node, *args, **kwargs):
            _kernel_names.append(f"extern_out:{str(node)[:200]}")
            return _orig_extern_out(self_, node, *args, **kwargs)
        PythonWrapperCodegen.generate_extern_kernel_out = _traced_extern_out
    except Exception as e:
        print(f"  [WARN] kernel name tracing unavailable: {e}", file=sys.stderr)

    _tracing_installed = True


def _reset_tracing():
    _heuristics.clear()
    _kernel_names.clear()


def _kernel_template(kernel_name):
    if kernel_name.startswith('extern_'):
        kn_lower = kernel_name.lower()
        if any(k in kn_lower for k in ['mm', 'gemm', 'addmm', 'bmm', 'linear']):
            return 'extern_gemm'
        if 'conv' in kn_lower:
            return 'extern_conv'
        return kernel_name[:80]
    return re.sub(r'_\d+$', '', kernel_name)


def _infer_schedule_families(heuristics, kernel_names):
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
# Oracle (same as _time_worker.py)
# ---------------------------------------------------------------------------
_ENV_ERROR_PATTERNS = [
    "does not support bfloat16 compilation natively",
    "does not support float16 compilation natively",
    "bf16 is not supported",
    "cuda out of memory", "no space left on device",
    "too many blocks in cooperative launch",
]


def _is_trivial_crash(error_msg):
    if not error_msg:
        return False
    msg = error_msg.lower()
    return any(p in msg for p in _ENV_ERROR_PATTERNS)


def _has_grad_inputs(inputs):
    return any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in inputs
    )


def _compare_grads(eager_inputs, compiled_inputs, dtype_str):
    atol = 1e-6 if 'float64' in dtype_str else 1e-3
    max_diff = 0.0
    for ei, ci in zip(eager_inputs, compiled_inputs):
        if not isinstance(ei, torch.Tensor) or not ei.requires_grad:
            continue
        eg = ei.grad
        cg = ci.grad
        if eg is None and cg is None:
            continue
        if eg is None or cg is None:
            return float('inf'), 'CATASTROPHIC_diff'
        if eg.shape != cg.shape:
            return float('inf'), 'CATASTROPHIC_diff'
        d = (eg.float() - cg.float()).abs().max().item()
        max_diff = max(max_diff, d)

    if max_diff > 10.0:
        return max_diff, 'CATASTROPHIC_diff'
    elif max_diff > atol * 10:
        return max_diff, 'SUSPICIOUS_diff'
    return max_diff, None


def execute_program(code, meta, device='cuda'):
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

        needs_backward = _has_grad_inputs(inputs)

        eager_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        eager_out = fn(*eager_in)
        result['eager_ok'] = True

        if needs_backward and isinstance(eager_out, torch.Tensor) and eager_out.requires_grad:
            eager_out.backward(torch.ones_like(eager_out))

        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, backend='inductor')
        comp_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        compiled_out = compiled_fn(*comp_in)
        result['compiled_ok'] = True

        if needs_backward and isinstance(compiled_out, torch.Tensor) and compiled_out.requires_grad:
            compiled_out.backward(torch.ones_like(compiled_out))

        dtype_str = meta.get('dtype', 'torch.float32')
        atol = 1e-6 if 'float64' in dtype_str else 1e-3
        if isinstance(eager_out, torch.Tensor) and isinstance(compiled_out, torch.Tensor):
            if eager_out.shape != compiled_out.shape:
                result['bug_type'] = 'shape_mismatch'
                result['max_diff'] = float('inf')
            else:
                diff = (eager_out.detach().float() - compiled_out.detach().float()).abs()
                max_diff = diff.max().item()
                result['max_diff'] = max_diff
                if max_diff > 10.0:
                    result['bug_type'] = 'CATASTROPHIC_diff'
                elif max_diff > atol * 10:
                    result['bug_type'] = 'SUSPICIOUS_diff'

        if needs_backward and result['bug_type'] is None:
            grad_diff, grad_bug = _compare_grads(eager_in, comp_in, dtype_str)
            if grad_bug:
                result['max_diff'] = grad_diff
                result['bug_type'] = grad_bug
                result['error'] = 'backward_gradient_mismatch'

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
# NNSmith generation (same as _time_worker.py)
# ---------------------------------------------------------------------------
_nnsmith_ctx = None


def _init_nnsmith(seed):
    global _nnsmith_ctx
    if _nnsmith_ctx is not None:
        return _nnsmith_ctx

    from nnsmith.materialize.torch import TorchModelCUDA
    from nnsmith.narrow_spec import auto_opset
    from nnsmith.backends.torchjit import TorchJITFactory

    factory = TorchJITFactory(target='cuda')
    opset = auto_opset(TorchModelCUDA, factory)
    _nnsmith_ctx = {'ModelType': TorchModelCUDA, 'opset': opset}
    return _nnsmith_ctx


def execute_nnsmith(seed_i, device='cuda'):
    from nnsmith.graph_gen import model_gen
    from nnsmith.util import set_seed

    _reset_tracing()
    result = {
        'eager_ok': False, 'compiled_ok': False,
        'max_diff': 0.0, 'bug_type': None, 'error': None,
        'trivial_crash': False, 'dtype': 'float32',
    }

    try:
        ctx = _init_nnsmith(seed_i)
        set_seed(seed_i)
        max_nodes = pyrandom.choice([3, 5, 8, 10, 15])

        gen = model_gen(opset=ctx['opset'], seed=seed_i, max_nodes=max_nodes)
        ir = gen.make_concrete()
        model = ctx['ModelType'].from_gir(ir)
        model.refine_weights()

        net = model.torch_model.eval()

        if model.sat_inputs is not None:
            inputs = {k: v.to(device) for k, v in model.sat_inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in net.get_random_inps().items()}

        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                result['dtype'] = str(v.dtype).replace('torch.', '')
                break

        with torch.no_grad():
            eager_out = net(**inputs)
        result['eager_ok'] = True

        torch._dynamo.reset()
        compiled_net = torch.compile(net, backend='inductor')
        with torch.no_grad():
            compiled_out = compiled_net(**inputs)
        result['compiled_ok'] = True

        if isinstance(eager_out, (tuple, list)):
            eager_list = list(eager_out)
            compiled_list = list(compiled_out)
        else:
            eager_list = [eager_out]
            compiled_list = [compiled_out]

        max_diff = 0.0
        for eo, co in zip(eager_list, compiled_list):
            if isinstance(eo, torch.Tensor) and isinstance(co, torch.Tensor):
                if eo.shape != co.shape:
                    result['bug_type'] = 'shape_mismatch'
                    result['max_diff'] = float('inf')
                    break
                d = (eo.float() - co.float()).abs().max().item()
                max_diff = max(max_diff, d)

        result['max_diff'] = max_diff
        if result['bug_type'] is None:
            if max_diff > 10.0:
                result['bug_type'] = 'CATASTROPHIC_diff'
            elif max_diff > 1e-2:
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
# Main loop with coverage tracking
# ---------------------------------------------------------------------------

def run_with_coverage(mode, seed, time_budget, device, sample_interval=30):
    _install_tracing()

    states = enumerate_target_states() if mode == 'targeted' else None

    kernel_templates = set()
    families_seen = set()
    bugs_found = 0
    trivial_crashes = 0
    gen_failures = 0
    total_ok = 0

    coverage_timeseries = []

    if mode == 'nnsmith':
        print(f"  Initializing NNSmith (seed={seed})...", file=sys.stderr)
        _init_nnsmith(seed)
        print(f"  NNSmith ready.", file=sys.stderr)

    cov = create_coverage_tracker()
    cov.start()

    t0 = time.time()
    last_sample = t0
    i = 0

    # Initial coverage sample (time=0)
    cov.stop()
    snap = sample_coverage(cov)
    snap['time_sec'] = 0.0
    snap['iteration'] = 0
    snap['programs_ok'] = 0
    snap['kernel_diversity'] = 0
    snap['families'] = 0
    snap['bugs'] = 0
    coverage_timeseries.append(snap)
    cov.start()

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break

        # Generate + execute
        if mode == 'targeted':
            state = states[i % len(states)]
            code, meta = generate_targeted_program(
                family=state['family'],
                dtype_str=state['dtype'],
                n_fused_target=state['n_fused'],
                pattern=state.get('pattern', 'auto'),
                seed=seed + i,
            )
            result = execute_program(code, meta, device)
        elif mode == 'random':
            code, meta = generate_random_program(seed=seed + i)
            result = execute_program(code, meta, device)
        elif mode == 'random_boundary':
            code, meta = generate_random_boundary_program(seed=seed + i)
            result = execute_program(code, meta, device)
        elif mode == 'nnsmith':
            seed_i = seed * 100000 + i
            result = execute_nnsmith(seed_i, device)
            meta = {'dtype': f"torch.{result.get('dtype', 'float32')}"}
            if result.get('error') and not result.get('eager_ok'):
                gen_failures += 1
                i += 1
                continue
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if result['eager_ok'] and result['compiled_ok']:
            total_ok += 1

        # Kernel tracking
        sched_families = _infer_schedule_families(_heuristics, _kernel_names)
        families_seen.update(sched_families)
        for name in _kernel_names:
            tmpl = _kernel_template(name)
            if tmpl:
                kernel_templates.add(tmpl)

        if result.get('bug_type'):
            if result.get('trivial_crash'):
                trivial_crashes += 1
            else:
                bugs_found += 1

        # Sample coverage periodically
        now = time.time()
        if now - last_sample >= sample_interval:
            cov.stop()
            snap = sample_coverage(cov)
            snap['time_sec'] = round(now - t0, 1)
            snap['iteration'] = i + 1
            snap['programs_ok'] = total_ok
            snap['kernel_diversity'] = len(kernel_templates)
            snap['families'] = len(families_seen)
            snap['bugs'] = bugs_found
            coverage_timeseries.append(snap)
            cov.start()
            last_sample = now

            print(
                f"  [{mode}][{i+1:5d}] "
                f"lines={snap['lines_covered']} "
                f"branches={snap['branches_covered']} "
                f"files={snap['files_touched']} "
                f"KD={len(kernel_templates)} "
                f"t={snap['time_sec']:.0f}/{time_budget}s",
                file=sys.stderr,
            )

        # Memory management
        if (i + 1) % 500 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        i += 1

    # Final sample
    cov.stop()
    final_snap = sample_coverage(cov)
    final_elapsed = time.time() - t0
    final_snap['time_sec'] = round(final_elapsed, 1)
    final_snap['iteration'] = i
    final_snap['programs_ok'] = total_ok
    final_snap['kernel_diversity'] = len(kernel_templates)
    final_snap['families'] = len(families_seen)
    final_snap['bugs'] = bugs_found
    coverage_timeseries.append(final_snap)

    # Get per-file coverage breakdown for the final state
    data = cov.get_data()
    file_coverage = {}
    for f in data.measured_files():
        rel = os.path.relpath(f, INDUCTOR_PATH)
        lines = data.lines(f)
        arcs = data.arcs(f)
        file_coverage[rel] = {
            'lines': len(lines) if lines else 0,
            'branches': len(arcs) if arcs else 0,
        }

    # Top-20 most-covered files
    top_files = sorted(
        file_coverage.items(), key=lambda x: x[1]['lines'], reverse=True
    )[:20]

    output = {
        'mode': mode,
        'seed': seed,
        'time_budget': time_budget,
        'sample_interval': sample_interval,
        'iterations_completed': i,
        'elapsed_sec': round(final_elapsed, 1),
        'total_ok': total_ok,
        'gen_failures': gen_failures,
        'bugs_found': bugs_found,
        'trivial_crashes': trivial_crashes,
        'kernel_diversity': len(kernel_templates),
        'families_seen': sorted(families_seen),
        'final_coverage': {
            'lines_covered': final_snap['lines_covered'],
            'branches_covered': final_snap['branches_covered'],
            'files_touched': final_snap['files_touched'],
        },
        'top_files': {k: v for k, v in top_files},
        'coverage_timeseries': coverage_timeseries,
    }

    print(json.dumps(output, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['targeted', 'random', 'random_boundary', 'nnsmith'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time-budget', type=int, required=True,
                        help='Wall-clock seconds to run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample-interval', type=int, default=30,
                        help='Coverage sampling interval in seconds (default: 30)')
    args = parser.parse_args()

    run_with_coverage(
        args.mode, args.seed, args.time_budget, args.device,
        sample_interval=args.sample_interval,
    )
