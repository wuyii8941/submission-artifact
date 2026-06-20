#!/usr/bin/env python3
"""
Time-controlled experiment worker: runs one (mode, seed) combination
with a fixed wall-clock budget.

Stops after `time_budget` seconds regardless of saturation state.
This enables fair comparison: same time, different methods, what do you get?

Primary metric: Kernel Diversity = |unique kernel templates|
  A kernel template = (scheduling_prefix, fused_op_sequence), i.e. the kernel
  name with the trailing counter stripped. This captures structurally distinct
  compiler outputs without counter noise.

Secondary: bug candidates, families reached.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import random as pyrandom
from collections import defaultdict

import torch
import torch._inductor
torch._inductor.config.force_disable_caches = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELFUZZ_PATH = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, KERNELFUZZ_PATH)

from targeted_generator import generate_targeted_program, enumerate_target_states
from random_generator import generate_random_program
from random_boundary_generator import generate_random_boundary_program
from kernel.state_space import KernelStateSpace

# ---------------------------------------------------------------------------
# Kernel tracing
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

        # Hook extern kernel dispatch to detect GEMM/conv
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


# ---------------------------------------------------------------------------
# Kernel Diversity: unique kernel templates
# ---------------------------------------------------------------------------
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


def _has_boundary_shape(meta):
    """Check if any shape dimension is near a power-of-2 boundary."""
    shape = meta.get('shape')
    if not shape:
        return False
    if isinstance(shape, str):
        shape = shape.strip('()')
        dims = [int(x.strip()) for x in shape.split(',') if x.strip()]
    elif isinstance(shape, (list, tuple)):
        dims = list(shape)
    else:
        return False
    for d in dims:
        if d <= 1:
            continue
        # Near a power-of-2 boundary: within ±1 of 2^k
        import math
        log2 = math.log2(d)
        nearest_pow2 = 2 ** round(log2)
        if abs(d - nearest_pow2) <= 1 and d != nearest_pow2:
            return True
        # Near Inductor thresholds: 512, 1024, 2048
        for thresh in [512, 1024, 2048, 4096]:
            if abs(d - thresh) <= 1 and d != thresh:
                return True
    return False


def _extract_decision_tuple(meta, heuristics, kernel_names, result):
    """Extract (schedule, dtype, backward) decision tuple for one program."""
    sched_families = _infer_schedule_families(heuristics, kernel_names)
    if not sched_families:
        sched = 'no_kernel'
    elif len(sched_families) == 1:
        sched = next(iter(sched_families))
    else:
        sched = '+'.join(sorted(sched_families))

    dtype = meta.get('dtype', 'torch.float32').replace('torch.', '')

    backward = False
    if meta.get('has_backward') == 'True' or meta.get('prog_type') == 'backward':
        backward = True
    pattern = meta.get('pattern', '')
    if pattern and pattern != 'auto':
        backward = True

    return (sched, dtype, backward)


def _kernel_template(kernel_name):
    """Extract the kernel template identity by stripping the trailing counter.

    triton_poi_fused_sin_relu_0 → triton_poi_fused_sin_relu
    triton_red_fused_sum_add_1  → triton_red_fused_sum_add
    extern_out:ExternKernel...  → kept as-is (already unique by content)

    This captures (scheduling_family × op_sequence) = structurally distinct
    compiler output, without the compilation-order counter noise.
    """
    if kernel_name.startswith('extern_'):
        # Normalize extern kernel identity by type
        kn_lower = kernel_name.lower()
        if any(k in kn_lower for k in ['mm', 'gemm', 'addmm', 'bmm', 'linear']):
            return 'extern_gemm'
        if 'conv' in kn_lower:
            return 'extern_conv'
        return kernel_name[:80]
    # Strip trailing _\d+ counter
    return re.sub(r'_\d+$', '', kernel_name)


# ---------------------------------------------------------------------------
# Oracle
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
    """Check if any input requires grad (signals backward testing)."""
    return any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in inputs
    )


def _compare_grads(eager_inputs, compiled_inputs, dtype_str):
    """Compare gradients on inputs that require grad. Returns (max_diff, bug_type)."""
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
    """Execute program through differential oracle (forward + backward)."""
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

        # Eager
        eager_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        eager_out = fn(*eager_in)
        result['eager_ok'] = True

        # Eager backward
        if needs_backward and isinstance(eager_out, torch.Tensor) and eager_out.requires_grad:
            eager_out.backward(torch.ones_like(eager_out))

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

        # Compiled backward
        if needs_backward and isinstance(compiled_out, torch.Tensor) and compiled_out.requires_grad:
            compiled_out.backward(torch.ones_like(compiled_out))

        # Forward differential comparison
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

        # Backward differential comparison (even if forward matches)
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
# NNSmith generation
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
    """Generate one NNSmith model, run eager vs compiled."""
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
# Main run loop (time-controlled)
# ---------------------------------------------------------------------------
def run_time_controlled(mode, seed, time_budget, device, checkpoint_path=None):
    _install_tracing()

    kss = KernelStateSpace()
    kernel_templates = set()  # primary metric: unique kernel templates

    states = enumerate_target_states() if mode in ('targeted', 'targeted_blind') else None
    bugs = []
    bug_sigs = set()
    raw_triggers = 0
    trivial_crashes = 0
    families_seen = set()
    timeseries = []
    total_ok = 0
    gen_failures = 0
    elapsed_offset = 0.0
    i_start = 0
    # Interaction region tracking: set of (schedule, dtype, backward) tuples
    regions_seen = set()
    region_bug_counts = defaultdict(int)  # region -> bug count
    region_sample_counts = defaultdict(int)  # region -> total samples

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            kernel_templates = set(ckpt.get('kernel_templates', []))
            bug_sigs = set(ckpt.get('bug_sigs', []))
            bugs = ckpt.get('bugs', [])
            families_seen = set(ckpt.get('families_seen', []))
            raw_triggers = ckpt.get('raw_triggers', 0)
            trivial_crashes = ckpt.get('trivial_crashes', 0)
            total_ok = ckpt.get('total_ok', 0)
            gen_failures = ckpt.get('gen_failures', 0)
            elapsed_offset = ckpt.get('elapsed', 0.0)
            i_start = ckpt.get('iteration', 0)
            timeseries = ckpt.get('timeseries', [])
            regions_seen = set(tuple(r) for r in ckpt.get('regions_seen', []))
            region_bug_counts = defaultdict(int, {
                tuple(json.loads(k)): v
                for k, v in ckpt.get('region_bug_counts', {}).items()
            })
            region_sample_counts = defaultdict(int, {
                tuple(json.loads(k)): v
                for k, v in ckpt.get('region_sample_counts', {}).items()
            })
            print(f"  Resumed from checkpoint: iter={i_start} elapsed={elapsed_offset:.0f}s "
                  f"KD={len(kernel_templates)} bugs={len(bugs)} regions={len(regions_seen)}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}", file=sys.stderr)

    if mode == 'nnsmith':
        print(f"  Initializing NNSmith (seed={seed})...", file=sys.stderr)
        _init_nnsmith(seed)
        print(f"  NNSmith ready.", file=sys.stderr)

    t0 = time.time() - elapsed_offset
    i = i_start

    def _save_checkpoint():
        if not checkpoint_path:
            return
        ckpt_data = {
            'kernel_templates': sorted(kernel_templates),
            'bug_sigs': sorted(bug_sigs),
            'bugs': bugs,
            'families_seen': sorted(families_seen),
            'regions_seen': sorted(list(r) for r in regions_seen),
            'region_bug_counts': {
                json.dumps(list(k)): v for k, v in region_bug_counts.items()
            },
            'region_sample_counts': {
                json.dumps(list(k)): v for k, v in region_sample_counts.items()
            },
            'raw_triggers': raw_triggers,
            'trivial_crashes': trivial_crashes,
            'total_ok': total_ok,
            'gen_failures': gen_failures,
            'elapsed': round(time.time() - t0, 1),
            'iteration': i + 1,
            'timeseries': timeseries,
        }
        tmp = checkpoint_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(ckpt_data, f)
        os.replace(tmp, checkpoint_path)

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
        elif mode == 'targeted_blind':
            family = pyrandom.choices(
                ['pointwise', 'reduction', 'persistent_reduction',
                 'extern_gemm', 'scan', 'extern_conv'],
                weights=[25, 17, 13, 13, 9, 9],
                k=1,
            )[0]
            dtype_str = pyrandom.choice([
                'torch.float32', 'torch.float64', 'torch.float16',
            ])
            n_fused = pyrandom.choice(['1', '2-3', '4+'])
            code, meta = generate_targeted_program(
                family=family,
                dtype_str=dtype_str,
                n_fused_target=n_fused,
                pattern='auto',
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

        # --- Coverage tracking ---
        dtype = meta.get('dtype', 'torch.float32').replace('torch.', '')

        if _kernel_names:
            kss.record_from_kernels(_kernel_names, dtype, _heuristics)

        sched_families_for_tracking = _infer_schedule_families(_heuristics, _kernel_names)
        families_seen.update(sched_families_for_tracking)

        for name in _kernel_names:
            tmpl = _kernel_template(name)
            if tmpl:
                kernel_templates.add(tmpl)

        # --- Decision tuple & interaction region tracking ---
        decision = _extract_decision_tuple(meta, _heuristics, _kernel_names, result)
        regions_seen.add(decision)
        region_sample_counts[decision] += 1
        sched_families = _infer_schedule_families(_heuristics, _kernel_names)
        has_boundary = _has_boundary_shape(meta) or bool(meta.get('boundary_dim'))

        # Bug tracking (raw triggers, no dedup — consistent with v4)
        if result.get('bug_type'):
            if result.get('trivial_crash'):
                trivial_crashes += 1
            else:
                raw_triggers += 1
                region_bug_counts[decision] += 1
                bug_meta = {k: str(v) for k, v in meta.items()
                            if k not in ('state',)}
                bug_meta['_schedule_families'] = sorted(sched_families)
                bug_meta['_decision'] = list(decision)
                bug_meta['_has_boundary'] = str(has_boundary)
                bugs.append({
                    'idx': i,
                    'time_sec': round(elapsed, 1),
                    'bug_type': result['bug_type'],
                    'max_diff': result['max_diff'],
                    'error': (result.get('error') or '')[:100],
                    'meta': bug_meta,
                })

        # Timeseries (every 10 iterations)
        if i % 10 == 0 or i == 0:
            timeseries.append({
                'iter': i,
                'time_sec': round(elapsed, 1),
                'kernel_diversity': len(kernel_templates),
                'bugs': len(bugs),
                'raw_triggers': raw_triggers,
                'trivial': trivial_crashes,
                'families': len(families_seen),
                'regions': len(regions_seen),
            })

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  [{mode}][{i+1:5d}] KD={len(kernel_templates)} "
                  f"bugs={len(bugs)}({raw_triggers}) triv={trivial_crashes} "
                  f"fam={len(families_seen)} rgn={len(regions_seen)} "
                  f"t={elapsed:.0f}/{time_budget}s", file=sys.stderr)
            _save_checkpoint()

        # Memory management
        if (i + 1) % 500 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        i += 1

    elapsed = time.time() - t0

    # Final timeseries point
    timeseries.append({
        'iter': i,
        'time_sec': round(elapsed, 1),
        'kernel_diversity': len(kernel_templates),
        'bugs': len(bugs),
        'raw_triggers': raw_triggers,
        'trivial': trivial_crashes,
        'families': len(families_seen),
        'regions': len(regions_seen),
    })

    _save_checkpoint()
    kss_summary = kss.get_summary()

    output = {
        'mode': mode,
        'seed': seed,
        'time_budget': time_budget,
        'iterations_completed': i,
        'elapsed_sec': round(elapsed, 1),
        'total_ok': total_ok,
        'gen_failures': gen_failures,
        'kernel_diversity': len(kernel_templates),
        'families_seen': sorted(families_seen),
        'n_families': len(families_seen),
        'regions_seen': sorted(list(r) for r in regions_seen),
        'n_regions': len(regions_seen),
        'region_bug_counts': {
            json.dumps(list(k)): v for k, v in region_bug_counts.items()
        },
        'region_sample_counts': {
            json.dumps(list(k)): v for k, v in region_sample_counts.items()
        },
        'bugs_found': len(bugs),
        'raw_triggers': raw_triggers,
        'trivial_crashes': trivial_crashes,
        'bugs': bugs,
        'kss_per_family': {
            fam: {'hits': info['hits'], 'total': info['total']}
            for fam, info in kss_summary['per_family'].items()
        },
        'timeseries': timeseries,
    }

    print(json.dumps(output, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['targeted', 'random', 'random_boundary', 'nnsmith', 'targeted_blind'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time-budget', type=int, required=True,
                        help='Wall-clock seconds to run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file path for restart recovery')
    args = parser.parse_args()

    run_time_controlled(args.mode, args.seed, args.time_budget, args.device,
                        checkpoint_path=args.checkpoint)
