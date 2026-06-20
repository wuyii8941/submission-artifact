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
try:
    import torch._inductor
    torch._inductor.config.force_disable_caches = True
except (AttributeError, ImportError):
    pass

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

    try:
        from torch._inductor.codegen.triton import TritonKernel
        _orig_codegen = TritonKernel.codegen_kernel

        def _traced_codegen(self_, *args, **kwargs):
            if getattr(self_, 'persistent_reduction', False):
                _heuristics.append('persistent_reduction')
            elif getattr(self_, 'inside_reduction', False):
                _heuristics.append('reduction')
            else:
                _heuristics.append('pointwise')
            return _orig_codegen(self_, *args, **kwargs)
        TritonKernel.codegen_kernel = _traced_codegen
    except Exception:
        pass

    try:
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel
        _orig_ss_codegen = TritonSplitScanKernel.codegen_kernel

        def _traced_ss_codegen(self_, *args, **kwargs):
            _heuristics.append('split_scan')
            return _orig_ss_codegen(self_, *args, **kwargs)
        TritonSplitScanKernel.codegen_kernel = _traced_ss_codegen
    except Exception:
        pass

    try:
        try:
            from torch._inductor.codegen.wrapper import PythonWrapperCodegen as WrapperCls
        except ImportError:
            from torch._inductor.codegen.wrapper import WrapperCodeGen as WrapperCls

        _orig_gen = WrapperCls.generate_kernel_call

        def _traced_gen(self_, kernel_name, *args, **kwargs):
            _kernel_names.append(str(kernel_name))
            return _orig_gen(self_, kernel_name, *args, **kwargs)
        WrapperCls.generate_kernel_call = _traced_gen

        _orig_extern_out = WrapperCls.generate_extern_kernel_out
        def _traced_extern_out(self_, node, *args, **kwargs):
            _kernel_names.append(f"extern_out:{str(node)[:200]}")
            return _orig_extern_out(self_, node, *args, **kwargs)
        WrapperCls.generate_extern_kernel_out = _traced_extern_out

        _orig_extern_alloc = WrapperCls.generate_extern_kernel_alloc
        def _traced_extern_alloc(self_, node, *args, **kwargs):
            _kernel_names.append(f"extern_alloc:{str(node)[:200]}")
            return _orig_extern_alloc(self_, node, *args, **kwargs)
        WrapperCls.generate_extern_kernel_alloc = _traced_extern_alloc
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
    if meta.get('has_backward') is True or meta.get('has_backward') == 'True' or meta.get('prog_type') == 'backward':
        backward = True
    pattern = meta.get('pattern', '')
    if pattern and 'bwd' in pattern:
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


def _fused_op_sig(kernel_templates):
    """Extract shape-independent fused-op signature from kernel templates.

    'triton_poi_fused_sin_relu_cos' → frozenset({'sin', 'relu', 'cos'})
    Multiple templates → union of all op sets + scheduling prefix.
    Returns a hashable string for dedup.
    """
    parts = []
    for tmpl in sorted(kernel_templates):
        if tmpl.startswith('extern_'):
            parts.append(tmpl)
            continue
        m = re.match(r'triton_(poi|red|per|hel)_fused_(.*)', tmpl)
        if m:
            sched = m.group(1)
            ops = frozenset(m.group(2).split('_'))
            parts.append(f"{sched}:{'+'.join(sorted(ops))}")
        else:
            parts.append(tmpl)
    return '|'.join(parts) if parts else ''


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


def _is_obvious_fp(bug_type, meta, max_diff):
    """Quick FP filter for real-time tracking. Conservative — only filters
    cases that are definitely not real bugs."""
    dtype = meta.get('dtype', 'torch.float32')
    if 'bfloat16' in dtype:
        return True
    if 'float16' in dtype and bug_type != 'compile_crash':
        return True
    if bug_type == 'compile_crash' and 'Triton PTX codegen error' in str(meta.get('_error', '')):
        return True
    return False


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
        result['code'] = f'# NNSmith seed={seed_i} max_nodes={max_nodes}\n{repr(net)}'

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
# WhiteFox: pre-generated test execution
# ---------------------------------------------------------------------------
_whitefox_tests = None

def _init_whitefox(seed):
    global _whitefox_tests
    if _whitefox_tests is not None:
        return

    test_dir = os.environ.get('WHITEFOX_TEST_DIR',
                os.path.join(SCRIPT_DIR, '..', 'whitefox_baseline', 'torch-inductor-generated'))
    _whitefox_tests = []
    for step_dir in sorted(os.listdir(test_dir)):
        step_path = os.path.join(test_dir, step_dir)
        if not os.path.isdir(step_path):
            continue
        for opt_dir in sorted(os.listdir(step_path)):
            opt_path = os.path.join(step_path, opt_dir)
            if not os.path.isdir(opt_path):
                continue
            for f in sorted(os.listdir(opt_path)):
                if f.endswith('.py'):
                    _whitefox_tests.append(os.path.join(opt_path, f))

    pyrandom.seed(seed)
    pyrandom.shuffle(_whitefox_tests)
    print(f"  WhiteFox: {len(_whitefox_tests)} pre-generated tests loaded", file=sys.stderr)


def execute_whitefox(idx, device='cuda'):
    """Run one WhiteFox test in a subprocess to prevent memory leaks."""
    import subprocess as _sp
    _reset_tracing()
    test_file = _whitefox_tests[idx % len(_whitefox_tests)]
    worker_script = os.path.join(SCRIPT_DIR, '_whitefox_subprocess.py')
    env = os.environ.copy()
    try:
        proc = _sp.run(
            [sys.executable, worker_script, test_file, device],
            capture_output=True, text=True, timeout=120, env=env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            result = json.loads(proc.stdout.strip().split('\n')[-1])
            # Recover the raw code for trigger saving
            with open(test_file) as f:
                result['code'] = f.read()
            # Feed subprocess-captured kernels into parent tracing buffers so
            # downstream kss / kernel_templates accounting works for WhiteFox.
            sub_kn = result.pop('kernel_names', [])
            sub_h = result.pop('heuristics', [])
            if sub_kn:
                _kernel_names.extend(sub_kn)
            if sub_h:
                _heuristics.extend(sub_h)
            return result
        else:
            return {
                'eager_ok': False, 'compiled_ok': False,
                'max_diff': 0.0, 'bug_type': None,
                'error': f'subprocess rc={proc.returncode}: {proc.stderr[-200:] if proc.stderr else ""}',
                'trivial_crash': False, 'dtype': 'float32',
            }
    except _sp.TimeoutExpired:
        return {
            'eager_ok': False, 'compiled_ok': False,
            'max_diff': 0.0, 'bug_type': None,
            'error': 'subprocess timeout (120s)',
            'trivial_crash': False, 'dtype': 'float32',
        }
    except Exception as e:
        return {
            'eager_ok': False, 'compiled_ok': False,
            'max_diff': 0.0, 'bug_type': None,
            'error': f'subprocess error: {e}',
            'trivial_crash': False, 'dtype': 'float32',
        }


def _execute_whitefox_inprocess(idx, device='cuda'):
    """Original in-process version (kept for reference, causes memory leaks)."""
    _reset_tracing()
    result = {
        'eager_ok': False, 'compiled_ok': False,
        'max_diff': 0.0, 'bug_type': None, 'error': None,
        'trivial_crash': False, 'dtype': 'float32',
    }

    test_file = _whitefox_tests[idx % len(_whitefox_tests)]

    try:
        with open(test_file) as f:
            raw_code = f.read()

        code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n" + raw_code
        ns = {}
        exec(code, ns)

        model = None
        for v in ns.values():
            if isinstance(v, torch.nn.Module):
                model = v
                break
        if model is None and 'Model' in ns:
            model = ns['Model']()

        if model is None:
            result['error'] = 'no model found'
            return result

        model = model.to(device).eval()

        input_tensors = []
        for k, v in ns.items():
            if isinstance(v, torch.Tensor) and k not in ('m', 'model'):
                input_tensors.append(v.to(device))

        if not input_tensors:
            result['error'] = 'no inputs found'
            return result

        for t in input_tensors:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                result['dtype'] = str(t.dtype).replace('torch.', '')
                break

        result['code'] = raw_code

        with torch.no_grad():
            eager_out = model(*input_tensors)
        result['eager_ok'] = True

        torch._dynamo.reset()
        compiled_model = torch.compile(model, backend='inductor')
        with torch.no_grad():
            compiled_out = compiled_model(*input_tensors)
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
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Main run loop (time-controlled)
# ---------------------------------------------------------------------------
def run_time_controlled(mode, seed, time_budget, device, checkpoint_path=None):
    _install_tracing()

    kss = KernelStateSpace()
    kernel_templates = set()  # primary metric: unique kernel templates
    template_sample_counts = defaultdict(int)  # template -> programs that hit it (A(k))

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

    # Unique bug tracking via fingerprint dedup
    bug_fingerprints = {}  # fingerprint -> first trigger info
    bug_op_sigs = set()    # unique fused-op signatures (shape-independent)
    real_bug_fps = set()   # fingerprints after obvious-FP filtering
    # Trigger saving directory
    triggers_dir = None
    if checkpoint_path:
        triggers_dir = os.path.join(os.path.dirname(checkpoint_path),
                                    'triggers', f'{mode}_seed{seed}')
        os.makedirs(triggers_dir, exist_ok=True)

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
            template_sample_counts = defaultdict(int, ckpt.get('template_sample_counts', {}))
            bug_fingerprints = ckpt.get('bug_fingerprints', {})
            bug_op_sigs = set(ckpt.get('bug_op_sigs', []))
            real_bug_fps = set(ckpt.get('real_bug_fps', []))
            print(f"  Resumed from checkpoint: iter={i_start} elapsed={elapsed_offset:.0f}s "
                  f"KD={len(kernel_templates)} bugs={len(bugs)} unique={len(bug_fingerprints)} regions={len(regions_seen)}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}", file=sys.stderr)

    if mode == 'nnsmith':
        print(f"  Initializing NNSmith (seed={seed})...", file=sys.stderr)
        _init_nnsmith(seed)
        print(f"  NNSmith ready.", file=sys.stderr)
    elif mode == 'whitefox':
        print(f"  Initializing WhiteFox (seed={seed})...", file=sys.stderr)
        _init_whitefox(seed)
        print(f"  WhiteFox ready.", file=sys.stderr)

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
            'template_sample_counts': dict(template_sample_counts),
            'raw_triggers': raw_triggers,
            'trivial_crashes': trivial_crashes,
            'total_ok': total_ok,
            'gen_failures': gen_failures,
            'elapsed': round(time.time() - t0, 1),
            'iteration': i + 1,
            'timeseries': timeseries,
            'bug_fingerprints': bug_fingerprints,
            'bug_op_sigs': sorted(bug_op_sigs),
            'real_bug_fps': sorted(real_bug_fps),
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
        elif mode == 'whitefox':
            result = execute_whitefox(i, device)
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

        iter_templates = set()
        for name in _kernel_names:
            tmpl = _kernel_template(name)
            if tmpl:
                kernel_templates.add(tmpl)
                iter_templates.add(tmpl)
        for tmpl in iter_templates:
            template_sample_counts[tmpl] += 1

        # --- Decision tuple & interaction region tracking ---
        decision = _extract_decision_tuple(meta, _heuristics, _kernel_names, result)
        regions_seen.add(decision)
        region_sample_counts[decision] += 1
        sched_families = _infer_schedule_families(_heuristics, _kernel_names)
        has_boundary = _has_boundary_shape(meta) or bool(meta.get('boundary_dim'))

        # Bug tracking with fingerprint dedup + trigger saving
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

                # Compute bug fingerprint from kernel templates + bug type
                cur_templates = sorted(set(
                    _kernel_template(n) for n in _kernel_names
                    if _kernel_template(n)
                ))
                fp_str = f"{result['bug_type']}|{'|'.join(cur_templates)}"
                fp = hashlib.md5(fp_str.encode()).hexdigest()[:16]
                is_new = fp not in bug_fingerprints

                op_sig = _fused_op_sig(cur_templates)
                if op_sig:
                    bug_op_sigs.add(f"{result['bug_type']}|{op_sig}")

                bug_meta['_error'] = (result.get('error') or '')[:200]
                if not _is_obvious_fp(result['bug_type'], bug_meta, result['max_diff']):
                    real_bug_fps.add(fp)

                if is_new:
                    bug_fingerprints[fp] = {
                        'idx': i, 'time_sec': round(elapsed, 1),
                        'bug_type': result['bug_type'],
                        'max_diff': result['max_diff'],
                        'kernel_templates': cur_templates,
                        'device': device,
                    }
                    # Save trigger program source code
                    if triggers_dir:
                        trigger_file = os.path.join(triggers_dir, f'trigger_{len(bug_fingerprints):04d}_iter{i}.py')
                        try:
                            trigger_code = ''
                            if mode in ('targeted', 'targeted_blind', 'random', 'random_boundary'):
                                trigger_code = code
                            elif mode == 'whitefox':
                                wf_path = _whitefox_tests[i % len(_whitefox_tests)]
                                with open(wf_path) as _wf:
                                    trigger_code = f"# source: {wf_path}\n" + _wf.read()
                            elif mode == 'nnsmith':
                                trigger_code = f"# NNSmith seed_i={seed * 100000 + i}\n# Reproduce: execute_nnsmith({seed * 100000 + i}, '{device}')\n"
                            trigger_code += f"\n# --- trigger info ---\n# fingerprint: {fp}\n# bug_type: {result['bug_type']}\n# max_diff: {result['max_diff']}\n# kernel_templates: {cur_templates}\n"
                            with open(trigger_file, 'w') as _tf:
                                _tf.write(trigger_code)
                        except Exception:
                            pass

                bugs.append({
                    'idx': i,
                    'time_sec': round(elapsed, 1),
                    'bug_type': result['bug_type'],
                    'max_diff': result['max_diff'],
                    'error': (result.get('error') or '')[:100],
                    'meta': bug_meta,
                    'fingerprint': fp,
                    'is_new_bug': is_new,
                })

        # Timeseries (every 10 iterations)
        if i % 10 == 0 or i == 0:
            sat_counts = {
                K: sum(1 for v in template_sample_counts.values() if v >= K)
                for K in [1, 10, 50, 100, 500]
            }
            timeseries.append({
                'iter': i,
                'time_sec': round(elapsed, 1),
                'kernel_diversity': len(kernel_templates),
                'bugs': len(bugs),
                'unique_bugs': len(bug_fingerprints),
                'unique_op_sigs': len(bug_op_sigs),
                'real_bugs': len(real_bug_fps),
                'raw_triggers': raw_triggers,
                'trivial': trivial_crashes,
                'families': len(families_seen),
                'regions': len(regions_seen),
                'sat': sat_counts,
            })

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  [{mode}][{i+1:5d}] KD={len(kernel_templates)} "
                  f"unique={len(bug_fingerprints)} real={len(real_bug_fps)} opsig={len(bug_op_sigs)} "
                  f"raw={raw_triggers} triv={trivial_crashes} "
                  f"fam={len(families_seen)} rgn={len(regions_seen)} "
                  f"t={elapsed:.0f}/{time_budget}s", file=sys.stderr)
            _save_checkpoint()

        # Memory management
        if (i + 1) % 200 == 0:
            torch._dynamo.reset()
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
        'unique_bugs': len(bug_fingerprints),
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
        'template_sample_counts': dict(template_sample_counts),
        'bugs_found': len(bugs),
        'unique_bugs': len(bug_fingerprints),
        'unique_op_sigs': len(bug_op_sigs),
        'real_bugs': len(real_bug_fps),
        'raw_triggers': raw_triggers,
        'trivial_crashes': trivial_crashes,
        'bugs': bugs,
        'bug_fingerprints': bug_fingerprints,
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
                        choices=['targeted', 'random', 'random_boundary', 'nnsmith', 'targeted_blind', 'whitefox'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time-budget', type=int, required=True,
                        help='Wall-clock seconds to run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file path for restart recovery')
    args = parser.parse_args()

    run_time_controlled(args.mode, args.seed, args.time_budget, args.device,
                        checkpoint_path=args.checkpoint)
