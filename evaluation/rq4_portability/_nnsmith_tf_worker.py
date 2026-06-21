#!/usr/bin/env python3
"""
NNSmith TF-XLA baseline worker: generates NNSmith models for TensorFlow,
runs 3-way differential oracle (eager vs Grappler vs XLA), tracks graph
diversity and findings.

Aligned with PyTorch NNSmith baseline:
  - Same budget/saturation protocol (iteration-based, patience detection)
  - Same oracle principle (differential: reference vs compiled)
  - Same seed structure
  - Graph diversity instead of Triton kernel L1/L2/L3
"""

import argparse
import json
import os
import sys
import time
import random as pyrandom
import traceback
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np

NNSMITH_PATH = os.environ.get('NNSMITH_PATH', os.path.join(os.path.dirname(sys.executable), '..', 'lib', 'python3.11', 'site-packages'))
if NNSMITH_PATH not in sys.path:
    sys.path.insert(0, NNSMITH_PATH)

import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# NNSmith TF model generation
# ---------------------------------------------------------------------------
def _init_nnsmith(device='cuda'):
    from nnsmith.materialize.tensorflow import TFModelCUDA, TFModelCPU
    from nnsmith.narrow_spec import auto_opset
    from nnsmith.backends.xla import XLAFactory

    target = 'cuda' if device == 'cuda' else 'cpu'
    ModelType = TFModelCUDA if device == 'cuda' else TFModelCPU
    factory = XLAFactory(target=target)
    opset = auto_opset(ModelType, factory)

    return {
        'ModelType': ModelType,
        'factory': factory,
        'opset': opset,
    }


def _to_f64(t):
    if isinstance(t, tf.Tensor):
        t = t.numpy()
    arr = np.asarray(t)
    if np.issubdtype(arr.dtype, np.complexfloating):
        return np.abs(arr).astype(np.float64)
    return arr.astype(np.float64)


def _max_diff_np(a, b):
    a = _to_f64(a)
    b = _to_f64(b)
    if a.shape != b.shape:
        return float('inf'), 'shape_mismatch'
    if a.size == 0:
        return 0.0, 'empty'
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        return float('inf'), 'nan_mismatch'
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return 0.0, 'all_nonfinite'
    return float(np.abs(a[mask] - b[mask]).max()), 'ok'


_ENV_ERROR_PATTERNS = [
    'cuda out of memory', 'no space left on device',
    'resource exhausted', 'oom',
]


def _is_trivial(error_msg):
    if not error_msg:
        return False
    msg = error_msg.lower()
    return any(p in msg for p in _ENV_ERROR_PATTERNS)


def _generate_and_execute(ctx, seed_i, device='cuda'):
    """Generate one NNSmith TF model, run 3-way oracle."""
    from nnsmith.graph_gen import model_gen
    from nnsmith.materialize.tensorflow import np_dict_from_tf, tf_dict_from_np
    from nnsmith.util import set_seed

    result = {
        'eager_ok': False, 'grappler_ok': False, 'xla_ok': False,
        'findings': [], 'bug_type': None, 'error': None,
        'trivial': False, 'dtype': 'float32', 'op_names': [],
    }

    try:
        set_seed(seed_i)
        max_nodes = pyrandom.choice([3, 5, 8, 10, 15])

        gen = model_gen(
            opset=ctx['opset'],
            seed=seed_i,
            max_nodes=max_nodes,
        )
        ir = gen.make_concrete()
        model = ctx['ModelType'].from_gir(ir)
        model.refine_weights()
    except Exception as e:
        result['error'] = f'gen: {type(e).__name__}: {str(e)[:200]}'
        return result

    try:
        inputs_tf = model.random_inputs()

        for v in inputs_tf.values():
            if v.dtype.is_floating:
                result['dtype'] = str(v.dtype.name)
                break

        op_names = []
        for inst in ir.insts:
            op_names.append(type(inst.iexpr.op).__name__)
        result['op_names'] = sorted(set(op_names))

    except Exception as e:
        result['error'] = f'input: {type(e).__name__}: {str(e)[:200]}'
        return result

    dev = '/GPU:0' if device == 'cuda' else '/CPU:0'

    # 1. Eager
    try:
        eager_out = model.run_eagerly(inputs_tf)
        eager_np = {k: v.numpy() for k, v in eager_out.items()}
        result['eager_ok'] = True
    except Exception as e:
        result['error'] = f'eager: {type(e).__name__}: {str(e)[:200]}'
        return result

    # 2. Grappler (tf.function without XLA)
    try:
        with tf.device(dev):
            grappler_fn = tf.function(model.net, jit_compile=False)
            grappler_out = grappler_fn(**inputs_tf)
            grappler_np = {k: v.numpy() for k, v in grappler_out.items()}
        result['grappler_ok'] = True
    except Exception as e:
        err = f'grappler: {type(e).__name__}: {str(e)[:200]}'
        result['error'] = err
        if _is_trivial(err):
            result['trivial'] = True
        else:
            result['bug_type'] = 'grappler_crash'
            result['findings'].append({'type': 'grappler_crash', 'error': err[:100]})
        return result

    # 3. XLA (jit_compile=True)
    try:
        xla_callable = ctx['factory'].make_backend(model)
        inputs_np = np_dict_from_tf(inputs_tf)
        xla_np = xla_callable(inputs_np)
        result['xla_ok'] = True
    except Exception as e:
        err = f'xla: {type(e).__name__}: {str(e)[:200]}'
        result['error'] = err
        if _is_trivial(err):
            result['trivial'] = True
        else:
            result['bug_type'] = 'xla_crash'
            result['findings'].append({'type': 'xla_crash', 'error': err[:100]})
        return result

    # 4. Differential comparison
    for k in eager_np:
        if k not in grappler_np or k not in xla_np:
            continue

        # eager vs grappler
        diff_eg, reason = _max_diff_np(eager_np[k], grappler_np[k])
        if reason == 'ok' and diff_eg > 1e-3:
            result['findings'].append({
                'type': 'eager_vs_grappler', 'key': k,
                'max_diff': diff_eg,
            })
        elif reason not in ('ok', 'empty', 'all_nonfinite'):
            result['findings'].append({
                'type': f'eager_vs_grappler_{reason}', 'key': k,
            })

        # eager vs xla
        diff_ex, reason = _max_diff_np(eager_np[k], xla_np[k])
        if reason == 'ok' and diff_ex > 1e-2:
            result['findings'].append({
                'type': 'eager_vs_xla', 'key': k,
                'max_diff': diff_ex,
            })
        elif reason not in ('ok', 'empty', 'all_nonfinite'):
            result['findings'].append({
                'type': f'eager_vs_xla_{reason}', 'key': k,
            })

        # grappler vs xla
        diff_gx, reason = _max_diff_np(grappler_np[k], xla_np[k])
        if reason == 'ok' and diff_gx > 1e-2:
            result['findings'].append({
                'type': 'grappler_vs_xla', 'key': k,
                'max_diff': diff_gx,
            })
        elif reason not in ('ok', 'empty', 'all_nonfinite'):
            result['findings'].append({
                'type': f'grappler_vs_xla_{reason}', 'key': k,
            })

    if result['findings']:
        types = [f['type'] for f in result['findings']]
        if any('crash' in t for t in types):
            result['bug_type'] = 'crash'
        elif any('shape_mismatch' in t or 'nan_mismatch' in t for t in types):
            result['bug_type'] = 'structural_diff'
        else:
            max_d = max((f.get('max_diff', 0) for f in result['findings']), default=0)
            result['bug_type'] = 'CATASTROPHIC_diff' if max_d > 10.0 else 'SUSPICIOUS_diff'

    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_nnsmith_tf(seed, max_budget, min_budget, patience, device, time_budget=None):
    print(f"  Initializing NNSmith TF (seed={seed})...", file=sys.stderr)
    ctx = _init_nnsmith(device)
    print(f"  NNSmith TF ready: {len(ctx['opset'])} ops", file=sys.stderr)

    graph_signatures = set()
    bugs = []
    trivial_crashes = 0
    timeseries = []
    total_ok = 0
    gen_failures = 0

    last_gd = 0
    last_gd_change_iter = 0
    t0 = time.time()

    finding_types = defaultdict(int)
    final_iter = max_budget

    for i in range(max_budget):
        seed_i = seed * 100000 + i

        result = _generate_and_execute(ctx, seed_i, device)

        if result.get('error') and not result.get('eager_ok'):
            gen_failures += 1
            continue

        if result['eager_ok'] and result['grappler_ok'] and result['xla_ok']:
            total_ok += 1

        # Graph diversity: (op_sig, dtype, has_findings)
        op_sig = tuple(result.get('op_names', []))
        dtype_str = result.get('dtype', 'float32')
        has_findings = bool(result.get('findings'))
        sigma = (op_sig, dtype_str, has_findings)
        graph_signatures.add(sigma)

        # Bugs
        if result.get('bug_type'):
            if result.get('trivial'):
                trivial_crashes += 1
            else:
                for f in result.get('findings', []):
                    finding_types[f['type']] += 1
                bugs.append({
                    'idx': i,
                    'bug_type': result['bug_type'],
                    'findings': result['findings'][:3],
                    'dtype': dtype_str,
                    'op_sig': list(op_sig),
                })

        # Saturation tracking (based on graph diversity)
        current_gd = len(graph_signatures)
        if current_gd > last_gd:
            last_gd = current_gd
            last_gd_change_iter = i

        # Timeseries
        if i % 10 == 0 or i == 0:
            timeseries.append({
                'iter': i,
                'graph_diversity': current_gd,
                'bugs': len(bugs),
                'trivial': trivial_crashes,
                'gen_failures': gen_failures,
                'total_ok': total_ok,
            })

        # Progress
        if (i + 1) % 200 == 0 or (result.get('bug_type') and not result.get('trivial')):
            elapsed = time.time() - t0
            bug_marker = f" BUG:{result['bug_type']}" if (
                result.get('bug_type') and not result.get('trivial')) else ""
            print(f"  [{i+1:5d}/{max_budget}] GD={current_gd} "
                  f"bugs={len(bugs)} triv={trivial_crashes} fail={gen_failures} "
                  f"t={elapsed:.0f}s{bug_marker}", file=sys.stderr)

        # Time budget check
        if time_budget and (time.time() - t0) >= time_budget:
            print(f"  TIME BUDGET reached at iter {i}: {time.time()-t0:.0f}s",
                  file=sys.stderr)
            final_iter = i + 1
            timeseries.append({
                'iter': i,
                'graph_diversity': current_gd,
                'bugs': len(bugs),
                'trivial': trivial_crashes,
                'gen_failures': gen_failures,
                'total_ok': total_ok,
            })
            break

        # Saturation check
        if i >= min_budget and (i - last_gd_change_iter) >= patience:
            print(f"  SATURATED at iter {i}: GD={current_gd} unchanged "
                  f"for {patience} iterations", file=sys.stderr)
            final_iter = i + 1
            timeseries.append({
                'iter': i,
                'graph_diversity': current_gd,
                'bugs': len(bugs),
                'trivial': trivial_crashes,
                'gen_failures': gen_failures,
                'total_ok': total_ok,
            })
            break

    elapsed = time.time() - t0

    output = {
        'mode': 'nnsmith_tf',
        'seed': seed,
        'budget_used': final_iter,
        'elapsed_sec': round(elapsed, 1),
        'total_ok': total_ok,
        'gen_failures': gen_failures,
        'graph_diversity': len(graph_signatures),
        'bugs_found': len(bugs),
        'trivial_crashes': trivial_crashes,
        'finding_types': dict(finding_types),
        'bugs': bugs,
        'saturated_at': last_gd_change_iter,
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
    parser.add_argument('--time-budget', type=int, default=None,
                        help='Wall-clock time budget in seconds (overrides iteration budget)')
    args = parser.parse_args()

    run_nnsmith_tf(args.seed, args.max_budget, args.min_budget,
                   args.patience, args.device, time_budget=args.time_budget)
