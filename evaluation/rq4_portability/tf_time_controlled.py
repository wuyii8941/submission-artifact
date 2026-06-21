#!/usr/bin/env python3
"""
TF Time-Controlled Experiment: targeted vs random generation.

Compares:
  - Targeted: uses the 121 template cases from targeted_consistency_fuzzer.py
    (structured by T1-T7 categories targeting specific compilation seams)
  - Random: random TF ops with random shapes through the same 3-way oracle
    (eager vs Grappler vs XLA)

Metric: cross-pipeline inconsistencies (diff triggers) found in fixed time budget.

This validates the state-targeting principle at pipeline-level granularity
(TensorFlow) in addition to kernel-level (PyTorch).

Output: results/tf_time_controlled_results.json
"""

import argparse
import json
import os
import sys
import time
import random as pyrandom
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent
TF_CROSS_DIR = Path(os.environ.get("TF_CROSS_DIR", str(SCRIPT_DIR)))
sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Oracle: 3-way differential (eager vs Grappler vs XLA)
# ---------------------------------------------------------------------------

def run_3way(fn, inputs, atol=1e-5, rtol=1e-4):
    """Run function in 3 modes, compare outputs. Returns finding dict or None."""
    result = {'eager_ok': False, 'grappler_ok': False, 'xla_ok': False,
              'findings': []}

    try:
        # Eager
        eager_out = fn(*[tf.identity(x) if isinstance(x, tf.Tensor) else x for x in inputs])
        result['eager_ok'] = True
    except Exception as e:
        result['eager_error'] = f"{type(e).__name__}: {str(e)[:200]}"
        return result

    try:
        # Grappler (tf.function without XLA)
        grappler_fn = tf.function(fn, jit_compile=False)
        grappler_out = grappler_fn(*[tf.identity(x) if isinstance(x, tf.Tensor) else x for x in inputs])
        result['grappler_ok'] = True
    except Exception as e:
        result['grappler_error'] = f"{type(e).__name__}: {str(e)[:200]}"
        result['findings'].append({
            'type': 'grappler_crash',
            'error': f"{type(e).__name__}: {str(e)[:100]}",
        })
        result['outcome_pattern'] = (0, -1, -1)  # grappler crash
        return result

    try:
        # XLA (jit_compile=True)
        xla_fn = tf.function(fn, jit_compile=True)
        xla_out = xla_fn(*[tf.identity(x) if isinstance(x, tf.Tensor) else x for x in inputs])
        result['xla_ok'] = True
    except Exception as e:
        result['xla_error'] = f"{type(e).__name__}: {str(e)[:200]}"
        result['findings'].append({
            'type': 'xla_crash',
            'error': f"{type(e).__name__}: {str(e)[:100]}",
        })
        result['outcome_pattern'] = (1, 0, 0)  # xla crash, eager≈grappler
        return result

    # Compare outputs
    def to_np(x):
        if isinstance(x, tf.Tensor):
            return x.numpy()
        if isinstance(x, (tuple, list)):
            return [to_np(v) for v in x]
        return np.asarray(x)

    def compare(a, b, name):
        a_np = to_np(a)
        b_np = to_np(b)
        if isinstance(a_np, list):
            for i, (aa, bb) in enumerate(zip(a_np, b_np)):
                finding = compare_arrays(aa, bb, f"{name}[{i}]", atol, rtol)
                if finding:
                    return finding
            return None
        return compare_arrays(a_np, b_np, name, atol, rtol)

    def compare_arrays(a, b, name, atol, rtol):
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return None
        if a.shape != b.shape:
            return {'type': 'shape_mismatch', 'pair': name, 'a': list(a.shape), 'b': list(b.shape)}
        if a.dtype.kind in 'fc':
            aa = a.astype(np.float64)
            bb = b.astype(np.float64)
            # NaN mismatch
            a_nan = np.isnan(aa)
            b_nan = np.isnan(bb)
            if not np.array_equal(a_nan, b_nan):
                return {'type': 'nan_mismatch', 'pair': name,
                        'a_nan': int(np.sum(a_nan)), 'b_nan': int(np.sum(b_nan))}
            mask = np.isfinite(aa) & np.isfinite(bb)
            if np.any(mask):
                diff = np.abs(aa[mask] - bb[mask])
                max_diff = float(np.max(diff))
                if max_diff > atol + rtol * np.max(np.abs(bb[mask])):
                    return {'type': 'numerical_diff', 'pair': name,
                            'max_diff': max_diff}
        else:
            if not np.array_equal(a, b):
                return {'type': 'value_mismatch', 'pair': name}
        return None

    # eager vs grappler
    finding = compare(eager_out, grappler_out, 'eager_vs_grappler')
    if finding:
        result['findings'].append(finding)

    # eager vs xla
    finding = compare(eager_out, xla_out, 'eager_vs_xla')
    if finding:
        result['findings'].append(finding)

    # grappler vs xla
    finding = compare(grappler_out, xla_out, 'grappler_vs_xla')
    if finding:
        result['findings'].append(finding)

    # Compilation outcome pattern: 3-bit (eg=match?, ex=match?, gx=match?)
    eg = 1 if not any(f.get('pair') == 'eager_vs_grappler' for f in result['findings']) else 0
    ex = 1 if not any(f.get('pair') == 'eager_vs_xla' for f in result['findings']) else 0
    gx = 1 if not any(f.get('pair') == 'grappler_vs_xla' for f in result['findings']) else 0
    result['outcome_pattern'] = (eg, ex, gx)

    return result


# ---------------------------------------------------------------------------
# Random TF program generator
# ---------------------------------------------------------------------------

# Named ops for graph-op-signature extraction (parallel to PyTorch fused_op_set)
RANDOM_OPS_UNARY = [
    ('sin',        lambda x: tf.math.sin(x)),
    ('cos',        lambda x: tf.math.cos(x)),
    ('exp',        lambda x: tf.math.exp(x)),
    ('log',        lambda x: tf.math.log(tf.abs(x) + 1e-8)),
    ('relu',       lambda x: tf.nn.relu(x)),
    ('sigmoid',    lambda x: tf.nn.sigmoid(x)),
    ('tanh',       lambda x: tf.math.tanh(x)),
    ('abs',        lambda x: tf.math.abs(x)),
    ('negative',   lambda x: tf.math.negative(x)),
    ('sqrt',       lambda x: tf.math.sqrt(tf.abs(x))),
    ('square',     lambda x: tf.math.square(x)),
    ('sign',       lambda x: tf.math.sign(x)),
    ('elu',        lambda x: tf.nn.elu(x)),
    ('selu',       lambda x: tf.nn.selu(x)),
    ('erf',        lambda x: tf.math.erf(x)),
    ('erfc',       lambda x: tf.math.erfc(x)),
    ('floor',      lambda x: tf.math.floor(x)),
    ('ceil',       lambda x: tf.math.ceil(x)),
    ('reciprocal', lambda x: tf.math.reciprocal(tf.abs(x) + 1e-8)),
]

RANDOM_OPS_REDUCE = [
    ('reduce_sum',  lambda x: tf.reduce_sum(x, axis=-1)),
    ('reduce_mean', lambda x: tf.reduce_mean(x, axis=-1)),
    ('reduce_max',  lambda x: tf.reduce_max(x, axis=-1)),
    ('reduce_min',  lambda x: tf.reduce_min(x, axis=-1)),
    ('reduce_prod', lambda x: tf.reduce_prod(x, axis=-1)),
    ('reduce_var',  lambda x: tf.math.reduce_variance(x, axis=-1)),
    ('reduce_std',  lambda x: tf.math.reduce_std(x, axis=-1)),
    ('argmin',      lambda x: tf.math.argmin(x, axis=-1)),
    ('argmax',      lambda x: tf.math.argmax(x, axis=-1)),
]

RANDOM_OPS_SCAN = [
    ('cumsum',  lambda x: tf.math.cumsum(x, axis=-1)),
    ('cumprod', lambda x: tf.math.cumprod(tf.nn.sigmoid(x), axis=-1)),
]

DTYPES = [tf.float32] * 60 + [tf.float64] * 15 + [tf.float16] * 10 + [tf.complex64] * 10 + [tf.bfloat16] * 5


def generate_random_tf_program(seed):
    """Generate a random TF function + inputs. Returns (fn, inputs, meta).
    meta includes 'op_sig': frozenset of op names used (for graph diversity)."""
    pyrandom.seed(seed)
    np.random.seed(seed % (2**31))

    dtype = pyrandom.choice(DTYPES)
    ndim = pyrandom.choice([1, 2, 3, 4])
    shape = tuple(pyrandom.randint(1, 256) for _ in range(ndim))

    # Clamp
    total = 1
    for s in shape:
        total *= s
    if total > 2**20:
        scale = (2**20 / total) ** (1.0 / ndim)
        shape = tuple(max(1, int(s * scale)) for s in shape)

    prog_type = pyrandom.choices(
        ['pointwise', 'reduce', 'scan', 'matmul', 'mixed'],
        weights=[30, 25, 10, 15, 20],
        k=1
    )[0]

    op_names = []

    if dtype.is_complex:
        x_np = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(
            np.complex64 if dtype == tf.complex64 else np.complex128)
    else:
        x_np = np.random.randn(*shape).astype(np.float32)

    x = tf.constant(x_np, dtype=dtype)
    inputs = (x,)

    if prog_type == 'pointwise':
        n_ops = pyrandom.randint(1, 4)
        chosen = [pyrandom.choice(RANDOM_OPS_UNARY) for _ in range(n_ops)]
        op_names = [c[0] for c in chosen]
        ops = [c[1] for c in chosen]
        def fn(x, _ops=ops):
            for op in _ops:
                x = op(x)
            return x

    elif prog_type == 'reduce':
        pre_chosen = [pyrandom.choice(RANDOM_OPS_UNARY) for _ in range(pyrandom.randint(0, 2))]
        red_chosen = pyrandom.choice(RANDOM_OPS_REDUCE)
        op_names = [c[0] for c in pre_chosen] + [red_chosen[0]]
        pre_ops = [c[1] for c in pre_chosen]
        red_op = red_chosen[1]
        def fn(x, _pre=pre_ops, _red=red_op):
            for op in _pre:
                x = op(x)
            return _red(x)

    elif prog_type == 'scan':
        scan_chosen = pyrandom.choice(RANDOM_OPS_SCAN)
        post_chosen = pyrandom.choice(RANDOM_OPS_UNARY)
        op_names = [scan_chosen[0], post_chosen[0]]
        scan_op = scan_chosen[1]
        post_op = post_chosen[1]
        def fn(x, _scan=scan_op, _post=post_op):
            return _post(_scan(x))

    elif prog_type == 'matmul':
        M = pyrandom.randint(2, 128)
        K = pyrandom.randint(2, 128)
        N = pyrandom.randint(2, 128)
        if dtype.is_complex:
            a_np = (np.random.randn(M, K) + 1j * np.random.randn(M, K)).astype(np.complex64)
            b_np = (np.random.randn(K, N) + 1j * np.random.randn(K, N)).astype(np.complex64)
        else:
            a_np = np.random.randn(M, K).astype(np.float32)
            b_np = np.random.randn(K, N).astype(np.float32)
        a = tf.constant(a_np, dtype=dtype)
        b = tf.constant(b_np, dtype=dtype)
        inputs = (a, b)
        post_chosen = pyrandom.choice(RANDOM_OPS_UNARY)
        op_names = ['matmul', post_chosen[0]]
        post_op = post_chosen[1]
        def fn(a, b, _post=post_op):
            return _post(tf.matmul(a, b))
        meta_shape = (M, K, N)

    else:  # mixed
        chosen = [pyrandom.choice(RANDOM_OPS_UNARY) for _ in range(pyrandom.randint(1, 3))]
        red_chosen = pyrandom.choice(RANDOM_OPS_REDUCE)
        op_names = [c[0] for c in chosen] + [red_chosen[0]]
        ops = [c[1] for c in chosen]
        red_op = red_chosen[1]
        def fn(x, _ops=ops, _red=red_op):
            for op in _ops:
                x = op(x)
            return _red(x)

    meta = {'seed': seed, 'dtype': str(dtype.name),
            'shape': meta_shape if prog_type == 'matmul' else shape,
            'prog_type': prog_type, 'op_sig': tuple(sorted(set(op_names)))}

    return fn, inputs, meta


# ---------------------------------------------------------------------------
# Targeted TF: load cases from targeted_consistency_fuzzer
# ---------------------------------------------------------------------------

def load_targeted_cases(seed, max_per_group=10):
    """Load targeted cases from the TF fuzzer."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from targeted_consistency_fuzzer import all_cases
    cases = all_cases(seed)
    return cases


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(mode, seed, time_budget, device='cpu'):
    """Run one (mode, seed) experiment."""
    results = {
        'mode': mode,
        'seed': seed,
        'time_budget': time_budget,
        'iterations': 0,
        'findings': [],
        'pipeline_states_seen': set(),
        'finding_types': Counter(),
        'errors': 0,
    }
    # Graph diversity tracking: σ(P) = (op_sig, dtype, outcome_pattern)
    graph_signatures = set()
    diversity_curve = []  # [(time, diversity_count), ...]

    if mode == 'targeted':
        cases = load_targeted_cases(seed)
        pyrandom.seed(seed)
        pyrandom.shuffle(cases)

    t0 = time.time()
    i = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break

        try:
            if mode == 'targeted':
                case = cases[i % len(cases)]
                # Vary seed for different input instances
                np.random.seed(seed + i)
                fn = case.fn
                input_set = case.input_sets[i % len(case.input_sets)]
                meta = {'group': case.group, 'name': case.name, 'seed': seed + i,
                        'tags': case.tags}
                oracle_result = run_3way(fn, input_set, atol=case.atol, rtol=case.rtol)

            else:  # random
                fn, inputs, meta = generate_random_tf_program(seed * 100000 + i)
                oracle_result = run_3way(fn, inputs)

        except Exception as e:
            results['errors'] += 1
            i += 1
            continue

        results['iterations'] += 1

        # Track pipeline states reached
        if oracle_result['eager_ok']:
            results['pipeline_states_seen'].add('eager')
        if oracle_result['grappler_ok']:
            results['pipeline_states_seen'].add('grappler')
        if oracle_result['xla_ok']:
            results['pipeline_states_seen'].add('xla')

        # Track findings
        if oracle_result['findings']:
            for f in oracle_result['findings']:
                finding = {
                    'iter': i,
                    'time_sec': round(elapsed, 1),
                    'type': f['type'],
                    'meta': {k: str(v) for k, v in meta.items()},
                }
                if 'max_diff' in f:
                    finding['max_diff'] = f['max_diff']
                if 'pair' in f:
                    finding['pair'] = f['pair']
                results['findings'].append(finding)
                results['finding_types'][f['type']] += 1

        # Graph diversity: σ(P) = (op_sig, dtype, outcome_pattern)
        # op_sig: analogous to fused_op_set in PyTorch kernel template τ(k)
        # dtype: input dtype regime
        # outcome_pattern: compiler behavior observable (which pipelines agree/disagree)
        outcome = oracle_result.get('outcome_pattern', (1, 1, 1))
        if mode == 'targeted':
            op_sig = (meta.get('group', ''), meta.get('name', ''))
            # Extract dtype from actual input tensors (compiler-side observable)
            dtype_str = 'float32'
            for inp in input_set:
                if isinstance(inp, tf.Tensor):
                    dtype_str = str(inp.dtype.name)
                    break
        else:
            op_sig = meta.get('op_sig', ())
            dtype_str = meta.get('dtype', 'float32')
        sigma = (op_sig, dtype_str, outcome)
        graph_signatures.add(sigma)

        # Record diversity curve at regular intervals
        if (i + 1) % 10 == 0:
            diversity_curve.append((round(elapsed, 1), len(graph_signatures)))

        if (i + 1) % 50 == 0:
            n_findings = len(results['findings'])
            print(f"  [{mode}][{i+1:5d}] findings={n_findings} "
                  f"GD={len(graph_signatures)} "
                  f"t={elapsed:.0f}/{time_budget}s", file=sys.stderr, flush=True)

        i += 1

    elapsed = time.time() - t0

    # Final diversity snapshot
    diversity_curve.append((round(elapsed, 1), len(graph_signatures)))

    # Convert sets/counters for JSON
    output = {
        'mode': mode,
        'seed': seed,
        'time_budget': time_budget,
        'elapsed_sec': round(elapsed, 1),
        'iterations': results['iterations'],
        'errors': results['errors'],
        'n_findings': len(results['findings']),
        'pipeline_states_seen': sorted(results['pipeline_states_seen']),
        'finding_types': dict(results['finding_types']),
        'graph_diversity': len(graph_signatures),
        'diversity_curve': diversity_curve,
        'findings': results['findings'],
    }
    return output


def run_single_subprocess(mode, seed, time_budget):
    """Run one experiment in an isolated subprocess to avoid XLA memory leaks."""
    import subprocess as sp
    tmp_out = SCRIPT_DIR / 'results' / f'_tmp_{mode}_{seed}.json'
    cmd = [sys.executable, __file__, '--single', mode, str(seed),
           str(time_budget), str(tmp_out)]
    proc = sp.run(cmd, capture_output=True, text=True,
                  timeout=time_budget + 120)
    if proc.returncode != 0:
        print(f"  SUBPROCESS FAILED (rc={proc.returncode}): {proc.stderr[-500:]}", file=sys.stderr)
        return None
    with open(tmp_out) as f:
        result = json.load(f)
    tmp_out.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time-budget', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--modes', nargs='+', default=['targeted', 'random'])
    parser.add_argument('--single', nargs=4, metavar=('MODE', 'SEED', 'BUDGET', 'OUTPATH'),
                        help='Internal: run single experiment in subprocess')
    args = parser.parse_args()

    # Subprocess worker mode
    if args.single:
        mode, seed, budget, outpath = args.single
        result = run_experiment(mode, int(seed), int(budget))
        with open(outpath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        return

    all_results = []
    total = len(args.modes) * len(args.seeds)
    done = 0

    for mode in args.modes:
        for seed in args.seeds:
            done += 1
            print(f"\n[{done}/{total}] {mode} seed={seed} budget={args.time_budget}s",
                  file=sys.stderr, flush=True)
            result = run_single_subprocess(mode, seed, args.time_budget)
            if result is None:
                print(f"  -> SKIPPED (subprocess crash)", file=sys.stderr, flush=True)
                continue
            all_results.append(result)
            gd = result.get('graph_diversity', '?')
            print(f"  -> findings={result['n_findings']} GD={gd} iters={result['iterations']} "
                  f"types={result['finding_types']}", file=sys.stderr, flush=True)

    # Aggregate
    agg = {}
    for mode in args.modes:
        runs = [r for r in all_results if r['mode'] == mode]
        if not runs:
            continue
        findings = [r['n_findings'] for r in runs]
        iters = [r['iterations'] for r in runs]
        gd_vals = [r.get('graph_diversity', 0) for r in runs]

        crash_counts = [sum(1 for f in r['findings'] if 'crash' in f['type']) for r in runs]
        diff_counts = [sum(1 for f in r['findings'] if 'crash' not in f['type']) for r in runs]

        agg[mode] = {
            'total_findings': {'mean': round(float(np.mean(findings)), 1),
                              'std': round(float(np.std(findings)), 1),
                              'values': findings},
            'crash_findings': {'mean': round(float(np.mean(crash_counts)), 1),
                              'std': round(float(np.std(crash_counts)), 1),
                              'values': crash_counts},
            'diff_findings': {'mean': round(float(np.mean(diff_counts)), 1),
                             'std': round(float(np.std(diff_counts)), 1),
                             'values': diff_counts},
            'graph_diversity': {'mean': round(float(np.mean(gd_vals)), 1),
                               'std': round(float(np.std(gd_vals)), 1),
                               'values': gd_vals},
            'iterations': {'mean': round(float(np.mean(iters)), 1),
                          'values': iters},
        }

    output = {
        'experiment': 'tf_time_controlled',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'time_budget_sec': args.time_budget,
            'seeds': args.seeds,
            'modes': args.modes,
        },
        'per_run': all_results,
        'aggregated': agg,
    }

    results_dir = SCRIPT_DIR / 'results'
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / 'tf_time_controlled_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Results: {out_path}", file=sys.stderr)
    for mode in args.modes:
        if mode in agg:
            a = agg[mode]
            print(f"  {mode}: findings={a['total_findings']['mean']}±{a['total_findings']['std']} "
                  f"GD={a['graph_diversity']['mean']}±{a['graph_diversity']['std']} "
                  f"(crash={a['crash_findings']['mean']}, diff={a['diff_findings']['mean']}) "
                  f"iters={a['iterations']['mean']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    print(json.dumps({'aggregated': agg}, indent=2, default=str))


if __name__ == '__main__':
    main()
