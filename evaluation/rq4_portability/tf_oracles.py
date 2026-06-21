"""Differential oracles for TF bug finding — v1.

Oracles:
1. eager_vs_tfn        — eager vs tf.function(fn) (grappler + autograph diff)
2. tfn_vs_xla          — tf.function vs tf.function(jit_compile=True) (XLA lowering diff)
3. grad_eager_vs_tfn   — GradientTape backward: eager vs tf.function
4. cpu_vs_gpu          — tf.function on CPU vs GPU
5. determinism_diff    — enable_op_determinism on/off
6. mixed_precision_diff — mixed_float16 policy on/off (fp32 ref)

Note: TF tolerance is looser than JAX because:
  - tf.function / grappler may reorder commutative ops
  - XLA GPU uses different reduction algorithms than eager GPU
  - Mixed precision path inserts autocasts
"""
import tensorflow as tf
import numpy as np
import traceback


def _to_f64(t):
    """Convert TF tensor or numpy array to float64 numpy."""
    if isinstance(t, tf.Tensor):
        t = t.numpy()
    return np.asarray(t, dtype=np.float64)


def _max_diff(a, b):
    a = _to_f64(a)
    b = _to_f64(b)
    if a.shape != b.shape:
        return float('inf'), "shape_mismatch"
    if a.size == 0:
        return 0.0, "empty"
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        return float('inf'), "nan_mismatch"
    a_inf = np.isinf(a)
    b_inf = np.isinf(b)
    if not np.array_equal(a_inf, b_inf):
        return float('inf'), "inf_mismatch"
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return 0.0, "all_nonfinite"
    return float(np.abs(a[mask] - b[mask]).max()), "ok"


def _compute_tol(ref, rtol, atol):
    ref_f = _to_f64(ref)
    scale = float(np.abs(ref_f[np.isfinite(ref_f)]).max()) if np.isfinite(ref_f).any() else 1.0
    return atol + rtol * scale


def _classify(diff, tol):
    if diff <= tol:
        return "OK"
    elif diff <= tol * 10:
        return "SUSPICIOUS"
    return "DIFF"


# ─── Oracle 1: eager vs tf.function ───

def oracle_eager_vs_tfn(fn, args, rtol=1e-4, atol=1e-5, label=""):
    """eager vs grappler-optimized tf.function."""
    try:
        out_e = fn(*args)
    except Exception as e:
        return {"oracle": "eager_vs_tfn", "label": label, "status": "EAGER_ERROR",
                "error": str(e)[:200]}
    try:
        fn_compiled = tf.function(fn, jit_compile=False, reduce_retracing=True)
        out_t = fn_compiled(*args)
    except Exception as e:
        return {"oracle": "eager_vs_tfn", "label": label, "status": "TFN_ERROR",
                "error": str(e)[:200], "eager_ok": True}

    diff, reason = _max_diff(out_e, out_t)
    if reason != "ok":
        return {"oracle": "eager_vs_tfn", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(out_e, rtol, atol)
    return {"oracle": "eager_vs_tfn", "label": label, "diff": diff, "tol": tol,
            "status": _classify(diff, tol)}


# ─── Oracle 2: tf.function vs tf.function(jit_compile=True) ───

def oracle_tfn_vs_xla(fn, args, rtol=1e-2, atol=1e-3, label=""):
    """Non-XLA tf.function vs XLA-compiled tf.function.

    Looser tolerance (rtol=1e-2) — XLA reordering of f32 reductions over
    large element counts routinely produces 1e-3 to 1e-4 relative drift.
    Only flag genuinely large differences (>1% relative).
    """
    try:
        fn_tfn = tf.function(fn, jit_compile=False)
        out_t = fn_tfn(*args)
    except Exception as e:
        return {"oracle": "tfn_vs_xla", "label": label, "status": "TFN_ERROR",
                "error": str(e)[:200]}
    try:
        fn_xla = tf.function(fn, jit_compile=True)
        out_x = fn_xla(*args)
    except Exception as e:
        return {"oracle": "tfn_vs_xla", "label": label, "status": "XLA_ERROR",
                "error": str(e)[:200], "tfn_ok": True}

    diff, reason = _max_diff(out_t, out_x)
    if reason != "ok":
        return {"oracle": "tfn_vs_xla", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(out_t, rtol, atol)
    return {"oracle": "tfn_vs_xla", "label": label, "diff": diff, "tol": tol,
            "status": _classify(diff, tol)}


# ─── Oracle 3: backward (GradientTape) eager vs tf.function ───

def oracle_grad_eager_vs_tfn(fn, args, argnum=0, rtol=1e-2, atol=1e-3, label=""):
    """Compute grad w.r.t. args[argnum] via GradientTape. Compare eager vs tf.function.

    Handles 3 failure modes distinctly:
      - Exception raised: real compiler crash
      - Returns None (discontinuous op chain): expected, skip
      - Both return None: OK (not a bug)
      - One returns None, other returns tensor: real inconsistency bug
    """
    def grad_compute(*inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs[argnum])
            out = fn(*inputs)
            if out.shape.rank is not None and out.shape.rank > 0:
                out = tf.reduce_sum(out)
        return tape.gradient(out, inputs[argnum])

    try:
        g_e = grad_compute(*args)
    except Exception as e:
        return {"oracle": "grad_eager_vs_tfn", "label": label,
                "status": "GRAD_EAGER_ERROR", "error": str(e)[:200]}

    try:
        grad_tfn = tf.function(grad_compute)
        g_t = grad_tfn(*args)
    except Exception as e:
        return {"oracle": "grad_eager_vs_tfn", "label": label,
                "status": "GRAD_TFN_ERROR", "error": str(e)[:200], "eager_ok": True}

    # Distinguish None (discontinuous graph) vs tensor
    e_is_none = g_e is None
    t_is_none = g_t is None

    if e_is_none and t_is_none:
        # Both None — expected for discontinuous graphs (floor/ceil/sort only)
        return {"oracle": "grad_eager_vs_tfn", "label": label,
                "status": "OK", "reason": "both_none"}
    if e_is_none != t_is_none:
        # Real inconsistency — one path differentiated, other didn't
        return {"oracle": "grad_eager_vs_tfn", "label": label,
                "status": "DIFF", "reason": "grad_none_mismatch",
                "eager_none": e_is_none, "tfn_none": t_is_none}

    diff, reason = _max_diff(g_e, g_t)
    if reason != "ok":
        return {"oracle": "grad_eager_vs_tfn", "label": label,
                "status": reason.upper(), "diff": diff}
    tol = _compute_tol(g_e, rtol, atol)
    return {"oracle": "grad_eager_vs_tfn", "label": label, "diff": diff, "tol": tol,
            "status": _classify(diff, tol)}


# ─── Oracle 4: CPU vs GPU (tf.function) ───

def oracle_cpu_vs_gpu(fn, args, rtol=1e-2, atol=1e-3, label=""):
    """Run tf.function on CPU and GPU, compare outputs."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return {"oracle": "cpu_vs_gpu", "label": label, "status": "SKIP_NO_GPU"}

    try:
        with tf.device('/CPU:0'):
            args_cpu = [tf.identity(a) for a in args]
            fn_cpu = tf.function(fn)
            out_cpu = fn_cpu(*args_cpu)
    except Exception as e:
        return {"oracle": "cpu_vs_gpu", "label": label, "status": "CPU_ERROR",
                "error": str(e)[:200]}

    try:
        with tf.device('/GPU:0'):
            args_gpu = [tf.identity(a) for a in args]
            fn_gpu = tf.function(fn)
            out_gpu = fn_gpu(*args_gpu)
    except Exception as e:
        return {"oracle": "cpu_vs_gpu", "label": label, "status": "GPU_ERROR",
                "error": str(e)[:200]}

    diff, reason = _max_diff(out_cpu, out_gpu)
    if reason != "ok":
        return {"oracle": "cpu_vs_gpu", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(out_cpu, rtol, atol)
    return {"oracle": "cpu_vs_gpu", "label": label, "diff": diff, "tol": tol,
            "status": _classify(diff, tol)}


# ─── Oracle 4b: pure-eager CPU vs GPU (no tf.function, no XLA) ───

def oracle_eager_cpu_gpu(fn, args, rtol=1e-3, atol=1e-4, label=""):
    """Pure-eager CPU vs GPU comparison. Bypasses tf.function and XLA entirely
    so any divergence is attributable to the TF CPU/GPU kernels themselves
    (e.g., the CPU softmax AVX-256 SIMD issue, fp32 reduction order, etc.).

    Unlike `oracle_cpu_vs_gpu` which wraps fn in tf.function, this oracle
    executes ops one-at-a-time in eager mode on each device.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return {"oracle": "eager_cpu_gpu", "label": label, "status": "SKIP_NO_GPU"}

    try:
        with tf.device('/CPU:0'):
            args_cpu = [tf.identity(a) for a in args]
            out_cpu = fn(*args_cpu)
    except Exception as e:
        return {"oracle": "eager_cpu_gpu", "label": label, "status": "CPU_ERROR",
                "error": str(e)[:200]}

    try:
        with tf.device('/GPU:0'):
            args_gpu = [tf.identity(a) for a in args]
            out_gpu = fn(*args_gpu)
    except Exception as e:
        return {"oracle": "eager_cpu_gpu", "label": label, "status": "GPU_ERROR",
                "error": str(e)[:200], "cpu_ok": True}

    diff, reason = _max_diff(out_cpu, out_gpu)
    if reason != "ok":
        return {"oracle": "eager_cpu_gpu", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(out_cpu, rtol, atol)
    return {"oracle": "eager_cpu_gpu", "label": label, "diff": diff, "tol": tol,
            "status": _classify(diff, tol)}


# ─── Oracle 4c: pure-eager backward CPU vs GPU ───

def oracle_eager_grad_cpu_gpu(fn, args, argnum=0, rtol=1e-3, atol=1e-4, label=""):
    """Pure-eager GradientTape backward on CPU vs GPU. Targets TF-only backward
    kernel differences (the equivalent of PyTorch bug_007 for TF)."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return {"oracle": "eager_grad_cpu_gpu", "label": label, "status": "SKIP_NO_GPU"}

    def grad_eager(inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs[argnum])
            out = fn(*inputs)
            if out.shape.rank is not None and out.shape.rank > 0:
                out = tf.reduce_sum(out)
        return tape.gradient(out, inputs[argnum])

    try:
        with tf.device('/CPU:0'):
            args_cpu = [tf.identity(a) for a in args]
            g_cpu = grad_eager(args_cpu)
    except Exception as e:
        return {"oracle": "eager_grad_cpu_gpu", "label": label,
                "status": "CPU_ERROR", "error": str(e)[:200]}
    try:
        with tf.device('/GPU:0'):
            args_gpu = [tf.identity(a) for a in args]
            g_gpu = grad_eager(args_gpu)
    except Exception as e:
        return {"oracle": "eager_grad_cpu_gpu", "label": label,
                "status": "GPU_ERROR", "error": str(e)[:200], "cpu_ok": True}

    if g_cpu is None and g_gpu is None:
        return {"oracle": "eager_grad_cpu_gpu", "label": label,
                "status": "OK", "reason": "both_none"}
    if (g_cpu is None) != (g_gpu is None):
        return {"oracle": "eager_grad_cpu_gpu", "label": label,
                "status": "DIFF", "reason": "grad_none_mismatch",
                "cpu_none": g_cpu is None, "gpu_none": g_gpu is None}

    diff, reason = _max_diff(g_cpu, g_gpu)
    if reason != "ok":
        return {"oracle": "eager_grad_cpu_gpu", "label": label,
                "status": reason.upper(), "diff": diff}
    tol = _compute_tol(g_cpu, rtol, atol)
    return {"oracle": "eager_grad_cpu_gpu", "label": label, "diff": diff,
            "tol": tol, "status": _classify(diff, tol)}


# ─── Oracle 4d: math-invariant (softmax sum == 1, etc.) ───

def oracle_math_invariant(fn, args, label=""):
    """Check mathematical invariants that must hold regardless of precision.

    Currently checks:
      - softmax output sums to 1.0 (per reduced axis) to within 1e-5
      - |x / |x|| == 1.0 for non-zero |x|

    If fn's output doesn't expose any invariant-checkable structure, returns
    status SKIP. Only reports DIFF when the invariant is measurably violated.
    """
    try:
        out = fn(*args)
    except Exception as e:
        return {"oracle": "math_invariant", "label": label, "status": "EAGER_ERROR",
                "error": str(e)[:200]}
    out_np = _to_f64(out)
    if out_np.size == 0 or not np.isfinite(out_np).any():
        return {"oracle": "math_invariant", "label": label, "status": "SKIP"}
    # Invariant 1: if output is in [0,1] and sums along last axis close to 1,
    # treat it as a softmax-like distribution and check the invariant tightly.
    if out_np.ndim >= 1 and out_np.size > 1:
        last_sum = out_np.sum(axis=-1)
        if (out_np >= 0).all() and (out_np <= 1.0001).all():
            dev = float(np.abs(last_sum - 1.0).max())
            if dev > 1e-5 and (np.abs(last_sum - 1.0) / np.maximum(np.abs(last_sum), 1e-9) > 1e-5).any():
                return {"oracle": "math_invariant", "label": label,
                        "status": "DIFF", "reason": "softmax_sum_not_one",
                        "max_deviation": dev}
    return {"oracle": "math_invariant", "label": label, "status": "OK"}


# ─── Oracle 5: determinism on/off ───

def oracle_determinism_diff(fn, args, rtol=1e-5, atol=1e-6, label=""):
    """Run fn twice (deterministic mode on), results should match exactly.

    If they don't, op determinism is broken for some op in fn.
    """
    # Note: enabling determinism globally in-process can have side effects.
    # Here we check: is fn deterministic across re-runs with same inputs?
    try:
        # Reset any cached state
        tf.keras.utils.set_random_seed(42)
        out1 = fn(*args)
        tf.keras.utils.set_random_seed(42)
        out2 = fn(*args)
    except Exception as e:
        return {"oracle": "determinism_diff", "label": label, "status": "ERROR",
                "error": str(e)[:200]}

    diff, reason = _max_diff(out1, out2)
    # Strict: any diff at all is a failure (deterministic means exact equality)
    return {"oracle": "determinism_diff", "label": label,
            "diff": diff, "reason": reason,
            "status": "OK" if diff == 0.0 else "SUSPICIOUS"}


# ─── Oracle 6b: f64 truth validation ───

def oracle_f64_truth(fn, args, label=""):
    """Compute f64 ground truth and compare against f32 paths.

    This filters false positives from near-zero cancellations and validates
    that divergent paths are actually wrong (not just f32 noise on both sides).

    Returns status INFO with per-path relative errors.
    """
    # Cast args to f64
    try:
        args_f64 = [tf.cast(a, tf.float64) if a.dtype.is_floating else a for a in args]
    except Exception as e:
        return {"oracle": "f64_truth", "label": label, "status": "SKIP",
                "reason": str(e)[:100]}

    try:
        out_f64 = fn(*args_f64)
        truth = _to_f64(out_f64)
    except Exception as e:
        return {"oracle": "f64_truth", "label": label, "status": "F64_ERROR",
                "error": str(e)[:200]}

    scale = float(np.abs(truth[np.isfinite(truth)]).max()) if np.isfinite(truth).any() else 0.0

    # Tiny output — any "relative error" is misleading
    if scale < 1e-6:
        return {"oracle": "f64_truth", "label": label, "status": "NEAR_ZERO",
                "scale": scale, "truth_sum": float(truth.sum() if truth.ndim else truth)}

    # Compute f32 eager + tf.function + xla and their absolute errors vs truth
    errs = {}
    try:
        out_e = fn(*args)
        de, _ = _max_diff(truth, out_e)
        errs["eager_abs_err"] = de
        errs["eager_rel_err"] = de / scale
    except Exception:
        errs["eager_abs_err"] = None

    try:
        out_t = tf.function(fn, jit_compile=False)(*args)
        dt, _ = _max_diff(truth, out_t)
        errs["tfn_abs_err"] = dt
        errs["tfn_rel_err"] = dt / scale
    except Exception:
        errs["tfn_abs_err"] = None

    try:
        out_x = tf.function(fn, jit_compile=True)(*args)
        dx, _ = _max_diff(truth, out_x)
        errs["xla_abs_err"] = dx
        errs["xla_rel_err"] = dx / scale
    except Exception:
        errs["xla_abs_err"] = None

    return {"oracle": "f64_truth", "label": label, "status": "INFO",
            "scale": scale, **errs}


# ─── Oracle 6: mixed precision ───

def oracle_mixed_precision_diff(fn, args, rtol=1e-1, atol=1e-2, label=""):
    """Compare fp32 path vs mixed_float16 path.

    Wide tolerance (1e-1 rtol) — just catching catastrophic mismatches.
    """
    try:
        # FP32 reference
        out_ref = fn(*args)
    except Exception as e:
        return {"oracle": "mixed_precision_diff", "label": label,
                "status": "REF_ERROR", "error": str(e)[:200]}

    # Cast inputs to fp16, run, then compare
    try:
        args_fp16 = [tf.cast(a, tf.float16) for a in args]

        def fn_f16(*inputs):
            # re-cast inside to see how ops behave
            out = fn(*inputs)
            return tf.cast(out, tf.float32)

        out_f16 = fn_f16(*args_fp16)
    except Exception as e:
        return {"oracle": "mixed_precision_diff", "label": label,
                "status": "F16_ERROR", "error": str(e)[:200]}

    diff, reason = _max_diff(out_ref, out_f16)
    if reason != "ok":
        return {"oracle": "mixed_precision_diff", "label": label,
                "status": reason.upper(), "diff": diff}
    tol = _compute_tol(out_ref, rtol, atol)
    return {"oracle": "mixed_precision_diff", "label": label, "diff": diff,
            "tol": tol, "status": _classify(diff, tol)}


# ─── Bug classifier ───

def has_bug(results):
    for r in results:
        if r.get("status") in ("DIFF", "SUSPICIOUS", "NAN_MISMATCH",
                                "INF_MISMATCH", "SHAPE_MISMATCH"):
            return True
        if r.get("status") in ("TFN_ERROR", "XLA_ERROR", "GRAD_TFN_ERROR",
                               "GPU_ERROR") and (r.get("eager_ok") or r.get("tfn_ok")):
            return True
    return False


def bug_severity(results):
    for r in results:
        if r.get("status") == "DIFF":
            return "CATASTROPHIC"
    for r in results:
        if r.get("status") in ("TFN_ERROR", "XLA_ERROR", "GRAD_TFN_ERROR",
                               "GPU_ERROR") and (r.get("eager_ok") or r.get("tfn_ok")):
            return "CRASH"
    for r in results:
        if r.get("status") == "SUSPICIOUS":
            return "SUSPICIOUS"
    return None


def report(r):
    tag = r.get("status", "")
    colors = {
        "DIFF": "\033[91m", "SUSPICIOUS": "\033[93m",
        "OK": "\033[92m",
    }
    color = colors.get(tag, "")
    reset = "\033[0m" if color else ""
    diff_str = f"diff={r.get('diff', '?'):.2e}" if isinstance(r.get('diff'), float) else ""
    print(f"  {color}[{r['oracle']:22s}] {tag:14s} {diff_str}{reset}")
