"""Differential oracles for JAX bug finding — v2.

Oracles:
1. jit_vs_eager     — pure fn: jit and eager must match
2. grad_jit_vs_eager — backward: grad(fn) jit vs eager
3. vmap_vs_loop     — vmap(fn) vs manual loop over batch dim
4. cpu_vs_gpu       — jit on CPU vs GPU
5. precision_matrix — matmul precision modes
6. config_diff      — x64 on/off, deterministic, etc.
"""
import jax
import jax.numpy as jnp
import numpy as np
import traceback


def _to_f64(x):
    """Safely convert to float64 numpy array."""
    return np.asarray(jax.device_get(x), dtype=np.float64)


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
    diff = np.abs(a[mask] - b[mask])
    return float(diff.max()), "ok"


def _compute_tol(ref, rtol, atol):
    ref_f = _to_f64(ref)
    scale = float(np.abs(ref_f[np.isfinite(ref_f)]).max()) if np.isfinite(ref_f).any() else 1.0
    return atol + rtol * scale


def _classify(diff, tol):
    """Graded divergence: OK / SUSPICIOUS / DIFF."""
    if diff <= tol:
        return "OK"
    elif diff <= tol * 10:
        return "SUSPICIOUS"
    else:
        return "DIFF"


# ─── Oracle 1: jit vs eager ───

def oracle_jit_vs_eager(fn, args, rtol=1e-5, atol=1e-5, label=""):
    try:
        out_e = fn(*args)
        jax.block_until_ready(out_e)
    except Exception as e:
        return {"oracle": "jit_vs_eager", "label": label, "status": "EAGER_ERROR",
                "error": str(e)[:200]}
    try:
        jax.clear_caches()
        out_j = jax.jit(fn)(*args)
        jax.block_until_ready(out_j)
    except Exception as e:
        return {"oracle": "jit_vs_eager", "label": label, "status": "JIT_ERROR",
                "error": str(e)[:200], "eager_ok": True}

    diff, reason = _max_diff(out_e, out_j)
    if reason != "ok":
        return {"oracle": "jit_vs_eager", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(out_e, rtol, atol)
    status = _classify(diff, tol)
    return {"oracle": "jit_vs_eager", "label": label, "diff": diff, "tol": tol,
            "status": status}


# ─── Oracle 2: backward (grad) jit vs eager ───

def oracle_grad_jit_vs_eager(fn, args, argnums=0, rtol=1e-4, atol=1e-5, label=""):
    """Compare jax.grad(fn) eager vs jit."""
    try:
        grad_fn = jax.grad(fn, argnums=argnums)
    except Exception as e:
        return {"oracle": "grad_jit_vs_eager", "label": label, "status": "GRAD_SETUP_ERROR",
                "error": str(e)[:200]}

    try:
        g_e = grad_fn(*args)
        jax.block_until_ready(g_e)
    except Exception as e:
        return {"oracle": "grad_jit_vs_eager", "label": label, "status": "GRAD_EAGER_ERROR",
                "error": str(e)[:200]}

    try:
        jax.clear_caches()
        g_j = jax.jit(grad_fn)(*args)
        jax.block_until_ready(g_j)
    except Exception as e:
        return {"oracle": "grad_jit_vs_eager", "label": label, "status": "GRAD_JIT_ERROR",
                "error": str(e)[:200], "eager_ok": True}

    diff, reason = _max_diff(g_e, g_j)
    if reason != "ok":
        return {"oracle": "grad_jit_vs_eager", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(g_e, rtol, atol)
    status = _classify(diff, tol)
    return {"oracle": "grad_jit_vs_eager", "label": label, "diff": diff, "tol": tol,
            "status": status}


# ─── Oracle 3: vmap vs manual loop ───

def oracle_vmap_vs_loop(fn, args, rtol=1e-5, atol=1e-5, label=""):
    """Compare vmap(fn) vs manual loop over first axis.

    Only works if all args have at least 1 dimension.
    """
    # Check all args have batch dim
    try:
        batch_size = args[0].shape[0]
        for a in args:
            if a.shape[0] != batch_size:
                return {"oracle": "vmap_vs_loop", "label": label, "status": "SKIP_SHAPE"}
    except (IndexError, AttributeError):
        return {"oracle": "vmap_vs_loop", "label": label, "status": "SKIP_SHAPE"}

    if batch_size > 16:
        # Too slow/memory-heavy to loop
        return {"oracle": "vmap_vs_loop", "label": label, "status": "SKIP_LARGE_BATCH"}

    # vmap result
    try:
        out_vmap = jax.jit(jax.vmap(fn))(*args)
        jax.block_until_ready(out_vmap)
    except Exception as e:
        return {"oracle": "vmap_vs_loop", "label": label, "status": "VMAP_ERROR",
                "error": str(e)[:200]}

    # manual loop
    try:
        outs = []
        for i in range(batch_size):
            sliced = tuple(a[i] for a in args)
            outs.append(jax.jit(fn)(*sliced))
        out_loop = jnp.stack(outs)
        jax.block_until_ready(out_loop)
    except Exception as e:
        return {"oracle": "vmap_vs_loop", "label": label, "status": "LOOP_ERROR",
                "error": str(e)[:200]}

    diff, reason = _max_diff(out_vmap, out_loop)
    if reason != "ok":
        return {"oracle": "vmap_vs_loop", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(out_loop, rtol, atol)
    status = _classify(diff, tol)
    return {"oracle": "vmap_vs_loop", "label": label, "diff": diff, "tol": tol,
            "status": status}


# ─── Oracle 4: CPU vs GPU ───

def oracle_cpu_vs_gpu(fn, args, rtol=1e-4, atol=1e-5, label=""):
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        return {"oracle": "cpu_vs_gpu", "label": label, "status": "SKIP_NO_GPU"}

    args_cpu = jax.tree.map(lambda x: jax.device_put(x, cpu), args)
    args_gpu = jax.tree.map(lambda x: jax.device_put(x, gpu), args)

    try:
        out_cpu = jax.jit(fn, device=cpu)(*args_cpu)
        jax.block_until_ready(out_cpu)
    except Exception as e:
        return {"oracle": "cpu_vs_gpu", "label": label, "status": "CPU_ERROR",
                "error": str(e)[:200]}

    try:
        out_gpu = jax.jit(fn, device=gpu)(*args_gpu)
        jax.block_until_ready(out_gpu)
    except Exception as e:
        return {"oracle": "cpu_vs_gpu", "label": label, "status": "GPU_ERROR",
                "error": str(e)[:200]}

    diff, reason = _max_diff(out_cpu, out_gpu)
    if reason != "ok":
        return {"oracle": "cpu_vs_gpu", "label": label, "status": reason.upper(),
                "diff": diff}
    tol = _compute_tol(out_cpu, rtol, atol)
    status = _classify(diff, tol)
    return {"oracle": "cpu_vs_gpu", "label": label, "diff": diff, "tol": tol,
            "status": status}


# ─── Oracle 5: matmul precision matrix ───

def oracle_precision_matrix(fn, args, label=""):
    results = {}
    for precision in ["default", "high", "highest"]:
        try:
            with jax.default_matmul_precision(precision):
                jax.clear_caches()
                out = jax.jit(fn)(*args)
                jax.block_until_ready(out)
                results[precision] = _to_f64(out)
        except Exception as e:
            return {"oracle": "precision_matrix", "label": label,
                    "status": f"ERROR_{precision.upper()}", "error": str(e)[:200]}

    diff_dh = float(np.abs(results["default"] - results["high"]).max())
    diff_hh = float(np.abs(results["high"] - results["highest"]).max())
    diff_dH = float(np.abs(results["default"] - results["highest"]).max())

    return {
        "oracle": "precision_matrix", "label": label,
        "default_vs_high": diff_dh, "high_vs_highest": diff_hh,
        "default_vs_highest": diff_dH,
        "status": "INFO",
    }


# ─── Oracle 6: config differential ───

def oracle_config_diff(fn, args, rtol=1e-5, atol=1e-5, label=""):
    """Run with different JAX configs and compare.

    Currently tests: x64 enabled vs disabled.
    """
    results = {}

    # x64 enabled (current default in fuzzer)
    try:
        jax.clear_caches()
        out = jax.jit(fn)(*args)
        jax.block_until_ready(out)
        results["x64_on"] = out
    except Exception as e:
        return {"oracle": "config_diff", "label": label, "status": "ERROR_X64_ON",
                "error": str(e)[:200]}

    # x64 disabled — re-cast inputs to float32
    try:
        args32 = tuple(
            a.astype(jnp.float32) if hasattr(a, 'dtype') and a.dtype == jnp.float64 else a
            for a in args
        )
        jax.clear_caches()
        out32 = jax.jit(fn)(*args32)
        jax.block_until_ready(out32)
        results["f32_only"] = out32
    except Exception as e:
        return {"oracle": "config_diff", "label": label, "status": "ERROR_F32",
                "error": str(e)[:200]}

    # Compare: large diff between f64 and f32 paths might indicate precision sensitivity
    diff, reason = _max_diff(results["x64_on"], results["f32_only"])
    return {
        "oracle": "config_diff", "label": label,
        "diff_x64_vs_f32": diff, "reason": reason,
        "status": "INFO",
    }


# ─── Multi-oracle runner ───

def run_all_oracles(fn, args, label="", run_backward=True, run_vmap=True,
                    run_cpu_gpu=False, run_precision=False, run_config=False):
    """Run multiple oracles and return list of results.

    Default: jit_vs_eager + backward + vmap (the fast ones).
    CPU vs GPU and precision matrix are slower, opt-in.
    """
    results = []

    # Always run jit vs eager
    r = oracle_jit_vs_eager(fn, args, label=label)
    results.append(r)

    # Backward oracle — fn must return scalar
    if run_backward:
        r = oracle_grad_jit_vs_eager(fn, args, label=label)
        results.append(r)

    # Vmap oracle — needs batch dim
    if run_vmap and all(hasattr(a, 'shape') and len(a.shape) >= 1 for a in args):
        r = oracle_vmap_vs_loop(fn, args, label=label)
        results.append(r)

    # CPU vs GPU
    if run_cpu_gpu:
        r = oracle_cpu_vs_gpu(fn, args, label=label)
        results.append(r)

    # Precision matrix
    if run_precision:
        r = oracle_precision_matrix(fn, args, label=label)
        results.append(r)

    # Config diff
    if run_config:
        r = oracle_config_diff(fn, args, label=label)
        results.append(r)

    return results


def has_bug(results):
    """Check if any oracle found a DIFF or SUSPICIOUS result."""
    for r in results:
        if r.get("status") in ("DIFF", "SUSPICIOUS", "NAN_MISMATCH",
                                "INF_MISMATCH", "SHAPE_MISMATCH"):
            return True
        # JIT errors when eager works = potential crash bug
        if r.get("status") in ("JIT_ERROR", "GRAD_JIT_ERROR", "VMAP_ERROR") and r.get("eager_ok"):
            return True
    return False


def bug_severity(results):
    """Return severity level: 'CATASTROPHIC', 'SUSPICIOUS', 'CRASH', or None."""
    for r in results:
        if r.get("status") == "DIFF":
            return "CATASTROPHIC"
    for r in results:
        if r.get("status") in ("JIT_ERROR", "GRAD_JIT_ERROR", "VMAP_ERROR") and r.get("eager_ok"):
            return "CRASH"
    for r in results:
        if r.get("status") == "SUSPICIOUS":
            return "SUSPICIOUS"
    return None


def report(r):
    tag = r.get("status", "")
    colors = {
        "DIFF": "\033[91m", "SUSPICIOUS": "\033[93m", "CATASTROPHIC": "\033[91m",
        "OK": "\033[92m", "INFO": "\033[94m",
    }
    color = colors.get(tag, "")
    reset = "\033[0m" if color else ""
    diff_str = f"diff={r.get('diff', '?'):.2e}" if isinstance(r.get('diff'), float) else ""
    print(f"  {color}[{r['oracle']:22s}] {tag:14s} {diff_str}{reset}")
