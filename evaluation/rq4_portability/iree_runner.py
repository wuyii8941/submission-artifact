"""IREE compile + run utilities for StableHLO differential fuzzing.

Compiles StableHLO MLIR → IREE flatbuffer → runs on specified backend.
Supports multi-backend differential (llvm-cpu, cuda, vulkan).

Key optimization: compile once per (mlir, backend), reuse for multiple inputs.
"""
import numpy as np
import traceback
from typing import Dict, List, Optional, Tuple

import iree.compiler as ireec
import iree.runtime as ireert

_compile_cache: Dict[Tuple[str, str], Optional[bytes]] = {}
_compile_errors: Dict[Tuple[str, str], str] = {}


def compile_stablehlo(mlir_text: str, backend: str = "llvm-cpu") -> Optional[bytes]:
    """Compile StableHLO MLIR to IREE flatbuffer (cached)."""
    key = (mlir_text, backend)
    if key in _compile_cache:
        return _compile_cache[key]
    try:
        extra_args = []
        if backend == "llvm-cpu":
            extra_args = ["--iree-llvmcpu-target-cpu=host"]
        elif backend == "cuda":
            extra_args = ["--iree-cuda-target=sm_75"]
        compiled = ireec.compile_str(
            mlir_text,
            target_backends=[backend],
            input_type="stablehlo",
            extra_args=extra_args,
        )
        _compile_cache[key] = compiled
        return compiled
    except Exception as e:
        _compile_cache[key] = None
        _compile_errors[key] = str(e)[:500]
        return None


def run_iree(compiled: bytes, inputs: List[np.ndarray],
             backend: str = "llvm-cpu") -> Optional[np.ndarray]:
    """Run compiled IREE module with given inputs."""
    driver_map = {
        "llvm-cpu": "local-task",
        "cuda": "cuda",
        "vulkan": "vulkan",
    }
    driver = driver_map.get(backend, "local-task")
    try:
        config = ireert.Config(driver)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
        ctx.add_vm_module(vm_module)
        f = ctx.modules.module.main
        result = f(*inputs)
        return np.array(result)
    except Exception:
        return None


def compile_and_run(mlir_text: str, inputs: List[np.ndarray],
                    backend: str = "llvm-cpu") -> dict:
    """Compile (cached) + run, return dict with result or error info."""
    result = {"backend": backend, "ok": False}
    compiled = compile_stablehlo(mlir_text, backend)
    if compiled is None:
        result["compile_ok"] = False
        key = (mlir_text, backend)
        result["compile_error"] = _compile_errors.get(key, "unknown")
        return result
    result["compile_ok"] = True

    try:
        driver = {"llvm-cpu": "local-task", "cuda": "cuda", "vulkan": "vulkan"}.get(backend, "local-task")
        config = ireert.Config(driver)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
        ctx.add_vm_module(vm_module)
        f = ctx.modules.module.main
        out = f(*inputs)
        result["output"] = np.array(out)
        result["ok"] = True
    except Exception as e:
        result["run_error"] = str(e)[:500]
    return result


def numpy_reference(op_type: str, op_name: str, inputs: List[np.ndarray],
                    axis: int = -1) -> Optional[np.ndarray]:
    """Compute numpy reference for comparison."""
    x = inputs[0]
    try:
        if op_type == "unary":
            fn = {
                "abs": np.abs, "ceil": np.ceil, "floor": np.floor,
                "negate": np.negative, "exponential": np.exp, "log": np.log,
                "logistic": lambda x: 1.0 / (1.0 + np.exp(-x)),
                "sine": np.sin, "cosine": np.cos, "tanh": np.tanh,
                "sqrt": np.sqrt, "rsqrt": lambda x: 1.0 / np.sqrt(x),
                "sign": np.sign, "round_nearest_even": np.round,
                "cbrt": np.cbrt,
                "log_plus_one": np.log1p,
                "exponential_minus_one": np.expm1,
            }.get(op_name)
            if fn is None:
                return None
            return fn(x)
        elif op_type == "binary":
            y = inputs[1]
            fn = {
                "add": np.add, "subtract": np.subtract,
                "multiply": np.multiply, "divide": np.divide,
                "maximum": np.maximum, "minimum": np.minimum,
                "power": np.power,
                "remainder": np.vectorize(lambda a, b: np.float32(np.fmod(a, b))),
                "atan2": np.arctan2,
            }.get(op_name)
            if fn is None:
                return None
            return fn(x, y)
        elif op_type == "reduce":
            fn = {
                "reduce_sum": lambda x, ax: np.sum(x, axis=ax),
                "reduce_max": lambda x, ax: np.max(x, axis=ax),
                "reduce_min": lambda x, ax: np.min(x, axis=ax),
                "reduce_prod": lambda x, ax: np.prod(x, axis=ax),
            }.get(op_name)
            if fn is None:
                return None
            return fn(x, axis)
        elif op_name == "softmax":
            e = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e / np.sum(e, axis=axis, keepdims=True)
    except Exception:
        return None
    return None


def compare_outputs(a: np.ndarray, b: np.ndarray,
                    rtol: float = 1e-4, atol: float = 1e-5) -> dict:
    """Compare two arrays, return diff stats."""
    if a.shape != b.shape:
        return {"match": False, "reason": f"shape_mismatch: {a.shape} vs {b.shape}"}
    abs_diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
    denom = np.where(denom < 1e-30, 1.0, denom)
    rel_diff = abs_diff / denom
    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    match = np.allclose(a, b, rtol=rtol, atol=atol)
    return {
        "match": match,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mean_abs_diff": float(abs_diff.mean()),
        "num_diffs": int(np.sum(~np.isclose(a, b, rtol=rtol, atol=atol))),
    }


def multi_backend_differential(mlir_text: str, inputs: List[np.ndarray],
                                backends: List[str] = None,
                                ref: Optional[np.ndarray] = None) -> dict:
    """Run on multiple backends, compare all pairs + optional numpy ref."""
    if backends is None:
        backends = ["llvm-cpu"]
    results = {}
    for backend in backends:
        results[backend] = compile_and_run(mlir_text, inputs, backend)

    comparisons = {}
    ok_backends = [b for b in backends if results[b]["ok"]]

    # Cross-backend comparisons
    for i, b1 in enumerate(ok_backends):
        for b2 in ok_backends[i+1:]:
            key = f"{b1}_vs_{b2}"
            comparisons[key] = compare_outputs(
                results[b1]["output"], results[b2]["output"]
            )

    # Compare each backend vs numpy ref
    if ref is not None:
        for b in ok_backends:
            key = f"{b}_vs_numpy"
            comparisons[key] = compare_outputs(results[b]["output"], ref)

    return {
        "backends": results,
        "comparisons": comparisons,
        "all_compile_ok": all(results[b].get("compile_ok", False) for b in backends),
        "all_run_ok": all(results[b]["ok"] for b in backends),
    }
