"""Targeted TensorFlow execution-mode consistency fuzzer.

Compares:
  - eager TensorFlow
  - tf.function(jit_compile=False), i.e. Grappler graph path
  - tf.function(jit_compile=True), i.e. XLA path

The cases are intentionally targeted at optimization/lowering decisions that
often change observable behavior: dtype conversion, layout rewrites, dynamic
shape specialization, index/shape lowering, control flow, stateful RNG, and
repair-oriented numeric invariants.

For new root-cause discovery, prefer the default `--groups discovery`. It avoids
known saturated basins and spends budget on underrepresented TensorFlow seams.
For maintainer-facing triage or regression of known root causes, use
`--groups issue_quality`.

Usage:
  ${ARTIFACT_ROOT}/tf_cross/harness/run_tf.sh \
    ${ARTIFACT_ROOT}/tf_cross/fuzzer/targeted_consistency_fuzzer.py --groups discovery

  ${ARTIFACT_ROOT}/tf_cross/harness/run_tf.sh \
    ${ARTIFACT_ROOT}/tf_cross/fuzzer/targeted_consistency_fuzzer.py --groups all --max-cases 0

  ${ARTIFACT_ROOT}/tf_cross/harness/run_tf.sh \
    ${ARTIFACT_ROOT}/tf_cross/fuzzer/targeted_consistency_fuzzer.py \
    --single-case fusion_nan where_dangerous_branch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


TensorTree = Any
InputSet = tuple[Any, ...]


@dataclass
class Case:
    group: str
    name: str
    fn: Callable[..., TensorTree]
    input_sets: list[InputSet]
    seed: int
    rtol: float = 1e-4
    atol: float = 1e-5
    tags: list[str] = field(default_factory=list)
    expect_xla_reject: bool = False


@dataclass
class ModeResult:
    ok: bool
    outputs: list[Any] | None = None
    error_type: str = ""
    error: str = ""
    detail: str = ""


@dataclass
class FuzzResult:
    group: str
    name: str
    seed: int
    status: str
    tags: list[str]
    findings: list[dict[str, Any]]
    modes: dict[str, dict[str, Any]]
    rtol: float
    atol: float
    tf_version: str = tf.__version__
    cuda_visible_devices: str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    triage_hint: dict[str, Any] = field(default_factory=dict)


def tensor(values: Any, dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.constant(values, dtype=dtype)


def randn(shape: tuple[int, ...], dtype: tf.dtypes.DType = tf.float32, scale: float = 1.0) -> tf.Tensor:
    return tf.constant((np.random.randn(*shape) * scale).astype(np.float32), dtype=dtype)


def clone_arg(x: Any) -> Any:
    if isinstance(x, tf.Tensor):
        return tf.identity(x)
    if isinstance(x, tuple):
        return tuple(clone_arg(v) for v in x)
    if isinstance(x, list):
        return [clone_arg(v) for v in x]
    return x


def to_numpy_tree(x: TensorTree) -> TensorTree:
    if isinstance(x, tf.Tensor):
        return x.numpy()
    if isinstance(x, tf.IndexedSlices):
        return (x.values.numpy(), x.indices.numpy(), None if x.dense_shape is None else x.dense_shape.numpy())
    if isinstance(x, tuple):
        return tuple(to_numpy_tree(v) for v in x)
    if isinstance(x, list):
        return [to_numpy_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: to_numpy_tree(x[k]) for k in sorted(x)}
    return np.asarray(x)


def summarize_tree(x: TensorTree) -> TensorTree:
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, tuple):
        return [summarize_tree(v) for v in x]
    if isinstance(x, list):
        return [summarize_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: summarize_tree(v) for k, v in x.items()}
    return str(type(x).__name__)


def compare_array(a: np.ndarray, b: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any] | None:
    if a.shape != b.shape:
        return {"type": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}

    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        aa = a.astype(np.complex128 if a.dtype.kind == "c" or b.dtype.kind == "c" else np.float64)
        bb = b.astype(np.complex128 if a.dtype.kind == "c" or b.dtype.kind == "c" else np.float64)
        a_nan = np.isnan(aa)
        b_nan = np.isnan(bb)
        if not np.array_equal(a_nan, b_nan):
            return {"type": "nan_mismatch", "a_nan": int(np.sum(a_nan)), "b_nan": int(np.sum(b_nan))}
        a_inf = np.isinf(aa)
        b_inf = np.isinf(bb)
        if not np.array_equal(a_inf, b_inf):
            return {"type": "inf_mismatch", "a_inf": int(np.sum(a_inf)), "b_inf": int(np.sum(b_inf))}
        mask = np.isfinite(aa) & np.isfinite(bb)
        if not np.any(mask):
            return None
        diff = np.abs(aa[mask] - bb[mask])
        max_diff = float(np.max(diff)) if diff.size else 0.0
        scale = float(max(np.max(np.abs(aa[mask])), 1.0)) if diff.size else 1.0
        tol = float(atol + rtol * scale)
        if max_diff > tol:
            return {"type": "numerical_diff", "max_diff": max_diff, "tol": tol, "scale": scale}
        return None

    if not np.array_equal(a, b):
        return {"type": "exact_diff", "a_sample": a.reshape(-1)[:8].tolist(), "b_sample": b.reshape(-1)[:8].tolist()}
    return None


def compare_tree(a: TensorTree, b: TensorTree, *, rtol: float, atol: float, path: str = "out") -> list[dict[str, Any]]:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [{"path": path, "type": "arity_mismatch", "a_len": len(a), "b_len": len(b)}]
        findings: list[dict[str, Any]] = []
        for i, (aa, bb) in enumerate(zip(a, b)):
            findings.extend(compare_tree(aa, bb, rtol=rtol, atol=atol, path=f"{path}[{i}]"))
        return findings

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [{"path": path, "type": "key_mismatch", "a_keys": sorted(a), "b_keys": sorted(b)}]
        findings = []
        for key in sorted(a):
            findings.extend(compare_tree(a[key], b[key], rtol=rtol, atol=atol, path=f"{path}.{key}"))
        return findings

    aa = np.asarray(a)
    bb = np.asarray(b)
    finding = compare_array(aa, bb, rtol=rtol, atol=atol)
    if finding is None:
        return []
    finding["path"] = path
    finding["a_dtype"] = str(aa.dtype)
    finding["b_dtype"] = str(bb.dtype)
    return [finding]


def run_mode(case: Case, mode: str) -> ModeResult:
    try:
        if mode == "eager":
            fn = case.fn
        elif mode == "grappler":
            fn = tf.function(case.fn, jit_compile=False, reduce_retracing=True)
        elif mode == "xla":
            fn = tf.function(case.fn, jit_compile=True, reduce_retracing=True)
        else:
            raise ValueError(f"unknown mode: {mode}")

        outputs = []
        for input_set in case.input_sets:
            args = tuple(clone_arg(x) for x in input_set)
            outputs.append(to_numpy_tree(fn(*args)))
        return ModeResult(ok=True, outputs=outputs)
    except Exception as e:
        return ModeResult(
            ok=False,
            error_type=type(e).__name__,
            error=str(e).splitlines()[0][:500],
            detail=traceback.format_exc(limit=5),
        )


def make_triage_hint(findings: list[dict[str, Any]], modes: dict[str, ModeResult]) -> dict[str, Any]:
    pairs = {str(f.get("pair", "")) for f in findings}
    finding_types = sorted({str(f.get("type", "")) for f in findings if f.get("type")})
    ok_modes = sorted(mode for mode, result in modes.items() if result.ok)
    error_modes = sorted(mode for mode, result in modes.items() if not result.ok)

    if any(f.get("type") == "mode_error" for f in findings):
        likely_component = "mode_support_or_lowering"
    elif {"eager_vs_xla", "grappler_vs_xla"} <= pairs and "eager_vs_grappler" not in pairs:
        likely_component = "xla_path"
    elif {"eager_vs_grappler", "eager_vs_xla"} <= pairs and "grappler_vs_xla" not in pairs:
        likely_component = "eager_kernel"
    elif "eager_vs_grappler" in pairs:
        likely_component = "grappler_or_graph_rewrite"
    elif findings:
        likely_component = "mode_consistency"
    else:
        likely_component = "none"

    semantic_types = {"nan_mismatch", "inf_mismatch", "exact_diff", "shape_mismatch", "mode_error"}
    severity = "semantic" if any(t in semantic_types for t in finding_types) else ("numeric" if findings else "none")

    return {
        "likely_component": likely_component,
        "severity": severity,
        "finding_types": finding_types,
        "ok_modes": ok_modes,
        "error_modes": error_modes,
    }


def run_case(case: Case) -> FuzzResult:
    tf.keras.backend.clear_session()
    modes = {mode: run_mode(case, mode) for mode in ("eager", "grappler", "xla")}
    findings: list[dict[str, Any]] = []

    for mode, result in modes.items():
        if not result.ok:
            if mode == "xla" and case.expect_xla_reject:
                continue
            findings.append({"pair": f"{mode}_execute", "type": "mode_error", "error_type": result.error_type, "error": result.error})

    pairs = [("eager", "grappler"), ("eager", "xla"), ("grappler", "xla")]
    for left, right in pairs:
        a = modes[left]
        b = modes[right]
        if not a.ok or not b.ok or a.outputs is None or b.outputs is None:
            continue
        if len(a.outputs) != len(b.outputs):
            findings.append({"pair": f"{left}_vs_{right}", "type": "run_count_mismatch"})
            continue
        for i, (aa, bb) in enumerate(zip(a.outputs, b.outputs)):
            for finding in compare_tree(aa, bb, rtol=case.rtol, atol=case.atol, path=f"run[{i}]"):
                finding["pair"] = f"{left}_vs_{right}"
                findings.append(finding)

    if not findings:
        status = "ok"
    elif any(f["type"] == "mode_error" for f in findings):
        status = "error_diff"
    else:
        status = "value_diff"

    return FuzzResult(
        group=case.group,
        name=case.name,
        seed=case.seed,
        status=status,
        tags=case.tags,
        findings=findings,
        modes={k: asdict(v) | {"outputs": summarize_tree(v.outputs) if v.outputs is not None else None} for k, v in modes.items()},
        rtol=case.rtol,
        atol=case.atol,
        triage_hint=make_triage_hint(findings, modes),
    )


def reduction_cases(seed: int) -> list[Case]:
    cases: list[Case] = []

    x = tensor([1e20, 1.0, -1e20, 3.0, -2.0, 1e-7, -1e-7], tf.float32)
    cases.append(Case(
        "reduction",
        "sum_extreme_reassociation",
        lambda a: tf.reduce_sum(a),
        [(x,)],
        seed=seed,
        rtol=0.0,
        atol=0.0,
        tags=["reassociation", "fp32", "extreme_values"],
    ))

    x2 = randn((17, 31, 13), tf.float32)
    cases.append(Case(
        "reduction",
        "var_non_power_two_axes",
        lambda a: tf.math.reduce_variance(tf.nn.softmax(a, axis=-1), axis=[0, 2]),
        [(x2,)],
        seed=seed,
        rtol=2e-4,
        atol=2e-5,
        tags=["softmax", "reduction", "non_power_two"],
    ))

    a = randn((9, 33), tf.float32, scale=3.0)
    b = randn((33, 17), tf.float32, scale=3.0)
    cases.append(Case(
        "reduction",
        "matmul_logsumexp_reduce",
        lambda x, y: tf.reduce_logsumexp(tf.matmul(x, y), axis=1),
        [(a, b)],
        seed=seed,
        rtol=2e-4,
        atol=2e-5,
        tags=["matmul", "logsumexp", "fusion"],
    ))

    return cases


def fusion_nan_cases(seed: int) -> list[Case]:
    x = tensor([-100.0, -1.0, -0.0, 0.0, 1e-20, 1.0, 88.0], tf.float32)
    y = randn((8, 16), tf.float32, scale=100.0)
    mask = tensor([True, True, True, False, False, True, False], tf.bool)
    return [
        Case(
            "fusion_nan",
            "where_dangerous_branch",
            lambda m, a: tf.where(m, a, tf.math.log(-tf.abs(a))),
            [(mask, x)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["where", "nan_propagation", "dead_branch"],
        ),
        Case(
            "fusion_nan",
            "mul_add_rounding",
            lambda a: tf.reduce_sum((a * tf.constant(1.0000001, tf.float32)) + tf.constant(1e-7, tf.float32), axis=1),
            [(y,)],
            seed=seed,
            rtol=1e-6,
            atol=1e-6,
            tags=["fma", "rounding", "fusion"],
        ),
        Case(
            "fusion_nan",
            "gelu_softmax_chain",
            lambda a: tf.reduce_sum(tf.nn.softmax(tf.nn.gelu(a), axis=-1), axis=-1),
            [(randn((7, 23), tf.float32, scale=8.0),)],
            seed=seed,
            rtol=1e-5,
            atol=1e-6,
            tags=["gelu", "softmax", "fusion"],
        ),
    ]


def dtype_cases(seed: int) -> list[Case]:
    floats = tensor([np.nan, np.inf, -np.inf, -0.0, -1.9, -1.5, -0.5, 0.5, 1.5, 1e20], tf.float32)
    ints = tensor([-128, -127, -2, -1, 0, 1, 2, 126, 127], tf.int8)
    shifts = tensor([0, 1, 2, 7, 8, 15, 31], tf.int32)
    return [
        Case(
            "dtype_cast",
            "float_to_int32_boundaries",
            lambda a: tf.cast(a, tf.int32),
            [(floats,)],
            seed=seed,
            tags=["cast", "nan_inf", "unspecified_boundary"],
        ),
        Case(
            "dtype_cast",
            "int8_mul_add_overflow",
            lambda a: tf.cast(tf.cast(a * a + tf.constant(17, tf.int8), tf.int8), tf.int32),
            [(ints,)],
            seed=seed,
            tags=["integer_overflow", "int8"],
        ),
        Case(
            "dtype_cast",
            "shift_boundaries",
            lambda a: tf.bitwise.left_shift(tf.ones_like(a), a),
            [(shifts,)],
            seed=seed,
            tags=["shift", "integer_boundary"],
        ),
    ]


def nan_semantics_cases(seed: int) -> list[Case]:
    """NaN/Inf-sensitive min/max family cases.

    These target operators where implementations often choose between
    NaN-propagating max/min and fmax/fmin-style behavior. The inputs are valid
    floating-point tensors and model NaNs that arise naturally from previous
    arithmetic.
    """

    vals = tensor([np.nan, 1.0, 3.0, 2.0, -np.inf, np.inf], tf.float32)
    vals2 = tensor([5.0, np.nan, -2.0, 7.0], tf.float32)
    pool_x = tensor(
        [[[[np.nan], [1.0], [4.0]], [[2.0], [3.0], [-1.0]], [[5.0], [6.0], [7.0]]]],
        tf.float32,
    )
    scatter_base = tensor([5.0, 5.0, 5.0, 5.0], tf.float32)
    scatter_indices = tensor([[1], [2]], tf.int32)
    scatter_updates = tensor([np.nan, -1.0], tf.float32)
    segment_data = tensor([np.nan, 1.0, 3.0, np.nan, 2.0], tf.float32)
    segment_ids = tensor([0, 0, 1, 1, 1], tf.int32)
    return [
        Case(
            "nan_semantics",
            "reduce_max_min_nan_vector",
            lambda a: (tf.reduce_max(a), tf.reduce_min(a)),
            [(vals,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["nan_propagation", "reduce_max", "reduce_min", "xla_cpu"],
        ),
        Case(
            "nan_semantics",
            "pairwise_maximum_minimum_nan",
            lambda a, b: (tf.math.maximum(a, b), tf.math.minimum(a, b)),
            [(vals[:4], vals2)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["nan_propagation", "maximum", "minimum", "xla_cpu"],
        ),
        Case(
            "nan_semantics",
            "max_pool_nan_window",
            lambda a: tf.nn.max_pool2d(a, ksize=2, strides=1, padding="VALID"),
            [(pool_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["nan_propagation", "max_pool2d", "window", "xla_gpu"],
        ),
        Case(
            "nan_semantics",
            "scatter_min_nan_update",
            lambda base, idx, upd: tf.tensor_scatter_nd_min(base, idx, upd),
            [(scatter_base, scatter_indices, scatter_updates)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["nan_propagation", "scatter", "duplicate_sensitive", "xla_gpu"],
        ),
        Case(
            "nan_semantics",
            "segment_max_min_nan",
            lambda data, ids: (tf.math.segment_max(data, ids), tf.math.segment_min(data, ids)),
            [(segment_data, segment_ids)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["nan_propagation", "segment", "reduce_max", "reduce_min"],
            expect_xla_reject=True,
        ),
        Case(
            "nan_semantics",
            "check_numerics_nan_silent",
            lambda a: tf.debugging.check_numerics(a, "nan_check"),
            [(tensor([1.0, 2.0, np.nan], tf.float32),)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["check_numerics", "nan", "xla_silent", "safety"],
        ),
        Case(
            "nan_semantics",
            "check_numerics_inf_silent",
            lambda a: tf.debugging.check_numerics(a, "inf_check"),
            [(tensor([1.0, np.inf], tf.float32),)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["check_numerics", "inf", "xla_silent", "safety"],
        ),
    ]


def cast_chain_cases(seed: int) -> list[Case]:
    """Lossy conversion chains where cast folding is dangerous."""

    # Values close to bf16/f16 boundaries plus a seeded tail. The explicit cast
    # chain should be observable because the intermediate dtype is narrower.
    boundary = np.array(
        [
            1.0 + 2.0**-8,
            1.0 + 2.0**-7,
            1.0 + 3.0 * 2.0**-8,
            -1.0 - 2.0**-8,
            0.333251953125,
            0.333984375,
            127.9375,
            128.125,
        ],
        dtype=np.float32,
    )
    tail = (np.random.RandomState(seed).randn(24).astype(np.float32) * 8.0)
    f32_values = tensor(np.concatenate([boundary, tail]), tf.float32)
    f64_values = tf.cast(f32_values, tf.float64)
    matrix_values = tensor(np.concatenate([boundary, tail[:8]]).reshape(4, 4), tf.float32)
    pred_values = tensor(([True, False] * 8), tf.bool)
    int_values = tensor([2**31 - 3, 2**31 - 1, 2**31, 2**31 + 1, 2**32 + 3, -2**31 - 1, -2**31], tf.int64)
    int_pred = tensor([True, False, True, False, True, False, True], tf.bool)

    def bf16_roundtrip(a):
        return tf.cast(tf.cast(a, tf.bfloat16), tf.float32)

    def f16_roundtrip(a):
        return tf.cast(tf.cast(a, tf.float16), tf.float32)

    return [
        Case(
            "cast_chain",
            "f32_bf16_f32_roundtrip_values",
            bf16_roundtrip,
            [(f32_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "bf16", "lossy_roundtrip", "xla_gpu"],
        ),
        Case(
            "cast_chain",
            "f64_bf16_f32_roundtrip_values",
            bf16_roundtrip,
            [(f64_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "bf16", "lossy_roundtrip", "xla_gpu"],
        ),
        Case(
            "cast_chain",
            "f32_f16_f32_roundtrip_values",
            f16_roundtrip,
            [(f32_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "fp16", "lossy_roundtrip", "xla_gpu"],
        ),
        Case(
            "cast_chain",
            "bf16_roundtrip_threshold_mask",
            lambda a: tf.cast(tf.not_equal(bf16_roundtrip(a), tf.cast(a, tf.float32)), tf.int32),
            [(f32_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "bf16", "amplified_bool_diff", "contract_violation"],
        ),
        Case(
            "cast_chain",
            "bf16_roundtrip_gather_index",
            lambda a: tf.gather(
                tf.constant([11, 22, 33, 44], tf.int32),
                tf.clip_by_value(tf.reduce_sum(tf.cast(tf.not_equal(bf16_roundtrip(a), tf.cast(a, tf.float32)), tf.int32)), 0, 3),
            ),
            [(f32_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "bf16", "amplified_index_diff", "gather"],
        ),
        Case(
            "cast_chain",
            "i64_i32_i64_roundtrip_boundaries",
            lambda a: tf.cast(tf.cast(a, tf.int32), tf.int64),
            [(int_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "integer", "lossy_roundtrip"],
        ),
        Case(
            "cast_chain",
            "i64_i32_i64_where_contract",
            lambda a, p: tf.where(p, tf.cast(tf.cast(a, tf.int32), tf.int64), a),
            [(int_values, int_pred)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "integer", "where", "contract_violation"],
        ),
        Case(
            "cast_chain",
            "i64_i32_i64_gather_index_contract",
            lambda a: tf.gather(
                tf.constant([7, 11, 13, 17], tf.int32),
                tf.clip_by_value(tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(tf.cast(a, tf.int32), tf.int64), a), tf.int32)), 0, 3),
            ),
            [(int_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "integer", "gather", "contract_violation"],
        ),
        Case(
            "cast_chain",
            "bf16_roundtrip_after_reshape_reduce",
            lambda a: tf.reduce_sum(tf.reshape(bf16_roundtrip(a), [2, 8]), axis=-1),
            [(matrix_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "bf16", "reshape", "reduce", "contract_violation"],
        ),
        Case(
            "cast_chain",
            "bf16_roundtrip_concat_contract",
            lambda a: tf.concat([bf16_roundtrip(a[:8]), bf16_roundtrip(a[8:])], axis=0),
            [(f32_values[:16],)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "bf16", "concat", "contract_violation"],
        ),
        Case(
            "cast_chain",
            "bf16_roundtrip_where_contract",
            lambda a, p: tf.where(p, bf16_roundtrip(a), a),
            [(f32_values[:16], pred_values)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["cast_chain", "bf16", "where", "contract_violation"],
        ),
    ]


def layout_cases(seed: int) -> list[Case]:
    x = randn((2, 9, 7, 3), tf.float32)
    filt = randn((3, 3, 3, 5), tf.float32)
    bias = randn((5,), tf.float32)
    block_x = tensor(np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3), tf.float32)
    bts_x = tensor(np.arange(4 * 2 * 2 * 1, dtype=np.float32).reshape(4, 2, 2, 1), tf.float32)
    crops = tensor([[0, 0], [0, 0]], tf.int32)
    return [
        Case(
            "layout",
            "conv_bias_relu_nhwc",
            lambda a, w, b: tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(a, w, strides=1, padding="SAME"), b)),
            [(x, filt, bias)],
            seed=seed,
            rtol=1e-4,
            atol=1e-5,
            tags=["conv2d", "layout_optimizer", "remapper"],
        ),
        Case(
            "layout",
            "transpose_conv_reduce",
            lambda a, w: tf.reduce_sum(tf.nn.conv2d(tf.transpose(a, [0, 2, 1, 3]), w, strides=1, padding="VALID"), axis=[1, 2]),
            [(x, filt)],
            seed=seed,
            rtol=1e-4,
            atol=1e-5,
            tags=["transpose", "conv2d", "reduction"],
        ),
        Case(
            "layout",
            "space_to_depth_depth_to_space_roundtrip",
            lambda a: tf.nn.depth_to_space(tf.nn.space_to_depth(a, block_size=2), block_size=2),
            [(block_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["space_to_depth", "depth_to_space", "layout_contract"],
        ),
        Case(
            "layout",
            "batch_to_space_space_to_batch_roundtrip",
            lambda a, c: tf.space_to_batch(tf.batch_to_space(a, block_shape=[2, 2], crops=c), block_shape=[2, 2], paddings=c),
            [(bts_x, crops)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["batch_to_space", "space_to_batch", "layout_contract"],
        ),
    ]


def dynamic_shape_cases(seed: int) -> list[Case]:
    return [
        Case(
            "dynamic_shape",
            "same_function_shape_sequence_reduce",
            lambda a: tf.reduce_sum(tf.nn.softmax(a, axis=-1), axis=-1),
            [(randn((1, 17), tf.float32),), (randn((3, 17), tf.float32),), (randn((2, 1), tf.float32),)],
            seed=seed,
            rtol=1e-5,
            atol=1e-6,
            tags=["shape_sequence", "specialization", "softmax"],
        ),
        Case(
            "dynamic_shape",
            "boolean_mask_shape_sequence",
            lambda a: tf.reduce_sum(tf.boolean_mask(a, a > 0.0)),
            [(randn((4, 5), tf.float32),), (randn((1, 5), tf.float32),), (randn((7, 5), tf.float32),)],
            seed=seed,
            rtol=1e-5,
            atol=1e-6,
            tags=["dynamic_output_shape", "mask"],
            expect_xla_reject=True,
        ),
    ]


def control_flow_cases(seed: int) -> list[Case]:
    x = randn((8,), tf.float32)
    return [
        Case(
            "control_flow",
            "cond_dead_branch_nan",
            lambda pred, a: tf.cond(pred, lambda: tf.reduce_sum(a), lambda: tf.reduce_sum(tf.math.log(-tf.abs(a)))),
            [(tf.constant(True), x)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["tf.cond", "dead_branch", "nan"],
        ),
        Case(
            "control_flow",
            "while_loop_accumulation",
            lambda a: tf.while_loop(
                lambda i, acc: i < tf.shape(a)[0],
                lambda i, acc: (i + 1, acc + tf.reduce_sum(a[i] * tf.cast(i + 1, a.dtype))),
                (tf.constant(0), tf.constant(0.0, tf.float32)),
            )[1],
            [(randn((11, 7), tf.float32),)],
            seed=seed,
            rtol=1e-5,
            atol=1e-6,
            tags=["while_loop", "accumulation"],
        ),
    ]


def state_rng_cases(seed: int) -> list[Case]:
    def stateful_random(a):
        return a + tf.random.uniform(tf.shape(a), dtype=a.dtype)

    def stateless_random(a, seed):
        return a + tf.random.stateless_uniform(tf.shape(a), seed=seed, dtype=a.dtype)

    return [
        Case(
            "state_rng",
            "stateful_rng_two_calls",
            stateful_random,
            [(tf.zeros((4, 4), tf.float32),), (tf.zeros((4, 4), tf.float32),)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["stateful_rng", "call_sequence"],
        ),
        Case(
            "state_rng",
            "stateless_rng_fixed_seed",
            stateless_random,
            [(tf.zeros((4, 4), tf.float32), tf.constant([123, 456], tf.int32))],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["stateless_rng", "determinism"],
        ),
    ]


def backward_cases(seed: int) -> list[Case]:
    def grad_relu_abs(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            out = tf.reduce_sum(tf.nn.relu(a) + tf.abs(a))
        return tape.gradient(out, a)

    def grad_reduce_broadcast(a, b):
        with tf.GradientTape() as tape:
            tape.watch([a, b])
            out = tf.reduce_sum(tf.nn.softmax(a + b, axis=-1) ** 2)
        ga, gb = tape.gradient(out, [a, b])
        return ga, gb

    return [
        Case(
            "backward",
            "nondiff_boundary_relu_abs",
            grad_relu_abs,
            [(tensor([-1.0, -0.0, 0.0, 1.0], tf.float32),)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gradient", "nondiff_boundary"],
        ),
        Case(
            "backward",
            "broadcast_softmax_gradient",
            grad_reduce_broadcast,
            [(randn((3, 7), tf.float32), randn((1, 7), tf.float32))],
            seed=seed,
            rtol=2e-4,
            atol=2e-5,
            tags=["gradient", "broadcast", "softmax"],
        ),
    ]


def ordering_tie_cases(seed: int) -> list[Case]:
    """Ordering/index ops where ties, signed zero, and NaN policy matter."""

    near = tensor([[1.0, 1.0 + 2.0**-23, 1.0, -0.0, 0.0, np.nan]], tf.float32)
    ties = tensor([[3.0, 3.0, 2.0, 1.0, 3.0], [0.0, -0.0, 0.0, -1.0, -1.0]], tf.float32)
    topk = tensor([[np.nan, 4.0, 4.0, -0.0, 0.0, 3.0]], tf.float32)
    topk_nan_back = tensor([[1.0, 2.0, 3.0, np.nan]], tf.float32)
    shaped = randn((3, 5), tf.float32, scale=1e-6) + tensor(
        [[1.0, 1.0, 1.0, 2.0, 2.0], [0.0, -0.0, 0.0, 1.0, 1.0], [3.0, 2.0, 2.0, 2.0, 1.0]],
        tf.float32,
    )
    return [
        Case(
            "ordering_ties",
            "top_k_nan_duplicate_tie",
            lambda a: tf.math.top_k(a, k=3, sorted=True),
            [(topk,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["top_k", "nan", "tie_breaking", "ordering"],
        ),
        Case(
            "ordering_ties",
            "top_k_nan_position_dependent",
            lambda a: tf.math.top_k(a, k=3, sorted=True),
            [(topk_nan_back,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["top_k", "nan", "position_dependent", "ordering"],
        ),
        Case(
            "ordering_ties",
            "argsort_signed_zero_stability",
            lambda a: tf.argsort(a, axis=-1, stable=True),
            [(ties,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argsort", "signed_zero", "stable_sort", "tie_breaking"],
        ),
        Case(
            "ordering_ties",
            "sort_signed_zero_values",
            lambda a: tf.sort(a, axis=-1, direction="ASCENDING"),
            [(ties,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["sort", "signed_zero", "tie_breaking"],
        ),
        Case(
            "ordering_ties",
            "argmax_near_equal_with_nan",
            lambda a: tf.argmax(a, axis=-1, output_type=tf.int32),
            [(near,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "near_equal", "nan", "tie_breaking"],
        ),
        Case(
            "ordering_ties",
            "top_k_near_equal_index_stability",
            lambda a: tf.math.top_k(a, k=2, sorted=True).indices,
            [(shaped,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["top_k", "near_equal", "index", "tie_breaking"],
        ),
    ]


def complex_boundary_cases(seed: int) -> list[Case]:
    """Complex branch-cut, signed-zero, and subnormal-sensitive cases."""

    tiny = np.float32(1e-40)
    branch = tf.constant(
        np.array(
            [
                -1.0 + 0.0j,
                -1.0 - 0.0j,
                -1.0 + tiny * 1j,
                -1.0 - tiny * 1j,
                0.0 + 0.0j,
                0.0 - 0.0j,
                tiny + tiny * 1j,
                tiny - tiny * 1j,
            ],
            dtype=np.complex64,
        )
    )
    denom = tf.constant(
        np.array([tiny + 0j, tiny + tiny * 1j, 1.0 - 0.0j, -1.0 + tiny * 1j], dtype=np.complex64)
    )
    numer = tf.constant(
        np.array([1.0 + 0j, 1.0 - 1.0j, tiny + 0j, tiny - tiny * 1j], dtype=np.complex64)
    )
    powers = tf.constant(np.array([-1.0 + tiny * 1j, -1.0 - tiny * 1j, tiny + tiny * 1j], dtype=np.complex64))
    return [
        Case(
            "complex_boundaries",
            "complex_log_branch_signed_zero",
            lambda z: tf.math.log(z),
            [(branch,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["complex", "log", "branch_cut", "signed_zero", "subnormal"],
        ),
        Case(
            "complex_boundaries",
            "complex_sqrt_branch_signed_zero",
            lambda z: tf.sqrt(z),
            [(branch,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["complex", "sqrt", "branch_cut", "signed_zero"],
        ),
        Case(
            "complex_boundaries",
            "complex_angle_signed_zero",
            lambda z: tf.math.angle(z),
            [(branch,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["complex", "angle", "signed_zero", "branch_cut"],
        ),
        Case(
            "complex_boundaries",
            "complex_divide_subnormal_denominator",
            lambda a, b: tf.math.divide(a, b),
            [(numer, denom)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["complex", "divide", "subnormal", "overflow_boundary"],
        ),
        Case(
            "complex_boundaries",
            "complex_pow_integer_boundary",
            lambda z: tf.pow(z, tf.constant(3.0, tf.complex64)),
            [(powers,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["complex", "pow", "branch_cut", "integer_exponent"],
        ),
    ]


def grad_boundary_cases(seed: int) -> list[Case]:
    """Gradient consistency around ties, dead branches, and nondiff boundaries."""

    tie_x = tensor([[-1.0, 0.0, -0.0, 1.0], [2.0, 2.0, -2.0, -2.0]], tf.float32)
    where_x = tensor([-1.0, -0.0, 0.0, 1.0, 2.0], tf.float32)
    clip_x = tensor([-1.0, -0.5, 0.0, 0.5, 1.0], tf.float32)
    max_a = tensor([1.0, 1.0, -0.0, 0.0, 2.0], tf.float32)
    max_b = tensor([1.0, 0.5, 0.0, -0.0, 2.0], tf.float32)

    def grad_abs(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            out = tf.reduce_sum(tf.abs(a))
        return tape.gradient(out, a)

    def grad_clip(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            out = tf.reduce_sum(tf.clip_by_value(a, -0.5, 0.5))
        return tape.gradient(out, a)

    def grad_where(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            out = tf.reduce_sum(tf.where(a >= 0.0, a * a, tf.math.log(-a)))
        return tape.gradient(out, a)

    def grad_maximum(a, b):
        with tf.GradientTape() as tape:
            tape.watch([a, b])
            out = tf.reduce_sum(tf.maximum(a, b))
        return tape.gradient(out, [a, b])

    def grad_reduce_max_tie(a):
        with tf.GradientTape() as tape:
            tape.watch(a)
            out = tf.reduce_sum(tf.reduce_max(a, axis=-1))
        return tape.gradient(out, a)

    return [
        Case(
            "grad_boundaries",
            "grad_abs_signed_zero",
            grad_abs,
            [(where_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gradient", "abs", "signed_zero", "nondiff_boundary"],
        ),
        Case(
            "grad_boundaries",
            "grad_clip_boundary",
            grad_clip,
            [(clip_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gradient", "clip_by_value", "boundary"],
        ),
        Case(
            "grad_boundaries",
            "grad_where_dead_branch_log",
            grad_where,
            [(where_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gradient", "where", "dead_branch", "nan_inf"],
        ),
        Case(
            "grad_boundaries",
            "grad_maximum_tie_split",
            grad_maximum,
            [(max_a, max_b)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gradient", "maximum", "tie", "nondiff_boundary"],
        ),
        Case(
            "grad_boundaries",
            "grad_reduce_max_tie_distribution",
            grad_reduce_max_tie,
            [(tie_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gradient", "reduce_max", "tie", "nondiff_boundary"],
        ),
    ]


def index_shape_cases(seed: int) -> list[Case]:
    """More typical discrete/index and shape-boundary cases.

    These avoid the already-known signed-zero, NaN min/max, and cast-chain root
    causes. They target everyday indexing/lowering decisions where graph/XLA
    paths may differ from eager semantics.
    """

    gather_params = tensor(np.arange(12, dtype=np.float32).reshape(4, 3), tf.float32)
    gather_indices = tensor([[0, 2], [3, 1]], tf.int32)
    one_hot_indices = tensor([[0, 2, -1], [3, 4, 1]], tf.int32)
    segment_data = tensor([1.0, 2.0, 3.0, 4.0], tf.float32)
    segment_ids = tensor([2, 0, 2, 4], tf.int32)
    slice_x = tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), tf.float32)
    unique_x = tensor([3, 1, 3, 2, 1, 4, 2], tf.int32)
    ragged_like = tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 0.0]], tf.float32)
    lengths = tensor([2, 1, 3], tf.int32)
    gather_nd_params = tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), tf.float32)
    gather_nd_indices = tensor([[[0], [2]], [[1], [0]]], tf.int32)
    scatter_base = tensor(np.arange(12, dtype=np.float32).reshape(3, 4), tf.float32)
    scatter_indices = tensor([[0, 1], [2, 3]], tf.int32)
    scatter_updates = tensor([100.0, 200.0], tf.float32)
    dynamic_slice_begin = tensor([0, 1, 0], tf.int32)
    dynamic_slice_size = tensor([2, 2, 3], tf.int32)

    return [
        Case(
            "index_shape",
            "gather_batch_dims_matrix_indices",
            lambda p, i: tf.gather(p, i, axis=0),
            [(gather_params, gather_indices)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gather", "index", "batch_like_indices"],
        ),
        Case(
            "index_shape",
            "one_hot_out_of_range_indices",
            lambda i: tf.one_hot(i, depth=4, on_value=1.0, off_value=-1.0, dtype=tf.float32),
            [(one_hot_indices,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["one_hot", "out_of_range", "index"],
        ),
        Case(
            "index_shape",
            "unsorted_segment_sum_empty_segments",
            lambda data, ids: tf.math.unsorted_segment_sum(data, ids, num_segments=6),
            [(segment_data, segment_ids)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["unsorted_segment", "empty_segment", "index"],
        ),
        Case(
            "index_shape",
            "strided_slice_negative_stride",
            lambda x: x[:, ::-1, 1:4:2],
            [(slice_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["strided_slice", "negative_stride", "shape_boundary"],
        ),
        Case(
            "index_shape",
            "unique_stable_order_values",
            lambda x: tf.unique(x),
            [(unique_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["unique", "stable_order", "index"],
        ),
        Case(
            "index_shape",
            "sequence_mask_ragged_like_reduce",
            lambda x, l: tf.reduce_sum(tf.where(tf.sequence_mask(l, maxlen=tf.shape(x)[1]), x, tf.zeros_like(x)), axis=-1),
            [(ragged_like, lengths)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["sequence_mask", "dynamic_shape", "ragged_like"],
        ),
        Case(
            "index_shape",
            "gather_nd_batched_inner_rows",
            lambda p, i: tf.gather_nd(p, i, batch_dims=1),
            [(gather_nd_params, gather_nd_indices)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["gather_nd", "batch_dims", "index"],
        ),
        Case(
            "index_shape",
            "tensor_scatter_update_distinct_indices",
            lambda b, i, u: tf.tensor_scatter_nd_update(b, i, u),
            [(scatter_base, scatter_indices, scatter_updates)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["scatter", "distinct_indices", "index"],
        ),
        Case(
            "index_shape",
            "dynamic_slice_normal_bounds",
            lambda x, b, s: tf.slice(x, b, s),
            [(slice_x, dynamic_slice_begin, dynamic_slice_size)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["slice", "dynamic_begin", "shape"],
        ),
    ]


def math_special_cases(seed: int) -> list[Case]:
    """Finite special-function boundary cases.

    These target libdevice/Eigen/XLA lowering differences for common numerical
    functions while avoiding NaN/Inf inputs as the primary trigger.
    """

    small_pos = tensor([1e-38, 1e-30, 1e-20, 1e-10, 1e-6, 0.1, 0.5, 1.0, 2.0, 10.0], tf.float32)
    normal_small_pos = tensor([1.2e-38, 2.0e-38, 1e-37, 1e-36, 1e-30, 1e-20, 1e-10, 1e-6], tf.float32)
    near_int = tensor([-3.999, -3.5, -2.001, -1.5, -0.999, -0.5, 0.5, 0.999, 1.001, 1.5, 2.001], tf.float32)
    moderate = tensor([-12.0, -8.0, -4.0, -1.0, -0.25, 0.25, 1.0, 4.0, 8.0, 12.0], tf.float32)
    prob = tensor([1e-6, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 0.9999, 1.0 - 1e-6], tf.float32)
    bessel_x = tensor([0.0, 1e-6, 0.1, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0], tf.float32)
    beta_a = tensor([0.1, 0.5, 1.0, 2.0, 5.0, 20.0], tf.float32)
    beta_b = tensor([20.0, 5.0, 2.0, 1.0, 0.5, 0.1], tf.float32)
    igamma_a = tensor([0.1, 0.2, 0.5, 1.0], tf.float32)
    igamma_normal_x = tensor([1.2e-38, 2.0e-38, 1e-37, 1e-36], tf.float32)

    return [
        Case(
            "math_special",
            "lgamma_finite_near_poles",
            lambda x: tf.math.lgamma(x),
            [(near_int,)],
            seed=seed,
            rtol=1e-6,
            atol=1e-6,
            tags=["lgamma", "finite", "near_integer", "special_function"],
        ),
        Case(
            "math_special",
            "digamma_finite_small_positive",
            lambda x: tf.math.digamma(x),
            [(small_pos,)],
            seed=seed,
            rtol=1e-5,
            atol=1e-5,
            tags=["digamma", "finite", "small_positive", "special_function"],
        ),
        Case(
            "math_special",
            "digamma_normal_small_isfinite",
            lambda x: tf.cast(tf.math.is_finite(tf.math.digamma(x)), tf.int32),
            [(normal_small_pos,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["digamma", "normal_float", "isfinite", "semantic_predicate", "special_function"],
        ),
        Case(
            "math_special",
            "digamma_normal_small_masked_reduce",
            lambda x: tf.reduce_sum(tf.where(tf.math.is_finite(tf.math.digamma(x)), tf.ones_like(x), tf.zeros_like(x))),
            [(normal_small_pos,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["digamma", "normal_float", "mask", "downstream_reduce", "special_function"],
        ),
        Case(
            "math_special",
            "erf_erfc_moderate_values",
            lambda x: (tf.math.erf(x), tf.math.erfc(x)),
            [(moderate,)],
            seed=seed,
            rtol=1e-6,
            atol=1e-7,
            tags=["erf", "erfc", "finite", "special_function"],
        ),
        Case(
            "math_special",
            "ndtri_probability_edges",
            lambda x: tf.math.ndtri(x),
            [(prob,)],
            seed=seed,
            rtol=1e-5,
            atol=1e-5,
            tags=["ndtri", "probability_edge", "finite", "special_function"],
        ),
        Case(
            "math_special",
            "igamma_igammac_shape_edges",
            lambda a, x: (tf.math.igamma(a, x), tf.math.igammac(a, x)),
            [(beta_a, small_pos[:6])],
            seed=seed,
            rtol=1e-5,
            atol=1e-6,
            tags=["igamma", "igammac", "finite", "special_function"],
        ),
        Case(
            "math_special",
            "igamma_normal_small_zero_predicate",
            lambda a, x: tf.cast(tf.equal(tf.math.igamma(a, x), tf.constant(0.0, tf.float32)), tf.int32),
            [(igamma_a, igamma_normal_x)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["igamma", "normal_float", "zero_predicate", "semantic_predicate", "special_function"],
        ),
        Case(
            "math_special",
            "igamma_normal_small_branch_sum",
            lambda a, x: tf.reduce_sum(
                tf.where(
                    tf.math.igamma(a, x) > tf.constant(0.0, tf.float32),
                    tf.constant(1.0, tf.float32),
                    tf.constant(-1.0, tf.float32),
                )
            ),
            [(igamma_a, igamma_normal_x)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["igamma", "normal_float", "branch", "downstream_reduce", "special_function"],
        ),
        Case(
            "math_special",
            "bessel_i0e_i1e_range",
            lambda x: (tf.math.bessel_i0e(x), tf.math.bessel_i1e(x)),
            [(bessel_x,)],
            seed=seed,
            rtol=1e-5,
            atol=1e-6,
            tags=["bessel", "i0e", "i1e", "finite", "special_function"],
        ),
        Case(
            "math_special",
            "zeta_finite_shape_edges",
            lambda x, q: tf.math.zeta(x, q),
            [(tensor([2.0, 3.0, 4.0, 8.0, 16.0, 32.0], tf.float32), beta_b)],
            seed=seed,
            rtol=1e-5,
            atol=1e-6,
            tags=["zeta", "finite", "special_function"],
            expect_xla_reject=True,
        ),
    ]


def normal_numeric_cases(seed: int) -> list[Case]:
    """Normal finite inputs with downstream discrete observables.

    This group deliberately avoids NaN/Inf/subnormal/signed-zero triggers. The
    goal is to find representative consistency cases that are easier to justify
    as normal numerical workloads.
    """

    rs = np.random.RandomState(seed)
    probs = tensor([1e-4, 3e-4, 1e-3, 1e-2, 0.1, 0.9, 0.99, 0.999, 0.9997, 0.9999], tf.float32)
    logits = tensor(np.array([-18.0, -12.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 12.0], np.float32), tf.float32)
    bessel = tensor(np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0], np.float32), tf.float32)
    gamma_near = tensor(np.array([0.125, 0.25, 0.5, 0.75, 1.25, 1.5, 2.5, 4.5, 8.5, 16.5], np.float32), tf.float32)
    mat = tensor((rs.randn(8, 8).astype(np.float32) * 0.25 + np.eye(8, dtype=np.float32)), tf.float32)
    vec = tensor((rs.randn(8).astype(np.float32) * 0.25), tf.float32)

    return [
        Case(
            "normal_numeric",
            "ndtri_normal_probability_threshold",
            lambda p: tf.cast(tf.math.ndtri(p) > tf.constant(3.71898, tf.float32), tf.int32),
            [(probs,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "ndtri", "threshold", "downstream_bool"],
        ),
        Case(
            "normal_numeric",
            "erfc_normal_tail_threshold",
            lambda x: tf.cast(tf.math.erfc(x) < tf.constant(1e-30, tf.float32), tf.int32),
            [(logits,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "erfc", "tail", "threshold"],
        ),
        Case(
            "normal_numeric",
            "bessel_i1e_normal_ordering",
            lambda x: tf.argsort(tf.math.bessel_i1e(x), stable=True),
            [(bessel,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "bessel", "ordering", "index"],
        ),
        Case(
            "normal_numeric",
            "lgamma_normal_fractional_threshold",
            lambda x: tf.cast(tf.math.lgamma(x) > tf.constant(1.0, tf.float32), tf.int32),
            [(gamma_near,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "lgamma", "threshold", "downstream_bool"],
        ),
        Case(
            "normal_numeric",
            "softplus_logsigmoid_normal_branch",
            lambda x: tf.cast((tf.nn.softplus(x) + tf.math.log_sigmoid(-x)) > tf.constant(0.0, tf.float32), tf.int32),
            [(logits,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "softplus", "log_sigmoid", "branch"],
        ),
        Case(
            "normal_numeric",
            "matvec_normal_threshold_index",
            lambda a, b: tf.argmax(tf.linalg.matvec(a, b), axis=-1, output_type=tf.int32),
            [(mat, vec)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "matvec", "argmax", "downstream_index"],
        ),
    ]


def invariant_numeric_cases(seed: int) -> list[Case]:
    """Normal finite mathematical invariants with downstream predicates.

    These cases are designed to be useful for maintainers: each one points to a
    specific mathematical contract and reports a boolean/reduced observable
    rather than only a raw floating-point delta.
    """

    xs = tensor(np.array([0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0], np.float32), tf.float32)
    signed_xs = tensor(np.array([-12.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 12.0], np.float32), tf.float32)
    probs = tensor(np.array([1e-4, 3e-4, 1e-3, 1e-2, 0.1, 0.25, 0.75, 0.9, 0.99, 0.999], np.float32), tf.float32)
    gamma_a = tensor(np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0], np.float32), tf.float32)
    gamma_x = tensor(np.array([1e-4, 0.01, 0.1, 1.0, 4.0, 12.0], np.float32), tf.float32)
    bessel = tensor(np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0], np.float32), tf.float32)

    return [
        Case(
            "invariant_numeric",
            "erf_erfc_sum_invariant",
            lambda x: tf.reduce_max(tf.abs((tf.math.erf(x) + tf.math.erfc(x)) - tf.constant(1.0, tf.float32))),
            [(signed_xs,)],
            seed=seed,
            rtol=0.0,
            atol=1e-7,
            tags=["normal_float", "erf", "erfc", "invariant", "repair_hint"],
        ),
        Case(
            "invariant_numeric",
            "igamma_complement_sum_invariant",
            lambda a, x: tf.reduce_max(tf.abs((tf.math.igamma(a, x) + tf.math.igammac(a, x)) - tf.constant(1.0, tf.float32))),
            [(gamma_a, gamma_x)],
            seed=seed,
            rtol=0.0,
            atol=1e-6,
            tags=["normal_float", "igamma", "igammac", "invariant", "repair_hint"],
        ),
        Case(
            "invariant_numeric",
            "ndtri_symmetry_invariant",
            lambda p: tf.reduce_max(tf.abs(tf.math.ndtri(p) + tf.math.ndtri(tf.constant(1.0, tf.float32) - p))),
            [(probs,)],
            seed=seed,
            rtol=0.0,
            atol=1e-5,
            tags=["normal_float", "ndtri", "symmetry", "invariant", "repair_hint"],
        ),
        Case(
            "invariant_numeric",
            "lgamma_recurrence_invariant",
            lambda x: tf.reduce_max(tf.abs((tf.math.lgamma(x + tf.constant(1.0, tf.float32)) - tf.math.lgamma(x)) - tf.math.log(x))),
            [(xs,)],
            seed=seed,
            rtol=0.0,
            atol=1e-5,
            tags=["normal_float", "lgamma", "log", "recurrence", "invariant"],
        ),
        Case(
            "invariant_numeric",
            "softplus_logsigmoid_identity",
            lambda x: tf.reduce_max(tf.abs((tf.nn.softplus(x) + tf.math.log_sigmoid(-x)) - x)),
            [(signed_xs,)],
            seed=seed,
            rtol=0.0,
            atol=1e-6,
            tags=["normal_float", "softplus", "log_sigmoid", "invariant"],
        ),
        Case(
            "invariant_numeric",
            "bessel_i0e_positive_predicate",
            lambda x: tf.reduce_sum(tf.cast(tf.math.bessel_i0e(x) > tf.constant(0.0, tf.float32), tf.int32)),
            [(bessel,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "bessel", "positive", "semantic_predicate"],
        ),
        Case(
            "invariant_numeric",
            "ndtri_symmetry_branch",
            lambda p: tf.cast(tf.abs(tf.math.ndtri(p) + tf.math.ndtri(tf.constant(1.0, tf.float32) - p)) > tf.constant(1e-3, tf.float32), tf.int32),
            [(probs,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["normal_float", "ndtri", "symmetry", "downstream_bool"],
        ),
    ]


def seam_discovery_cases(seed: int) -> list[Case]:
    """Cases for seams that are underrepresented in the high-signal regressions.

    This group is intentionally kept out of `issue_quality`: it is for escaping
    known basins such as cast elision, softmax-size-1, NaN min/max, and Grappler
    ArgMin/ArgMax plateaus. Prefer dense observable outputs so the existing
    comparison logic can stay simple and exact.
    """

    rs = np.random.RandomState(seed)
    values = tensor(rs.permutation(np.arange(12, dtype=np.float32)).reshape(3, 4), tf.float32)
    partitions = tensor(rs.randint(0, 3, size=(8,), dtype=np.int32), tf.int32)
    partition_data = tensor(rs.randint(-20, 21, size=(8,)).astype(np.float32), tf.float32)
    repeat_x = tensor(rs.randint(-10, 11, size=(3, 4)).astype(np.float32), tf.float32)
    repeat_counts = tensor(rs.randint(0, 4, size=(4,), dtype=np.int32), tf.int32)
    unique_counts_x = tensor(rs.randint(0, 6, size=(9,), dtype=np.int32), tf.int32)
    sorted_edges_np = np.sort(rs.randint(-5, 6, size=(7,)).astype(np.float32))
    sorted_edges = tensor(sorted_edges_np, tf.float32)
    search_values = tensor(rs.randint(-7, 8, size=(5,)).astype(np.float32) + np.float32(0.5 * (seed % 2)), tf.float32)
    reverse_x = tensor(rs.randint(-50, 51, size=(3, 4, 2)).astype(np.float32), tf.float32)
    reverse_lengths = tensor(rs.randint(0, 5, size=(3,), dtype=np.int32), tf.int32)
    roll_x = tensor(rs.randint(-20, 21, size=(4, 5)).astype(np.float32), tf.float32)
    roll_shift = [int(rs.randint(-3, 4)), int(rs.randint(-4, 5))]
    space_x = tensor(rs.randint(-30, 31, size=(1, 4, 4, 2)).astype(np.float32), tf.float32)
    sparse_values = rs.randint(-5, 6, size=(5,)).astype(np.float32)
    sparse_dense_rhs = rs.randint(-4, 5, size=(4, 2)).astype(np.float32)
    sparse_reduce_values = rs.randint(-9, 10, size=(4,)).astype(np.float32)
    embedding_params = rs.randint(-5, 6, size=(5, 3)).astype(np.float32)

    def top_k_indices(a):
        out = tf.nn.top_k(a, k=2, sorted=True)
        return out.values, out.indices

    def dynamic_partition_stitch_roundtrip(data, p):
        idx = tf.range(tf.shape(data)[0], dtype=tf.int32)
        parts = tf.dynamic_partition(data, p, num_partitions=3)
        idx_parts = tf.dynamic_partition(idx, p, num_partitions=3)
        return tf.dynamic_stitch(idx_parts, parts)

    def sparse_matmul_dense_equiv():
        sp = tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 3], [1, 1], [2, 0], [2, 2]], tf.int64),
            values=tf.constant(sparse_values, tf.float32),
            dense_shape=tf.constant([3, 4], tf.int64),
        )
        dense = tf.constant(sparse_dense_rhs)
        sparse_out = tf.sparse.sparse_dense_matmul(sp, dense)
        dense_out = tf.matmul(tf.sparse.to_dense(sp), dense)
        return tf.reduce_max(tf.abs(sparse_out - dense_out))

    def sparse_reduce_dense_equiv():
        sp = tf.SparseTensor(
            indices=tf.constant([[0, 1], [1, 0], [1, 3], [3, 2]], tf.int64),
            values=tf.constant(sparse_reduce_values, tf.float32),
            dense_shape=tf.constant([4, 5], tf.int64),
        )
        dense = tf.sparse.to_dense(sp)
        return tf.sparse.reduce_sum(sp, axis=1), tf.reduce_sum(dense, axis=1)

    def ragged_to_dense_row_sum():
        rt = tf.ragged.constant([[1.0, 2.0, 3.0], [], [-1.0, 4.0], [5.0]], dtype=tf.float32)
        dense = rt.to_tensor(default_value=0.0)
        return tf.reduce_sum(dense, axis=1)

    def set_intersection_dense():
        a = tf.constant([[1, 2, 3, 5], [3, 4, 5, 6]], tf.int32)
        b = tf.constant([[3, 5, 8, 9], [1, 3, 5, 7]], tf.int32)
        return tf.sparse.to_dense(tf.sets.intersection(a, b), default_value=-1)

    def embedding_lookup_sparse_dense_equiv():
        params = tf.constant(embedding_params)
        ids = tf.SparseTensor(
            indices=tf.constant([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 2]], tf.int64),
            values=tf.constant([0, 3, 1, 2, 4, 0], tf.int64),
            dense_shape=tf.constant([3, 3], tf.int64),
        )
        looked_up = tf.nn.embedding_lookup_sparse(params, ids, None, combiner="sum")
        dense_ids = tf.sparse.to_dense(ids, default_value=0)
        mask = tf.cast(tf.sparse.to_dense(tf.SparseTensor(ids.indices, tf.ones_like(ids.values, tf.float32), ids.dense_shape)), tf.float32)
        dense_equiv = tf.reduce_sum(tf.gather(params, dense_ids) * mask[..., None], axis=1)
        return tf.reduce_max(tf.abs(looked_up - dense_equiv))

    def space_to_batch_roundtrip(x):
        y = tf.space_to_batch(x, block_shape=[2, 2], paddings=[[0, 0], [0, 0]])
        return tf.batch_to_space(y, block_shape=[2, 2], crops=[[0, 0], [0, 0]])

    return [
        Case(
            "seam_discovery",
            "top_k_regular_values_shape_boundary",
            top_k_indices,
            [(values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["top_k", "index", "shape_boundary", "unmined_seam"],
        ),
        Case(
            "seam_discovery",
            "dynamic_partition_stitch_roundtrip",
            dynamic_partition_stitch_roundtrip,
            [(partition_data, partitions)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["dynamic_partition", "dynamic_stitch", "roundtrip", "unmined_seam"],
        ),
        Case(
            "seam_discovery",
            "sparse_dense_matmul_dense_equiv",
            sparse_matmul_dense_equiv,
            [()],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["sparse", "matmul", "dense_equiv", "unmined_seam"],
            expect_xla_reject=True,
        ),
        Case(
            "seam_discovery",
            "sparse_reduce_sum_dense_equiv",
            sparse_reduce_dense_equiv,
            [()],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["sparse", "reduce_sum", "dense_equiv", "unmined_seam"],
            expect_xla_reject=True,
        ),
        Case(
            "seam_discovery",
            "ragged_to_dense_row_sum",
            ragged_to_dense_row_sum,
            [()],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["ragged", "to_dense", "row_sum", "unmined_seam"],
            expect_xla_reject=True,
        ),
        Case(
            "seam_discovery",
            "sets_intersection_dense",
            set_intersection_dense,
            [()],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["sets", "intersection", "sparse_output", "unmined_seam"],
            expect_xla_reject=True,
        ),
        Case(
            "seam_discovery",
            "repeat_axis_counts",
            lambda x, r: tf.repeat(x, r, axis=1),
            [(repeat_x, repeat_counts)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["repeat", "dynamic_counts", "shape", "unmined_seam"],
        ),
        Case(
            "seam_discovery",
            "unique_with_counts_stable_order",
            lambda x: tf.unique_with_counts(x),
            [(unique_counts_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["unique_with_counts", "stable_order", "index", "unmined_seam"],
            expect_xla_reject=True,
        ),
        Case(
            "seam_discovery",
            "searchsorted_left_right_duplicates",
            lambda edges, vals: (tf.searchsorted(edges, vals, side="left"), tf.searchsorted(edges, vals, side="right")),
            [(sorted_edges, search_values)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["searchsorted", "duplicates", "index", "unmined_seam"],
        ),
        Case(
            "seam_discovery",
            "reverse_sequence_lengths",
            lambda x, l: tf.reverse_sequence(x, l, seq_axis=1, batch_axis=0),
            [(reverse_x, reverse_lengths)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["reverse_sequence", "per_row_lengths", "shape", "unmined_seam"],
        ),
        Case(
            "seam_discovery",
            "roll_multi_axis",
            lambda x: tf.roll(x, shift=roll_shift, axis=[0, 1]),
            [(roll_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["roll", "multi_axis", "shape", "unmined_seam"],
        ),
        Case(
            "seam_discovery",
            "space_to_batch_batch_to_space_roundtrip",
            space_to_batch_roundtrip,
            [(space_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["space_to_batch", "batch_to_space", "roundtrip", "unmined_seam"],
        ),
        Case(
            "seam_discovery",
            "embedding_lookup_sparse_dense_equiv",
            embedding_lookup_sparse_dense_equiv,
            [()],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["embedding_lookup_sparse", "dense_equiv", "unmined_seam"],
            expect_xla_reject=True,
        ),
    ]


def amplifier_cases(seed: int) -> list[Case]:
    extreme = tensor([1e20, 1.0, -1e20, 3.0, -2.0, 1e-7, -1e-7], tf.float32)
    logits1 = randn((8, 1), tf.float32, scale=32.0)
    cast_values = tensor(
        np.array(
            [
                1.0 + 2.0**-8,
                1.0 + 2.0**-7,
                1.0 + 3.0 * 2.0**-8,
                -1.0 - 2.0**-8,
                0.333251953125,
                0.333984375,
            ],
            dtype=np.float32,
        ),
        tf.float32,
    )
    cond_x = randn((8,), tf.float32)
    arg_x = tensor([[1.0, -1.0, -5.0, 2.0]], tf.float32)
    arg_rows = tensor([[1.0, -1.0, -5.0, 2.0], [0.25, -0.25, 3.0, -4.0]], tf.float32)
    relu6_min_x = tensor([[-1.0, -2.0, 0.5, 3.0]], tf.float32)
    relu6_max_x = tensor([[7.0, 8.0, 1.0, 2.0]], tf.float32)
    maximum_x = tensor([[-1.0, -2.0, 0.5, 3.0]], tf.float32)
    minimum_x = tensor([[2.0, 3.0, 0.0, -1.0]], tf.float32)
    floor_argmin_x = tensor([[-1.1, -1.9, 0.0, 2.0]], tf.float32)
    floor_argmax_x = tensor([[2.1, 2.9, 1.0, 0.0]], tf.float32)
    ceil_argmin_x = tensor([[1.9, 1.1, 3.0, 4.0]], tf.float32)
    ceil_argmax_x = tensor([[1.1, 1.9, 0.0, -1.0]], tf.float32)
    rint_argmin_x = tensor([[1.49, 1.01, 2.2, 3.0]], tf.float32)
    rint_argmax_x = tensor([[1.01, 1.49, 0.0, -1.0]], tf.float32)
    sign_argmin_x = tensor([[-1.0, -5.0, 2.0, 3.0]], tf.float32)
    sign_argmax_x = tensor([[1.0, 5.0, -2.0, -3.0]], tf.float32)
    return [
        Case(
            "amplifiers",
            "reassociation_threshold_bool",
            lambda a: tf.cast(tf.reduce_sum(a) >= tf.constant(1.0, tf.float32), tf.int32),
            [(extreme,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["amplified_bool_diff", "reassociation", "threshold"],
        ),
        Case(
            "amplifiers",
            "reassociation_gather_index",
            lambda a: tf.gather(
                tf.constant([10, 20, 30, 40], tf.int32),
                tf.clip_by_value(tf.cast(tf.reduce_sum(a), tf.int32), 0, 3),
            ),
            [(extreme,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["amplified_index_diff", "reassociation", "gather"],
        ),
        Case(
            "amplifiers",
            "softmax_size1_exact_equal_count",
            lambda a: tf.reduce_sum(tf.cast(tf.equal(tf.nn.softmax(a, axis=-1), tf.constant(1.0, tf.float32)), tf.int32)),
            [(logits1,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["contract_violation", "exact_value", "softmax_size1"],
        ),
        Case(
            "amplifiers",
            "bf16_roundtrip_boundary_contract",
            lambda a: tf.cast(tf.not_equal(tf.cast(tf.cast(a, tf.bfloat16), tf.float32), a), tf.int32),
            [(cast_values,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["contract_violation", "cast_chain", "bf16_boundary"],
        ),
        Case(
            "amplifiers",
            "cond_dead_branch_is_finite",
            lambda pred, a: tf.cast(
                tf.math.is_finite(tf.cond(pred, lambda: tf.reduce_sum(a), lambda: tf.reduce_sum(tf.math.log(-tf.abs(a))))),
                tf.int32,
            ),
            [(tf.constant(True), cond_x)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["nan_inf_mismatch", "tf.cond", "dead_branch"],
        ),
        Case(
            "amplifiers",
            "argmin_non_strict_monotonic_relu",
            lambda a: tf.argmin(tf.nn.relu(a), axis=-1, output_type=tf.int32),
            [(arg_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmax_non_strict_monotonic_relu",
            lambda a: tf.argmax(tf.nn.relu(a), axis=-1, output_type=tf.int32),
            [(arg_rows,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmin_clip_by_value_plateau",
            lambda a: tf.argmin(tf.clip_by_value(a, -1.0, 1.0), axis=-1, output_type=tf.int32),
            [(arg_rows,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "clip_by_value", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmax_floor_plateau",
            lambda a: tf.argmax(tf.floor(a), axis=-1, output_type=tf.int32),
            [(floor_argmax_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "floor", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmin_floor_plateau",
            lambda a: tf.argmin(tf.floor(a), axis=-1, output_type=tf.int32),
            [(floor_argmin_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "floor", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmin_ceil_plateau",
            lambda a: tf.argmin(tf.math.ceil(a), axis=-1, output_type=tf.int32),
            [(ceil_argmin_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "ceil", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmax_ceil_plateau",
            lambda a: tf.argmax(tf.math.ceil(a), axis=-1, output_type=tf.int32),
            [(ceil_argmax_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "ceil", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmin_rint_plateau",
            lambda a: tf.argmin(tf.raw_ops.Rint(x=a), axis=-1, output_type=tf.int32),
            [(rint_argmin_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "rint", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmax_rint_plateau",
            lambda a: tf.argmax(tf.raw_ops.Rint(x=a), axis=-1, output_type=tf.int32),
            [(rint_argmax_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "rint", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmin_sign_plateau",
            lambda a: tf.argmin(tf.sign(a), axis=-1, output_type=tf.int32),
            [(sign_argmin_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "sign", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmax_sign_plateau",
            lambda a: tf.argmax(tf.sign(a), axis=-1, output_type=tf.int32),
            [(sign_argmax_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "sign", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmin_relu6_low_plateau",
            lambda a: tf.argmin(tf.nn.relu6(a), axis=-1, output_type=tf.int32),
            [(relu6_min_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "relu6", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmax_relu6_high_plateau",
            lambda a: tf.argmax(tf.nn.relu6(a), axis=-1, output_type=tf.int32),
            [(relu6_max_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "relu6", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmin_maximum_low_plateau",
            lambda a: tf.argmin(tf.maximum(a, tf.constant(0.0, tf.float32)), axis=-1, output_type=tf.int32),
            [(maximum_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmin", "maximum", "non_strict_monotonic", "grappler"],
        ),
        Case(
            "amplifiers",
            "argmax_minimum_high_plateau",
            lambda a: tf.argmax(tf.minimum(a, tf.constant(1.0, tf.float32)), axis=-1, output_type=tf.int32),
            [(minimum_x,)],
            seed=seed,
            rtol=0.0,
            atol=0.0,
            tags=["argmax", "minimum", "non_strict_monotonic", "grappler"],
        ),
    ]


def all_cases(seed: int) -> list[Case]:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cases: list[Case] = []
    cases.extend(reduction_cases(seed))
    cases.extend(fusion_nan_cases(seed))
    cases.extend(dtype_cases(seed))
    cases.extend(nan_semantics_cases(seed))
    cases.extend(cast_chain_cases(seed))
    cases.extend(layout_cases(seed))
    cases.extend(dynamic_shape_cases(seed))
    cases.extend(control_flow_cases(seed))
    cases.extend(state_rng_cases(seed))
    cases.extend(backward_cases(seed))
    cases.extend(ordering_tie_cases(seed))
    cases.extend(complex_boundary_cases(seed))
    cases.extend(grad_boundary_cases(seed))
    cases.extend(index_shape_cases(seed))
    cases.extend(math_special_cases(seed))
    cases.extend(normal_numeric_cases(seed))
    cases.extend(invariant_numeric_cases(seed))
    cases.extend(seam_discovery_cases(seed))
    cases.extend(amplifier_cases(seed))
    return cases


ISSUE_QUALITY_CASE_NAMES = {
    # Cast/lowering contracts: normal finite values where an optimizer may
    # incorrectly remove a user-visible narrowing conversion.
    "f32_bf16_f32_roundtrip_values",
    "f64_bf16_f32_roundtrip_values",
    "f32_f16_f32_roundtrip_values",
    "bf16_roundtrip_threshold_mask",
    "bf16_roundtrip_gather_index",
    "bf16_roundtrip_boundary_contract",
    "bf16_roundtrip_after_reshape_reduce",
    "bf16_roundtrip_concat_contract",
    "bf16_roundtrip_where_contract",
    "i64_i32_i64_roundtrip_boundaries",
    "i64_i32_i64_where_contract",
    "i64_i32_i64_gather_index_contract",
    # Graph/lowering contracts with ordinary shapes and non-extreme values.
    "conv_bias_relu_nhwc",
    "space_to_depth_depth_to_space_roundtrip",
    "batch_to_space_space_to_batch_roundtrip",
    "same_function_shape_sequence_reduce",
    "while_loop_accumulation",
    "broadcast_softmax_gradient",
    "gather_batch_dims_matrix_indices",
    "one_hot_out_of_range_indices",
    "unsorted_segment_sum_empty_segments",
    "strided_slice_negative_stride",
    "unique_stable_order_values",
    "sequence_mask_ragged_like_reduce",
    "gather_nd_batched_inner_rows",
    "tensor_scatter_update_distinct_indices",
    "dynamic_slice_normal_bounds",
    # Repair-oriented normal finite numeric signals.
    "erfc_normal_tail_threshold",
    "bessel_i1e_normal_ordering",
    "lgamma_normal_fractional_threshold",
    "softplus_logsigmoid_normal_branch",
    "matvec_normal_threshold_index",
    "erf_erfc_sum_invariant",
    "igamma_complement_sum_invariant",
    "lgamma_recurrence_invariant",
    "softplus_logsigmoid_identity",
    "bessel_i0e_positive_predicate",
    # Confirmed strong Grappler rewrite pattern; no overflow/subnormal trigger.
    "argmin_non_strict_monotonic_relu",
    "argmax_non_strict_monotonic_relu",
    "argmin_clip_by_value_plateau",
    "argmax_floor_plateau",
    "argmin_floor_plateau",
    "argmin_ceil_plateau",
    "argmax_ceil_plateau",
    "argmin_rint_plateau",
    "argmax_rint_plateau",
    "argmin_sign_plateau",
    "argmax_sign_plateau",
    "argmin_relu6_low_plateau",
    "argmax_relu6_high_plateau",
    "argmin_maximum_low_plateau",
    "argmax_minimum_high_plateau",
    # Exact contract; no edge numeric input.
    "softmax_size1_exact_equal_count",
}


def selected_cases_for_seed(groups: list[str], seed: int, max_cases: int) -> list[Case]:
    cases = all_cases(seed)
    if "discovery" in groups:
        cases = [c for c in cases if c.group in {"seam_discovery", "index_shape"}]
        known_basin_names = {
            "ndtri_normal_probability_threshold",
            "ndtri_symmetry_invariant",
            "softmax_size1_exact_equal_count",
            "argmin_non_strict_monotonic_relu",
            "argmax_non_strict_monotonic_relu",
        }
        cases = [c for c in cases if c.name not in known_basin_names]
    elif "issue_quality" in groups:
        cases = [c for c in cases if c.name in ISSUE_QUALITY_CASE_NAMES]
    elif "smoke" in groups:
        smoke_names = {
            "f32_bf16_f32_roundtrip_values",
            "conv_bias_relu_nhwc",
            "same_function_shape_sequence_reduce",
            "broadcast_softmax_gradient",
            "unsorted_segment_sum_empty_segments",
            "igamma_complement_sum_invariant",
            "argmin_non_strict_monotonic_relu",
        }
        cases = [c for c in cases if c.name in smoke_names]
    elif "all" not in groups:
        wanted = set(groups)
        cases = [c for c in cases if c.group in wanted]

    if max_cases > 0:
        by_group: dict[str, int] = {}
        limited = []
        for case in cases:
            seen = by_group.get(case.group, 0)
            if seen < max_cases:
                limited.append(case)
                by_group[case.group] = seen + 1
        cases = limited
    return cases


def selected_cases(groups: list[str], seed: int, max_cases: int, iters: int) -> list[Case]:
    cases: list[Case] = []
    for i in range(iters):
        cases.extend(selected_cases_for_seed(groups, seed + i, max_cases))
    return cases


def write_result(path: Path, result: FuzzResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def run_isolated(script: Path, group: str, name: str, seed: int, timeout: int) -> FuzzResult:
    cmd = [sys.executable, str(script), "--single-case", group, name, "--seed", str(seed)]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        detail = ((e.stderr or "") + "\n" + (e.stdout or ""))[-4000:]
        return FuzzResult(
            group=group,
            name=name,
            seed=seed,
            status="worker_timeout",
            tags=[],
            findings=[{"type": "worker_timeout", "timeout": timeout, "detail": detail}],
            modes={},
            rtol=0.0,
            atol=0.0,
        )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return FuzzResult(**json.loads(line[len("RESULT_JSON "):]))
    detail = (proc.stderr or proc.stdout)[-4000:]
    return FuzzResult(
        group=group,
        name=name,
        seed=seed,
        status="worker_crash",
        tags=[],
        findings=[{"type": "worker_crash", "returncode": proc.returncode, "detail": detail}],
        modes={},
        rtol=0.0,
        atol=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="discovery", help="Comma list: discovery,issue_quality,smoke,all,reduction,fusion_nan,dtype_cast,nan_semantics,cast_chain,layout,dynamic_shape,control_flow,state_rng,backward,ordering_ties,complex_boundaries,grad_boundaries,index_shape,math_special,normal_numeric,invariant_numeric,seam_discovery,amplifiers")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit cases per group. 0 means no limit.")
    parser.add_argument("--iters", type=int, default=1, help="Number of seeded variants to run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-isolate", action="store_true", help="Run all cases in this process.")
    parser.add_argument("--single-case", nargs=2, metavar=("GROUP", "NAME"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    cases = selected_cases(groups, args.seed, args.max_cases, args.iters)

    if args.single_case:
        group, name = args.single_case
        matches = [c for c in all_cases(args.seed) if c.group == group and c.name == name]
        if not matches:
            raise KeyError(f"unknown case: {group}/{name}")
        result = run_case(matches[0])
        print("RESULT_JSON " + json.dumps(asdict(result), sort_keys=True))
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "results" / "targeted_consistency" / f"tf_targeted_consistency_{timestamp}.jsonl"
    if out.exists():
        out.unlink()

    print(f"tf={tf.__version__}")
    print(f"gpus={[d.name for d in tf.config.list_physical_devices('GPU')]}")
    print(f"groups={groups} cases={len(cases)} seed={args.seed} iters={args.iters}")
    print(f"out={out}")
    print()

    results: list[FuzzResult] = []
    script = Path(__file__).resolve()
    for case in cases:
        if args.no_isolate:
            result = run_case(case)
        else:
            result = run_isolated(script, case.group, case.name, case.seed, args.timeout)
        results.append(result)
        write_result(out, result)
        marker = result.status.upper()
        suffix = ""
        if result.findings:
            first = result.findings[0]
            suffix = f" {first.get('pair', first.get('type', ''))}: {first.get('type', '')}"
        print(f"[{marker:12s}] seed={result.seed:<6d} {result.group:14s} {result.name}{suffix}", flush=True)

    summary: dict[str, int] = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    print()
    print("summary=" + json.dumps(summary, sort_keys=True))
    interesting = [r for r in results if r.status != "ok"]
    if interesting:
        print("interesting:")
        for result in interesting[:30]:
            print(f"  {result.status:12s} seed={result.seed:<6d} {result.group:14s} {result.name} tags={','.join(result.tags)}")
    return 1 if interesting else 0


if __name__ == "__main__":
    raise SystemExit(main())
