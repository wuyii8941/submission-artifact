#!/usr/bin/env python3
"""Reproduce IREE bug: Resize nearest + half_pixel numerical error.

IREE gives wrong results for Resize(mode=nearest, coordinate_transformation_mode=half_pixel)
on CPU (llvm-cpu). CUDA variant times out during compilation.

Tests: 1x1x4x4, 1x1x7x7, 1x3x8x8 with scales 1.5, 2.0, 3.0
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import subprocess
import os
import sys

# IREE Python API
from iree.compiler import compile_str
from iree import runtime as ireert
import onnxruntime as ort

IREE_IMPORT = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-import-onnx"


def make_resize_model(input_shape, scales_hw, coord_mode="half_pixel", mode="nearest"):
    """Create ONNX Resize model (opset 11+)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_shape))

    # Full scales: [1, 1, scale_h, scale_w]
    scales_val = np.array([1.0, 1.0, scales_hw[0], scales_hw[1]], dtype=np.float32)
    out_h = int(input_shape[2] * scales_hw[0])
    out_w = int(input_shape[3] * scales_hw[1])
    output_shape = [input_shape[0], input_shape[1], out_h, out_w]

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)

    # roi and scales as initializers; use 3 inputs (no sizes)
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(scales_val, name="scales")

    attrs = {
        "mode": mode,
        "coordinate_transformation_mode": coord_mode,
    }
    if mode == "nearest":
        attrs["nearest_mode"] = "floor"

    node = helper.make_node(
        "Resize",
        inputs=["X", "roi", "scales"],
        outputs=["Y"],
        **attrs,
    )

    graph = helper.make_graph([node], "resize_test", [X], [Y],
                              initializer=[roi, scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def run_ort_model(model, inputs):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts,
                                 providers=["CPUExecutionProvider"])
    feed = {inp.name: v for inp, v in zip(sess.get_inputs(), inputs)}
    return sess.run(None, feed)


def run_iree_model(model, inputs, backend="llvm-cpu"):
    onnx_path = f"/tmp/_repro_resize_{os.getpid()}.onnx"
    onnx.save(model, onnx_path)

    r = subprocess.run([IREE_IMPORT, onnx_path], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, f"iree-import-onnx failed: {r.stderr[:500]}"

    extra_args = []
    if backend == "llvm-cpu":
        extra_args = ["--iree-llvmcpu-target-cpu=host"]

    try:
        compiled = compile_str(r.stdout, input_type="auto",
                              target_backends=[backend],
                              extra_args=extra_args)
    except Exception as e:
        return None, f"compile_str failed: {str(e)[:500]}"

    driver = "local-task" if backend == "llvm-cpu" else "cuda"
    config = ireert.Config(driver)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
    ctx.add_vm_module(vm_module)

    for func_name in ["resize_test", "test", "main"]:
        try:
            f = getattr(ctx.modules.module, func_name)
            break
        except AttributeError:
            continue
    else:
        # List available functions
        avail = dir(ctx.modules.module)
        return None, f"No test/main/resize_test function found. Available: {avail}"

    result = f(*inputs)
    if hasattr(result, 'to_host'):
        return [np.array(result)], None
    return [np.array(r) for r in result] if isinstance(result, (list, tuple)) else [np.array(result)], None


def main():
    print("=" * 70)
    print("IREE Bug Reproduction: Resize nearest + half_pixel numerical error")
    print("=" * 70)

    test_cases = [
        ((1, 1, 4, 4), (1.5, 1.5)),
        ((1, 1, 4, 4), (2.0, 2.0)),
        ((1, 1, 4, 4), (3.0, 3.0)),
        ((1, 1, 7, 7), (1.5, 1.5)),
        ((1, 1, 7, 7), (2.0, 2.0)),
        ((1, 1, 7, 7), (3.0, 3.0)),
        ((1, 3, 8, 8), (1.5, 1.5)),
        ((1, 3, 8, 8), (2.0, 2.0)),
        ((1, 3, 8, 8), (3.0, 3.0)),
    ]

    any_bug = False
    np.random.seed(42)

    for input_shape, scales in test_cases:
        print(f"\n--- Test: shape={input_shape}, scales={scales} ---")

        x = np.random.randn(*input_shape).astype(np.float32)
        model = make_resize_model(input_shape, scales)

        # ORT reference
        ort_out = run_ort_model(model, [x])
        print(f"  ORT output shape: {ort_out[0].shape}")

        # IREE CPU
        iree_out, err = run_iree_model(model, [x], backend="llvm-cpu")
        if err:
            print(f"  IREE CPU ERROR: {err}")
            any_bug = True
            continue

        print(f"  IREE output shape: {iree_out[0].shape}")

        # Compare
        ref = ort_out[0]
        got = iree_out[0]

        if ref.shape != got.shape:
            print(f"  SHAPE MISMATCH: ORT={ref.shape} vs IREE={got.shape}")
            any_bug = True
            continue

        diff = np.abs(ref - got)
        max_diff = float(np.max(diff))
        n_differ = int(np.sum(diff > 1e-5))
        total = ref.size

        if max_diff > 1e-5:
            print(f"  NUMERICAL BUG: max_diff={max_diff:.6e}, {n_differ}/{total} elements differ")
            # Show first few differences
            flat_ref = ref.flatten()
            flat_got = got.flatten()
            flat_diff = diff.flatten()
            worst_idx = np.argsort(flat_diff)[-5:][::-1]
            for idx in worst_idx:
                if flat_diff[idx] > 1e-5:
                    print(f"    idx={idx}: ORT={flat_ref[idx]:.6f}, IREE={flat_got[idx]:.6f}, diff={flat_diff[idx]:.6e}")
            any_bug = True
        else:
            print(f"  OK: max_diff={max_diff:.6e}")

    print("\n" + "=" * 70)
    if any_bug:
        print("RESULT: BUG REPRODUCED - Resize nearest + half_pixel gives wrong results on IREE CPU")
        print("\nONNX Spec validity: coordinate_transformation_mode='half_pixel' with")
        print("mode='nearest' is valid per ONNX spec (opset 11+). ORT handles it correctly.")
    else:
        print("RESULT: No numerical differences found")
    print("=" * 70)


if __name__ == "__main__":
    main()
