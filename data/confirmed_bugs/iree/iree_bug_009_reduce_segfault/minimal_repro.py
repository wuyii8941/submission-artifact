#!/usr/bin/env python3
"""Reproduce IREE bug: ReduceMax/ReduceMin multi-axis SEGFAULT.

IREE crashes with exit code -11 (SEGFAULT) when compiling ReduceMax/ReduceMin
with keepdims=0 and axes=[0, -1] on input shape [1, 8, 1].
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import subprocess
import os
import sys
import signal

IREE_IMPORT = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-import-onnx"

# Try importing IREE Python API
try:
    from iree.compiler import compile_str
    from iree import runtime as ireert
    HAS_IREE_API = True
except ImportError:
    HAS_IREE_API = False

import onnxruntime as ort


def make_reduce_model(op_type, input_shape, axes, keepdims=0):
    """Create ONNX Reduce model (opset 18, axes as input)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_shape))

    # For opset 18, axes is an input tensor
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name="axes")

    # Calculate output shape
    ndim = len(input_shape)
    norm_axes = [(a % ndim) for a in axes]
    if keepdims:
        out_shape = [1 if i in norm_axes else s for i, s in enumerate(input_shape)]
    else:
        out_shape = [s for i, s in enumerate(input_shape) if i not in norm_axes]
        if not out_shape:
            out_shape = []  # scalar

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape if out_shape else [])

    node = helper.make_node(
        op_type,
        inputs=["X", "axes"],
        outputs=["Y"],
        keepdims=keepdims,
    )

    graph = helper.make_graph([node], "reduce_test", [X], [Y], initializer=[axes_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def run_ort_model(model, inputs):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts,
                                 providers=["CPUExecutionProvider"])
    feed = {inp.name: v for inp, v in zip(sess.get_inputs(), inputs)}
    return sess.run(None, feed)


def test_with_iree_cli(model, inputs, backend="llvm-cpu"):
    """Test using CLI subprocess to capture SEGFAULT exit codes."""
    onnx_path = f"/tmp/_repro_reduce_{os.getpid()}.onnx"
    onnx.save(model, onnx_path)

    # Step 1: iree-import-onnx
    r = subprocess.run([IREE_IMPORT, onnx_path], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, f"iree-import-onnx failed (rc={r.returncode}): {r.stderr[:500]}"

    mlir_text = r.stdout
    mlir_path = f"/tmp/_repro_reduce_{os.getpid()}.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    # Step 2: iree-compile (as subprocess to catch SEGFAULT)
    iree_compile = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-compile"
    vmfb_path = f"/tmp/_repro_reduce_{os.getpid()}.vmfb"

    compile_cmd = [
        iree_compile,
        mlir_path,
        f"--iree-hal-target-backends={backend}",
        "--iree-llvmcpu-target-cpu=host",
        "-o", vmfb_path,
        "--iree-input-type=auto",
    ]

    try:
        r2 = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "iree-compile timed out"

    return r2.returncode, r2.stderr[:1000]


def test_with_iree_api(model, inputs, backend="llvm-cpu"):
    """Test using IREE Python API (may crash the process on SEGFAULT)."""
    if not HAS_IREE_API:
        return None, "IREE Python API not available"

    onnx_path = f"/tmp/_repro_reduce_{os.getpid()}.onnx"
    onnx.save(model, onnx_path)

    r = subprocess.run([IREE_IMPORT, onnx_path], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, f"iree-import-onnx failed: {r.stderr[:500]}"

    try:
        compiled = compile_str(r.stdout, input_type="auto",
                              target_backends=[backend],
                              extra_args=["--iree-llvmcpu-target-cpu=host"])
    except Exception as e:
        return None, f"compile_str raised: {type(e).__name__}: {str(e)[:500]}"

    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
    ctx.add_vm_module(vm_module)

    for func_name in ["reduce_test", "test", "main"]:
        try:
            f = getattr(ctx.modules.module, func_name)
            break
        except AttributeError:
            continue
    else:
        avail = dir(ctx.modules.module)
        return None, f"No function found. Available: {avail}"

    result = f(*inputs)
    if hasattr(result, 'to_host'):
        return [np.array(result)], None
    return [np.array(r) for r in result] if isinstance(result, (list, tuple)) else [np.array(result)], None


def main():
    print("=" * 70)
    print("IREE Bug Reproduction: ReduceMax/ReduceMin multi-axis SEGFAULT")
    print("=" * 70)

    np.random.seed(42)

    test_cases = [
        # (op_type, input_shape, axes, keepdims)
        ("ReduceMax", (1, 8, 1), [0, -1], 0),
        ("ReduceMin", (1, 8, 1), [0, -1], 0),
        # Also test with explicit positive axes
        ("ReduceMax", (1, 8, 1), [0, 2], 0),
        ("ReduceMin", (1, 8, 1), [0, 2], 0),
        # Test with keepdims=1 for comparison
        ("ReduceMax", (1, 8, 1), [0, -1], 1),
        ("ReduceMin", (1, 8, 1), [0, -1], 1),
        # Different shapes with multi-axis
        ("ReduceMax", (3, 4, 5), [0, 2], 0),
        ("ReduceMin", (3, 4, 5), [0, 2], 0),
    ]

    for op_type, input_shape, axes, keepdims in test_cases:
        print(f"\n--- {op_type}: shape={input_shape}, axes={axes}, keepdims={keepdims} ---")

        x = np.random.randn(*input_shape).astype(np.float32)
        model = make_reduce_model(op_type, input_shape, axes, keepdims)

        # ORT reference
        try:
            ort_out = run_ort_model(model, [x])
            print(f"  ORT output shape: {ort_out[0].shape}, value: {ort_out[0].flatten()[:8]}")
        except Exception as e:
            print(f"  ORT failed: {e}")
            continue

        # IREE via subprocess (safe for SEGFAULT detection)
        rc, stderr = test_with_iree_cli(model, [x])

        if rc is None:
            print(f"  IREE error: {stderr}")
        elif rc == -11:
            print(f"  IREE SEGFAULT (exit code -11)!")
            print(f"  stderr: {stderr[:300]}")
        elif rc == -6:
            print(f"  IREE ABORT (exit code -6)!")
            print(f"  stderr: {stderr[:300]}")
        elif rc != 0:
            print(f"  IREE compile failed (exit code {rc})")
            print(f"  stderr: {stderr[:300]}")
        else:
            print(f"  IREE compiled successfully (rc=0)")
            # If compilation succeeded, try running via API
            iree_out, err = test_with_iree_api(model, [x])
            if err:
                print(f"  IREE runtime error: {err}")
            else:
                diff = np.abs(ort_out[0] - iree_out[0])
                max_diff = float(np.max(diff))
                print(f"  IREE output: {iree_out[0].flatten()[:8]}, max_diff={max_diff:.6e}")

    print(f"\n{'=' * 70}")
    print("ONNX Spec validity: ReduceMax/ReduceMin with multi-axis reduction")
    print("and keepdims=0 is valid per ONNX spec (opset 18).")
    print("axes=[0, -1] is equivalent to axes=[0, 2] for a 3D tensor.")
    print("=" * 70)


if __name__ == "__main__":
    main()
