"""IREE Bug: float64 (double) operations produce corrupted results

IREE mishandles float64 tensors, producing corrupted output for any
operation on float64 data. The corruption pattern suggests buffer
misalignment (reading 4 bytes per element instead of 8, or similar).

This affects:
- float64 inputs with any op (e.g., Abs of [1,2,3,4] -> [1,2,0,0])
- Cast from float32 to float64 (garbage output)
- Cast from float64 to float32 (garbage output)

Oracle: ONNX Runtime
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import subprocess

IREE_IMPORT = "/data1/tzh/tvm_fuzz/bin/iree-import-onnx"


def run_iree(model, inputs):
    from iree.compiler import compile_str
    from iree import runtime as ireert

    onnx_path = "/tmp/_iree_f64_bug.onnx"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    r = subprocess.run([IREE_IMPORT, onnx_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"import failed: {r.stderr[:300]}")
    compiled = compile_str(
        r.stdout, input_type="onnx", target_backends=["llvm-cpu"],
        extra_args=["--iree-llvmcpu-target-cpu=generic"],
    )
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm = ireert.VmModule.copy_buffer(ctx.instance, compiled)
    ctx.add_vm_module(vm)
    result = ctx.modules.module.test(*inputs)
    if hasattr(result, "to_host"):
        return result.to_host()
    return [r.to_host() for r in result]


def run_ort(model, inputs):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts)
    feed = {inp.name: v for inp, v in zip(sess.get_inputs(), inputs)}
    return sess.run(None, feed)


if __name__ == "__main__":
    print("IREE float64 corruption bug reproduction")
    print("=" * 50)

    # Test 1: float64 input → Abs → float64 output
    print("\n1. float64 Abs (identity-like):")
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    X = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, None)
    node = helper.make_node("Abs", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    ort_out = run_ort(model, [x])[0]
    iree_out = run_iree(model, [x])
    print(f"  Input:     {x}")
    print(f"  ORT:       {ort_out}")
    print(f"  IREE:      {iree_out}")
    print(f"  Match: {np.allclose(ort_out, iree_out)}")

    # Test 2: Cast float32 → float64
    print("\n2. Cast float32 → float64:")
    x32 = np.array([-1.5, 0.0, 1.5, 3.0], dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, None)
    node = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.DOUBLE)
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    ort_out = run_ort(model, [x32])[0]
    iree_out = run_iree(model, [x32])
    print(f"  Input:     {x32}")
    print(f"  ORT:       {ort_out}")
    print(f"  IREE:      {iree_out}")
    print(f"  Match: {np.allclose(ort_out, iree_out)}")

    # Test 3: Cast float64 → float32
    print("\n3. Cast float64 → float32:")
    x64 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    X = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT)
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    ort_out = run_ort(model, [x64])[0]
    iree_out = run_iree(model, [x64])
    print(f"  Input:     {x64}")
    print(f"  ORT:       {ort_out}")
    print(f"  IREE:      {iree_out}")
    print(f"  Match: {np.allclose(ort_out, iree_out)}")

    # Test 4: float32 operations work correctly (for comparison)
    print("\n4. float32 Abs (works correctly):")
    x32 = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Abs", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    ort_out = run_ort(model, [x32])[0]
    iree_out = run_iree(model, [x32])
    print(f"  Input:     {x32}")
    print(f"  ORT:       {ort_out}")
    print(f"  IREE:      {iree_out}")
    print(f"  Match: {np.allclose(ort_out, iree_out)}")
