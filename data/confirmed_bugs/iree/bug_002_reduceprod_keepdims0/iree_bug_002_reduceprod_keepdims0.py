"""IREE Bug: ReduceProd with keepdims=0 crashes during compilation

When ReduceProd reduces along a subset of axes with keepdims=0 (rank reduction),
the IREE compiler generates an invalid tensor.reshape op where the shape operand
rank doesn't match the result tensor rank.

This only affects ReduceProd — other reduce ops (ReduceMax, ReduceMin, ReduceMean,
ReduceL1, ReduceL2) work correctly with keepdims=0.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import subprocess

IREE_IMPORT = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-import-onnx"


def try_compile(desc, model):
    from iree.compiler import compile_str

    onnx_path = "/tmp/_iree_rp_bug.onnx"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    r = subprocess.run([IREE_IMPORT, onnx_path], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {desc}: iree-import-onnx FAIL")
        return
    try:
        compiled = compile_str(
            r.stdout,
            input_type="onnx",
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        print(f"  {desc}: OK")
    except Exception as e:
        err = str(e)
        if "tensor.reshape" in err:
            print(f"  {desc}: CRASH — tensor.reshape rank mismatch")
        else:
            print(f"  {desc}: CRASH — {err[:200]}")


if __name__ == "__main__":
    print("IREE ReduceProd keepdims=0 compilation crash")
    print("=" * 50)

    print("\nReduceProd keepdims=0 (CRASHES for output rank > 0):")
    for shape, axes in [
        ([3, 4], [0]),
        ([3, 4], [1]),
        ([3, 4, 5], [1]),
        ([3, 4, 5], [0, 2]),
        ([2, 3, 4, 5], [1]),
    ]:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("ReduceProd", ["X"], ["Y"], axes=axes, keepdims=0)
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        try_compile(f"shape={shape} axes={axes}", model)

    print("\nReduceProd keepdims=0 reducing to scalar (OK):")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("ReduceProd", ["X"], ["Y"], axes=[0, 1], keepdims=0)
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    try_compile("shape=[3,4] axes=[0,1] (scalar)", model)

    print("\nReduceProd keepdims=1 (all OK):")
    for axes in [[0], [1], [0, 1]]:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("ReduceProd", ["X"], ["Y"], axes=axes, keepdims=1)
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        try_compile(f"axes={axes} keepdims=1", model)

    print("\nOther reduce ops with keepdims=0 (all OK):")
    for op in ["ReduceMax", "ReduceMin", "ReduceMean", "ReduceL1", "ReduceL2"]:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node(op, ["X"], ["Y"], axes=[1], keepdims=0)
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        try_compile(f"{op} axes=[1] keepdims=0", model)
