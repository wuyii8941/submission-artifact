"""IREE Bug: Unique op causes compiler SEGFAULT (signal 11)

The ONNX Unique op is imported into MLIR successfully, but iree-compile
crashes with SIGSEGV (signal 11) during compilation.

Compilers should never segfault on valid input.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import subprocess

IREE_IMPORT = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-import-onnx"


if __name__ == "__main__":
    print("IREE Unique compiler SEGFAULT reproduction")
    print("=" * 50)

    x = np.array([3, 1, 2, 1, 3, 2, 4], dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [7])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Unique", ["X"], ["Y"], sorted=1)
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)

    onnx.save(model, "/tmp/unique.onnx")
    r = subprocess.run([IREE_IMPORT, "/tmp/unique.onnx"], capture_output=True, text=True)
    print(f"Import: {'OK' if r.returncode == 0 else 'FAIL'}")

    if r.returncode == 0:
        print(f"MLIR:\n{r.stdout}")
        from iree.compiler import compile_str
        try:
            compiled = compile_str(
                r.stdout, input_type="onnx", target_backends=["llvm-cpu"],
                extra_args=["--iree-llvmcpu-target-cpu=generic"],
            )
            print(f"Compile: OK ({len(compiled)} bytes)")
        except Exception as e:
            err = str(e)
            if "Error code: -11" in err:
                print("Compile: SEGFAULT (signal 11) — compiler crashed!")
            else:
                print(f"Compile: FAIL — {err[:200]}")
