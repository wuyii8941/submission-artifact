"""IREE Bug: ReverseSequence produces wrong results when batch_axis != 0

ReverseSequence with batch_axis=1, time_axis=0 produces incorrect output.
The default configuration (batch_axis=0, time_axis=1) works correctly.

Oracle: ONNX Runtime
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

IREE_IMPORT = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-import-onnx"


def run(model, inputs):
    model = onnx.shape_inference.infer_shapes(model)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts)
    ort_out = sess.run(None, {i.name: v for i, v in zip(sess.get_inputs(), inputs)})[0]

    onnx.save(model, "/tmp/_rs_bug.onnx")
    r = subprocess.run([IREE_IMPORT, "/tmp/_rs_bug.onnx"], capture_output=True, text=True)
    c = compile_str(r.stdout, input_type="onnx", target_backends=["llvm-cpu"],
                    extra_args=["--iree-llvmcpu-target-cpu=generic"])
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm = ireert.VmModule.copy_buffer(ctx.instance, c)
    ctx.add_vm_module(vm)
    iree_out = ctx.modules.module.test(*inputs).to_host()
    return ort_out, iree_out


if __name__ == "__main__":
    print("IREE ReverseSequence axis bug reproduction")
    print("=" * 50)

    x = np.array([[1, 4, 7],
                  [2, 5, 8],
                  [3, 6, 9],
                  [0, 0, 0]], dtype=np.float32)
    lens = np.array([3, 2, 1], dtype=np.int64)

    print("\nbatch_axis=1, time_axis=0 (WRONG):")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])
    L = helper.make_tensor_value_info("L", TensorProto.INT64, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    n = helper.make_node("ReverseSequence", ["X", "L"], ["Y"],
                         batch_axis=1, time_axis=0)
    g = helper.make_graph([n], "test", [X, L], [Y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    ort_out, iree_out = run(m, [x, lens])
    print(f"  Input:\n{x}")
    print(f"  lens={lens}")
    print(f"  ORT (correct):\n{ort_out}")
    print(f"  IREE (wrong):\n{iree_out}")
    print(f"  Max diff: {np.max(np.abs(ort_out - iree_out))}")

    print("\nbatch_axis=0, time_axis=1 (OK):")
    x2 = x.T.copy()
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x2.shape))
    L = helper.make_tensor_value_info("L", TensorProto.INT64, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    n = helper.make_node("ReverseSequence", ["X", "L"], ["Y"],
                         batch_axis=0, time_axis=1)
    g = helper.make_graph([n], "test", [X, L], [Y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    ort_out2, iree_out2 = run(m, [x2, lens])
    print(f"  ORT:\n{ort_out2}")
    print(f"  IREE:\n{iree_out2}")
    print(f"  Match: {np.allclose(ort_out2, iree_out2)}")
