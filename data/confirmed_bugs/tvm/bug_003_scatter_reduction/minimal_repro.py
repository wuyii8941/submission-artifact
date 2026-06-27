"""Bug 005: ONNX ScatterElements ignores 'reduction' attribute.

TVM version: 0.23.0 (also on main)
Status: CONFIRMED BUG (unfixed on main)

Description:
  TVM's ONNX frontend ScatterElements converter ignores the 'reduction'
  attribute (v16+). When reduction="add" or "mul", TVM still uses the
  default "none" (overwrite) behavior, producing wrong results.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

np.random.seed(42)

for reduction in ["add", "mul"]:
    shape = (4, 4)
    X = np.ones(shape, dtype=np.float32)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    updates = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
    idx_info = helper.make_tensor_value_info("idx", TensorProto.INT64, [2, 2])
    upd_info = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2, 2])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(shape))
    node = helper.make_node("ScatterElements", ["X", "idx", "updates"], ["Y"],
                            axis=0, reduction=reduction)
    graph = helper.make_graph([node], "test", [X_info, idx_info, upd_info], [Y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    # ORT
    sess = ort.InferenceSession(model.SerializeToString())
    ort_out = sess.run(None, {"X": X, "idx": indices, "updates": updates})[0]

    # TVM
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    mod = from_onnx(model)
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    exe = tvm.relax.build(pipeline(mod), target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_out = vm["main"](
        tvm.runtime.tensor(X, device=tvm.cpu()),
        tvm.runtime.tensor(indices, device=tvm.cpu()),
        tvm.runtime.tensor(updates, device=tvm.cpu()),
    ).numpy()

    match = np.allclose(tvm_out, ort_out)
    print(f"reduction='{reduction}': TVM matches ORT? {match}")
    if not match:
        print(f"  ORT output:\n{ort_out}")
        print(f"  TVM output:\n{tvm_out}")
        print(f"  BUG: TVM ignores reduction='{reduction}', uses overwrite instead")
    print()
