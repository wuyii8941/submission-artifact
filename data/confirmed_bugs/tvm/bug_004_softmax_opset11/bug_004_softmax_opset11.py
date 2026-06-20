"""Bug 004: ONNX Softmax/Hardmax/LogSoftmax opset <13 semantics not implemented.

TVM version: 0.23.0 (also on main)
Status: CONFIRMED BUG (unfixed on main)

Description:
  TVM's ONNX frontend only implements Softmax v13 semantics (direct axis softmax).
  ONNX opset <13 Softmax has DIFFERENT semantics: flatten to 2D at axis,
  compute softmax on last dim, reshape back.

  This causes silently wrong results for any ONNX model using opset <=12.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

np.random.seed(42)
shape = (2, 3, 4)
X = np.random.randn(*shape).astype(np.float32)

X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(shape))
node = helper.make_node("Softmax", ["X"], ["Y"])  # default axis
graph = helper.make_graph([node], "test", [X_info], [Y_info])

model11 = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
model13 = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# ORT correctly handles both opsets
sess11 = ort.InferenceSession(model11.SerializeToString())
ort11 = sess11.run(None, {"X": X})[0]

sess13 = ort.InferenceSession(model13.SerializeToString())
ort13 = sess13.run(None, {"X": X})[0]

print(f"ORT opset11 == ORT opset13? {np.allclose(ort11, ort13)}")
print("(Expected: False, because opset 11 flattens to 2D at axis=1)")

# TVM
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

for name, model, ort_ref in [("opset11", model11, ort11), ("opset13", model13, ort13)]:
    mod = from_onnx(model)
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    exe = tvm.relax.build(pipeline(mod), target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_out = vm["main"](tvm.runtime.tensor(X, device=tvm.cpu())).numpy()

    match = np.allclose(tvm_out, ort_ref, atol=1e-5)
    print(f"\nTVM {name}: matches ORT? {match}")
    if not match:
        print(f"  max_diff = {np.max(np.abs(tvm_out - ort_ref)):.6f}")
        print(f"  BUG: TVM uses opset13 semantics for opset11 model!")
