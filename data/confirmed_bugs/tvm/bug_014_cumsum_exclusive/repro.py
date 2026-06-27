"""Bug 3: CumSum with exclusive=1 crashes TVM ONNX frontend.

Error: "Exclusive option not yet supported."

Root cause: The TVM ONNX frontend CumSum converter (onnx_frontend.py line ~1389)
has an explicit assertion that rejects exclusive=1:
    assert not attr.get("exclusive", False), "Exclusive option not yet supported."

ONNX spec validity: YES -- exclusive=1 is a valid ONNX CumSum attribute (opset 11+).
It means the i-th output element is the cumulative sum of elements 0..i-1 (excluding i).

Severity: MEDIUM -- explicitly unimplemented feature, not a logic error. But still
a valid ONNX model that ORT handles correctly.

Tested on TVM v0.23.0 (commit 75ce1b4d4), ORT 1.24.4.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Build model
np.random.seed(42)
x = np.random.randn(5).astype(np.float32)
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [5])
out = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
axis_init = numpy_helper.from_array(np.array(0, dtype=np.int32), "axis")
node = helper.make_node("CumSum", ["x", "axis"], ["y"], exclusive=1, reverse=0)
graph = helper.make_graph([node], "test", [inp], [out], initializer=[axis_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model = onnx.shape_inference.infer_shapes(model)

# ORT reference
import onnxruntime as ort
opts = ort.SessionOptions()
opts.log_severity_level = 3
sess = ort.InferenceSession(model.SerializeToString(), opts,
                            providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"x": x})
print(f"ORT: {ort_out[0]}")
print(f"  (exclusive cumsum: first element is 0, then cumsum of preceding elements)")

# TVM
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

shape_dict = {"x": [5]}
try:
    mod = from_onnx(model, shape_dict)
    print("TVM: conversion succeeded (bug may be fixed)")
except Exception as e:
    print(f"TVM: CRASHED -- {e}")
