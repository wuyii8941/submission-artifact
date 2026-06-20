#!/usr/bin/env python3
# === PROVENANCE: Class B — issue-body lifted ===============================
# Code lifted from upstream issue body via web extraction. Not guaranteed
# byte-identical to the original issue post; verify against upstream before
# relying. See BUG_MAPPING.md > Reproducer Provenance Audit.
# Verified on TVM 0.24.dev0 (LD_LIBRARY_PATH=/data1/tzh/tvm_fuzz/lib):
# crashes with tvm.error.InternalError 'PermuteDims expects ... tensor
# ndim is 5 while the given number of axes is 4' — exact match.
# ==========================================================================
"""
Bug: Relax/ONNX GridSample._impl_v16 unconditionally permutes the grid from
[N,H,W,2] to [N,2,H,W] and dispatches to the 2D image.grid_sample, crashing
on 5D volumetric inputs.

Upstream issue: <upstream issue/PR, anonymized> (closed/fixed by PR #<anonymized>)

Error message:
    PermuteDims expects the number of input axes to equal the ndim of the
    input tensor. However, the tensor ndim is 5 while the given number of
    axes is 4

Oracle: ORT runs the 5D model successfully; TVM ONNX frontend crashes.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from tvm.relax.frontend.onnx import from_onnx

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4, 4])
G = helper.make_tensor_value_info("G", TensorProto.FLOAT, [1, 4, 4, 4, 3])
Y = onnx.ValueInfoProto(); Y.name = "Y"
node = helper.make_node(
    "GridSample", ["X", "G"], ["Y"],
    mode="nearest", padding_mode="zeros", align_corners=0,
)
g = helper.make_graph([node], "g", [X, G], [Y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])

x = np.random.randn(1, 1, 4, 4, 4).astype(np.float32)
grid = np.random.uniform(-1.2, 1.2, size=(1, 4, 4, 4, 3)).astype(np.float32)

# ORT works
ort_out = ort.InferenceSession(m.SerializeToString()).run(None, {"X": x, "G": grid})[0]
print("ORT output shape:", ort_out.shape)

# TVM ONNX frontend crashes here
inf = onnx.shape_inference.infer_shapes(m)
mod = from_onnx(inf)
print("TVM Relax module built — unexpected, bug should have crashed")
