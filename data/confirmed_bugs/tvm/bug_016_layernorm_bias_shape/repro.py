#!/usr/bin/env python3
# === PROVENANCE: Class B — issue-body lifted ===============================
# Code lifted from upstream issue body via web extraction. Not guaranteed
# byte-identical to the original issue post; verify against upstream before
# relying. See BUG_MAPPING.md > Reproducer Provenance Audit.
# Verified on TVM 0.24.dev0 (LD_LIBRARY_PATH=${ARTIFACT_ROOT}/tvm_fuzz/lib):
# crashes with tvm.error.InternalError 'Op(relax.nn.layer_norm) requires
# ... However, there exists [T.int64(8)] and [T.int64(3)] that are
# unequal' — confirms the [3]-shaped synthesized zero bias bug.
# ==========================================================================
"""
Bug: Relax/ONNX LayerNormalization without optional bias synthesizes a zero
bias of shape data.shape[1] instead of gamma_shape, causing a dimension
mismatch in relax.nn.layer_norm.

Upstream issue: <upstream issue/PR, anonymized> (closed/fixed by PR #<anonymized>)

For input shape [2,3,4,8] with axis=-1, gamma has shape [8] but the
synthesized zero bias gets shape [3] (= data.shape[1]). Legalization fails
with a size-mismatch InternalError.

This breaks PyTorch transformers exported as ONNX with LayerNorm(bias=False).

Oracle: ORT runs the model successfully; TVM ONNX frontend crashes in
from_onnx() when constructing the bias-less LayerNormalization.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
from tvm.relax.frontend.onnx import from_onnx

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4, 8])
Y = onnx.ValueInfoProto(); Y.name = "Y"
W = numpy_helper.from_array(np.ones((8,), dtype=np.float32), "W")

# axis = -1 (default), no B (bias) input — triggers the synthesized-bias path.
node = helper.make_node(
    "LayerNormalization", ["X", "W"], ["Y"], axis=-1, epsilon=1e-5,
)
g = helper.make_graph([node], "g", [X], [Y], initializer=[W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

x = np.random.randn(2, 3, 4, 8).astype(np.float32)
ort_out = ort.InferenceSession(m.SerializeToString()).run(None, {"X": x})[0]
print("ORT output shape:", ort_out.shape)  # (2, 3, 4, 8)

# TVM frontend crashes with size-mismatch in relax.nn.layer_norm.
inf = onnx.shape_inference.infer_shapes(m)
mod = from_onnx(inf)
print("TVM Relax module built — unexpected, bug should have crashed")
