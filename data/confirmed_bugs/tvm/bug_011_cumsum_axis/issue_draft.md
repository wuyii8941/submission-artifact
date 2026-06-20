# [Bug][Relax][ONNX] CumSum ignores axis parameter, always reduces along axis 0

## Expected behavior

`CumSum` with `axis=1` on a 2D input should compute cumulative sums along each row.

## Actual behavior

TVM hardcodes `axis=0` when the axis input is a `relax.Var` (which is the standard ONNX format — axis is an input tensor, not an attribute). The cumulative sum is always computed along axis 0 regardless of the actual axis value.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

x = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
A = helper.make_tensor_value_info("axis", TensorProto.INT32, [])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("CumSum", ["X", "axis"], ["Y"])
graph = helper.make_graph([node], "test", [X, A], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
model = onnx.shape_inference.infer_shapes(model)

axis_val = np.array(1, dtype=np.int32)

# ORT: correct (cumsum along axis 1)
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": x, "axis": axis_val})[0]

# TVM: wrong (cumsum along axis 0)
mod = from_onnx(model)
exe = tvm.relax.build(
    tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])(mod), target="llvm"
)
vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
tvm_out = vm["main"](
    tvm.runtime.tensor(x, device=tvm.cpu()),
    tvm.runtime.tensor(axis_val, device=tvm.cpu()),
).numpy()

print("Expected (cumsum axis=1):")
print(ort_out)
# [[ 1  3  6 10]
#  [ 5 11 18 26]
#  [ 9 19 30 42]]

print("Got (cumsum axis=0 — WRONG):")
print(tvm_out)
# [[ 1  2  3  4]
#  [ 6  8 10 12]
#  [15 18 21 24]]

print(f"Max diff: {np.max(np.abs(tvm_out - ort_out))}")  # 18.0
```

## Root cause

In `onnx_frontend.py`, `CumSum._impl_v14` hardcodes `axis=0` when the axis input is a `relax.Var`:

```python
class CumSum(OnnxOpConverter):
    <upstream maintainer>
    def _impl_v14(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis = get_constant(inputs[1], params)
        
        if isinstance(axis, relax.Constant):
            axis = int(axis.data.numpy())
        elif isinstance(axis, relax.Var):
            axis = 0  # BUG: hardcoded instead of resolving the actual value
```

In ONNX, `axis` is always provided as an input tensor. When it arrives as a `relax.Var` (not resolved to a constant), the converter silently defaults to 0.

Note: PR #18137 fixed the `reverse` handling for CumSum but did not address this hardcoded axis.

## Suggested fix

Use `get_constant` to resolve the axis value, or raise an error if it cannot be resolved at graph construction time:

```python
if isinstance(axis, relax.Constant):
    axis = int(axis.data.numpy())
elif isinstance(axis, relax.Var):
    raise ValueError(
        "CumSum requires a constant axis value. "
        "Please ensure the axis input is provided as a graph initializer."
    )
```

A better approach: check if the axis is available in `params` (model initializers), which is the common case in real ONNX models.

## Environment

- TVM: latest main (commit ca68bef6b), also v0.23.0
- Python: 3.11
- OS: Linux
