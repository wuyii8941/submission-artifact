# [Bug][Relax][ONNX] ScatterElements ignores `reduction` attribute, silently produces wrong results

## Expected behavior

`ScatterElements` with `reduction="add"` should **accumulate** updates onto existing values (i.e., `data[i] += update`). Similarly, `reduction="mul"` should multiply, and `reduction="min"`/`"max"` should take element-wise min/max.

## Actual behavior

TVM always **overwrites** the data regardless of the `reduction` attribute, producing wrong results without any error or warning.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

data = np.ones((3, 3), dtype=np.float32)
indices = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
updates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
I = helper.make_tensor_value_info("I", TensorProto.INT64, [2, 3])
U = helper.make_tensor_value_info("U", TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("ScatterElements", ["X", "I", "U"], ["Y"],
                        axis=0, reduction="add")
graph = helper.make_graph([node], "test", [X, I, U], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model = onnx.shape_inference.infer_shapes(model)

# ORT: correct (add updates to existing values)
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": data, "I": indices, "U": updates})[0]

# TVM: wrong (overwrites instead of adding)
mod = from_onnx(model)
exe = tvm.relax.build(
    tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])(mod), target="llvm"
)
vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
tvm_out = vm["main"](
    tvm.runtime.tensor(data, device=tvm.cpu()),
    tvm.runtime.tensor(indices, device=tvm.cpu()),
    tvm.runtime.tensor(updates, device=tvm.cpu()),
).numpy()

print("ORT (correct, reduction='add'):")
print(ort_out)
# [[6. 1. 1.]    ← 1 + 1 + 4 = 6
#  [1. 8. 1.]    ← 1 + 2 + 5 = 8
#  [1. 1. 10.]]  ← 1 + 3 + 6 = 10

print("TVM (wrong, overwrites):")
print(tvm_out)
# [[4. 1. 1.]    ← last write wins (4 overwrites 1)
#  [1. 5. 1.]    ← last write wins (5 overwrites 2)
#  [1. 1. 6.]]   ← last write wins (6 overwrites 3)
```

## Root cause

In `onnx_frontend.py`, `ScatterElements` only has `_impl_v11` which does not read the `reduction` attribute:

```python
class ScatterElements(OnnxOpConverter):
    <upstream maintainer>
    def _impl_v11(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 0)
        return relax.op.scatter_elements(inputs[0], inputs[1], inputs[2], axis=axis)
        # Missing: reduction = attr.get("reduction", b"none")
```

Note that the `ScatterND` converter in the same file correctly handles reduction (with a `_reduction_check` method), but `ScatterElements` does not.

## Suggested fix

Add `_impl_v16` (opset 16 introduced `reduction`):

```python
<upstream maintainer>
def _impl_v16(cls, bb, inputs, attr, params):
    axis = attr.get("axis", 0)
    reduction = attr.get("reduction", b"none")
    if isinstance(reduction, bytes):
        reduction = reduction.decode("utf-8")
    return relax.op.scatter_elements(
        inputs[0], inputs[1], inputs[2], axis=axis, reduction=reduction
    )
```

## Impact

Any ONNX model using `ScatterElements` with `reduction="add"` (common in embedding gradient accumulation, GNN message passing, sparse update patterns) will silently produce wrong results.

## Environment

- TVM: latest main (commit ca68bef6b), also v0.23.0
- Python: 3.11
- OS: Linux
