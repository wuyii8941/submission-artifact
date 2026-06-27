# [Bug][Relax][ONNX] Gather with negative indices reads out-of-bounds memory

## Expected behavior

ONNX `Gather` supports negative indices: index `-1` refers to the last element along the axis, `-2` to the second-to-last, etc. Per the spec: *"All index values are expected to be within bounds [-s, s-1] along axis of size s."*

## Actual behavior

TVM passes negative indices directly to `relax.op.take` without normalization. This causes **out-of-bounds memory reads**, returning garbage values (often ~1e+30 magnitude). TVM itself warns: *"Fast mode segfaults when there are out-of-bounds indices."*

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

data = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=np.float32)
indices = np.array([0, -1], dtype=np.int64)  # row 0 and last row (row 2)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
I = helper.make_tensor_value_info("I", TensorProto.INT64, [2])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("Gather", ["X", "I"], ["Y"], axis=0)
graph = helper.make_graph([node], "test", [X, I], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model = onnx.shape_inference.infer_shapes(model)

# ORT: correct
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": data, "I": indices})[0]
print("ORT output:")
print(ort_out)
# [[ 1  2  3  4]    ← row 0
#  [ 9 10 11 12]]   ← row 2 (index -1)

# TVM: reads garbage memory
mod = from_onnx(model)
exe = tvm.relax.build(
    tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])(mod), target="llvm"
)
vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
tvm_out = vm["main"](
    tvm.runtime.tensor(data, device=tvm.cpu()),
    tvm.runtime.tensor(indices, device=tvm.cpu()),
).numpy()
print("TVM output:")
print(tvm_out)
# [[ 1.000e+00  2.000e+00  3.000e+00  4.000e+00]
#  [ 4.524e+30  2.245e+33  ...        ...       ]]  ← GARBAGE from out-of-bounds read
```

## Root cause

In `onnx_frontend.py`, `Gather._impl_v13` passes indices directly to `relax.op.take` without normalizing negative values:

```python
class Gather(OnnxOpConverter):
    <upstream maintainer>
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        indices = inputs[1]
        axis = attr.get("axis", 0)
        # ...
        return relax.op.take(data, indices, axis)
        # Missing: normalize negative indices (indices += dim_size where indices < 0)
```

The same issue also affects `GatherElements`, which passes negative indices to `relax.op.gather_elements` without normalization.

## Severity

**High** — This is a memory safety issue:
- Silent data corruption (returns garbage values without error)
- Potential segfault in certain configurations (TVM warns about this)
- Negative indices are common in ONNX models (e.g., selecting the last element)

## Suggested fix

Normalize negative indices before the `take` call:

```python
<upstream maintainer>
def _impl_v13(cls, bb, inputs, attr, params):
    data = inputs[0]
    indices = inputs[1]
    axis = attr.get("axis", 0)
    # Normalize negative indices
    dim_size = relax.const(int(data.struct_info.shape[axis]), "int64")
    zero = relax.const(0, "int64")
    neg_mask = relax.op.less(indices, zero)
    indices = relax.op.where(neg_mask, relax.op.add(indices, dim_size), indices)
    return relax.op.take(data, indices, axis)
```

## Environment

- TVM: latest main (commit ca68bef6b), also v0.23.0
- Python: 3.11
- OS: Linux
