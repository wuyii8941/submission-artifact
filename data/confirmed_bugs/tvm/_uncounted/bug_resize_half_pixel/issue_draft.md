# [Bug][Relax][ONNX] Resize produces wrong results with half_pixel/pytorch_half_pixel coordinate transformation and non-integer scales

## Expected behavior

`Resize` with `coordinate_transformation_mode=half_pixel` (or `pytorch_half_pixel`, `asymmetric`) and non-integer scale factors (e.g., 2.5x) should compute source coordinates as:
```
src = (dst + 0.5) / scale - 0.5
```
and produce results matching ONNX Runtime.

## Actual behavior

TVM computes incorrect source coordinates, leading to wrong pixel mapping. For a 3×3 input with scale=2.5, the max absolute difference vs ORT is **4.0** (nearest mode) and **0.63** (linear mode).

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
roi = numpy_helper.from_array(np.array([], dtype=np.float32), "roi")
sc = numpy_helper.from_array(np.array([1, 1, 2.5, 2.5], dtype=np.float32), "scales")
node = helper.make_node("Resize", ["X", "roi", "scales"], ["Y"],
                        mode="nearest",
                        coordinate_transformation_mode="half_pixel",
                        nearest_mode="floor")
graph = helper.make_graph([node], "test", [X], [Y], initializer=[roi, sc])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model = onnx.shape_inference.infer_shapes(model)

# ORT
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": x})[0]

# TVM
mod = from_onnx(model)
exe = tvm.relax.build(
    tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])(mod), target="llvm")
vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
tvm_out = vm["main"](tvm.runtime.tensor(x, device=tvm.cpu())).numpy()

print("ORT output (7x7):")
print(ort_out.reshape(7, 7))
# [[0. 0. 0. 0. 1. 1. 2.]
#  [0. 0. 0. 0. 1. 1. 2.]  ...

print("TVM output (7x7):")
print(tvm_out.reshape(7, 7))
# [[0. 0. 0. 1. 1. 1. 2.]
#  [0. 0. 0. 1. 1. 1. 2.]  ...

print("Max diff:", np.abs(ort_out - tvm_out).max())  # 4.0
```

## Affected configurations

| mode | coordinate_transformation_mode | max_diff vs ORT |
|------|-------------------------------|-----------------|
| nearest | half_pixel | 4.0 |
| nearest | pytorch_half_pixel | 4.0 |
| linear | half_pixel | 0.63 |
| linear | asymmetric | 0.46 |
| linear | pytorch_half_pixel | 0.63 |

Trigger condition: non-integer scale factor (e.g., 2.5) on odd spatial dims (e.g., 3×3). Integer scales (2x, 3x) appear correct.

## Root cause

The coordinate transformation for `half_pixel` mode applies `floor((dst + 0.5) / scale - 0.5)` but the implementation appears to use a different rounding or offset, causing the source index to be off by one pixel for certain output positions.

## Environment

- TVM: 0.24.dev0, commit 0b0afd8dd (2026-04-24)
- Python: 3.11
- OS: Linux
