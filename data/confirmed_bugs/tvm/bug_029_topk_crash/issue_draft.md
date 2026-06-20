# [Bug][Relax][ONNX] TopK operator crashes during from_onnx() conversion

## Expected behavior

`from_onnx()` should successfully convert ONNX models containing `TopK` operators.

## Actual behavior

Any ONNX model with a `TopK` operator crashes with:
```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

This is 100% reproducible — no TopK model can be converted.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import tvm
from tvm.relax.frontend.onnx import from_onnx

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5, 10])
K_init = numpy_helper.from_array(np.array([3], dtype=np.int64), "K")
Y_val = helper.make_tensor_value_info("Values", TensorProto.FLOAT, None)
Y_idx = helper.make_tensor_value_info("Indices", TensorProto.INT64, None)
node = helper.make_node("TopK", ["X", "K"], ["Values", "Indices"], axis=1)
graph = helper.make_graph([node], "test", [X], [Y_val, Y_idx],
                          initializer=[K_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model = onnx.shape_inference.infer_shapes(model)

mod = from_onnx(model)  # TypeError: only 0-dimensional arrays can be converted to Python scalars
```

## Root cause

The TopK converter receives `K` as a 1-element tensor constant (shape `[1]`), but tries to extract it as a Python scalar directly. The 1-D tensor needs `.item()` or indexing `[0]` to extract the scalar value.

Tested with multiple shapes and axis values — all crash:
- `shape=(5, 10), k=3, axis=1` → CRASH
- `shape=(5, 10), k=3, axis=0` → CRASH
- `shape=(3, 4, 5), k=2, axis=2` → CRASH
- `shape=(100,), k=5, axis=0` → CRASH

## Environment

- TVM: 0.24.dev0, commit 0b0afd8dd (2026-04-24)
- Python: 3.11
- OS: Linux
