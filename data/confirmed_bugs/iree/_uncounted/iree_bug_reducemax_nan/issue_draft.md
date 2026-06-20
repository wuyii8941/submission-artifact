# ReduceMax/ReduceMin over-propagates NaN — returns NaN for rows where non-NaN value should win

## Expected behavior

For `ReduceMax([[NaN, 1.0], [2.0, NaN]], axis=1)`:
- Row 0 `[NaN, 1.0]`: NaN is present → result should be NaN (per ORT)
- Row 1 `[2.0, NaN]`: non-NaN value 2.0 should dominate → result should be 2.0 (per ORT)

Expected: `[NaN, 2.0]`

## Actual behavior

IREE returns `[NaN, NaN]` — NaN infects row 1 even though it has a valid non-NaN maximum.

```
Input:
[[nan  1.]
 [ 2. nan]]

ReduceMax(axis=1):
  ORT:  [nan  2.]
  IREE: [nan nan]

ReduceMin(axis=1):
  ORT:  [nan  2.]
  IREE: [nan nan]
```

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

x = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
node = helper.make_node("ReduceMax", ["X"], ["Y"], axes=[1], keepdims=0)
graph = helper.make_graph([node], "main", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model = onnx.shape_inference.infer_shapes(model)

# ORT
sess = ort.InferenceSession(model.SerializeToString())
print("ORT:", sess.run(None, {"X": x})[0])  # [nan  2.]

# IREE (via import_onnx + compile_str pipeline)
# ... (same pipeline as Sign bug)
# Result: [nan nan]
```

Full reproducer at: `confirmed_bugs/iree/iree_bug_reducemax_nan/minimal_repro.py`

## Root cause

The linalg reduction lowering uses an operation that always propagates NaN (likely `arith.maximumf` with IEEE 754 NaN-propagation semantics), regardless of whether the non-NaN value should "win." This over-propagates NaN compared to ORT's behavior.

Interestingly, TVM has the **opposite** bug on the same input: TVM **under-propagates** NaN (`[1.0, NaN]` instead of `[NaN, 2.0]`), using `fmax` semantics where NaN is treated as missing. Both diverge from ORT, but in opposite directions.

## Note

While the ONNX spec is not fully normative on NaN behavior in reductions, ONNX Runtime handles it consistently and this divergence can cause silent numerical differences when switching runtimes.

## Environment

- iree-base-compiler: 3.12.0rc20260515
- iree-base-runtime: 3.12.0rc20260515
- Python: 3.11
- OS: Linux
