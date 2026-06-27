# [Bug][ONNX] SoftmaxCrossEntropyLoss with weight + reduction='mean' computes wrong result

## What happened?

`SoftmaxCrossEntropyLoss` with a non-uniform `weight` tensor and `reduction='mean'` produces incorrect results. The `sum` and `none` reductions are correct even with weights, and `mean` without weights is also correct.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

np.random.seed(42)
x = np.random.randn(4, 5).astype(np.float32)
target = np.array([0, 1, 4, 3], dtype=np.int64)
weight = np.array([1.0, 2.0, 1.0, 0.5, 1.5], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
T = helper.make_tensor_value_info("T", TensorProto.INT64, [4])
W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [5])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("SoftmaxCrossEntropyLoss", ["X", "T", "W"], ["Y"],
                        reduction="mean")
graph = helper.make_graph([node], "test", [X, T, W], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model = onnx.shape_inference.infer_shapes(model)

# ORT
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": x, "T": target, "W": weight})[0]

# IREE
onnx.save(model, "/tmp/sce.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/sce.onnx"], capture_output=True, text=True)
compiled = compile_str(r.stdout, input_type="onnx", target_backends=["llvm-cpu"])
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, compiled))
iree_out = ctx.modules.module.test(x, target, weight).to_host()

print(f"ORT:  {ort_out}")   # [1.6834847]
print(f"IREE: {iree_out}")  # [2.1043556] — WRONG
```

## Pattern

| reduction | weight | Result |
|-----------|--------|--------|
| `mean` | none | OK |
| `mean` | uniform `[1,1,1,1,1]` | OK |
| **`mean`** | **non-uniform `[1,2,1,0.5,1.5]`** | **WRONG (diff=0.42)** |
| `sum` | non-uniform | OK |
| `none` | non-uniform | OK |

## Root cause

The ONNX spec defines the weighted mean as:

```
loss = sum(CE_i * weight[target_i]) / sum(weight[target_i])
```

IREE appears to divide by `N` (number of samples) instead of `sum(weight[target_i])`. This explains why:
- `sum` is correct (no division)
- `none` is correct (no reduction)
- `mean` without weight is correct (`sum(weight[target_i]) == N` when all weights are 1)
- `mean` with non-uniform weight is wrong (denominator differs)

Verification: `ORT=8.417/5.0=1.683` vs `IREE=8.417/4=2.104` → IREE divides by N=4 instead of sum(weights)=5.0.

## Environment

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu
- Python: 3.11
