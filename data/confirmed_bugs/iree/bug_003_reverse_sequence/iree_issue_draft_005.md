# [Bug][ONNX] ReverseSequence produces wrong results when batch_axis=1, time_axis=0

> **Status: CONFIRMED BY DEVELOPER**
> GitHub issue: <upstream issue, anonymized>

## What happened?

`ReverseSequence` with `batch_axis=1, time_axis=0` produces incorrect output. The default configuration (`batch_axis=0, time_axis=1`) works correctly.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

x = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9],
              [0, 0, 0]], dtype=np.float32)
lens = np.array([3, 2, 1], dtype=np.int64)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])
L = helper.make_tensor_value_info("L", TensorProto.INT64, [3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("ReverseSequence", ["X", "L"], ["Y"],
                        batch_axis=1, time_axis=0)
graph = helper.make_graph([node], "test", [X, L], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model = onnx.shape_inference.infer_shapes(model)

# ORT (correct)
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": x, "L": lens})[0]

# IREE
onnx.save(model, "/tmp/rs.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/rs.onnx"], capture_output=True, text=True)
compiled = compile_str(r.stdout, input_type="onnx", target_backends=["llvm-cpu"])
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, compiled))
iree_out = ctx.modules.module.test(x, lens).to_host()

print(f"ORT:\n{ort_out}")
print(f"IREE:\n{iree_out}")
```

**Expected (ORT):**
```
[[3. 5. 7.]
 [2. 4. 8.]
 [1. 6. 9.]
 [0. 0. 0.]]
```

**Actual (IREE):**
```
[[3. 4. 7.]
 [2. 5. 8.]
 [1. 6. 9.]
 [0. 0. 0.]]
```

## Pattern

| batch_axis | time_axis | Result |
|------------|-----------|--------|
| 0 | 1 | OK |
| **1** | **0** | **WRONG** |

The default axis configuration works correctly. Only non-default axes trigger the bug.

## Root cause hypothesis

The lowering of `ReverseSequence` likely hardcodes or swaps the batch/time axis roles, so that when `batch_axis=1, time_axis=0` is specified, the reversal is applied along the wrong dimension or with incorrect indexing.

## Environment

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu
- Python: 3.11
