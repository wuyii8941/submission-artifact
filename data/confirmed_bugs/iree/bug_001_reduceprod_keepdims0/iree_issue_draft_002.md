# [Bug][ONNX] ReduceProd with keepdims=0 crashes: tensor.reshape rank mismatch

> **Status: CONFIRMED BY DEVELOPER — fixed upstream in torch-mlir (PR <anon>)**
> GitHub issue (IREE): <upstream issue, anonymized>
> GitHub issue (LLVM): <upstream issue, anonymized>

## What happened?

`ReduceProd` with `keepdims=0` fails during compilation with `'tensor.reshape' op length of shape operand differs from the result's tensor rank` when reducing along a subset of axes (output rank > 0).

`keepdims=1` works for all configurations. Other reduction ops (`ReduceMax`, `ReduceMin`, `ReduceMean`, `ReduceL1`, `ReduceL2`) all work correctly with `keepdims=0`.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
from iree.compiler import compile_str
import subprocess

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("ReduceProd", ["X"], ["Y"], axes=[1], keepdims=0)
graph = helper.make_graph([node], "test", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model = onnx.shape_inference.infer_shapes(model)

onnx.save(model, "/tmp/rp.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/rp.onnx"], capture_output=True, text=True)
compiled = compile_str(r.stdout, input_type="onnx", target_backends=["llvm-cpu"])
# Error: 'tensor.reshape' op length of shape operand differs from the result's tensor rank
```

## Pattern

| Config | Result |
|--------|--------|
| `ReduceProd axes=[0] keepdims=0` (shape [3,4]) | **CRASH** |
| `ReduceProd axes=[1] keepdims=0` (shape [3,4]) | **CRASH** |
| `ReduceProd axes=[1] keepdims=0` (shape [3,4,5]) | **CRASH** |
| `ReduceProd axes=[0,2] keepdims=0` (shape [3,4,5]) | **CRASH** |
| `ReduceProd axes=[0,1] keepdims=0` (scalar output) | OK |
| `ReduceProd axes=[0] keepdims=1` | OK |
| `ReduceProd axes=[1] keepdims=1` | OK |
| `ReduceMax axes=[1] keepdims=0` | OK |
| `ReduceMin axes=[1] keepdims=0` | OK |
| `ReduceMean axes=[1] keepdims=0` | OK |
| `ReduceL1 axes=[1] keepdims=0` | OK |
| `ReduceL2 axes=[1] keepdims=0` | OK |

The crash occurs only when: (1) the op is `ReduceProd`, (2) `keepdims=0`, and (3) the result is not a scalar.

## Analysis

The `onnx.ReduceProd` lowering appears to have a different code path from other reduction ops. When `keepdims=0`, the result tensor has fewer dimensions than the input, and the `tensor.reshape` generated in the ReduceProd lowering uses the wrong rank for its shape operand. Other reduce ops handle this rank reduction correctly, so this is likely a ReduceProd-specific issue in the torch-to-linalg lowering.

## Environment

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu
- Python: 3.11
- OS: Linux
