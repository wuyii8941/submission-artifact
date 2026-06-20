# [Bug][ONNX] Unique op causes compiler SEGFAULT (signal 11)

## What happened?

Compiling a valid ONNX model containing the `Unique` op causes the IREE compiler to crash with a segmentation fault (signal 11). The ONNX-to-MLIR import step succeeds; the crash occurs during `iree-compile`.

Compilers should never segfault on valid input.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
from iree.compiler import compile_str
import subprocess

x = np.array([3, 1, 2, 1, 3, 2, 4], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [7])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("Unique", ["X"], ["Y"], sorted=1)
graph = helper.make_graph([node], "test", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model = onnx.shape_inference.infer_shapes(model)

# Step 1: Import succeeds
onnx.save(model, "/tmp/unique.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/unique.onnx"],
                    capture_output=True, text=True)
assert r.returncode == 0, "Import failed"
print("Import: OK")
print(f"MLIR output:\n{r.stdout}")

# Step 2: Compile crashes with SEGFAULT
try:
    compiled = compile_str(r.stdout, input_type="onnx",
                           target_backends=["llvm-cpu"])
    print(f"Compile: OK ({len(compiled)} bytes)")
except Exception as e:
    if "signal 11" in str(e).lower() or "Error code: -11" in str(e):
        print("Compile: SEGFAULT (signal 11)")
    else:
        print(f"Compile: FAIL — {str(e)[:300]}")
```

**Result:**
```
Import: OK
Compile: SEGFAULT (signal 11)
```

## Notes

- The `Unique` op is a valid ONNX op (opset 11+) with well-defined semantics.
- The crash is unconditional — it occurs regardless of input data or tensor size.
- Only the first output (`Y`, the unique values) is requested; even requesting all 4 outputs still crashes.

## Environment

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu
- Python: 3.11
