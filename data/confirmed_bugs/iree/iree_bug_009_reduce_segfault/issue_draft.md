# [Bug][ONNX] ReduceMax/ReduceMin with multi-axis + negative index causes SEGFAULT

## What happened?

Compiling a valid ONNX model with `ReduceMax` or `ReduceMin` using **multiple axes where at least one is negative** causes the IREE compiler to crash with exit code 245 (SEGFAULT). The IREE error handler prints "Please report issues to <upstream tracker>".

Compilers should never segfault on valid input.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import subprocess

x = np.random.randn(1, 8, 1).astype(np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 1])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [])
axes_init = numpy_helper.from_array(np.array([0, -1], dtype=np.int64), "axes")
node = helper.make_node("ReduceMax", ["X", "axes"], ["Y"], keepdims=0)
graph = helper.make_graph([node], "test", [X], [Y], initializer=[axes_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model = onnx.shape_inference.infer_shapes(model)

onnx.save(model, "/tmp/reduce_segfault.onnx")

# Import succeeds
r1 = subprocess.run(["iree-import-onnx", "/tmp/reduce_segfault.onnx"],
                     capture_output=True, text=True, timeout=120)
assert r1.returncode == 0, f"Import failed: {r1.stderr}"
print("Import: OK")

# Save MLIR and compile — SEGFAULT
with open("/tmp/reduce_segfault.mlir", "w") as f:
    f.write(r1.stdout)

r2 = subprocess.run(["iree-compile", "/tmp/reduce_segfault.mlir",
                      "--iree-hal-target-backends=llvm-cpu",
                      "--iree-llvmcpu-target-cpu=host",
                      "--iree-input-type=auto",
                      "-o", "/tmp/reduce_segfault.vmfb"],
                     capture_output=True, text=True, timeout=120)
print(f"Compile exit code: {r2.returncode}")
# Exit code: -11 (SIGSEGV)
```

## Configurations tested

| Op | axes | keepdims | Result |
|---|---|---|---|
| ReduceMax | [0, -1] | 0 | **CRASH (exit 245)** |
| ReduceMin | [0, -1] | 0 | **CRASH (exit 245)** |
| ReduceMax | [0, 2] | 0 | OK |
| ReduceMin | [0, 2] | 0 | OK |
| ReduceMax | [0, -1] | 1 | OK |
| ReduceMin | [0, -1] | 1 | OK |
| ReduceMax | [-1] (single) | 0 | OK |

The crash requires: multi-axis + at least one negative axis + keepdims=0.

## Notes

- All parameters are valid per ONNX spec (opset 18). Negative axes are explicitly permitted.
- This is distinct from the ReduceProd crash (iree_issue_draft_002), which is a `tensor.reshape` rank mismatch error, not a segfault.
- The crash occurs in the torch-mlir/linalg lowering pass, not during import.
- ONNX Runtime handles all these cases correctly.

## Environment

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu and cuda (both crash)
- Python: 3.11
