# [Bug][ONNX] Resize with nearest + half_pixel + floor produces wrong results

## What happened?

`Resize` with `mode="nearest"`, `coordinate_transformation_mode="half_pixel"`, and `nearest_mode="floor"` produces **incorrect output** on the `llvm-cpu` backend. Some output positions read from wrong input indices, and others read uninitialized memory (garbage values like 5.9e+14).

On CUDA, the same model causes the compiler to hang indefinitely.

Other combinations of these attributes work correctly:
- `nearest_mode="round_prefer_floor"` / `"round_prefer_ceil"` / `"ceil"` → OK
- `coordinate_transformation_mode="asymmetric"` → OK

The bug is specifically the interaction between `half_pixel` + `floor`.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

np.random.seed(42)
x = np.random.randn(1, 1, 4, 4).astype(np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8, 8])
roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
scales = numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), name="scales")
node = helper.make_node("Resize", ["X", "roi", "scales"], ["Y"],
                        mode="nearest",
                        coordinate_transformation_mode="half_pixel",
                        nearest_mode="floor")
graph = helper.make_graph([node], "test", [X], [Y], initializer=[roi, scales])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
model = onnx.shape_inference.infer_shapes(model)

# ORT: correct
sess = ort.InferenceSession(model.SerializeToString(),
                            providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"X": x})[0]

# IREE
onnx.save(model, "/tmp/resize_test.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/resize_test.onnx"],
                    capture_output=True, text=True, timeout=120)
compiled = compile_str(r.stdout, input_type="auto",
                       target_backends=["llvm-cpu"],
                       extra_args=["--iree-llvmcpu-target-cpu=host"])
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
ctx.add_vm_module(vm_module)
f = ctx.modules.module.test
iree_out = np.array(f(x))

diff = np.abs(ort_out - iree_out)
print(f"Max diff: {diff.max():.6e}")     # e.g., 3.35e+02 (reads garbage)
print(f"Diffs: {(diff > 1e-5).sum()}/{ort_out.size}")  # 15/64
```

**Result:** max_diff up to 1.70e+19 (uninitialized memory reads), 15-141 elements differ depending on input shape and scale.

## Affected configurations

| Input shape | Scales | Max diff | Elements wrong |
|---|---|---|---|
| 1×1×4×4 | 2.0×2.0 | 3.35e+02 | 15/64 |
| 1×1×4×4 | 1.5×1.5 | 2.15e+00 | 11/36 |
| 1×1×4×4 | 3.0×3.0 | 3.27e+00 | 23/144 |
| 1×1×7×7 | 2.0×2.0 | 1.70e+19 | 27/196 |
| 1×3×8×8 | 2.0×2.0 | 3.24e+00 | 93/768 |

All parameter values are valid per ONNX spec (opset 11+).

## Environment

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu (CUDA hangs)
- Python: 3.11
