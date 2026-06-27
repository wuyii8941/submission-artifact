# [Bug][ONNX] float64 demotion produces corrupted output due to buffer size mismatch

> **Status: CONFIRMED BY DEVELOPER / labeled `bug`**
> GitHub issue: <upstream issue, anonymized for review>
> Developer/root-cause note: <upstream issue, anonymized for review>

## What happened?

Simple ONNX operations on `float64` tensors produce corrupted output in IREE. For example, `Abs([1.0, 2.0, 3.0, 4.0])` returns `[1.0, 2.0, 0.0, 0.0]`.

The maintainer-facing root-cause analysis identifies this as a compiler conversion issue: the `f64` ABI is preserved while the function body is demoted to `f32`, creating an invalid storage-size mismatch across `hal.tensor.import/export`.

## Minimal reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

X = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, None)
node = helper.make_node("Abs", ["X"], ["Y"])
graph = helper.make_graph([node], "test", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model = onnx.shape_inference.infer_shapes(model)

onnx.save(model, "/tmp/f64.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/f64.onnx"], capture_output=True, text=True)
compiled = compile_str(
    r.stdout,
    input_type="onnx",
    target_backends=["llvm-cpu"],
    extra_args=["--iree-vm-target-extension-f64"],
)

config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, compiled))
out = ctx.modules.module.test(x).to_host()

print("Input:   ", x)
print("Expected:", x)
print("Got:     ", out)
```

## Observed behavior

```text
Input:    [1. 2. 3. 4.]
Expected: [1. 2. 3. 4.]
Got:      [1. 2. 0. 0.]
```

## Root cause

`DemoteF64ToF32Pass` rewrites the computation body to `f32` but bridges the public ABI using typed `as` bitcasts on `hal.tensor.import` and `hal.tensor.export`. This makes the ABI view `tensor<4xf64>` while the converted storage path is `tensor<4xf32>`.

The resulting byte sizes differ: 32 bytes are expected for four `f64` elements, but the demoted storage path uses 16 bytes. Downstream HAL lowering trusts the bitcast contract, so the undersized buffer survives until runtime and causes silent data corruption.

## Why this is a strong paper case

- Uses normal finite `float64` values.
- Minimal operator: unary `Abs`.
- Silent wrong output, not just a rejected program.
- Root cause is localized to a concrete compiler pass and ABI/storage contract.
- Developer analysis confirms this is compiler-side, not a runtime-only issue.

## Environment

- iree-base-compiler: 3.11.0
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu
- Python: 3.11
- OS: Linux
