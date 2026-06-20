# Sign operator returns 0 for NaN inputs instead of NaN

## Expected behavior

`Sign(NaN)` should return `NaN`, matching ONNX Runtime behavior.

## Actual behavior

IREE returns `0` for NaN inputs.

```
Input: [nan, 1.0, nan, -2.0, 0.0]
ORT:   [nan, 1.0, nan, -1.0, 0.0]
IREE:  [0.0, 1.0, 0.0, -1.0, 0.0]
```

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import iree.compiler as ic
import iree.runtime as ir
import tempfile, subprocess, os

x = np.array([np.nan, 1.0, np.nan, -2.0, 0.0], dtype=np.float32)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5])
node = helper.make_node("Sign", ["X"], ["Y"])
graph = helper.make_graph([node], "main", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model = onnx.shape_inference.infer_shapes(model)

# ORT reference
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": x})[0]

# IREE: import ONNX → compile → run
with tempfile.TemporaryDirectory() as d:
    onnx.save(model, f"{d}/model.onnx")
    subprocess.run(["python", "-m", "iree.compiler.tools.import_onnx",
                    f"{d}/model.onnx", "-o", f"{d}/model.mlir"], check=True)
    with open(f"{d}/model.mlir") as f:
        mlir = f.read()
    compiled = ic.compile_str(mlir, target_backends=["llvm-cpu"])
    config = ir.Config("local-task")
    ctx = ir.SystemContext(config=config)
    vm = ir.VmModule.copy_buffer(ctx.instance, compiled)
    ctx.add_vm_module(vm)
    iree_out = np.array(ctx.modules.module.main(
        ir.asdevicearray(ctx.modules.module._context.config.device, x)).to_host())

print(f"ORT:  {ort_out}")   # [nan  1. nan -1.  0.]
print(f"IREE: {iree_out}")  # [ 0.  1.  0. -1.  0.]
```

## Root cause

The Sign lowering uses a comparison chain (`x > 0 → 1, x < 0 → -1, else → 0`) which maps NaN to 0 since IEEE 754 comparisons with NaN return false.

## Note

While the ONNX spec is not fully normative on NaN behavior for Sign, ONNX Runtime (the reference implementation) consistently propagates NaN. IREE's behavior can cause silent numerical divergence when switching runtimes.

## Environment

- iree-base-compiler: 3.12.0rc20260515
- iree-base-runtime: 3.12.0rc20260515
- Python: 3.11
- OS: Linux
