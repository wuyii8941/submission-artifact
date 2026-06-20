# [Bug][ONNX] DepthToSpace mode="CRD" produces wrong results for blocksize >= 3

## What happened?

`DepthToSpace` with `mode="CRD"` produces **incorrect output** when `blocksize >= 3`. The `DCR` mode works correctly for all blocksizes, and `CRD` mode works for `blocksize=2`.

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

blocksize = 3
C = blocksize * blocksize * 2  # 18
x = np.arange(1 * C * 3 * 3, dtype=np.float32).reshape(1, C, 3, 3)

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, C, 3, 3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
node = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=blocksize, mode="CRD")
graph = helper.make_graph([node], "test", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model = onnx.shape_inference.infer_shapes(model)

# ORT reference
sess = ort.InferenceSession(model.SerializeToString())
ort_out = sess.run(None, {"X": x})[0]

# IREE
onnx.save(model, "/tmp/d2s.onnx")
r = subprocess.run(["iree-import-onnx", "/tmp/d2s.onnx"], capture_output=True, text=True)
compiled = compile_str(r.stdout, input_type="onnx", target_backends=["llvm-cpu"])
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
ctx.add_vm_module(vm_module)
iree_out = ctx.modules.module.test(x).to_host()

print(f"ORT  (first 8): {ort_out.flatten()[:8]}")
# [ 0.  9. 18.  1. 10. 19.  2. 11.]
print(f"IREE (first 8): {iree_out.flatten()[:8]}")
# [ 0.  9.  1. 10.  2. 11. 18. 27.]
print(f"Max diff: {np.max(np.abs(ort_out - iree_out))}")
# 68.0
```

## Test matrix

| blocksize | DCR mode | CRD mode |
|-----------|----------|----------|
| 2         | OK       | OK       |
| 3         | OK       | **WRONG** (max_diff=68) |
| 4         | OK       | **WRONG** (max_diff=86) |
| 5         | OK       | **WRONG** (max_diff=208) |

## Analysis

For CRD mode, the ONNX spec defines the reshape/transpose as:
- Reshape to `(B, C', blocksize, blocksize, H, W)`
- Transpose to `(B, C', H, blocksize, W, blocksize)`
- Reshape to `(B, C', H*blocksize, W*blocksize)`

The CRD mode lowering appears to use an incorrect permutation when `blocksize >= 3`. The fact that `blocksize=2` works but `blocksize=3+` fails suggests a dimension ordering bug in the transpose indices, possibly an off-by-one in the permutation generation that only manifests when the blocksize dimension is > 2.

## Environment

- iree-base-compiler: 3.11.0 (IREE compiler version 3.11.0rc20260316)
- iree-base-runtime: 3.11.0
- Backend: llvm-cpu
- Python: 3.11
- OS: Linux
