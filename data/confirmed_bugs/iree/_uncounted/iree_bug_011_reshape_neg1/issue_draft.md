## [Bug] ONNX Reshape with dynamic shape containing -1 compiles successfully but crashes at runtime with OUT_OF_RANGE

### What happened?

When an ONNX `Reshape` op receives its target shape as a **runtime input** (graph input, not initializer) containing `-1` (the ONNX standard sentinel for "infer this dimension"), the model **compiles without error** but **crashes at runtime** with an `OUT_OF_RANGE` buffer access violation.

The compiled MLIR correctly shows a dynamic output type (`!torch.vtensor<[?,?],f32>`), so the compiler accepts the dynamic shape. However, at runtime the `-1` value is not resolved and leaks through as a literal unsigned 64-bit dimension, producing an `adjusted_length` near `2^64`.

Note: when the target shape is a **constant initializer** (not a graph input), Reshape works correctly — the compiler resolves `-1` at compile time.

### Steps to reproduce

```bash
# 1. Create a minimal ONNX model (shape is a graph input, not initializer)
python3 -c "
import numpy as np, onnx
from onnx import helper, TensorProto

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 4])
S = helper.make_tensor_value_info('shape', TensorProto.INT64, [2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)
node = helper.make_node('Reshape', ['X', 'shape'], ['Y'])
graph = helper.make_graph([node], 'test', [X, S], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 9
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, '/tmp/reshape_neg1.onnx')
np.save('/tmp/reshape_x.npy', np.random.randn(2,3,4).astype(np.float32))
np.save('/tmp/reshape_s.npy', np.array([2, -1], dtype=np.int64))
"

# 2. Import + compile (succeeds without error)
iree-import-onnx /tmp/reshape_neg1.onnx | \
  iree-compile --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=generic -o /tmp/reshape.vmfb -

# 3. Run (crashes)
iree-run-module --module=/tmp/reshape.vmfb --function=test \
  --input=@/tmp/reshape_x.npy --input=@/tmp/reshape_s.npy --output=-
```

**Runtime error:**
```
OUT_OF_RANGE; attempted to access an address outside of the valid buffer range
(offset=0, adjusted_length=18446744073709551608, buffer byte_length=96)
```

The `adjusted_length` of `0xFFFFFFFFFFFFFFF8` is `-8` as unsigned, confirming the `-1` sentinel leaked through unresolved.

### Expected behavior

Either:
- **Preferred**: resolve `-1` at runtime so `shape=[2, -1]` on input `[2, 3, 4]` produces output `[2, 12]` (matching ONNX spec and ONNX Runtime), or
- **Acceptable**: reject dynamic-input Reshape with `-1` at **compile time** with a clear error, rather than producing a binary that crashes.

### What works vs. what fails

| Shape source | Target shape | Result |
|---|---|---|
| Graph input | `[6, 4]` (no -1) | OK |
| Graph input | `[24]` (no -1) | OK |
| Graph input | `[2, -1]` | **Runtime crash** |
| Graph input | `[-1, 4]` | **Runtime crash** |
| Graph input | `[1, -1]` | **Runtime crash** |
| Graph input | `[0, 3, -1]` | **Runtime crash** |
| Initializer | `[2, -1]` | OK (resolved at compile time) |

### Environment

- IREE compiler: `3.11.0rc20260316` @ `e4a3b0405d7d`
- Target: `llvm-cpu` (generic)
- ONNX: 1.17.0
- OS: Ubuntu 20.04
