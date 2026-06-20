# [Bug][Relax] NaN handling inconsistency in Relu, Sign, ReduceMax, ReduceMin

## Summary

Multiple operators handle NaN differently from ONNX Runtime:

1. **Relu(NaN) → 0** (ORT: NaN)
2. **Sign(NaN) → 0** (ORT: NaN)
3. **ReduceMax/ReduceMin** — position-dependent NaN behavior:
   - `ReduceMax([NaN, 1.0]) → 1.0` (ORT: NaN)
   - `ReduceMax([2.0, NaN]) → NaN` (ORT: NaN for row 1, 2.0 for row 2)

## Reproduction

```python
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

def run_tvm(model, inputs):
    model = onnx.shape_inference.infer_shapes(model)
    mod = from_onnx(model)
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    exe = tvm.relax.build(pipeline(mod), target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_ins = [tvm.runtime.tensor(v, device=tvm.cpu()) for v in inputs]
    return vm["main"](*tvm_ins).numpy()

# --- Part 1: Relu and Sign ---
x = np.array([np.nan, 1.0, np.nan, -2.0], dtype=np.float32)
for op_name in ["Relu", "Sign"]:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    node = helper.make_node(op_name, ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    sess = ort.InferenceSession(model.SerializeToString())
    ort_out = sess.run(None, {"X": x})[0]
    tvm_out = run_tvm(model, [x])
    print(f"{op_name}({x}):")
    print(f"  ORT: {ort_out}")   # [nan  1. nan  0.] / [nan  1. nan -1.]
    print(f"  TVM: {tvm_out}")   # [0. 1. 0. 0.]    / [0.  1.  0. -1.]

# --- Part 2: ReduceMax/ReduceMin ---
x2 = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
for op_name in ["ReduceMax", "ReduceMin"]:
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    node = helper.make_node(op_name, ["X", "axes"], ["Y"], keepdims=0)
    graph = helper.make_graph([node], "test", [X], [Y], initializer=[axes_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    sess = ort.InferenceSession(model.SerializeToString())
    ort_out = sess.run(None, {"X": x2})[0]
    tvm_out = run_tvm(model, [x2])
    print(f"{op_name}([[nan,1],[2,nan]], axis=1):")
    print(f"  ORT: {ort_out}")   # [nan  2.]
    print(f"  TVM: {tvm_out}")   # [ 1. nan]
```

## Root cause

- **Relu**: Lowered to `max(x, 0)` which uses `fmax` semantics where NaN is treated as missing value
- **Sign**: Lowered to a comparison chain (`x > 0 → 1, x < 0 → -1, else 0`) which maps NaN to 0 since `NaN > 0` and `NaN < 0` are both false
- **ReduceMax/Min**: Left-fold reduction with `fmax`/`fmin` — NaN propagation depends on whether NaN appears in the accumulator or the new element, leading to position-dependent results

## Note

We acknowledge that the ONNX spec does not normatively require NaN propagation for these operators. However, ONNX Runtime (the reference implementation) propagates NaN consistently, and TVM's behavior differs in ways that can cause silent numerical divergence when migrating models between runtimes. We report this as a behavioral inconsistency for awareness.

## Environment

- TVM: 0.24.dev0, commit 0b0afd8dd (2026-04-24)
- Python: 3.11
- OS: Linux
