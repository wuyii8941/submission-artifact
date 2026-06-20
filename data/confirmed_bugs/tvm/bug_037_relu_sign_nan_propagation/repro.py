"""Bug 030: ONNX frontend Relu/Sign/ReduceMax/ReduceMin NaN handling diverges from ORT.

TVM version: 0.24.dev0, <commit, anonymized> (2026-04-24)

Related: bug_019 (reduce_max/min NaN CPU vs CUDA, Relax IR level).
This bug covers the same NaN class from the ONNX frontend perspective,
comparing against ORT as the reference implementation.

Description:
  Part 1 — Relu and Sign:
    Relu(NaN) returns 0 instead of NaN (ORT returns NaN)
    Sign(NaN) returns 0 instead of NaN (ORT returns NaN)

  Root cause: Relu lowered to max(x, 0) using fmax semantics (NaN = missing);
  Sign uses comparison chain where NaN falls through to default 0.

  Part 2 — ReduceMax/ReduceMin:
    Position-dependent NaN behavior (same class as bug_019):
    ReduceMax([NaN, 1.0]) = 1.0  (ORT: NaN)  — NaN ignored when first
    ReduceMax([2.0, NaN]) = NaN  (ORT: 2.0)  — NaN propagated when second
    This is the fold-order-dependent fmax/fmin behavior.

Note: ONNX spec is ambiguous on NaN for these ops, but ORT handles it
consistently. Filed as behavioral inconsistency report.

Trigger: NaN values in input tensors. 100% reproducible.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


def run_ort(model, inputs):
    sess = ort.InferenceSession(model.SerializeToString())
    feed = {}
    for i, inp_meta in enumerate(sess.get_inputs()):
        feed[inp_meta.name] = inputs[i]
    return sess.run(None, feed)[0]


def run_tvm(model, inputs):
    model_inf = onnx.shape_inference.infer_shapes(model)
    mod = from_onnx(model_inf)
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    exe = tvm.relax.build(pipeline(mod), target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_ins = [tvm.runtime.tensor(v, device=tvm.cpu()) for v in inputs]
    return vm["main"](*tvm_ins).numpy()


if __name__ == "__main__":
    print("=== Part 1: Relu and Sign NaN propagation ===")
    x = np.array([np.nan, 1.0, np.nan, -2.0], dtype=np.float32)
    for op_name in ["Relu", "Sign"]:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
        node = helper.make_node(op_name, ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        ort_out = run_ort(model, [x])
        tvm_out = run_tvm(model, [x])

        nan_match = np.array_equal(np.isnan(ort_out), np.isnan(tvm_out))
        status = "OK" if nan_match else "WRONG"
        print(f"[{status}] {op_name}({x}):")
        print(f"  ORT: {ort_out}")
        print(f"  TVM: {tvm_out}")
        print()

    print("=== Part 2: ReduceMax/ReduceMin NaN position-dependent ===")
    x2 = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
    for op_name in ["ReduceMax", "ReduceMin"]:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
        node = helper.make_node(op_name, ["X", "axes"], ["Y"], keepdims=0)
        graph = helper.make_graph([node], "test", [X], [Y], initializer=[axes_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        ort_out = run_ort(model, [x2])
        tvm_out = run_tvm(model, [x2])

        nan_match = np.array_equal(np.isnan(ort_out), np.isnan(tvm_out))
        status = "OK" if nan_match else "WRONG"
        print(f"[{status}] {op_name}([[nan, 1.0], [2.0, nan]], axis=1):")
        print(f"  ORT: {ort_out}  (NaN: {np.isnan(ort_out)})")
        print(f"  TVM: {tvm_out}  (NaN: {np.isnan(tvm_out)})")
        print()
