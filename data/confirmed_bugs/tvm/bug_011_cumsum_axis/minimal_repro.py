"""Bug 011: CumSum produces wrong results for axis != 0.

CumSum works correctly for axis=0 but gives wrong values for other axes.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def run_ort(model, inputs):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, inputs)[0]


def run_tvm(model, inputs):
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    model_inf = onnx.shape_inference.infer_shapes(model)
    mod = from_onnx(model_inf)
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    exe = tvm.relax.build(pipeline(mod), target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_inputs = [tvm.runtime.tensor(v, device=tvm.cpu()) for v in inputs.values()]
    return vm["main"](*tvm_inputs).numpy()


if __name__ == "__main__":
    x = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]], dtype=np.float32)

    for axis_val in [0, 1]:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        A = helper.make_tensor_value_info("axis", TensorProto.INT32, [])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        cs = helper.make_node("CumSum", ["X", "axis"], ["Y"])
        graph = helper.make_graph([cs], "test", [X, A], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

        inputs = {"X": x, "axis": np.array(axis_val, dtype=np.int32)}
        ort_out = run_ort(model, inputs)
        tvm_out = run_tvm(model, inputs)
        diff = np.abs(tvm_out - ort_out).max()
        status = "OK" if diff < 1e-5 else "WRONG"
        print(f"axis={axis_val}: max_diff={diff:.6f}  {status}")
        if status == "WRONG":
            print(f"  Expected (ORT):\n{ort_out}")
            print(f"  Got (TVM):\n{tvm_out}")
