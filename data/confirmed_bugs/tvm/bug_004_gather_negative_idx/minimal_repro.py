"""Bug 010: Gather produces wrong results with negative indices.

ONNX spec: negative indices mean indexing from the end (e.g., -1 = last element).
TVM doesn't handle this, producing garbage values.
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
    mod = from_onnx(model)
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    exe = tvm.relax.build(pipeline(mod), target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_inputs = [tvm.runtime.tensor(v, device=tvm.cpu()) for v in inputs.values()]
    return vm["main"](*tvm_inputs).numpy()


if __name__ == "__main__":
    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]], dtype=np.float32)

    # Positive indices (should work)
    pos_idx = np.array([0, 2], dtype=np.int64)
    # Negative indices (same elements: 0 → 0, -1 → 2)
    neg_idx = np.array([0, -1], dtype=np.int64)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    I = helper.make_tensor_value_info("I", TensorProto.INT64, [2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    gather = helper.make_node("Gather", ["X", "I"], ["Y"], axis=0)
    graph = helper.make_graph([gather], "test", [X, I], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    for idx, tag in [(pos_idx, "positive"), (neg_idx, "negative")]:
        inputs = {"X": data, "I": idx}
        ort_out = run_ort(model, inputs)
        tvm_out = run_tvm(model, inputs)
        match = "OK" if np.allclose(ort_out, tvm_out) else "WRONG"
        print(f"{tag} indices {idx}: {match}")
        if match == "WRONG":
            print(f"  ORT: {ort_out}")
            print(f"  TVM: {tvm_out}")
