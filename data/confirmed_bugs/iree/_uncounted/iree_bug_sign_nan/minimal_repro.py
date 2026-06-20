"""IREE Bug I1: Sign(NaN) returns 0 instead of NaN.

Per ORT behavior, Sign(NaN) should return NaN.
IREE returns 0 — the sign comparison chain doesn't check for NaN.

Same class as TVM bug T3 (Sign NaN non-propagation).

IREE version: 20241107.1071
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import tempfile, subprocess, os, sys

IREE_PYTHON = "${ARTIFACT_ROOT}/tvm_fuzz/bin/python3"
ENV = os.environ.copy()
ENV["LD_PRELOAD"] = "${ARTIFACT_ROOT}/tvm_fuzz/lib/libstdc++.so.6"


def run_ort(model, inputs):
    sess = ort.InferenceSession(model.SerializeToString())
    feed = {sess.get_inputs()[i].name: inputs[i] for i in range(len(inputs))}
    return sess.run(None, feed)[0]


def run_iree(model, inputs, backend="llvm-cpu"):
    model = onnx.shape_inference.infer_shapes(model)
    with tempfile.TemporaryDirectory() as d:
        onnx_path = os.path.join(d, "model.onnx")
        mlir_path = os.path.join(d, "model.mlir")
        onnx.save(model, onnx_path)
        r = subprocess.run(
            [IREE_PYTHON, "-m", "iree.compiler.tools.import_onnx",
             onnx_path, "-o", mlir_path],
            capture_output=True, text=True, timeout=60, env=ENV)
        if r.returncode != 0:
            raise RuntimeError(f"import failed: {r.stderr[-300:]}")

        inp_paths = []
        for i, arr in enumerate(inputs):
            p = os.path.join(d, f"in_{i}.npy")
            np.save(p, arr)
            inp_paths.append(p)

        script = f"""
import iree.compiler as ireec
import iree.runtime as ireert
import numpy as np, sys

with open('{mlir_path}') as f:
    mlir = f.read()
compiled = ireec.compile_str(mlir, target_backends=['{backend}'],
    extra_args=['--iree-llvmcpu-target-cpu=host'])
config = ireert.Config('local-task')
ctx = ireert.SystemContext(config=config)
vm = ireert.VmModule.copy_buffer(ctx.instance, compiled)
ctx.add_vm_module(vm)
f = ctx.modules.module.main
inputs = [np.load(p) for p in {inp_paths!r}]
result = f(*inputs)
np.save('{d}/out_0.npy', np.array(result))
print(1)
"""
        r = subprocess.run([IREE_PYTHON, "-c", script],
                          capture_output=True, text=True, timeout=120, env=ENV)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        return np.load(os.path.join(d, "out_0.npy"))


if __name__ == "__main__":
    x = np.array([np.nan, 1.0, np.nan, -2.0, 0.0], dtype=np.float32)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5])
    node = helper.make_node("Sign", ["X"], ["Y"])
    graph = helper.make_graph([node], "main", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_out = run_ort(model, [x])
    print(f"Sign({x}):")
    print(f"  ORT:  {ort_out}")

    iree_out = run_iree(model, [x])
    nan_match = np.array_equal(np.isnan(ort_out), np.isnan(iree_out))
    status = "OK" if nan_match else "WRONG"
    print(f"  IREE: {iree_out}")
    print(f"  [{status}] NaN positions match: {nan_match}")
    print(f"  Expected NaN at positions [0, 2], got: {np.where(np.isnan(iree_out))[0]}")
