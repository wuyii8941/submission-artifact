"""IREE Bug I2: ReduceMax/ReduceMin NaN over-propagation.

IREE over-propagates NaN in reductions:
  ReduceMax([[NaN, 1.0], [2.0, NaN]]) axis=1
  ORT:  [NaN, 2.0]  (NaN propagated for row with leading NaN, value wins for row with trailing NaN)
  IREE: [NaN, NaN]  (NaN infects all rows)

Contrast with TVM which under-propagates:
  TVM:  [1.0, NaN]  (value wins for row with leading NaN, NaN propagated for row with trailing NaN)

Both diverge from ORT, but in opposite directions — position-dependent fold order.

IREE version: 20241107.1071
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import tempfile, subprocess, os

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
    x = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
    print(f"Input:\n{x}\n")

    for op_name in ["ReduceMax", "ReduceMin"]:
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
        node = helper.make_node(op_name, ["X"], ["Y"], axes=[1], keepdims=0)
        graph = helper.make_graph([node], "main", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        ort_out = run_ort(model, [x])
        iree_out = run_iree(model, [x])
        nan_match = np.array_equal(np.isnan(ort_out), np.isnan(iree_out))
        status = "OK" if nan_match else "WRONG"
        print(f"{op_name}([[nan, 1.0], [2.0, nan]], axis=1):")
        print(f"  ORT:  {ort_out}  (NaN: {np.isnan(ort_out)})")
        print(f"  IREE: {iree_out}  (NaN: {np.isnan(iree_out)})")
        print(f"  [{status}]")
        print()
