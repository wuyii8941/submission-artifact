"""IREE Bug: SoftmaxCrossEntropyLoss with weight + reduction='mean' produces wrong results

IREE computes the wrong weighted mean for SCELoss. The 'sum' and 'none' reductions
are correct; only 'mean' with non-uniform weights is wrong.

The ONNX spec defines weighted mean as:
  loss = sum(CE_i * weight[target_i]) / sum(weight[target_i])
IREE appears to compute:
  loss = sum(CE_i * weight[target_i]) / N

Oracle: ONNX Runtime
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import subprocess
from iree.compiler import compile_str
from iree import runtime as ireert

IREE_IMPORT = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-import-onnx"


def run(model, inputs):
    model = onnx.shape_inference.infer_shapes(model)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts)
    ort_out = sess.run(None, {i.name: v for i, v in zip(sess.get_inputs(), inputs)})[0]

    onnx.save(model, "/tmp/_sce_bug.onnx")
    r = subprocess.run([IREE_IMPORT, "/tmp/_sce_bug.onnx"], capture_output=True, text=True)
    c = compile_str(r.stdout, input_type="onnx", target_backends=["llvm-cpu"],
                    extra_args=["--iree-llvmcpu-target-cpu=generic"])
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm = ireert.VmModule.copy_buffer(ctx.instance, c)
    ctx.add_vm_module(vm)
    iree_out = ctx.modules.module.test(*inputs).to_host()
    return ort_out, iree_out


if __name__ == "__main__":
    print("IREE SCELoss + weight + mean bug reproduction")
    print("=" * 50)

    np.random.seed(42)
    x = np.random.randn(4, 5).astype(np.float32)
    target = np.array([0, 1, 4, 3], dtype=np.int64)
    weight = np.array([1.0, 2.0, 1.0, 0.5, 1.5], dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
    T = helper.make_tensor_value_info("T", TensorProto.INT64, [4])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    for red in ["mean", "sum", "none"]:
        for use_w, w_label in [(False, "no_weight"), (True, "weight")]:
            ins = ["X", "T"] + (["W"] if use_w else [])
            inputs_vi = [X, T] + ([W] if use_w else [])
            inputs_np = [x, target] + ([weight] if use_w else [])
            n = helper.make_node("SoftmaxCrossEntropyLoss", ins, ["Y"], reduction=red)
            g = helper.make_graph([n], "test", inputs_vi, [Y])
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
            try:
                ort_out, iree_out = run(m, inputs_np)
                diff = float(np.max(np.abs(ort_out.astype(np.float64) - iree_out.astype(np.float64))))
                status = "OK" if diff < 1e-4 else f"WRONG (diff={diff:.4f})"
                print(f"  {red:5s} {w_label:12s}: ORT={ort_out.flatten()[:3]}  IREE={iree_out.flatten()[:3]}  {status}")
            except Exception as e:
                print(f"  {red:5s} {w_label:12s}: ERROR — {str(e)[:100]}")
