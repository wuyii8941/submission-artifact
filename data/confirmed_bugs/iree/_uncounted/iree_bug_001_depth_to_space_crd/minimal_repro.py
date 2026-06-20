"""IREE Bug: DepthToSpace mode=CRD produces wrong results for blocksize >= 3

IREE returns incorrect output for DepthToSpace with mode="CRD" when blocksize >= 3.
DCR mode works correctly for all blocksizes, and CRD mode works for blocksize=2.

Oracle: ONNX Runtime
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import subprocess
import sys

IREE_IMPORT = "${ARTIFACT_ROOT}/tvm_fuzz/bin/iree-import-onnx"


def run_ort(model, inputs):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts)
    feed = {inp.name: v for inp, v in zip(sess.get_inputs(), inputs)}
    return sess.run(None, feed)


def run_iree(model, inputs):
    from iree.compiler import compile_str
    from iree import runtime as ireert

    onnx_path = "/tmp/_iree_d2s_bug.onnx"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    r = subprocess.run([IREE_IMPORT, onnx_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"iree-import-onnx failed: {r.stderr[:500]}")
    compiled = compile_str(
        r.stdout,
        input_type="onnx",
        target_backends=["llvm-cpu"],
        extra_args=["--iree-llvmcpu-target-cpu=generic"],
    )
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled)
    ctx.add_vm_module(vm_module)
    f = ctx.modules.module.test
    result = f(*inputs)
    if hasattr(result, "to_host"):
        return result.to_host()
    return [r.to_host() for r in result]


def test_depth_to_space(blocksize, mode):
    C = blocksize * blocksize * 2
    x = np.arange(1 * C * 3 * 3, dtype=np.float32).reshape(1, C, 3, 3)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=blocksize, mode=mode)
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    ort_out = run_ort(model, [x])[0]
    try:
        iree_out = run_iree(model, [x])
    except Exception as e:
        print(f"  bs={blocksize} mode={mode}: IREE ERROR: {e}")
        return

    if ort_out.shape != iree_out.shape:
        print(f"  bs={blocksize} mode={mode}: SHAPE MISMATCH ort={ort_out.shape} iree={iree_out.shape}")
        return

    diff = float(np.max(np.abs(ort_out.astype(np.float64) - iree_out.astype(np.float64))))
    match = "OK" if diff < 1e-5 else f"WRONG (max_diff={diff:.2f})"
    print(f"  bs={blocksize} mode={mode}: {match}")
    if diff >= 1e-5:
        print(f"    ORT output (first 8 values):  {ort_out.flatten()[:8]}")
        print(f"    IREE output (first 8 values): {iree_out.flatten()[:8]}")


if __name__ == "__main__":
    print("IREE DepthToSpace CRD mode bug reproduction")
    print("=" * 50)
    print()
    print("DCR mode (expect all OK):")
    for bs in [2, 3, 4, 5]:
        test_depth_to_space(bs, "DCR")

    print()
    print("CRD mode (expect WRONG for bs >= 3):")
    for bs in [2, 3, 4, 5]:
        test_depth_to_space(bs, "CRD")
