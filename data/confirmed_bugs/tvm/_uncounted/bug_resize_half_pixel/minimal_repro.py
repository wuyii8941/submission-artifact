"""TVM Bug: Resize nearest+half_pixel produces wrong results for non-integer scales.

When using Resize with mode='nearest', coordinate_transformation_mode='half_pixel',
nearest_mode='floor', and a non-integer scale factor (e.g., 2.5x), TVM computes
incorrect source coordinates. The bug manifests when the input spatial dimension is
odd (e.g., 3x3) — the output pixel mapping differs from ONNX Runtime.

Root cause: The half_pixel coordinate transformation formula
  src_coord = (dst_coord + 0.5) / scale - 0.5
followed by floor() is implemented incorrectly in TVM's Resize lowering,
leading to off-by-one source pixel selection.

Also affects: linear interpolation with non-integer scales (asymmetric, pytorch_half_pixel).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort


def run_ort(model, x):
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {'X': x})[0]


def run_tvm(model, x):
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    model_inf = onnx.shape_inference.infer_shapes(model)
    mod = from_onnx(model_inf)
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    exe = tvm.relax.build(pipeline(mod), target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    return vm["main"](tvm.runtime.tensor(x, device=tvm.cpu())).numpy()


def make_resize_model(mode, coord_mode, nearest_mode="floor", scales=[1,1,2.5,2.5]):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), "roi")
    sc = numpy_helper.from_array(np.array(scales, dtype=np.float32), "scales")
    attrs = {"mode": mode, "coordinate_transformation_mode": coord_mode}
    if mode == "nearest":
        attrs["nearest_mode"] = nearest_mode
    node = helper.make_node("Resize", ["X", "roi", "scales"], ["Y"], **attrs)
    graph = helper.make_graph([node], "test", [X], [Y], initializer=[roi, sc])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


if __name__ == "__main__":
    x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    print("Input:")
    print(x.reshape(3, 3))
    print()

    configs = [
        ("nearest", "half_pixel"),
        ("nearest", "pytorch_half_pixel"),
        ("linear", "half_pixel"),
        ("linear", "asymmetric"),
        ("linear", "pytorch_half_pixel"),
    ]

    for mode, coord_mode in configs:
        model = make_resize_model(mode, coord_mode)
        ort_out = run_ort(model, x)
        tvm_out = run_tvm(model, x)
        diff = np.abs(ort_out - tvm_out).max()
        status = "OK" if diff < 1e-5 else "WRONG"
        print(f"[{status}] mode={mode}, coord={coord_mode}: max_diff={diff:.6f}")
        if status == "WRONG":
            print(f"  ORT:\n{ort_out.reshape(ort_out.shape[2], ort_out.shape[3])}")
            print(f"  TVM:\n{tvm_out.reshape(tvm_out.shape[2], tvm_out.shape[3])}")
        print()
