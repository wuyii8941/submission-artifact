"""IREE Bug 011: Reshape with -1 (infer dimension) causes runtime OUT_OF_RANGE error.

When ONNX Reshape receives a shape containing -1 (the standard "infer this
dimension" sentinel), IREE compiles successfully but at runtime the -1 is
treated as a literal dimension value, causing a buffer access violation.

Expected: shape=[2, -1] on input [2,3,4] -> output [2, 12]
Actual:   Runtime error: OUT_OF_RANGE; attempted to access an address
          outside of the valid buffer range (adjusted_length=0xFFFFFFFFFFFFFFF8)

The adjusted_length values (~2^64 - small_number) confirm the -1 sentinel
leaks through as an unsigned 64-bit integer.

ORT handles this correctly for all tested shapes.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import subprocess
import tempfile
import os
import shutil

IREE_BIN = "${ARTIFACT_ROOT}/tvm_fuzz/bin"
IREE_IMPORT = f"{IREE_BIN}/iree-import-onnx"
IREE_COMPILE = f"{IREE_BIN}/iree-compile"
IREE_RUN = f"{IREE_BIN}/iree-run-module"


def test_reshape(x_shape, target_shape):
    """Test Reshape with given target shape. Returns (success, detail)."""
    x = np.random.randn(*x_shape).astype(np.float32)
    shape = np.array(target_shape, dtype=np.int64)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape))
    S = helper.make_tensor_value_info("shape", TensorProto.INT64, [len(target_shape)])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
    graph = helper.make_graph([node], "test", [X, S], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model)

    tmpdir = tempfile.mkdtemp(prefix="iree_reshape_")
    try:
        onnx.save(model, f"{tmpdir}/m.onnx")
        np.save(f"{tmpdir}/x.npy", x)
        np.save(f"{tmpdir}/s.npy", shape)

        r = subprocess.run([IREE_IMPORT, f"{tmpdir}/m.onnx"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, f"import failed: {r.stderr[:200]}"

        r2 = subprocess.run(
            [IREE_COMPILE, "--iree-hal-target-backends=llvm-cpu",
             "--iree-llvmcpu-target-cpu=generic", "-o", f"{tmpdir}/m.vmfb", "-"],
            input=r.stdout, capture_output=True, text=True, timeout=60)
        if r2.returncode != 0:
            return False, f"compile failed: {r2.stderr[:200]}"

        r3 = subprocess.run(
            [IREE_RUN, f"--module={tmpdir}/m.vmfb", "--function=test",
             f"--input=@{tmpdir}/x.npy", f"--input=@{tmpdir}/s.npy",
             f"--output=@{tmpdir}/out.npy"],
            capture_output=True, text=True, timeout=30)
        if r3.returncode != 0:
            return False, f"runtime failed: {r3.stderr[:300]}"

        out = np.load(f"{tmpdir}/out.npy")
        return True, f"output shape={out.shape}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    np.random.seed(42)
    print("IREE Reshape -1 bug reproduction")
    print("=" * 60)

    test_cases = [
        # (input_shape, target_shape, description)
        ([2, 3, 4], [6, 4], "static reshape (no -1)"),
        ([2, 3, 4], [24], "flatten (no -1)"),
        ([2, 3, 4], [2, -1], "-1 infer last dim"),
        ([2, 3, 4], [-1, 4], "-1 infer first dim"),
        ([2, 3, 4], [1, -1], "-1 infer with 1"),
        ([2, 3, 4], [0, 3, -1], "0 (keep) + -1 infer"),
        ([3, 4, 5], [3, -1], "-1 on 3D->2D"),
        ([3, 4, 5], [-1, 5], "-1 on 3D->2D first"),
        ([6], [-1, 2], "-1 on 1D->2D"),
        ([6], [2, -1], "-1 on 1D->2D second"),
    ]

    pass_count = 0
    fail_count = 0
    for x_shape, target_shape, desc in test_cases:
        ok, detail = test_reshape(x_shape, target_shape)
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        print(f"  [{status}] {desc}: input={x_shape} shape={target_shape}")
        if not ok:
            print(f"         {detail[:200]}")

    print(f"\n{pass_count} passed, {fail_count} failed")
    if fail_count > 0:
        print("\nConclusion: Any Reshape with -1 in the target shape causes")
        print("a runtime OUT_OF_RANGE buffer access violation.")
