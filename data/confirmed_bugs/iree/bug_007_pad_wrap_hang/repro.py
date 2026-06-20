#!/usr/bin/env python3
"""Bug #036: Pad mode=wrap causes compiler infinite loop.

iree-compile hangs indefinitely when compiling Pad with mode="wrap".
Other modes (constant, reflect) work correctly.
mode="edge" fails to compile (known Bug #023) but doesn't hang.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import subprocess
import tempfile
import os

IREE_IMPORT = "/data1/tzh/tvm_fuzz/bin/iree-import-onnx"
IREE_COMPILE = "/data1/tzh/tvm_fuzz/bin/iree-compile"

print("=" * 60)
print("Bug #036: Pad wrap mode compiler hang")
print("=" * 60)

for mode in ["constant", "reflect", "edge", "wrap"]:
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    pads_init = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), "pads")
    if mode == "constant":
        cv = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "cv")
        nodes = [helper.make_node("Pad", ["X", "pads", "cv"], ["Y"], mode=mode)]
        graph = helper.make_graph(nodes, "test", [X], [Y], initializer=[pads_init, cv])
    else:
        nodes = [helper.make_node("Pad", ["X", "pads"], ["Y"], mode=mode)]
        graph = helper.make_graph(nodes, "test", [X], [Y], initializer=[pads_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)

    # ORT reference
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(model.SerializeToString(), opts,
                                    providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"X": x})
        ort_ok = f"OK shape={ort_out[0].shape}"
    except:
        ort_ok = "FAIL"

    tmpdir = tempfile.mkdtemp(prefix="iree_036_")
    onnx.save(model, os.path.join(tmpdir, "model.onnx"))
    r = subprocess.run([IREE_IMPORT, os.path.join(tmpdir, "model.onnx"),
                       "-o", os.path.join(tmpdir, "model.mlir")],
                      capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"  mode={mode}: IMPORT FAIL (ORT: {ort_ok})")
        continue

    try:
        r = subprocess.run([IREE_COMPILE, "--iree-hal-target-backends=llvm-cpu",
                           "--iree-llvmcpu-target-cpu=generic",
                           os.path.join(tmpdir, "model.mlir"),
                           "-o", os.path.join(tmpdir, "model.vmfb")],
                          capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            is_crash = "Stack dump" in r.stderr
            tag = "CRASH" if is_crash else "COMPILE_FAIL"
            print(f"  mode={mode}: {tag} (ORT: {ort_ok})")
        else:
            print(f"  mode={mode}: compile OK (ORT: {ort_ok})")
    except subprocess.TimeoutExpired:
        print(f"  mode={mode}: TIMEOUT >30s (ORT: {ort_ok})")
