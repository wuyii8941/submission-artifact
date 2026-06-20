"""Bug 029: TopK operator crashes during ONNX frontend conversion.

Status: CONFIRMED BUG (NEW)
TVM version: 0.24.dev0, commit 0b0afd8dd (2026-04-24)

Description:
  from_onnx() crashes with "only 0-dimensional arrays can be converted to
  Python scalars" for ANY model containing a TopK operator.

  The K input is passed as a 1-element tensor constant (shape [1]), but the
  converter tries to extract it as a Python scalar without .item() or [0].

  This makes the entire TopK operator unusable through the ONNX frontend.
  100% reproducible on every shape/axis combination tested.

Trigger: any TopK op with constant K input (standard ONNX usage).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import tvm
from tvm.relax.frontend.onnx import from_onnx


def make_topk_model(shape, k, axis):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
    K_init = numpy_helper.from_array(np.array([k], dtype=np.int64), "K")
    Y_val = helper.make_tensor_value_info("Values", TensorProto.FLOAT, None)
    Y_idx = helper.make_tensor_value_info("Indices", TensorProto.INT64, None)
    node = helper.make_node("TopK", ["X", "K"], ["Values", "Indices"], axis=axis)
    graph = helper.make_graph([node], "test", [X], [Y_val, Y_idx],
                              initializer=[K_init])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


if __name__ == "__main__":
    configs = [
        ((5, 10), 3, 1),
        ((5, 10), 3, 0),
        ((3, 4, 5), 2, 2),
        ((100,), 5, 0),
    ]

    for shape, k, axis in configs:
        model = make_topk_model(shape, k, axis)
        model_inf = onnx.shape_inference.infer_shapes(model)
        try:
            mod = from_onnx(model_inf)
            print(f"[OK] TopK shape={shape} k={k} axis={axis}")
        except Exception as e:
            print(f"[CRASH] TopK shape={shape} k={k} axis={axis}: {e}")
