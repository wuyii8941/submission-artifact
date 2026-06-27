"""Bug 032: ReorderPermuteDimsAfterConcat crashes on negative concat axis.

The ReorderPermuteDimsAfterConcat pass crashes with "Index -1 out of bounds 4"
when the concat op uses a negative axis (axis=-1). The pass reads the axis value
directly without normalizing negative indices.

Minimal reproduction: two 4D tensors → permute_dims → concat(axis=-1).
"""

import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R
import traceback


def make_mod():
    bb = relax.BlockBuilder()
    x = relax.Var('x', relax.TensorStructInfo((1, 4, 8, 8), 'float32'))
    y = relax.Var('y', relax.TensorStructInfo((1, 4, 8, 8), 'float32'))
    with bb.function('main', [x, y]):
        with bb.dataflow():
            xt = bb.emit(R.permute_dims(x, [0, 2, 3, 1]))  # NCHW -> NHWC
            yt = bb.emit(R.permute_dims(y, [0, 2, 3, 1]))  # NCHW -> NHWC
            out = bb.emit_output(R.concat([xt, yt], axis=-1))  # concat on C dim
        bb.emit_func_output(out)
    return bb.finalize()


if __name__ == "__main__":
    mod = make_mod()

    # Without pass — works fine
    pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
    mod_l = pipeline(mod)
    exe = tvm.relax.build(mod_l, target='llvm')
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    inp = [tvm.runtime.tensor(np.ones((1, 4, 8, 8), dtype=np.float32), device=tvm.cpu()),
           tvm.runtime.tensor(np.ones((1, 4, 8, 8), dtype=np.float32) * 2, device=tvm.cpu())]
    out_ok = vm['main'](*inp).numpy()
    print(f"Without pass: OK, shape={out_ok.shape}")

    # With ReorderPermuteDimsAfterConcat — crashes
    print("\nApplying ReorderPermuteDimsAfterConcat pass...")
    try:
        pipeline = tvm.ir.transform.Sequential([
            relax.transform.ReorderPermuteDimsAfterConcat(),
            relax.transform.LegalizeOps()
        ])
        mod_l = pipeline(mod)
        print("Pass applied OK (bug not reproduced)")
    except Exception as e:
        print(f"CRASH: {e}")
        print("\nBUG CONFIRMED: ReorderPermuteDimsAfterConcat crashes on negative axis")

    # Workaround: use axis=3 instead of axis=-1
    print("\n--- Workaround: positive axis ---")
    bb = relax.BlockBuilder()
    x = relax.Var('x', relax.TensorStructInfo((1, 4, 8, 8), 'float32'))
    y = relax.Var('y', relax.TensorStructInfo((1, 4, 8, 8), 'float32'))
    with bb.function('main', [x, y]):
        with bb.dataflow():
            xt = bb.emit(R.permute_dims(x, [0, 2, 3, 1]))
            yt = bb.emit(R.permute_dims(y, [0, 2, 3, 1]))
            out = bb.emit_output(R.concat([xt, yt], axis=3))  # positive axis
        bb.emit_func_output(out)
    mod_pos = bb.finalize()
    try:
        pipeline = tvm.ir.transform.Sequential([
            relax.transform.ReorderPermuteDimsAfterConcat(),
            relax.transform.LegalizeOps()
        ])
        mod_l = pipeline(mod_pos)
        print("With positive axis=3: OK")
    except Exception as e:
        print(f"With positive axis=3: CRASH: {e}")
