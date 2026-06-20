"""Bug 020: isnan/isinf/isfinite crash — CodeGenVM not implemented.

Status: CONFIRMED BUG (NEW)
TVM version: latest main (commit ca68bef6b, 2026-04-20)

Description:
  R.isnan(), R.isinf(), and R.isfinite() all crash at build time with:
    InternalError: CodeGenVM cannot handle this intrinsic now: Op(relax.isnan)

  Crashes on BOTH CPU (llvm) and CUDA targets. These ops are registered in
  the Relax op registry but have no CodeGenVM implementation.

Impact:
  Models using isnan/isinf/isfinite checks cannot be compiled at all.
"""
import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R


def _make_var(name, shape, dtype="float32"):
    return relax.Var(name, relax.TensorStructInfo(shape, dtype))


shape = (4, 8)
bb = relax.BlockBuilder()
x = _make_var("x", shape, "float32")
with bb.function("main", [x]):
    with bb.dataflow():
        out = bb.emit(R.isnan(x))
        gv = bb.emit_output(out)
    bb.emit_func_output(gv)
mod = bb.get()

pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
mod_l = pipeline(mod)

try:
    exe = relax.build(mod_l, target="llvm")
    print("Build succeeded (unexpected)")
except Exception as e:
    print(f"Build CRASHED (expected):")
    print(f"  {type(e).__name__}: {e}")
