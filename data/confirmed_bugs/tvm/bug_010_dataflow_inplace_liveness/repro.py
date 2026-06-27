"""Bug 034: DataflowUseInplaceCalls corrupts concat input via in-place multiply.

When two expand_dims produce identical tensors (a and b from same input x),
and multiply(a, b) feeds into concat([prod, b]), the DataflowUseInplaceCalls
pass incorrectly allows multiply to write in-place to b's buffer.

This means when concat reads b, it gets the overwritten value (x^2) instead
of the original (x). The liveness analysis fails to account for b being
used both as a multiply input AND a later concat input.

Severity: CORRECTNESS BUG — silent wrong results, no crash or warning.

TVM version: 0.24.dev0, commit 0b0afd8dd (2026-04-24)
"""

import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R
from tvm.relax.transform import LegalizeOps


def build_and_run(mod, inputs_np, passes):
    pipeline = tvm.ir.transform.Sequential(passes)
    mod_lowered = pipeline(mod)
    exe = tvm.relax.build(mod_lowered, target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_inputs = [tvm.runtime.tensor(x, device=tvm.cpu()) for x in inputs_np]
    out = vm["main"](*tvm_inputs)
    if isinstance(out, (list, tuple)):
        return [o.numpy() for o in out]
    return [out.numpy()]


# Minimal reproducer
bb = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((4,), "float32"))
with bb.function("main", [x]):
    with bb.dataflow():
        a = bb.emit(R.expand_dims(x, axis=1))      # (4,1)
        b = bb.emit(R.expand_dims(x, axis=1))      # (4,1)
        prod = bb.emit(R.multiply(a, b))             # (4,1) = x^2
        out = bb.emit(R.concat([prod, b], axis=1))   # (4,2) = [x^2, x]
        gv = bb.emit_output(out)
    bb.emit_func_output(gv)
mod = bb.finalize()

x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

# Without DataflowUseInplaceCalls — CORRECT
out_correct = build_and_run(mod, [x_np], [LegalizeOps()])
print("Without DataflowUseInplaceCalls (correct):")
print(out_correct[0])
print(f"  col0 (x^2): {out_correct[0][:, 0]}")
print(f"  col1 (x):   {out_correct[0][:, 1]}")

# With DataflowUseInplaceCalls — INCORRECT
out_buggy = build_and_run(
    mod, [x_np],
    [relax.transform.DataflowUseInplaceCalls(), LegalizeOps()])
print("\nWith DataflowUseInplaceCalls (BUG):")
print(out_buggy[0])
print(f"  col0 (x^2): {out_buggy[0][:, 0]}")
print(f"  col1 (should be x, got x^2): {out_buggy[0][:, 1]}")

max_diff = float(np.abs(out_correct[0] - out_buggy[0]).max())
print(f"\nMax difference: {max_diff}")
assert max_diff > 0, "Bug not reproduced"
print("BUG CONFIRMED: in-place multiply overwrites concat's input buffer")
