"""Bug 036: maximum/minimum/relu/clip do not propagate NaN (first operand).

TVM's T.max/T.min compile to x86 maxss/minss instructions which return the
second operand when the first is NaN, violating IEEE 754 NaN propagation.

Affected: R.maximum, R.minimum, R.nn.relu, R.clip, R.argmax
Not affected: R.nn.leakyrelu, R.nn.silu, R.nn.gelu (use comparison paths)

Pattern: max(NaN, x) → x (wrong), max(x, NaN) → NaN (correct, by accident)
relu(NaN) → max(NaN, 0) → 0 (wrong, should be NaN)
clip(NaN, lo, hi) → hi (wrong, should be NaN)

Severity: CORRECTNESS BUG — NaN silently disappears, masking numerical errors.
TVM version: 0.24.dev0, commit 0b0afd8dd (2026-04-24)
"""

import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R
from tvm.relax.transform import LegalizeOps


def build_and_run(mod, inputs_np):
    pipeline = tvm.ir.transform.Sequential([LegalizeOps()])
    mod_low = pipeline(mod)
    exe = tvm.relax.build(mod_low, target="llvm")
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_inputs = [tvm.runtime.tensor(x, device=tvm.cpu()) for x in inputs_np]
    return vm["main"](*tvm_inputs).numpy()


# === Bug 1: maximum(NaN, x) returns x instead of NaN ===
print("=== maximum(NaN, x) ===")
bb = relax.BlockBuilder()
a = relax.Var("a", relax.TensorStructInfo((4,), "float32"))
b = relax.Var("b", relax.TensorStructInfo((4,), "float32"))
with bb.function("main", [a, b]):
    with bb.dataflow():
        gv = bb.emit_output(bb.emit(R.maximum(a, b)))
    bb.emit_func_output(gv)
mod = bb.finalize()

A = np.array([np.nan, 1.0, np.nan, 0.0], np.float32)
B = np.array([1.0, np.nan, np.nan, np.nan], np.float32)
out = build_and_run(mod, [A, B])
ref = np.maximum(A, B)

print(f"A:     {A}")
print(f"B:     {B}")
print(f"TVM:   {out}")
print(f"NumPy: {ref}")
print(f"Match: {np.array_equal(out, ref, equal_nan=True)}")
assert not np.array_equal(out, ref, equal_nan=True), "Bug not reproduced"
print("BUG CONFIRMED: max(NaN, x) = x, should be NaN\n")

# === Bug 2: relu(NaN) returns 0 instead of NaN ===
print("=== relu(NaN) ===")
bb2 = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((4,), "float32"))
with bb2.function("main", [x]):
    with bb2.dataflow():
        gv = bb2.emit_output(bb2.emit(R.nn.relu(x)))
    bb2.emit_func_output(gv)
mod2 = bb2.finalize()

X = np.array([np.nan, -1.0, np.nan, 2.0], np.float32)
out2 = build_and_run(mod2, [X])
ref2 = np.maximum(X, 0)

print(f"Input: {X}")
print(f"TVM:   {out2}")
print(f"NumPy: {ref2}")
assert not np.isnan(out2[0]), "Bug not reproduced"
print("BUG CONFIRMED: relu(NaN) = 0, should be NaN\n")

# === Bug 3: clip(NaN, lo, hi) returns hi instead of NaN ===
print("=== clip(NaN, -1, 1) ===")
bb3 = relax.BlockBuilder()
x3 = relax.Var("x", relax.TensorStructInfo((4,), "float32"))
with bb3.function("main", [x3]):
    with bb3.dataflow():
        gv = bb3.emit_output(bb3.emit(R.clip(x3, -1.0, 1.0)))
    bb3.emit_func_output(gv)
mod3 = bb3.finalize()

X3 = np.array([np.nan, 0.5, -2.0, np.inf], np.float32)
out3 = build_and_run(mod3, [X3])
ref3 = np.clip(X3, -1.0, 1.0)

print(f"Input: {X3}")
print(f"TVM:   {out3}")
print(f"NumPy: {ref3}")
assert not np.isnan(out3[0]), "Bug not reproduced"
print("BUG CONFIRMED: clip(NaN, -1, 1) = 1, should be NaN\n")

# === Demonstrate silent danger: relu swallows NaN ===
print("=== Real-world impact: relu hides NaN from overflow ===")
bb4 = relax.BlockBuilder()
x4 = relax.Var("x", relax.TensorStructInfo((1, 4), "float32"))
with bb4.function("main", [x4]):
    with bb4.dataflow():
        r = bb4.emit(R.nn.relu(x4))
        s = bb4.emit(R.sum(r, axis=-1))
        gv = bb4.emit_output(s)
    bb4.emit_func_output(gv)
mod4 = bb4.finalize()

D = np.array([[1.0, 2.0, np.nan, 4.0]], np.float32)
out4 = build_and_run(mod4, [D])
ref4 = np.maximum(D, 0).sum(axis=-1)
print(f"Input:  {D}")
print(f"TVM relu+sum:  {out4}  (NaN was silently zeroed)")
print(f"NumPy relu+sum: {ref4}  (NaN correctly propagated)")
