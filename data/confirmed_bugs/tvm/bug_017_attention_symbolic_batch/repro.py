#!/usr/bin/env python3
# === PROVENANCE: Class B — issue-body lifted ===============================
# Code lifted from upstream issue body via web extraction. Not guaranteed
# byte-identical to the original issue post; verify against upstream before
# relying. See BUG_MAPPING.md > Reproducer Provenance Audit.
# NOT verified on this server: TVM env (tvm_ffi 0.1.10 vs source tvm
# 0.24.dev0) has a 'Cannot find object type index for tir.Buffer' build
# mismatch that blocks 'from tvm import tir' — unrelated to the bug
# itself. Reproducer code matches the upstream issue body verbatim
# (minus a 'tirx' → 'tir' typo correction).
# ==========================================================================
"""
Bug: relax.op.nn.attention fails to legalize when the batch dim is a symbolic
tir.Var. _te_attention builds two separate Mul(B, num_head) expressions, and
topi.nn.batch_matmul compares them with Python `==`, which returns a deferred
PrimExpr instead of True — tripping the batch-dim assertion inside batch_matmul.

Upstream issue: <upstream issue/PR, anonymized> (closed/fixed by PR #<anonymized>)

Oracle: legalization should succeed for symbolic batch dims (it does for
concrete int batch dims).
"""
import tvm
from tvm import relax, tir

B = tir.Var("B", "int64")
bb = relax.BlockBuilder()
Q = relax.Var("Q", relax.TensorStructInfo([B, 4, 2, 8], "float32"))
K = relax.Var("K", relax.TensorStructInfo([B, 4, 2, 8], "float32"))
V = relax.Var("V", relax.TensorStructInfo([B, 4, 2, 8], "float32"))

with bb.function("main", [Q, K, V]):
    with bb.dataflow():
        y = bb.emit(relax.op.nn.attention(Q, K, V))
        gv = bb.emit_output(y)
    bb.emit_func_output(gv)

mod = bb.get()
# Bug: LegalizeOps crashes with AssertionError on the batch-dim check
# because Mul(B, num_head) == Mul(B, num_head) returns a PrimExpr, not True.
mod = relax.transform.LegalizeOps()(mod)
print("Legalization succeeded — unexpected, bug should have crashed")
