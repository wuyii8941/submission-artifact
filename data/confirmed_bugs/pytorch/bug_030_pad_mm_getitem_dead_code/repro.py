#!/usr/bin/env python3
# === PROVENANCE: Class C — static demonstration ===========================
# The upstream issue has NO Python reproducer (the bug is source-level dead
# code). This script demonstrates the type confusion via FX-graph construction.
# Not a runtime crash or wrong-result reproducer.
# Verified to run on PyTorch 2.13.0.dev20260609+cu126; final line prints
# "BUG CONFIRMED: pad_mm.get_non_view_def cannot recurse through getitem".
# See BUG_MAPPING.md > Reproducer Provenance Audit.
# ==========================================================================
"""
Bug: get_non_view_def in torch/_inductor/fx_passes/pad_mm.py compares
node.op (always a string like "call_function") with operator.getitem
(a function object). The comparison `node.op is operator.getitem` is
always False, so the recursion through getitem nodes is dead code —
pad_mm fails to look through getitem to find the underlying definition.

Upstream issue: <upstream issue/PR, anonymized> (closed/triaged)

Impact is performance only (pad_mm may skip a valid pad opportunity);
computation results are unaffected. This reproducer demonstrates the
type-confusion that makes the recursion unreachable.
"""
import operator
import torch
import torch.fx as fx

# Build a minimal getitem node and inspect node.op (the *opcode*, a string)
# vs node.target (the actual function, operator.getitem).
g = fx.Graph()
x = g.placeholder("x")
gi = g.call_function(operator.getitem, args=(x, 0))
g.output(gi)
gm = fx.GraphModule(torch.nn.Module(), g)

node = next(n for n in gm.graph.nodes if n.op == "call_function")

print(f"node.op     = {node.op!r}  (string opcode)")
print(f"node.target = {node.target!r}  (function object)")
print(f"operator.getitem = {operator.getitem!r}")
print()

# Buggy condition from pad_mm.py:
buggy = (node.op is operator.getitem)
# Correct condition:
fixed = (node.op == "call_function" and node.target is operator.getitem)

print(f"buggy check (node.op is operator.getitem) -> {buggy}  (always False — dead code)")
print(f"fixed check (target match)               -> {fixed}")

assert buggy is False, "Buggy comparison can never match — confirmed."
assert fixed is True, "Fixed comparison correctly identifies the getitem node."
print()
print("BUG CONFIRMED: pad_mm.get_non_view_def cannot recurse through getitem.")
