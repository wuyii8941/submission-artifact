#!/usr/bin/env python3
# === PROVENANCE: Class C — static demonstration ===========================
# The upstream issue has NO Python reproducer (the bug is source-level dead
# code). This script demonstrates the type confusion via class introspection.
# Not a runtime crash or wrong-result reproducer.
# Verified to run on PyTorch 2.13.0.dev20260609+cu126; final line prints
# "BUG CONFIRMED: extern-kernel preference ... is unreachable".
# See BUG_MAPPING.md > Reproducer Provenance Audit.
# ==========================================================================
"""
Bug: pick_deterministic_choice() checks the wrong type — ExternKernelChoice
instead of ExternKernelCaller — so the deterministic-mode preference for
extern (cuBLAS/ATen) kernels is dead code.

Upstream issue: <upstream issue/PR, anonymized> (closed/triaged)

In torch/_inductor/select_algorithm.py, pick_deterministic_choice (~line 3771)
and _classify_kernel_operation (~line 3650) both do:

    isinstance(choice, ExternKernelChoice)

But ExternKernelChoice is a *factory* class that inherits directly from
`object`, while the elements of the `choices` list are ExternKernelCaller
instances (subclass of ChoiceCaller). The check is therefore always False;
pick_deterministic_choice always falls through to `return choices[0]`.

This reproducer demonstrates the type mismatch via class introspection — the
buggy isinstance check cannot match anything the function ever receives.
"""
from torch._inductor.select_algorithm import ExternKernelChoice, ExternKernelCaller

print(f"ExternKernelChoice MRO: {[c.__name__ for c in ExternKernelChoice.__mro__]}")
print(f"ExternKernelCaller MRO: {[c.__name__ for c in ExternKernelCaller.__mro__]}")
print()

# The two classes have no inheritance relationship — yet the buggy code
# filters a list of ExternKernelCaller instances with an isinstance check
# for ExternKernelChoice. The filter is always empty.
related = (issubclass(ExternKernelCaller, ExternKernelChoice)
           or issubclass(ExternKernelChoice, ExternKernelCaller))
assert not related, "Classes are unrelated — confirmed"

# Simulate what pick_deterministic_choice receives at runtime.
class _FakeExtern(ExternKernelCaller):
    def __init__(self): pass
class _FakeTriton:
    pass

choices = [_FakeExtern(), _FakeTriton()]

# Buggy filter (current code):
externs_buggy = [c for c in choices if isinstance(c, ExternKernelChoice)]
# Correct filter (proposed fix):
externs_fixed = [c for c in choices if isinstance(c, ExternKernelCaller)]

print(f"Buggy filter matches: {len(externs_buggy)} (always 0 — dead code)")
print(f"Fixed filter matches: {len(externs_fixed)} (extern kernel is found)")

assert len(externs_buggy) == 0
assert len(externs_fixed) == 1
print()
print("BUG CONFIRMED: extern-kernel preference under "
      "torch._inductor.config.deterministic=True is unreachable.")
