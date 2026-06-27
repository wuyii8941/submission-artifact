"""
Bug: Copy-paste typo in scheduler.py prologue fusion check — node1 checked
twice instead of node1 and node2.

In scheduler.py line 6610, the template prologue fusion guard:
    if node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation():
checks node1 twice. The second check should be node2:
    if node1.has_aliasing_or_mutation() or node2.has_aliasing_or_mutation():

This means if node2 (the template) has aliasing or mutation, the prologue
fusion proceeds when it should be rejected, potentially producing incorrect
results in fused kernels where the template has side effects.

The bug is in the can_fuse() method of the Inductor scheduler, which decides
whether two nodes can be fused into a single kernel. The prologue fusion
path fuses a pointwise op into a template (e.g., matmul) kernel. The check
was added to ensure both nodes are functional (no aliasing/mutation), but
due to the typo, only node1 is validated.

Root cause: Copy-paste error. The condition should check both node1 and
node2 for aliasing or mutation before allowing fusion.
"""

# This is a code-level bug in PyTorch Inductor's scheduler, not a numerical
# error that can be triggered by a simple test. The reproducer below
# demonstrates where the bug is in the source code.

import torch
import inspect
import textwrap


def show_bug():
    """Show the buggy line in the scheduler source code."""
    from torch._inductor import scheduler

    source = inspect.getsource(scheduler)
    lines = source.split('\n')

    print("Bug location: torch/_inductor/scheduler.py")
    print()

    for i, line in enumerate(lines):
        if 'node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation()' in line:
            start = max(0, i - 3)
            end = min(len(lines), i + 4)
            for j in range(start, end):
                marker = ">>> BUG" if j == i else "       "
                print(f"  {marker} | {lines[j]}")
            print()
            print("  The second 'node1' should be 'node2'.")
            print("  Current:  node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation()")
            print("  Correct:  node1.has_aliasing_or_mutation() or node2.has_aliasing_or_mutation()")
            return True

    print("Bug not found (may have been fixed)")
    return False


if __name__ == '__main__':
    found = show_bug()
    if found:
        print("\nBUG confirmed: copy-paste typo in prologue fusion guard")
