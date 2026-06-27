# [Inductor] Copy-paste typo in scheduler prologue fusion guard: `node1` checked twice instead of `node1` and `node2`

## 🐛 Describe the bug

In `torch/_inductor/scheduler.py`, the `can_fuse()` method has a copy-paste typo in the template prologue fusion guard. The condition at line 6610 checks `node1.has_aliasing_or_mutation()` **twice** instead of checking both `node1` and `node2`:

```python
# Line 6610 (BUGGY):
if node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation():
    why("template prologue can only fuse functional pointwise nodes")
    return False
```

The second `node1` should be `node2`:

```python
# Correct:
if node1.has_aliasing_or_mutation() or node2.has_aliasing_or_mutation():
    why("template prologue can only fuse functional pointwise nodes")
    return False
```

The same pattern is used correctly elsewhere in the same function:

```python
# Line 6377 (correct):
if node1.has_aliasing_or_mutation() or node2.has_aliasing_or_mutation():
    return None
```

As a result, if `node2` (the template node) has aliasing or mutation, the prologue fusion guard does not reject the fusion.

## Reproducer

```python
import inspect
from torch._inductor import scheduler

source = inspect.getsource(scheduler)
lines = source.split('\n')

for i, line in enumerate(lines):
    if 'node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation()' in line:
        for j in range(max(0, i-3), min(len(lines), i+4)):
            marker = ">>>" if j == i else "   "
            print(f"  {marker} {lines[j]}")
        print(f"\n  BUG: second 'node1' should be 'node2'")
        break
```

Output:
```
               if node1.get_buffer_names() & unsupported_prologue_args:
                   why("prologue fusion not implemented for kernel for these inputs")
                   return False
   >>>
               if node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation():
                   why("template prologue can only fuse functional pointwise nodes")
                   return False

     BUG: second 'node1' should be 'node2'
```

## Fix

```diff
-            if node1.has_aliasing_or_mutation() or node1.has_aliasing_or_mutation():
+            if node1.has_aliasing_or_mutation() or node2.has_aliasing_or_mutation():
                 why("template prologue can only fuse functional pointwise nodes")
                 return False
```

## Versions

- PyTorch: 2.13.0.dev20260513+cu126
- File: `torch/_inductor/scheduler.py`, line 6610

cc <upstream maintainer> <upstream maintainer>
