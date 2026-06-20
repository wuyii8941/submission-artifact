# [Inductor] `pick_deterministic_choice` checks `ExternKernelChoice` instead of `ExternKernelCaller` — deterministic mode doesn't prefer extern kernels

## 🐛 Describe the bug

In `torch/_inductor/select_algorithm.py`, `pick_deterministic_choice()` (called when `torch._inductor.config.deterministic = True`) tries to prefer extern (ATen/cuBLAS) kernels over Triton templates. However, it checks `isinstance(choice, ExternKernelChoice)` instead of `isinstance(choice, ExternKernelCaller)`.

`ExternKernelChoice` is a standalone factory class (inherits from `object`), while the actual elements in the `choices` list are `ExternKernelCaller` instances (inherits from `ChoiceCaller`). The isinstance check is always `False`, so the function always falls through to `return choices[0]`.

The same issue exists in `_classify_kernel_operation()` at line 3650.

## Code

```python
# select_algorithm.py, line 3771-3779
def pick_deterministic_choice(self, choices: list[ChoiceCaller]) -> ChoiceCaller:
    assert len(choices) >= 2
    externs = [
        choice for choice in choices if isinstance(choice, ExternKernelChoice)
        #                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                          should be ExternKernelCaller
    ]
    if len(externs) > 0:
        return externs[0]
    else:
        return choices[0]  # ← always reaches here

# select_algorithm.py, line 3650
elif isinstance(choice, ExternKernelChoice):  # same bug
    choice_name = choice.name
```

Class hierarchy:

```
object
└── ExternKernelChoice       ← factory; creates ExternKernelCaller instances (line 3024)

ChoiceCaller                 ← base for elements in `choices` list
├── ExternKernelCaller       ← wraps an ExternKernelChoice (line 3242)
├── TritonTemplateCaller
└── ...
```

Every other isinstance check in the same file correctly uses `ExternKernelCaller`:

```python
# lines 3932, 3940, 3987, 4008, 4026, 4051, 4869, 5132, 5247 — all use ExternKernelCaller
if isinstance(choice, ExternKernelCaller):  ✓
```

## Suggested Fix

```python
# line 3774
choice for choice in choices if isinstance(choice, ExternKernelCaller)

# line 3650
elif isinstance(choice, ExternKernelCaller):
```

## Impact

When `torch._inductor.config.deterministic = True`, the algorithm selector is supposed to prefer extern (cuBLAS) kernels over Triton templates for reproducibility. Instead, it picks `choices[0]` regardless of type.

In `_classify_kernel_operation()`, extern kernel names (`mm`, `bmm`, `addmm`, etc.) are never matched, so the classifier falls through to shape heuristics or `"other"`, affecting autotuning logging and filtering.

## Versions

```
PyTorch: 2.13.0.dev20260513+cu126
```

cc <upstream maintainer> <upstream maintainer>
