# `torch.compile` silently produces wrong results for `index_add`/`index_copy`/`index_fill`/`index_put` on expanded (stride-0) tensors

**Filed:** <upstream issue, anonymized for review>

## Summary

When tensors created via `.expand()` (introducing stride-0 dimensions) are compiled with `torch.compile(backend='inductor')`, indexed write operations produce silently incorrect forward results. The compiler fails to materialize expanded dimensions, causing writes to different logical rows to collide in shared underlying memory.

- Eager mode: correct
- `aot_eager`: correct
- Inductor: **WRONG RESULTS** (silent)

Workaround: `.clone()` or `.contiguous()` after `.expand()`.

## Minimal reproducer

```python
import torch

def fn(src, idx):
    x = torch.zeros(1, 8, device="cuda").expand(4, -1)  # stride-0 in dim 0
    return x.index_add(0, idx, src)

src = torch.ones(3, 8, device="cuda")
idx = torch.tensor([0, 1, 0], device="cuda")

eager = fn(src, idx)
compiled = torch.compile(fn, backend="inductor")(src, idx)

print("Eager row 0:", eager[0].tolist())      # [2, 2, ...] correct
print("Compiled row 0:", compiled[0].tolist()) # [3, 3, ...] WRONG
```

## Affected ops

- `index_add`
- `index_copy`
- `index_fill`
- `index_put`

## Root cause

Same bug class as bug_007/025 (stride-0 clone not materialized) but through `index_*` ops.
