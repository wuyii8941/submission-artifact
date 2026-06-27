# `torch.compile` + `torch.while_loop`: stride mismatch crash during backward pass

**Filed:** <upstream issue, anonymized for review>

## Summary

`torch.compile(fullgraph=True)` with the Inductor backend crashes when running backward through a `torch.while_loop` that modifies carried tensor state. The error is a stride mismatch between the carried input (stride `(0, 0)`) and the body output (stride `(64, 1)`).

- Eager mode: works fine
- `aot_eager` backend: works fine
- Inductor (forward only): works fine
- Inductor (with backward): **CRASHES**

## Error

```
InductorError: UncapturedHigherOrderOpError: Expected carried_inputs and body_output
to have same metadata but found:
pair[1] differ in 'stride: (0, 0) vs (64, 1)'
```

## Minimal reproducer

```python
import torch

x = torch.randn(4, 64, device='cuda', requires_grad=True)

def cond_fn(i, x):
    return i < 5

def body_fn(i, x):
    return i + 1, x * 0.9 + 0.1

def fn(x):
    _, result = torch.while_loop(cond_fn, body_fn, (torch.tensor(0, device='cuda'), x))
    return result.sum()

torch._dynamo.reset()
torch.compile(fn, fullgraph=True)(x).backward()  # CRASHES
```

## Root cause

Inductor-specific. Related to #<anon>/#<anon> (forward-pass while_loop issues) but backward trigger is novel.
