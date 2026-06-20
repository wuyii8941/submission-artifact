"""
BUG 055: torch._foreach_sub(tensors, other_tensors, alpha=k) silently ignores alpha under torch.compile

Root cause: In lowering.py:7963, `_foreach_sub.List` is registered WITHOUT `allow_alpha=True`,
while `_foreach_add.List` (line 7953) correctly passes `allow_alpha=True`.

The alpha parameter is silently dropped during lowering, so compiled _foreach_sub always
computes `a - b` instead of `a - alpha*b`.

File: torch/_inductor/lowering.py, line 7963
Fix: Change `register_foreach_pointwise(aten._foreach_sub.List, sub)` to
     `register_foreach_pointwise(aten._foreach_sub.List, sub, allow_alpha=True)`

Confirmed on PyTorch 2.13.0.dev20260513+cu126
"""
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = False


def foreach_sub_alpha(a, b):
    return torch._foreach_sub(a, b, alpha=2.0)


torch.manual_seed(42)
a = [torch.randn(4, device='cuda') for _ in range(3)]
b = [torch.randn(4, device='cuda') for _ in range(3)]

# Eager result (correct): a_i - 2.0 * b_i
eager = foreach_sub_alpha(a, b)
print(f"Eager (_foreach_sub, alpha=2.0):")
for i, t in enumerate(eager):
    print(f"  [{i}]: {t.tolist()}")

# Compiled result (WRONG: alpha=2.0 is silently ignored, computes a_i - b_i)
torch._dynamo.reset()
compiled = torch.compile(foreach_sub_alpha)(
    [t.clone() for t in a], [t.clone() for t in b]
)
print(f"\nCompiled (_foreach_sub, alpha=2.0):")
for i, t in enumerate(compiled):
    print(f"  [{i}]: {t.tolist()}")

# Verify: compiled matches alpha=1 (no alpha), NOT alpha=2
no_alpha = torch._foreach_sub(a, b)
print(f"\nVerification — compiled matches alpha=1 (alpha silently ignored):")
for i, (na, c) in enumerate(zip(no_alpha, compiled)):
    diff = (na - c).abs().max().item()
    print(f"  [{i}] compiled vs no_alpha: diff={diff}")

print(f"\nVerification — compiled does NOT match eager (alpha=2):")
for i, (e, c) in enumerate(zip(eager, compiled)):
    diff = (e - c).abs().max().item()
    print(f"  [{i}] compiled vs eager: diff={diff}")

# Contrast: _foreach_add with alpha=2.0 works correctly
def foreach_add_alpha(a, b):
    return torch._foreach_add(a, b, alpha=2.0)

eager_add = foreach_add_alpha(a, b)
torch._dynamo.reset()
compiled_add = torch.compile(foreach_add_alpha)(
    [t.clone() for t in a], [t.clone() for t in b]
)
print(f"\nContrast: _foreach_add with alpha=2.0 works correctly:")
for i, (e, c) in enumerate(zip(eager_add, compiled_add)):
    diff = (e - c).abs().max().item()
    print(f"  [{i}] add compiled vs eager: diff={diff}")

assert all(
    (na - c).abs().max().item() == 0.0
    for na, c in zip(no_alpha, compiled)
), "Expected compiled to match no-alpha result"
assert any(
    (e - c).abs().max().item() > 0.1
    for e, c in zip(eager, compiled)
), "Expected compiled to NOT match eager alpha=2 result"
print("\nBUG CONFIRMED: _foreach_sub alpha parameter silently ignored under torch.compile")
