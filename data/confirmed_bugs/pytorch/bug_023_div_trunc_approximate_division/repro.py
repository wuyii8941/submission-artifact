"""
BUG 053: torch.div(rounding_mode='trunc') produces off-by-one results under torch.compile

Root cause: In lowering.py:7104, the 'trunc' path uses plain `div(a, b)` which maps to
Triton's approximate division (reciprocal-based, ~2^-10 precision). The 'floor' path at
line 7101 was already fixed to use `_div_rn` (IEEE round-to-nearest division), but the
same fix was not applied to the 'trunc' path.

When the true quotient is very close to (but not equal to) an integer, the approximate
division can push the result past the integer boundary, causing trunc() to produce a
result off by 1.

Example: div_trunc(14.999999, 3.0)
  True quotient: 4.99999968... -> trunc should give 4.0
  Approx quotient: ~5.0         -> trunc gives 5.0 (WRONG)

File: torch/_inductor/lowering.py, line 7104
Fix: Change `trunc(div(a, b))` to `trunc(_div_rn(a, b))`, same as the floor path.

Confirmed on PyTorch 2.13.0.dev20260513+cu126
"""
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = False


def div_trunc(a, b):
    return torch.div(a, b, rounding_mode='trunc')


# Test values where the quotient is very close to an integer
a = torch.tensor([14.999999, 26.999999, 29.999999, 13.999999, 27.999999],
                 device='cuda', dtype=torch.float32)
b = torch.tensor([3.0, 3.0, 3.0, 7.0, 7.0],
                 device='cuda', dtype=torch.float32)

# Eager result (correct)
out_eager = div_trunc(a, b)
print(f"Eager:    {out_eager.tolist()}")
# Expected: [4.0, 8.0, 9.0, 1.0, 3.0]

# Compiled result (wrong)
torch._dynamo.reset()
out_compiled = torch.compile(div_trunc)(a.clone(), b.clone())
print(f"Compiled: {out_compiled.tolist()}")
# Actual:   [5.0, 9.0, 10.0, 2.0, 4.0] -- off by 1!

diff = out_eager - out_compiled
print(f"Diff:     {diff.tolist()}")
assert (diff.abs() > 0).any(), "Bug not reproduced"
print("\nBUG CONFIRMED: div_trunc produces off-by-one results under torch.compile")

# The bug also affects EXACT integer quotients (both directions):
print("\n--- Exact integer quotient case ---")
a2 = torch.tensor([148.0, 1073.0, 2112.0], device='cuda')
b2 = torch.tensor([37.0, 37.0, 33.0], device='cuda')
out_e2 = div_trunc(a2, b2)
torch._dynamo.reset()
out_c2 = torch.compile(div_trunc)(a2.clone(), b2.clone())
print(f"Eager:    {out_e2.tolist()}")  # [4.0, 29.0, 64.0]
print(f"Compiled: {out_c2.tolist()}")  # [3.0, 28.0, 63.0] -- off by -1!

# Verify: the floor path does NOT have this issue (it uses _div_rn)
def div_floor(a, b):
    return torch.div(a, b, rounding_mode='floor')

out_floor_e = div_floor(a, b)
torch._dynamo.reset()
out_floor_c = torch.compile(div_floor)(a.clone(), b.clone())
print(f"\nFloor eager:    {out_floor_e.tolist()}")
print(f"Floor compiled: {out_floor_c.tolist()}")
print(f"Floor matches:  {torch.equal(out_floor_e, out_floor_c)}")
