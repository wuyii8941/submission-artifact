"""
Isolate the bug to a single constant in Inductor's generated kernel.

Strategy:
  1. Verify buf0 (the pool output) is a normal contiguous tensor with
     the expected stride (3185, 49, 7, 1). If this is wrong then the
     bug is in the pool kernel / buffer allocator, not the reduction kernel.
  2. Compute the expected sum directly from buf0 in eager PyTorch.
     Compare to the inductor-compiled sum. If they differ the bug is
     in the kernel that READS from buf0.
  3. Extract the exact Inductor-generated Triton reduction kernel verbatim,
     and also make a copy with ONE change: replace '3200' with '3185'.
     Run both against the SAME buf0. If the patched kernel is correct
     and the original is wrong, the bug is 100% in the constant that
     Inductor hard-coded into its generated source.

This isolates Inductor's codegen from Triton, CUDA, cuBLAS, the driver,
the pool kernel, the buffer allocator, and eager PyTorch.
"""
import torch
import triton
import triton.language as tl

print(f"PyTorch: {torch.__version__}")
print(f"Triton:  {triton.__version__}")
print()

torch.manual_seed(0)
x = torch.randn(2, 65, 8, 8, dtype=torch.float32, device="cuda")

# ---------------------------------------------------------------------------
# Step 1: produce buf0 in eager and verify its layout
# ---------------------------------------------------------------------------
buf0 = torch.nn.functional.adaptive_avg_pool2d(x, 7).contiguous()
print(f"buf0.shape   = {tuple(buf0.shape)}")
print(f"buf0.stride  = {buf0.stride()}")
print(f"buf0.numel   = {buf0.numel()}")
print(f"buf0.is_contiguous = {buf0.is_contiguous()}")
assert buf0.stride() == (3185, 49, 7, 1), "buf0 stride is NOT (3185, 49, 7, 1)"
assert buf0.numel() == 2 * 3185, "buf0 size mismatch"
print()

# ---------------------------------------------------------------------------
# Step 2: compute the expected final output directly from buf0 in eager
# ---------------------------------------------------------------------------
expected = buf0.flatten(1).sum(dim=-1)
print(f"Expected final output (eager, from buf0):")
print(f"  {expected.tolist()}")
print()

# ---------------------------------------------------------------------------
# Step 3a: The EXACT Inductor-generated reduction kernel (verbatim, bug=3200)
# ---------------------------------------------------------------------------
@triton.jit
def inductor_broken(in_ptr0, out_ptr0, xnumel, r0_numel,
                    XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 2
    r0_numel = 3185
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3200 * x0),  # <<< the bug: 3200
                       r0_mask & xmask,
                       eviction_policy="evict_first",
                       other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)


# ---------------------------------------------------------------------------
# Step 3b: SAME kernel with ONE character change: 3200 -> 3185
# ---------------------------------------------------------------------------
@triton.jit
def inductor_patched(in_ptr0, out_ptr0, xnumel, r0_numel,
                     XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 2
    r0_numel = 3185
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3185 * x0),  # <<< patched
                       r0_mask & xmask,
                       eviction_policy="evict_first",
                       other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)


def run(kernel, label):
    out = torch.empty(2, dtype=torch.float32, device="cuda")
    grid = (2,)
    # XBLOCK=1, R0_BLOCK=1024 — plausible config for this shape; exact value
    # doesn't matter because the kernel is correct for ANY R0_BLOCK that
    # covers r0_numel=3185.
    kernel[grid](buf0, out, 2, 3185, XBLOCK=1, R0_BLOCK=4096)
    print(f"  {label:25s}: {out.tolist()}")
    return out


print("Step 3: run BOTH kernels on the same buf0 and compare:")
out_broken  = run(inductor_broken,  "inductor (3200, as-is)")
out_patched = run(inductor_patched, "patched  (3200→3185)")
print(f"  {'expected (eager)':25s}: {expected.tolist()}")
print()

diff_broken  = (out_broken  - expected).abs().max().item()
diff_patched = (out_patched - expected).abs().max().item()
print(f"  diff(broken  vs expected) = {diff_broken:.4e}")
print(f"  diff(patched vs expected) = {diff_patched:.4e}")
print()

# ---------------------------------------------------------------------------
# Step 4: Full torch.compile comparison (sanity — the upstream behaviour)
# ---------------------------------------------------------------------------
def prog(x):
    y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
    return y.flatten(1).sum(dim=-1)

torch._dynamo.reset()
c = torch.compile(prog, backend="inductor")
out_compile = c(x.clone())
print(f"torch.compile(prog)(x)     : {out_compile.tolist()}")
print(f"  vs expected diff         : {(out_compile - expected).abs().max().item():.4e}")
print()

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("=" * 70)
print("VERDICT")
print("=" * 70)
if diff_broken > 1e-3 and diff_patched < 1e-3:
    print("  ✓ broken kernel (3200) produces wrong output")
    print("  ✓ patched kernel (3185) produces correct output")
    print("  → The bug is 100% in the hard-coded constant '3200'")
    print("    in Inductor's generated Triton kernel source.")
    print("  → It is NOT in Triton, CUDA, the driver, the buffer allocator,")
    print("    or the pool kernel — buf0 is correct, the SAME buf0 gives")
    print("    correct output when the constant is fixed.")
else:
    print("  UNEXPECTED — see diffs above")
