# FP Example: aoti_diff/1 (d4_cumsum_pointwise) — state-pollution false positive

> This file documents a FALSE POSITIVE found in the P0 run (2026-04-12) to
> calibrate LESSONS.md §8 (state-pollution FP class) and justify the P0.1
> AOTI verify gap fix.

## What the fuzzer reported

```
[FIRE][CAT] [NOVEL] [D4:aoti_diff] P3 src=corpus config=aoti preset=combo_pointwise_only
diff=3571.15576171875 kernels=266 prog=d4_cumsum_pointwise|torch.float32|b3s147h20|cfg=combo_pointwise_only
```

`aoti_diff`: eager and AOTI disagreed by max_diff=3571, 67% of elements wrong.

## Why it's a FP

Fresh-process verify across **10 random seeds** (2026-04-12):
- All 10 seeds: AOTI max_diff ∈ [4.9e-4, 1.2e-3] (fp32 noise scale)
- Parent process claimed 3571 → 6 orders of magnitude larger than fresh child
- Structure: cumsum + relu + silu + matmul + sum — no data-dependent shapes,
  no non-determinism, no sensitivity to specific input ranges

## Root cause

Long-process state pollution in the P0 fuzzer harness (~399 iter, single
python process). AOTI's `aoti_compile_and_package()` + `aoti_load_package()`
shares CUDA context and memory with the parent. After ~300 iters of
compile/load/cleanup cycles, the CUDA memory state may have corrupted the
result tensor — the `Failed to destroy CUDA event in AOTInductor model`
warnings in the P0 log (lines 2424-2441) are circumstantial evidence.

## Why the FP wasn't filtered by P0.1 verify

The `_verify_bug_in_fresh_process()` function had a short-circuit for AOTI bugs:
```python
if cfg_lbl == "aoti":
    return True, "aoti_verify_not_implemented_assume_ok"
```
This was based on the assumption that AOTI (which re-exports each iter) would
be less susceptible to state pollution. That assumption was wrong. The AOTI
.so loading process interacts with the CUDA allocator state and can be
polluted by prior iters.

## Action taken

Fix applied: replaced the AOTI verify short-circuit with a real AOTI
fresh-process verify driver (P0.1 AOTI verify gap fix, 2026-04-12).
