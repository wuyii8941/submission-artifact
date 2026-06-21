# WhiteFox Baseline Experiment

Empirical validation that WhiteFox's pass-level testing does not systematically
reach compilation-decision-conditioned (CDC) bugs.

## Setup

- **PyTorch**: 2.13.0.dev20260501+cu126
- **WhiteFox tests**: 61,000 pre-generated programs from
  [WhiteFox](https://github.com/ise-uiuc/WhiteFox) repository
  (`starcoder-generated/torch-inductor-generated/`)
- **Hardware**: NVIDIA Tesla T4 GPUs
- **Date**: May 14-15, 2026

## Configurations

| Config | GPU | Device | test_ad | Timeout | Tests Run |
|--------|-----|--------|---------|---------|-----------|
| GPU0 (forward-only) | 0 | cuda | False | 10s | 61,000 |
| GPU1 (+backward) | 1 | cuda | True | 15s | 58,020 |

GPU0 uses WhiteFox's default configuration (forward-only, `torch.no_grad()`,
`fn.train(False)`). GPU1 enables backward testing (`test_ad=True`) as a
fairness enhancement.

## Key Results

### GPU0: Forward-only (WhiteFox default)
- 574 raw bugs, **528 unique** (after dedup)
- 502 compilation failures (JIT_STATUS: Inductor codegen produces invalid C++)
- 26 silent wrong results (JIT_VALUE: numerical divergence eager vs compiled)
- **0 compilation-decision-conditioned bugs**
- Dominant finding: `pointwise_cumsum_replacement` float16 numerical drift (215 raw)

### GPU1: +Backward testing
- 3,848 raw bugs, **638 unique**
- 110 additional gradient-related bugs (FWD_GRAD + REV_GRAD + FWD_VALUE + REV_VALUE)
  that forward-only testing misses
- All gradient bugs are pass-level (backward implementation of individual passes)
- **0 compilation-decision-conditioned bugs**

### Combined
- **122,000 test executions** (61K x 2 configs), **0 CDC bugs**
- All bugs map 1:1 to specific optimization passes
- No bug depends on scheduling decisions, fusion structure, or template selection

## Comparison with DeepOpFuzz

| | WhiteFox (122K tests) | DeepOpFuzz |
|---|---|---|
| Pass-implementation bugs | 475+ | 0 |
| Decision-conditioned bugs | 0 | 10 (PyTorch) |
| Requires backward testing | 148 (with test_ad=True) | 5 |
| Dominant bug type | cumsum pass numerical drift | scheduling/fusion codegen |

The cumsum example is illustrative: WhiteFox finds numerical drift in the
`pointwise_cumsum_replacement` pass transformation (pass-level bug). DeepOpFuzz
finds a crash in the SplitScan codegen path triggered when dim>=129 causes a
scheduling decision change (decision-conditioned bug). Same operator, same
compiler, structurally different bug classes.

## File Structure

```
results_gpu0_cuda/          # Forward-only (complete)
  bugs.jsonl                # One JSON per bug: {file, category, optimization, errors, cdc_ops}
  summary.json              # Aggregate statistics
  experiment.log            # Timestamped execution log
  checkpoint.json           # Progress checkpoint

results_gpu1_cuda_ad/       # +Backward (58K/61K)
  bugs.jsonl
  summary.json
  experiment.log
  checkpoint.json

batch_runner.py             # Main experiment script
launch_full_experiment.sh   # Launch script for all GPUs
```

## Bug Categories

- `JIT_STATUS`: Inductor compilation failure (codegen C++ doesn't compile, or runtime error)
- `JIT_VALUE`: Silent numerical divergence between eager and compiled forward output
- `FWD_VALUE`: Forward value differs under backward-enabled test
- `FWD_GRAD`: Forward gradient incorrect
- `REV_VALUE`: Reverse-mode value incorrect
- `REV_GRAD`: Reverse-mode gradient incorrect

## Reproducing

```bash
# Requires WhiteFox repo at ${ARTIFACT_ROOT}/WhiteFox
# and PyTorch nightly at ${ARTIFACT_ROOT}/pt_nightly_env

# Forward-only (WhiteFox default)
CUDA_VISIBLE_DEVICES=0 python3 batch_runner.py \
    --gpu 0 --device cuda --timeout 10 --batch-size 50

# With backward testing
CUDA_VISIBLE_DEVICES=1 python3 batch_runner.py \
    --gpu 1 --device cuda --timeout 15 --batch-size 30 --test-ad
```

## Paper Reference

These results appear in Section 5.1.3 (State-Space Blind Spots of Existing
Fuzzers), "Empirical validation with WhiteFox" paragraph.
