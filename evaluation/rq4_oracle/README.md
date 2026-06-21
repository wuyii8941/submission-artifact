# Oracle Validation (RQ3 ablation prototype)

> **Note:** the paper now reports the oracle contribution as part of the
> RQ3 ablation (driven from `evaluation/rq2_benchmark/`). The scripts in
> this directory are an earlier prototype kept for completeness; they are
> not invoked by `REPRODUCE.md`.

Validates the multi-stage oracle pipeline's precision (low false-positive
rate) and necessity (crash-only oracles miss the majority of findings).

## Pipeline Stages

1. **Repeated Execution (Determinism)**: Program executed 3 times; if
   results are inconsistent across runs, marked FLAKY and filtered.

2. **Perturbation Stability**: For numerical differences, inputs
   perturbed with ε ~ U(-1e-4, 1e-4) three times. Difference must
   remain directionally consistent (same sign, similar magnitude).

3. **FP64 Backoff**: Program re-run at float64 precision. If the
   difference vanishes (< 1e-6), the original was floating-point
   precision noise, not a compiler bug.

4. **Cross-Configuration Checks** (in full fuzzer only): Compare
   eager, aot_eager, and inductor to distinguish frontend from backend.

## Scripts

| Script | Purpose |
|--------|---------|
| `validate_oracle.py` | Standalone 3-stage validation pipeline |

## Source

The full oracle pipeline is integrated into:
- `${ARTIFACT_ROOT}/kernelfuzz/fuzzer.py` (GraphFuzzerV4 class methods)
- `${ARTIFACT_ROOT}/kernelfuzz/bug_result.py` (`_verify_bug_in_fresh_process`)
- `${ARTIFACT_ROOT}/evaluation/dedup_analysis.py` (post-hoc filtering)

## Usage

```bash
# Demo mode: runs built-in test cases demonstrating each stage
python validate_oracle.py --demo

# Validate a specific program file (must define `program` and `_program_inputs`)
python validate_oracle.py --program /path/to/bug_repro.py
```

## Paper Claims (Table 6: Filtering Funnel)

```
Raw candidates (main campaign)              169
Deduplicated programs (5 seeds)             300
VALID_BUG (pass all 3 stages)               61
Confirmed upstream issues (root-cause dedup)  9
```

- All 9 PyTorch bugs reproduce deterministically (Stage 1 pass)
- Numerical-diff bugs (#anon-1, #anon-2, etc.) are perturbation-stable
- Crash/metadata bugs (#anon-3, #anon-4) confirmed via control configs
- Conservative pipeline: 61/300 → 20% survive → high precision
