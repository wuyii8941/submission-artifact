# RQ1: BugsInDLLs Motivating Study (API-Mode Validation)

Validates that compiler-aware guidance reaches known bug paths that
unguided API-level mutation misses, using BugsInDLLs as ground truth.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_bugsindlls_batch.py` | Batch orchestrator: iterates fuzzable APIs from CSV |
| `benchmark_full_oracle_triton.py` | Per-API fuzzer with all oracle types |
| `benchmark_full_oracle_bugsindlls.py` | Hybrid oracle variant for BugsInDLLs |
| `run_and_match_bugsindlls.py` | Post-hoc matcher: fuzzer output → CSV ground truth |
| `run_and_match_bugsindlls_tf.py` | TensorFlow variant of the matcher |
| `bug_discovery_enhancement.py` | Kernel-guided mutation strategy (ε-greedy) |
| `kernel_state_space.py` | API-mode kernel state space tracker |
| `ground_truth/` → `bugsindlls/` | Ground truth CSVs (PyTorch + TensorFlow) |

## Source

All scripts are symlinked from:
- `${ARTIFACT_ROOT}/FreeFuzz-baseline/kernel-guided/src/`
- `${ARTIFACT_ROOT}/FreeFuzz-baseline/bugsindlls/`

## Reproducing

```bash
# PyTorch batch (requires ~hours)
python run_bugsindlls_batch.py --scope pytorch \
    --output results/bugsindlls_pt_$(date +%Y%m%d)

# TensorFlow batch
python run_bugsindlls_batch.py --scope tensorflow \
    --output results/bugsindlls_tf_$(date +%Y%m%d)

# Match against ground truth
python run_and_match_bugsindlls.py \
    --batch-dir results/bugsindlls_pt_* \
    --ground-truth ground_truth/bug_dataset_pytorch.csv

python run_and_match_bugsindlls_tf.py \
    --batch-dir results/bugsindlls_tf_* \
    --ground-truth ground_truth/bug_dataset_tensorflow.csv
```

## Paper Claims (Table 2)

- PyTorch: 14 strict hits + 6 lenient = 20/39 reproduced bugs reached
- TensorFlow: 9 strict hits + 12 API-match = 21/28 reproduced bugs reached
- Guided mutation reaches bug paths that random misses (compile oracle
  detects silent diffs invisible to crash-only)

## Dependencies

- PyTorch nightly (with torch.compile / Inductor)
- TensorFlow 2.x (for TF experiments)
- FreeFuzz-baseline API models (`classes/torch_api.py`, `classes/tf_api.py`)
