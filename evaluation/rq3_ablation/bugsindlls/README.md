# RQ3: BugsInDLLs External-Validation Ablation

Backs the BugsInDLLs ablation reported in the paper's RQ3
(`tab:components`): API-mutation vs. operation-guided generation, each
combined with crash-only and full oracles, on a known-bug corpus.

## Files

| Path | Purpose |
|---|---|
| `run_bugsindlls_batch.py` | Batch orchestrator: iterates fuzzable APIs from the ground-truth CSV |
| `benchmark_full_oracle_triton.py` | Per-API fuzzer with all oracle types (crash + differential + gradient + metamorphic) |
| `benchmark_full_oracle_bugsindlls.py` | Hybrid oracle variant tuned for the BugsInDLLs corpus |
| `run_and_match_bugsindlls.py` | Post-hoc matcher: fuzzer output → CSV ground truth (PyTorch) |
| `run_and_match_bugsindlls_tf.py` | TensorFlow variant of the matcher |
| `bug_discovery_enhancement.py` | Kernel-guided mutation strategy (ε-greedy) |
| `kernel_state_space.py` | API-mode kernel state-space tracker |
| `ground_truth/bug_dataset_pytorch.csv` | PyTorch BugsInDLLs ground truth (39 reproducible bugs) |
| `ground_truth/bug_dataset_tensorflow.csv` | TensorFlow BugsInDLLs ground truth (28 reproducible bugs; CSV has 29 `Reproduced=Yes` rows but the denominator is 28 because one entry — issue 64576, `Status=Open`, no PR — has no accepted fix and is excluded. A second `Status=Open` entry (issue 64256) is retained because it carries an active fix PR (PR 64587) and is therefore considered confirmed-and-fixable; three other `Reproduced=Yes` entries have a blank PR# but are `Status=Closed` and are counted.) |
| `results/batch_summary_high_risk.json` | Per-API run summary backing the operation-guided row of `tab:components` (paths anonymized) |

> The driver scripts above call into the FreeFuzz API-model library
> (`classes/torch_api.py`, `classes/tf_api.py`). That library is not
> bundled in this artifact because it is a separate (anonymous-during-
> review) project; reviewers wanting to re-execute should pull it in
> via `KERNELFUZZ_FREEFUZZ_SRC`.

## Paper claims (`tab:components`)

| Configuration | PyTorch | TensorFlow |
|---|---|---|
| API mutation + crash-only | 3/39 (7.7%) | 2/28 (7.1%) |
| API mutation + full oracle | 14/39 (35.9%) | 9/28 (32.1%) |
| Operation-guided + full oracle | 20/39 (51.3%) | 21/28 (75.0%) |

The PyTorch row "20/39" decomposes into 14 strict root-cause matches + 6
lenient API-family matches; TF "21/28" decomposes into 9 strict + 12
API-family. The matcher scripts above implement both modes.

> **Reproducibility note:** The strict-match row is reproducible standalone.
> The lenient API-family match requires the FreeFuzz API-model library
> (loaded via `KERNELFUZZ_FREEFUZZ_SRC`; see Dependencies below) to determine
> API family membership. Without it, the verifiable lower bound is
> PT 14/39 (35.9%) strict and TF 9/28 (32.1%) strict — identical to the
> "API mutation + full oracle" row above and confirming that operation
> guidance adds coverage beyond API mutation alone.

## Reproducing

```bash
# PyTorch batch (requires ~hours; needs FreeFuzz API-model library on PYTHONPATH)
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

## Dependencies

- PyTorch nightly (with torch.compile / Inductor) for the PT batch
- TensorFlow 2.x for the TF batch
- FreeFuzz-baseline API model files (loaded via `KERNELFUZZ_FREEFUZZ_SRC`)
