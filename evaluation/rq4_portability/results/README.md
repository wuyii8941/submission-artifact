# RQ4 Portability Logs

The paper's RQ4 portability claim — **24 cross-framework bugs (7 IREE +
17 TVM)** — is backed by the per-bug reproducers and metadata under
`data/confirmed_bugs/iree/` and `data/confirmed_bugs/tvm/`, not by the
files in this directory.

This `results/` directory contains supplementary TF/XLA discovery-campaign
logs that document the budget-controlled fuzzing runs that fed the
primary TF/XLA campaign reported under RQ1 (`tab:bug-summary`,
TensorFlow/XLA row, 31 bugs).

| File | Content |
|---|---|
| `tf_1800s_results.json` | Aggregated 5-seed × 2-mode (targeted / random) × 1800 s TF/XLA budget runs. Contains per-run stats (iterations, finding-type counts, diversity curve sampled at 10× decimation) and NNSmith TF baseline comparison. Per-finding records were stripped to keep the file ≤ 50 KB; the full ~10 MB version with every `{iter, time_sec, type, meta, pair}` record lives at `paper/evaluation/rq5_cross_framework/results/tf_1800s_results.json` in the development tree. |

The IREE and TVM reproducers themselves are minimal Python scripts in
`data/confirmed_bugs/{iree,tvm}/bug_*` directories; reviewers can
re-execute them with the per-bug reproducer instructions in
`data/confirmed_bugs/BUG_MAPPING.md`.
