# RQ4: Cross-Framework Portability

> **Directory naming note:** earlier drafts numbered this RQ as RQ5 and
> some scripts still carry the `rq5_*` prefix or `cross_framework`
> suffix. The paper refers to it as RQ4.

This directory contains the portability case studies on **IREE** and
**TVM** plus the supporting TF/XLA time-controlled budget runs.

The paper's portability claim:

> On IREE and TVM, the reused strategies find 24 previously unknown
> bugs (7 on IREE, 17 on TVM).

Bug reproducers and per-bug summaries live in
`../../data/confirmed_bugs/iree/` (7 directories) and
`../../data/confirmed_bugs/tvm/` (17 directories).

## Scripts (development tree; not all needed to validate paper claims)

| Script | Purpose |
|---|---|
| `iree_runner.py` | IREE compilation + ONNX Runtime differential oracle |
| `tf_time_controlled.py` | Budget-controlled TF/XLA fuzzer used to produce the time-curve JSON in `results/` |
| `run_tf_1800s_parallel.sh` | Five-seed launcher around `tf_time_controlled.py` |
| `_nnsmith_tf_worker.py`, `run_nnsmith_tf_baseline.py` | NNSmith TF baseline for the comparison logs in `results/` |
| `tf_oracles.py`, `tf_graph_gen.py`, `tf_coverage.py` | TF-side oracle / generator / coverage hooks |

> `tf_main.py`, the JAX scripts, `pattern_migration_test.py`, and
> `targeted_consistency_fuzzer.py` are exploratory drivers from the
> development tree that are not exercised by the published numbers.

## Reproducing

The IREE and TVM portability bugs are reproduced from the minimized
reproducers under `${ARTIFACT_ROOT}/data/confirmed_bugs/iree/` and
`${ARTIFACT_ROOT}/data/confirmed_bugs/tvm/`. The TF/XLA budget-controlled
runs use `run_tf_1800s_parallel.sh`.

```bash
# TF/XLA budget runs (5 seeds × 3 modes × 1800 s)
bash run_tf_1800s_parallel.sh
```

## Per-Bug Counts (paper `tab:bug-summary`)

| Stack | Silent | Crash | Total | Fixed | Open |
|---|---|---|---|---|---|
| IREE | 3 | 4 | 7 | 2 | 5 |
| TVM | 8 | 9 | 17 | 16 | 1 |
| **Cross-framework total** | **11** | **13** | **24** | **18** | **6** |

The TensorFlow/XLA portability evidence is reported separately as part
of the primary campaign (paper RQ1 / `tab:bug-summary` row for
TensorFlow/XLA, 31 bugs).

## Logs

Per-seed JSON and log files for the TF time-controlled runs are in
`results/`; see `results/README.md` for the file inventory.
