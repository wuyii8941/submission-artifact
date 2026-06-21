# Evaluation Experiments

The authoritative entry point for reproducing the paper's results is the
top-level **[`REPRODUCE.md`](../REPRODUCE.md)**, which maps each paper
table / figure to the script that produces it. This file is just an index
to the directories that script lives in.

## RQ → Directory Mapping

| Paper element | Directory | Driver script |
|---|---|---|
| RQ1: BugsInDLLs reachability (Table 2, Table 4) | `rq1_bugsindlls/` | `run_bugsindlls_batch.py` |
| RQ2: Generation efficiency (Figure 3) | `rq2_benchmark/` | `run_benchmark.py` (5 seeds × 4 modes × 4h budget) |
| RQ3: Ablation (Table 3) and per-kernel saturation (Table 5) | `rq2_benchmark/` | Same run as RQ2; saturation re-aggregated from each JSON's `timeseries` field |
| RQ4: Cross-framework portability (TF/IREE/TVM) | `rq5_cross_framework/` + `data/confirmed_bugs/{iree,tvm}/` | `run_tf_1800s_parallel.sh` |

Confirmed bugs (RQ1 main campaign) live in
`../data/confirmed_bugs/{pytorch,tensorflow,xla,iree,tvm}/`; see the
top-level catalog at `../data/confirmed_bugs/BUG_MAPPING.md`.

## Baseline replay scripts

- `whitefox_baseline/` — replays the 61k WhiteFox pre-generated programs
  under the KernelFuzz oracle.
- `nnsmith_baseline/` — runs NNSmith under the KernelFuzz oracle (used
  for the RQ2 efficiency comparison).

## Supporting / archived directories

These directories are kept for completeness but are **not** referenced by
`REPRODUCE.md` — they hold superseded experiments, secondary analyses,
and intermediate artifacts:

- `time_controlled/`, `rq2_coverage/`, `ablation/`, `baselines/`,
  `fusion_analysis/`, `nnsmith_native/` — earlier prototypes of the
  RQ2/RQ3 experiments; superseded by `rq2_benchmark/`.
- `rq4_oracle/` — earlier oracle-validation prototype.
- `analysis_*.py`, `dedup_analysis.py`, `derive_component_table.py`,
  `extract_*.py`, `merge_seeds.py`, `plot_causal_figures.py`,
  `retrospective_enrichment.py` — one-off analysis helpers; outputs of
  the ones actually used by the paper are produced by the drivers above
  or by `build_timecurve.py` inside `rq2_benchmark/`.

## Hardware

All experiments target CUDA on NVIDIA Tesla T4 / RTX A6000 with CUDA 12.x.
Some kernel scheduling decisions (e.g., cooperative reduction) depend on
SM count and may differ on other GPU architectures.
