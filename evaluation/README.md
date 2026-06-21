# Evaluation Experiments

The authoritative entry point for reproducing the paper's results is the
top-level **[`REPRODUCE.md`](../REPRODUCE.md)**, which maps each paper
table / figure to the script that produces it. This file is just an
index of the layout under `evaluation/`.

## Paper RQ → Directory Mapping

| Paper element | Directory | Driver script |
|---|---|---|
| RQ1 (Effectiveness, Table 1) | `rq1_effectiveness/` | (catalog only — see `data/confirmed_bugs/`) |
| RQ2 (Efficiency, Figure 3) | `rq2_efficiency/` | `run_benchmark.py` via `launch_template_runs.sh` |
| RQ3 (Ablation, Tables 3 + 5) | `rq2_efficiency/` (same runs as RQ2) | `run_benchmark.py` |
| RQ3 (BugsInDLLs validation, Table 4) | `rq3_ablation/bugsindlls/` | `run_bugsindlls_batch.py` |
| RQ4 (Portability, TF / IREE / TVM) | `rq4_portability/` + `data/confirmed_bugs/{iree,tvm}/` | `run_tf_1800s_parallel.sh` |

## Baseline replays

`baselines/` holds the WhiteFox and NNSmith helpers used by the RQ2
benchmark for the comparative columns:

- `baselines/whitefox_baseline/` — replays the 61k WhiteFox
  pre-generated programs under the DeepOpFuzz oracle.
- `baselines/nnsmith_baseline/` — runs NNSmith under the DeepOpFuzz oracle.
- `baselines/nnsmith_native/` — runs NNSmith with its own native oracle
  (supplementary cross-check, not in the paper's main tables).

## Archive

`archive/` contains earlier prototypes of the experiments and
one-off analysis scripts that are **not** invoked by `REPRODUCE.md`.
They are retained for completeness but should not be needed to
reproduce paper results:

- `archive/time_controlled/`, `archive/rq2_coverage/`,
  `archive/ablation/`, `archive/fusion_analysis/` — earlier prototypes
  of the RQ2/RQ3 experiments superseded by `rq2_efficiency/`.
- `archive/rq4_oracle/` — earlier oracle-validation prototype.
- `archive/analysis_*.py`, `archive/dedup_analysis.py`,
  `archive/derive_component_table.py`, `archive/extract_api_ablation.py`,
  `archive/merge_seeds.py`, `archive/plot_causal_figures.py`,
  `archive/retrospective_enrichment.py` — one-off analysis helpers.

## Hardware

All experiments target CUDA on NVIDIA Tesla T4 / RTX A6000 with CUDA 12.x.
Some kernel scheduling decisions (e.g., cooperative reduction) depend on
SM count and may differ on other GPU architectures.
