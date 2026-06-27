# Evaluation Experiments

The authoritative entry point for reproducing the paper's results is the
top-level **[`REPRODUCE.md`](../REPRODUCE.md)**, which maps each paper
table / figure to the script that produces it. This file is just an
index of the layout under `evaluation/`.

## Paper RQ → Directory Mapping

| Paper element | Directory | Driver script |
|---|---|---|
| RQ1 — Table I (`tab:bug-summary`) | `rq1_effectiveness/` | (catalog only — see `data/confirmed_bugs/`) |
| RQ1 — Table II (`tab:pipeline-gap`) | `rq1_effectiveness/baseline_replay/` | `launch_4h_runs.sh` + `attribute_table2.py` |
| RQ2 — Table III (`tab:budget-split`) + Figure 4 (`fig:rq2-efficiency`) | `rq2_efficiency/` | `run_benchmark.py` via `launch_template_runs.sh` |
| RQ3 — Table IV (`tab:rq3-contribution`) | `rq3_ablation/results/` (derived from RQ2 runs) | see `rq3_ablation/results/README.md` |
| RQ3 — Table V (`tab:components`, BugsInDLLs) | `rq3_ablation/bugsindlls/` | `run_bugsindlls_batch.py` |
| RQ3 — appendix metamorphic 6/6 + 4/4 | `rq3_ablation/metamorphic_results/` | (cached JSON; see README in that dir) |
| RQ4 — IREE + TVM portability | `rq4_portability/` + `data/confirmed_bugs/{iree,tvm}/` | `run_tf_1800s_parallel.sh` for the TF/XLA logs |

## Baseline replays

`baselines/` holds the WhiteFox and NNSmith helpers used by the RQ2
benchmark for the comparative columns:

- `baselines/whitefox_baseline/` — replays the 61k WhiteFox
  pre-generated programs under the DeepOpFuzz oracle.
- `baselines/nnsmith_baseline/` — runs NNSmith under the DeepOpFuzz oracle.
- `baselines/nnsmith_native/` — runs NNSmith with its own native oracle
  (supplementary cross-check, not in the paper's main tables).

## Hardware

All experiments target CUDA on NVIDIA Tesla T4 / RTX A6000 with CUDA 12.x.
Some kernel scheduling decisions (e.g., cooperative reduction) depend on
SM count and may differ on other GPU architectures.
