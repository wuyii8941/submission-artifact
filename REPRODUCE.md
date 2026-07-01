# Reproducing Paper Results

This document maps each major paper claim to the script that reproduces it,
along with expected runtime on a single NVIDIA T4 / A6000 GPU class.

## Quick reference

| Paper element | Script | Approx. budget |
|---|---|---|
| Table I (`tab:bug-summary`) | `data/confirmed_bugs/` + paper text | (catalog only) |
| Table II (`tab:pipeline-gap`, RQ1 4 h replay coverage of 30 PT/Inductor root causes) | `evaluation/rq1_effectiveness/baseline_replay/launch_4h_runs.sh` + `attribute_table2.py` | 4 h × 5 methods in parallel |
| Table III (`tab:budget-split`, RQ2 efficiency) | `evaluation/rq2_efficiency/launch_template_runs.sh` + `build_timecurve.py` | 4 h × 5 seeds × 4 modes |
| Table IV (`tab:rq3-contribution`, RQ3 ablation) | see RQ3 section below | derived from RQ2 runs |
| Table V (`tab:components`, RQ3 BugsInDLLs validation) | `evaluation/rq3_ablation/bugsindlls/run_bugsindlls_batch.py` | configurable |
| Figure 1 (`fig:inductor-regions`) | static (paper) | – |
| Figure 2 (`fig:behavior-gap`) | static (paper) | – |
| Figure 3 (`fig:overview`) | static (paper) | – |
| Figure 4 (`fig:rq2-efficiency`) | `evaluation/rq2_efficiency/run_benchmark.py` + `extract_curves.py` | RQ2 budget |

## RQ1 — Effectiveness (campaign bugs)

See `data/confirmed_bugs/{pytorch, tensorflow, xla, iree, tvm}/`. Each
subdirectory holds minimized reproducers for the bugs reported in the
paper. Each reproducer's header comment lists the exact stack version
that triggers it.

### Single-bug reproducer (recommended)

`tools/run_repro.sh <bug-dir>` is a stack-aware dispatcher that picks the
right interpreter, sets stack-specific environment (e.g. `LD_LIBRARY_PATH`
for TVM, `TF_CPP_MIN_LOG_LEVEL` for TensorFlow), and runs the canonical
`.py` in the bug directory. Set these once per shell:

```bash
export DOF_PT_PYTHON=/path/to/pt-nightly/venv/bin/python
export DOF_TF_PYTHON=/path/to/tf-nightly/venv/bin/python
export DOF_TVM_HOME=/path/to/tvm-install
export DOF_IREE_PYTHON=/path/to/iree-venv/bin/python
```

then:

```bash
tools/run_repro.sh data/confirmed_bugs/pytorch/bug_003_slice_scatter_backward
tools/run_repro.sh data/confirmed_bugs/tvm/bug_001_gemv_keyerror
```

### All bugs at once

`tools/run_all_repros.sh [stack]` loops over every reproducer (optionally
filtered to one stack), writes per-bug logs and a summary CSV under
`results/repro_run_<timestamp>/`. Each repro gets a 5-minute soft cap
(override with `DOF_REPRO_TIMEOUT`).

```bash
tools/run_all_repros.sh                 # all 86 bugs
tools/run_all_repros.sh pytorch         # just the PT bugs
```

Exit-code convention: `0` means the reproducer ran to completion (some
bugs use stdout markers like "BUG CONFIRMED" / "MISMATCH" rather than a
non-zero exit); non-zero means the bug manifested as a crash or error;
`124` is the dispatcher timeout.

## RQ2 — Efficiency (historical benchmark)

```bash
cd evaluation/rq2_efficiency
./launch_template_runs.sh        # 5 seeds × 4 modes × 4 h budget
                                 # ~12 h wall-clock on 8 GPUs in parallel
python build_timecurve.py        # produces the cross-seed efficiency CSV
python plot_kd_rootcause.py      # plots Figure 4
```

Random worker is wrapped with restart-on-crash logic
(`launch_random_restart.sh`) because random sampling occasionally
triggers CUDA illegal memory access in the generated programs.

**Pre-computed data for Table III columns:**

| Column | Source file |
|---|---|
| Bugs reached (2 h / 4 h) | `data_rootcause_5seed.csv` (5-seed mean) |
| Kernels reached | `results_sat_pilot/*_seed42.json` → `timeseries[-1]["sat"]["1"]` |
| Kernels stressed | `results_sat_pilot/*_seed42.json` → `timeseries[-1]["sat"]["10"]` |

See `evaluation/rq2_efficiency/README.md` for the full data-source map.

## RQ3 — Ablation (generation + oracle contribution)

The component ablation (Table IV / `tab:rq3-contribution`) is derived
from the same RQ2 benchmark runs by toggling the generation mode
(targeted vs random) and the oracle (full differential vs forward-only).
The summary JSON is at
`evaluation/rq3_ablation/results/ablation_rootcause_summary.json`;
`evaluation/rq3_ablation/results/README.md` documents how each row maps
to a `(mode, oracle)` pair and how the paper's crash-only rows are
derived from the benchmark's silent-correctness construction.

The external BugsInDLLs validation (Table V / `tab:components`) is a
separate driver:

```bash
cd evaluation/rq3_ablation/bugsindlls
python run_bugsindlls_batch.py
```

The reference-free / metamorphic rediscovery results reported in the
appendix are backed by JSON files under
`evaluation/rq3_ablation/metamorphic_results/`.

## RQ4 — Portability (IREE / TVM)

The 24 IREE + TVM bugs are evidenced by the per-bug reproducers in
`data/confirmed_bugs/iree/` and `data/confirmed_bugs/tvm/`. Supporting
discovery-campaign logs for the TF/XLA primary campaign live in
`evaluation/rq4_portability/results/`.

```bash
cd evaluation/rq4_portability
./run_tf_1800s_parallel.sh    # TF/XLA budget-controlled discovery campaign
# IREE / TVM bug-level repros live in data/confirmed_bugs/{iree,tvm}/
```

## Expected outputs

- CSVs under `results/` (created at run time)
- Per-run JSONs under `evaluation/<rq>/results_*/`
- Aggregated tables printed to stdout

## Known limitations

- Some random-generation runs (RQ2 baseline) hit transient CUDA errors
  and require the restart wrapper. Total wall-clock for the random
  baseline may exceed the 4-hour budget by a few minutes.
- External datasets (WhiteFox tests, BugsInDLLs corpus) must be
  obtained from their original projects; see `INSTALL.md`.
