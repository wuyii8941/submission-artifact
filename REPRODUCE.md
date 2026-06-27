# Reproducing Paper Results

This document maps each major paper claim to the script that reproduces it,
along with expected runtime on a single NVIDIA T4 / A6000 GPU class.

## Quick reference

| Paper element | Script | Approx. budget |
|---|---|---|
| Table 1 (bug summary) | `data/confirmed_bugs/` + paper text | (catalog only) |
| Table 2 (RQ1 4h replay coverage of 30 PT/Inductor root causes) | `evaluation/rq1_effectiveness/baseline_replay/launch_4h_runs.sh` + `attribute_table2.py` | 4h × 5 methods in parallel |
| Table 3 (RQ3 ablation) | `evaluation/rq2_efficiency/run_benchmark.py --modes targeted,random,...` | 4h × 5 seeds × 4 modes |
| Table 4 (RQ3 BugsInDLLs validation) | `evaluation/rq3_ablation/bugsindlls/run_bugsindlls_batch.py` | configurable |
| Table 5 (RQ3 per-kernel saturation) | `evaluation/rq2_efficiency/run_benchmark.py` (with worker patch) | 4h × 5 seeds × 4 modes |
| Figure 1 (Inductor stages) | static (paper) | – |
| Figure 2 (compiler-operation gap) | static (paper) | – |
| Figure 3 (RQ2 efficiency) | `evaluation/rq2_efficiency/run_benchmark.py` + `evaluation/extract_curves.py` | as above |

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
./launch_template_runs.sh        # 5 seeds × 4 modes × 4h budget
                                 # ~12h wall-clock on 8 GPUs in parallel
python build_timecurve.py        # produces the cross-seed efficiency CSV
python plot_kd_rootcause.py      # plots Figure 3
```

Random worker is wrapped with restart-on-crash logic
(`launch_random_restart.sh`) because random sampling occasionally
triggers CUDA illegal memory access in the generated programs.

## RQ3 — Ablation (generation + oracle contribution)

The component ablation (Table 3) comes from the same RQ2 benchmark runs
by toggling the generation mode and the oracle. The saturation table
(Table 5) comes from aggregating per-kernel hit counts in the same JSON
outputs:

```bash
python evaluation/rq2_efficiency/build_timecurve.py
# Per-kernel hit counts are recorded in the timeseries field of each
# JSON checkpoint produced by the worker.
```

The external BugsInDLLs validation (Table 4) is a separate driver:

```bash
cd evaluation/rq3_ablation/bugsindlls
python run_bugsindlls_batch.py
```

## RQ4 — Portability (TensorFlow / IREE / TVM)

```bash
cd evaluation/rq4_portability
./run_tf_1800s_parallel.sh    # TF cross-framework
# IREE / TVM repros in data/confirmed_bugs/{iree,tvm}/
```

## Expected outputs

- CSVs under `results/` (created at run time)
- Per-run JSONs under `evaluation/<rq>/results_*/`
- Aggregated tables printed to stdout

## Known limitations

- Some random-generation runs (RQ2 baseline) hit transient CUDA errors
  and require the restart wrapper. Total wall-clock for the random
  baseline may exceed the 4-hour budget by a few minutes.
- The saturation analysis (Table 5, Section 5.3) requires the
  worker patch that records per-template hit counts in each timeseries
  checkpoint. The patched worker is included.
- External datasets (WhiteFox tests, BugsInDLLs corpus) must be
  obtained from their original projects; see `INSTALL.md`.
