# Reproducing Paper Results

This document maps each major paper claim to the script that reproduces it,
along with expected runtime on a single NVIDIA T4 / A6000 GPU class.

## Quick reference

| Paper element | Script | Approx. budget |
|---|---|---|
| Table 1 (bug summary) | `data/confirmed_bugs/` + paper text | (catalog only) |
| Table 2 (RQ1 component matches) | `evaluation/rq1_bugsindlls/run_bugsindlls_batch.py` | hours, per stack |
| Table 3 (RQ3 ablation) | `evaluation/rq2_benchmark/run_benchmark.py --modes targeted,random,...` | 4h × 5 seeds × 4 modes |
| Table 4 (RQ3 BugsInDLLs validation) | `evaluation/rq1_bugsindlls/run_bugsindlls_batch.py` | configurable |
| Table 5 (RQ3 per-kernel saturation) | `evaluation/rq2_benchmark/run_benchmark.py` (with worker patch) | 4h × 5 seeds × 4 modes |
| Figure 1 (Inductor stages) | static (paper) | – |
| Figure 2 (compiler-operation gap) | static (paper) | – |
| Figure 3 (RQ2 efficiency) | `evaluation/rq2_benchmark/run_benchmark.py` + `evaluation/extract_curves.py` | as above |

## RQ1 — Effectiveness (campaign bugs)

See `data/confirmed_bugs/{pytorch, tensorflow, xla, iree, tvm}/`. Each
subdirectory holds minimized reproducers for the bugs reported in the
paper. To re-verify a bug, install the corresponding stack version
(listed in each repro's header comment) and run the Python file.

Example:

```bash
cd data/confirmed_bugs/pytorch
python bug_001_*/minimal_repro.py
```

## RQ2 — Efficiency (historical benchmark)

```bash
cd evaluation/rq2_benchmark
./launch_template_runs.sh        # 5 seeds × 4 modes × 4h budget
                                 # ~12h wall-clock on 8 GPUs in parallel
python build_timecurve.py        # produces the cross-seed efficiency CSV
python plot_kd_rootcause.py      # plots Figure 3
```

Random worker is wrapped with restart-on-crash logic
(`launch_random_restart.sh`) because random sampling occasionally
triggers CUDA illegal memory access in the generated programs.

## RQ3 — Ablation (generation + oracle contribution)

The ablation table comes from the same RQ2 benchmark runs by toggling
the generation mode and the oracle. The saturation table comes from
aggregating per-kernel hit counts in the same JSON outputs:

```bash
python evaluation/rq2_benchmark/build_timecurve.py
# Saturation analysis: aggregate template_sample_counts across seeds
python scripts/aggregate_saturation.py
```

## RQ4 — Portability (IREE / TVM)

```bash
cd evaluation/rq5_cross_framework
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
