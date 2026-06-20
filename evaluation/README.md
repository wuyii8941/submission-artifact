# Evaluation Experiments

All experiment scripts, raw results, and analysis for the paper's evaluation.

## RQ → Directory Mapping

| RQ | Question | Directory | Primary Output |
|----|----------|-----------|----------------|
| RQ1 | Bug effectiveness + BugsInDLLs reachability | `rq1_bugsindlls/` | Tables 1-3 |
| RQ2 | Generation comparison (targeting vs baselines) | `time_controlled/` | Tables 4-5 |
| RQ2 | Oracle necessity (crash-only vs differential) | `time_controlled/` | Tables 6-7 |
| RQ2 | NNSmith native oracle (multi-backend) | `nnsmith_native/` | Supplement |
| RQ3 | Coverage growth D(t) and saturation | `time_controlled/` (extended) | Figure |
| RQ4 | Oracle precision pipeline | `rq4_oracle/` | Tables 8-9 |
| RQ5 | Cross-framework transferability | `rq5_cross_framework/` | Table 10 |

## Directory Structure

```
evaluation/
├── README.md                          # This file
│
├── time_controlled/                   # RQ2 + RQ3: primary experiments
│   ├── run_time_controlled.py         #   300s comparison (Table 4)
│   ├── run_extended_coverage.py       #   1800s coverage curves (RQ3 figure)
│   ├── _time_worker.py                #   Worker (one mode × one seed)
│   ├── plot_coverage_curves.py        #   D(t) curve plotter
│   ├── extract_paper_numbers.py       #   LaTeX number extraction
│   └── results/
│       ├── time_controlled_results.json      # 300s data (Table 4)
│       └── extended_coverage_results.json    # 1800s data (RQ3)
│
├── nnsmith_baseline/                  # RQ2: NNSmith with KernelFuzz oracle
│   ├── run_nnsmith_baseline.py
│   ├── _nnsmith_worker.py
│   └── results/
│
├── nnsmith_native/                    # RQ2: NNSmith with native oracle
│   ├── run_nnsmith_native.py          #   Orchestrator (5 seeds × 300s)
│   ├── _nnsmith_native_worker.py      #   CPU-vs-CUDA + eager-vs-compiled
│   └── results/
│
├── ablation_v2/                       # RQ2: saturation-based ablation
│   ├── run_ablation_v2.py
│   ├── _ablation_worker_v2.py
│   └── results/
│
├── rq1_bugsindlls/                    # RQ1: BugsInDLLs benchmark validation
│   ├── run_bugsindlls_batch.py        #   Batch orchestrator
│   ├── benchmark_full_oracle_*.py     #   Per-API fuzzer
│   ├── run_and_match_bugsindlls*.py   #   Ground-truth matcher
│   └── ground_truth/                  #   BugsInDLLs CSV datasets
│
├── rq4_oracle/                        # RQ4: Oracle validation
│   └── validate_oracle.py             #   3-stage filter demo
│
├── rq5_cross_framework/               # RQ5: TF/XLA/IREE/TVM
│   ├── tf_main.py                     #   TensorFlow/XLA fuzzer
│   ├── iree_runner.py                 #   IREE runner
│   └── pattern_migration_test.py      #   Pattern transfer test
│
├── dedup_analysis.py                  # Bug deduplication analysis
├── extract_api_ablation.py            # RQ1 number extraction
├── extract_curves.py                  # Coverage curve CSV export
├── merge_seeds.py                     # Multi-seed result merger
├── retrospective_enrichment.py        # Template enrichment analysis
└── results/                           # Aggregated/merged results
    ├── dedup_analysis.json
    ├── ablation_5seed_merged.json
    └── coverage_curves.json
```

## Reproducing Key Experiments

### RQ2: Generation Comparison (Table 4)

```bash
# 300s budget, 5 seeds, 3 modes (targeted / random / NNSmith)
# ~25 min total (sequential) or ~5 min on 3 GPUs
CUDA_VISIBLE_DEVICES=0 python time_controlled/run_time_controlled.py \
    --time-budget 300 --gpu 0

# Extract LaTeX numbers
python time_controlled/extract_paper_numbers.py
```

### RQ3: Coverage Curves (Extended Budget)

```bash
# 1800s (30 min) budget for saturation evidence
# ~2.5h per mode (sequential) or ~2.5h on 3 GPUs

# Parallel on 3 GPUs:
CUDA_VISIBLE_DEVICES=0 python time_controlled/run_extended_coverage.py \
    --time-budget 1800 --modes targeted --gpu 0 &
CUDA_VISIBLE_DEVICES=1 python time_controlled/run_extended_coverage.py \
    --time-budget 1800 --modes random --gpu 1 &
CUDA_VISIBLE_DEVICES=2 python time_controlled/run_extended_coverage.py \
    --time-budget 1800 --modes nnsmith --gpu 2 &
wait

# Plot
python time_controlled/plot_coverage_curves.py \
    time_controlled/results/extended_coverage_results.json \
    -o figures/coverage_curves_1800s.pdf
```

### RQ2 Supplement: NNSmith Native Oracle

Runs NNSmith with its own multi-backend oracle (CPU vs CUDA torchjit)
alongside the KernelFuzz oracle (eager vs compiled), on the same
generated models.

```bash
CUDA_VISIBLE_DEVICES=0 python nnsmith_native/run_nnsmith_native.py \
    --time-budget 300
```

## Coverage Metric

Kernel diversity D = |unique kernel templates|, where a kernel template
is the kernel name with trailing compilation counter stripped:

```
triton_poi_fused_sin_relu_0  →  triton_poi_fused_sin_relu
triton_red_fused_sum_add_1   →  triton_red_fused_sum_add
extern_out:mm(...)           →  extern_gemm
```

No manual denominator; compare absolute D under equal wall-clock time.

## Hardware

All experiments target CUDA on NVIDIA Tesla T4 GPUs with CUDA 12.6.
Some kernel scheduling decisions (e.g., cooperative reduction) depend
on SM count and may differ on other GPU architectures.
