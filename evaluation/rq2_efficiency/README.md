# RQ2 Efficiency Data

This directory backs **Table III (`tab:budget-split`)** and **Figure 4 (`fig:rq2-efficiency`)** in the paper.

## Data files and what they back

| File / Directory | Backs | Notes |
|---|---|---|
| `data_rootcause_5seed.csv` | **Bugs reached** columns (2 h / 4 h) | 5-seed mean per method |
| `results_sat_pilot/*.json` | **Kernels reached** and **Kernels stressed** columns | Single seed (42); `sat['1']` = reached ≥1×, `sat['10']` = stressed ≥10× |
| `data_kd_5seed.csv` | Figure 4 kernel-diversity curves (not the table) | 5-seed mean `kernel_diversity` — different metric, see note below |
| `data_kd.csv` | Figure 4 seed-42 curve only | Single-seed `kernel_diversity` |
| `results_freefuzz_5seed/` | FreeFuzz rows in both table and figure | 5-seed structured JSON + logs |
| `ground_truth_bugs.csv` | 48-bug historical benchmark definition | Reproduced bugs on PyTorch 2.2.0 |
| `bug_labeling.csv` | Root-cause labels for the 48 bugs | Used by root-cause dedup |

## Kernels reached / stressed: `results_sat_pilot/`

The `results_sat_pilot/` JSONs are single-seed (seed 42) runs with kernel-saturation
tracking enabled.  Each timeseries entry contains a `sat` dict:

```json
"sat": {"1": 6372, "10": 317, "50": 55, "100": 34, "500": 8}
```

where the key is the hit-count threshold and the value is how many distinct
Triton kernel templates exceeded that threshold.  The paper's columns map as:

- **Kernels reached** ← `sat['1']` at the 4 h checkpoint
- **Kernels stressed** ← `sat['10']` at the 4 h checkpoint

Verified values at the 4 h checkpoint (kernel_diversity is a related but
distinct count; see note below):

| Method | Kernels reached (4 h) | Kernels stressed (4 h) |
|---|---|---|
| DeepOpFuzz (targeted) | 6,372 | 317 |
| Random | 7,439 | 77 |
| NNSmith | 80 | 67 |
| WhiteFox | 201 | 34 |
| FreeFuzz | 366–367 | 24 |

## Note on `data_kd_5seed.csv`

`data_kd_5seed.csv` records the 5-seed average of `kernel_diversity`
(unique kernel code hashes encountered during execution) sampled at
0.05 h intervals.  This metric is used to plot the time-evolution curve
in Figure 4 and is **not** the same as `sat['1']` (kernels reached):

- `kernel_diversity` counts unique kernel *hashes* across all test
  executions (including repeated invocations of the same template).
- `sat['1']` counts unique kernel *templates* hit at least once.

The two diverge because a single template may produce multiple hash
variants under different shape/dtype conditions.  The 5-seed mean in
`data_kd_5seed.csv` is also higher than the single-seed sat_pilot value
because it averages across five independent runs.  WhiteFox shows 0 in
`data_kd_5seed.csv` because the WhiteFox replay wrapper does not emit
hash-trackable kernel artifacts in the 5-seed harness; the correct
WhiteFox kernel counts (131/201 reached, 23/34 stressed) are in
`results_sat_pilot/whitefox_seed42.json`.
