# Table II — 4-hour baseline replay on 30 PyTorch/Inductor root causes

This directory reproduces Table II (`tab:pipeline-gap`) in the paper: a
per-method, per-Inductor-stage count of how many of the 30 PT/Inductor
root causes a method actually triggers in a 4-hour run on the bug-bearing
PyTorch version.

The experiment compares DeepOpFuzz against four baselines (Random,
FreeFuzz, NNSmith, WhiteFox) on **the same PT 2.11 environment**, **the
same 4-hour wall-clock budget**, and **the same eager-vs-compiled
differential oracle**. Each method runs with a single seed; this is a
budget-controlled audit, not a statistical comparison.

## Files

| File | Purpose |
|---|---|
| `launch_4h_runs.sh` | Drive all 5 runs in parallel on GPUs 0–4 |
| `bug_classification.py` | Canonical taxonomy of the 30 root causes (`region`, `structural_reqs`) |
| `attribute_table2.py` | Map saved trigger programs → root cause IDs → per-stage counts |
| `generate_table.py` | Emit the LaTeX snippet for the paper |
| `results/table2_attribution.json` | Cached output from our reference run (the numbers in the paper) |
| `results/table2.tex` | Generated LaTeX snippet for inclusion via `\input{...}` |

## Reproducing

1. Set up a PyTorch 2.11.0 + Triton 3.6.0 environment with the optional
   baseline dependencies (NNSmith, `ml_dtypes`, `hydra-core`,
   `z3-solver`, `onnx`). See `INSTALL.md` at the artifact root.

2. Point the runner at that environment and at the FreeFuzz pool:

   ```bash
   export DOF_PT_PYTHON=/path/to/pt211/bin/python
   export KF_FREEFUZZ_POOL=/path/to/freefuzz_pool
   ```

3. Launch:

   ```bash
   cd evaluation/rq1_effectiveness/baseline_replay
   ./launch_4h_runs.sh           # 4h wall-clock; 5 GPUs in parallel
   ```

4. Attribute findings to root causes:

   ```bash
   python attribute_table2.py --runs-root ./runs_4h
   ```

5. Emit the LaTeX snippet:

   ```bash
   python generate_table.py
   ```

The cached JSON at `results/table2_attribution.json` corresponds to our
reference run; `generate_table.py` reads it by default and reproduces
the paper's Table II.

## Attribution methodology

For each saved trigger program (one Python file per fingerprint-deduped
oracle violation), the script extracts source-level features via
operator-name regexes plus large-dim detection on tensor constructors.
A root cause is counted as **hit** iff at least one trigger program
contains all the structural features required to enter the faulty path.

Per-method results report **distinct root causes hit**, not raw trigger
counts; the raw counts are retained in the JSON only as a diagnostic.

A handful of root causes use overrides (`OVERRIDE_REQS` in
`attribute_table2.py`) to drop Inductor-internal flags that never appear
in user source code (e.g. `split_scan` for bug_011 — SplitScan is an
Inductor scheduling decision that any sufficiently large `cumsum` will
trigger). bug_051 (a scheduler prologue typo) is a code-level defect not
triggerable by any test program and is marked unreachable.

Three root causes in `bug_classification.py` (`bug_054`, `bug_065`,
`bug_066`) have `directory=""` because they are campaign-level root
causes with no standalone minimized reproducer bundled in this artifact.
Their triggers were found during the DeepOpFuzz PT campaign and classified
into the 30-root-cause taxonomy; the attributor skips directory lookup for
them and relies solely on the `structural_reqs` field when assigning credit.

## Diagnostic numbers

The reference run produced:

| Method | Trigger files saved | Distinct root causes hit |
|---|---:|---:|
| DeepOpFuzz | 543 | 13 / 30 |
| Random | 404 | 5 / 30 |
| FreeFuzz | 17 | 1 / 30 |
| NNSmith | 0 | 0 / 30 |
| WhiteFox | 2 | 1 / 30 |
| Any baseline (union) | — | 6 / 30 |

NNSmith's saved-trigger count of zero is consistent with its 4-hour
process completing 6,353 model generations and reaching 51 distinct
Triton kernel templates without ever raising the differential oracle —
the same oracle DeepOpFuzz uses. The structural ceiling is its opset:
none of `slice_scatter`, `diagonal_scatter`, `cumsum`, `cumprod`,
`adaptive_avg_pool`, `nested_tensor`, `inference_mode`, complex dtypes,
or `float8` primitives are in NNSmith's auto-detected opset on PT 2.11,
and NNSmith does not synthesize backward graphs. These features cover
17+ of the 30 root causes.

WhiteFox's 2 saved triggers are the two distinct fingerprints among 23
raw oracle hits; one matches bug_011 (`torch.cumsum` over a
large-dim fp16 tensor → Inductor SplitScan path).

FreeFuzz's 17 saved triggers come from its pre-generated API-mutation
pool (`KF_FREEFUZZ_POOL`); only one of them satisfies a root cause's
structural requirements.
