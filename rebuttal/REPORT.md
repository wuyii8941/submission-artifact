# Rebuttal results

Scripts and results of the experiments in our response. Each directory holds the script that computes its numbers and
the resulting JSON; `A3_oatest_tvm/`, `B3_filtering/`, `target_extraction/` and `C3_bug_ablation/` also hold the
scripts that run the experiments. Run records are not included.

| Answer | Directory | Figure |
|---|---|---|
| A-Q1 | `A1_feedback/` | R1 |
| A-Q2, C-Q3 | `bug_conditions.json` | — |
| A-Q2 | `tvm_bug_stages.json` | — |
| A-Q3 | `A3_oatest_tvm/` | R2 |
| B-Q1, B-Q2, C-Q1 | `target_extraction/` | — |
| B-Q3 | `B3_filtering/`, `bug_fix_status.json` | R3 |
| C-Q2 | `C2_coverage/` | R4 |
| C-Q3 | `C3_components/`, `C3_bug_ablation/` | — |

## A-Q1: target-hit feedback (`A1_feedback/results/summary.json`)

PyTorch 2.13 nightly, Tesla T4, 43 targets, seeds 42 / 123 / 456, 4 h per run. Feedback samples a target with weight
1/(1+hits); uniform selection gives every target the same share of attempts. Targets, generator, matcher and oracles
are the same in both arms.

- Both arms hit all 43 targets in every seed.
- Least-hit quarter of the targets (L25): 602.5 (feedback) and 530.2 (uniform) hits on average; at equal program counts
  +13.9% (+12.1%, +15.4%, +14.2% per seed).
- Time until every target has a confirmed hit: 70.5 s (feedback, 110–120 tests) and 139.0 s (uniform, 250–400 tests).

## A-Q2, C-Q3: condition-specific and backward-only bugs (`bug_conditions.json`)

One row per bug of the two primary campaigns, with the criteria. Of 62 bugs, 18 fail only at a specific shape, dtype or
layout and 19 show only in the backward pass. Of the 31 PyTorch/Inductor bugs, 12 fail only at a specific shape, dtype
or layout.

## A-Q2: stages of the TVM bugs (`tvm_bug_stages.json`)

8 of the 17 TVM bugs lie in the ONNX frontend, the model-loading stage; the other 9 lie in Relax passes (4), lowering to
TIR (3), GPU scheduling (1) and code generation (1). Each row gives the location and the root cause. OATest's evaluation
reports that the three bugs it missed all occur in the model loading stage (Section 4.3 of its paper; quoted in the file).

## A-Q3: OATest's TVM generator (`A3_oatest_tvm/results/a3_numbers.json`)

TVM `292ecfd`, 35 passes, seeds 42 / 123 / 456, 2 h per run. Arms: OATest's pattern choice (`pattern_uniform`),
target-aware pass choice without matcher and feedback (`pass_uniform`), and the same with them (`hit_feedback`). A hit
counts only when the target pass rewrote the IR.

- Passes hit at least ten times: 24.3, 32.0 and 33.0; at least once: 31.3, 34.0 and 35.0.
- RealizeVDevice: 1,045 programs generated for it by `hit_feedback`, 12 rewritten.

Runs: `setup_oatest.sh` (OATest at commit `7c28fe9f71` with `oatest_splice_graph.patch`), then `run_formal.sh` with
`TVM_HOME` set to a CPU build of TVM `292ecfd`; the 35 passes are in `targets_primary.json`.

## B-Q1, B-Q2, C-Q1: target derivation (`target_extraction/`)

`build_targets.sh` runs every step (PyTorch 2.13 nightly with CUDA and `expecttest`, TensorFlow 2.22 nightly). The hand-written rules
are in `pt_apply_criteria.py` and `pt_select_entries.py` (113 lines, PyTorch/Inductor) and in `tf_apply_criteria.py` and
`tf_rules.py` (92 lines, TF/XLA). Results
(`results/REPORT.json`, Tesla T4):

- The derived Inductor families equal the 8 submitted families; all 44 submitted path entries lie in the derived pools.
- The kernel choice Inductor records and the emitted kernels agree in 708 of 708 checked compilations.
- Persistent reduction switches at 1024 elements (`results/pt_conditions.json`); fusing operator pairs are in
  `results/pt_composition.json`.
- 43 target states on the Tesla T4 (`results/pt_target_states.json`).
- 396 TF/XLA target states (`results/tf_target_states.json`): the 480 combinations of the derived families, dtype
  classes, execution modes and input regimes, kept where the generator's reachability rule in the artifact holds.
- TF/XLA dtype classes, execution modes and input regimes equal the artifact's.

## B-Q3: candidate filtering (`B3_filtering/results/filter_audit.json`, `bug_fix_status.json`)

All 86 reported bugs are confirmed; 50 are fixed as of 2026-09-24 (43 at submission): PyTorch 23, TF/XLA 9, TVM 16,
IREE 2. `bug_fix_status.json` lists every bug and the counting rule.

Historical benchmark, PyTorch 2.2.0, seeds 42 / 123 / 456, 4 h per run. Means over the runs: 987.0 candidate failures,
401.0 deduplicated candidates, 280.3 labeled false positives by the benchmark classifier (DeepOpFuzz); 2,436.7,
1,835.3 and 1,689.0 (Random). Deduplicated candidates labeled false positives: 69.9% and 92.0%. Of DeepOpFuzz's false
positives, 93.9% are fp16/bf16 numerical differences.

Runs: `run.sh` runs the artifact's historical-benchmark worker (`evaluation/rq2_efficiency/_benchmark_worker.py`) for
DeepOpFuzz and Random with PyTorch 2.2.0; `filtering_audit.py` reads its output (`B3_DATA`).

## C-Q2: coverage under matched budgets (`C2_coverage/results/c2_numbers.json`)

PyTorch 2.13 nightly, Tesla T4, seeds 42 / 123 / 456, 1 h per run. Means for DeepOpFuzz, random ops with the same
boundary values, and Random:

- Operation targets reached: 100%, 88.4%, 71.3%.
- Branch coverage of the kernel-selection code: 23.9%, 23.9%, 22.7% (`results/coverage_per_file.json`).
- Programs at a switch point: 22.0%, 2.7%, 0.05%.

## C-Q3: generation components (`C3_components/results/results.json`, `C3_bug_ablation/results/results.json`)

Historical benchmark, PyTorch 2.2.0 + Triton 2.2.0, Tesla T4, seeds 42 / 123 / 456, same program counts, all eight
combinations of path entries, conditions and composition constraints. Benchmark bugs: 6 / 6 / 5 with none, 14 / 13 / 12
with all three. Main effects: path entries 6.3, conditions 0.5, composition constraints 1.0.

Confirmed PyTorch bugs, each reproducer run as is, with the whole captured graph compiled, and with each operator
compiled on its own (`run_mode.sh`, PyTorch 2.11–2.13): 24 of 31 reproduce; of the 21 that run per operator, 11 need
their operators compiled together.
