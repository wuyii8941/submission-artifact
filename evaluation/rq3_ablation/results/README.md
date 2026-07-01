# RQ3 Ablation Results

## `ablation_rootcause_summary.json`

Source of the four-cell ablation table (paper Table IV / `tab:rq3-contribution`).
Eight conditions are reported (two PyTorch versions × four ablations):

| Key | Mode | Oracle | Paper row | n_root_causes (220 / nightly) |
|---|---|---|---|---|
| `targeted_full` | operation-guided | full differential | **Full system** (paper: 11--15) | 16 / 11 |
| `random_full` | random | full differential | **w/o operation guidance** (paper: 8) | 8 / 8 |
| `targeted_fwd_only` | operation-guided | forward-only differential | (see derivation below) | 16 / 13 |
| `random_fwd_only` | random | forward-only differential | (see derivation below) | 8 / 7 |

The two `*_fwd_only` runs use a forward-output-only differential oracle:
they still compare eager vs.\ compiled forward outputs but skip the
backward-gradient and configuration-differential checks that the full
oracle performs.

### Derivation of the paper's "crash-only" rows

Paper Table IV reports **0** for both *w/o full oracle* (crash-only) and
*w/o both* (random + crash-only). These do not have dedicated runs in
`ablation_rootcause_summary.json`, but the numbers follow directly from
how the historical benchmark is constructed:

> *"A bug is included only if its minimized reproducer triggers a silent
> eager-vs-compiled correctness failure."* — paper §5.1 / `evaluation.tex`.

Every one of the 48 reproducible bugs in `ground_truth_bugs.csv` is
therefore a silent wrong-output bug — none produces an uncaught
exception, assertion failure, or other crash signal. Its categories are
`Graph Caching` (3), `Graph Semantic Capturing` (7), `In-Place Operation
Handling` (9), `Low-level Code Generation` (18), `Memory Layout
Conflict` (3), and `Operator Transformation` (8); none of these
categories includes crash-style failures.

A crash-only oracle (one that flags only exceptions, runtime errors, or
generated-code compilation failures) thus catches **0** of these bugs by
construction. The corresponding *w/o both* cell (random generation +
crash-only oracle) is **0** for the same reason: regardless of generation
strategy, a crash-only oracle cannot detect silent correctness bugs.

### Notes on the targeted-mode numbers

- The "Full system" row in the paper is written as a min--max range
  (11--15), reflecting the per-seed spread across five fixed seeds (the
  same seeds used in RQ2). The two snapshots in this file (16 on PT 2.2.0,
  11 on PT 2.13 nightly) provide complementary evidence spanning versions.
- The `targeted_fwd_only` numbers (16 / 13) show that dropping
  gradient and configuration checks alone does not yet floor detection
  on this benchmark: many bugs are caught by forward-output differential
  comparison only. The drop to 0 occurs only when the remaining
  differential signal is removed (crash-only).

Per-run JSON / log / ckpt files for these conditions live at
`rq2_benchmark/results_ablation_220/` and `results_ablation_nightly/`
in the development tree and are not bundled here to keep the
artifact size manageable.
