# FreeFuzz 5-Seed Logs (RQ2)

These per-seed logs back the FreeFuzz column in the paper's RQ2 table
(`tab:budget-split`): mean **2.6 ± 0.9 distinct root causes** per run,
with **24 stressed kernels** (≥10 hits) and **366 reached kernels**
(≥1 hit) at the 4 h checkpoint.

| File | Content |
|---|---|
| `freefuzz_seed{42,123,456,789,1024}.json` | per-seed structured run summary (iteration count, finding categories, kernel template list, etc.) |
| `freefuzz_seed*.log` | per-seed raw run log |

Seeds match the other RQ2 baselines (random / nnsmith / whitefox /
targeted), all of which are at the matching `results_*_5seed/` paths in
the development tree.

## How the paper's "2.6" relates to the raw counts

Each JSON exposes several bug-related fields that should not be
confused with the historical-benchmark *root cause* count reported in
the paper:

| JSON field | Meaning | Typical value |
|---|---|---|
| `bugs` | every flagged finding, including duplicates | ~50--70 per seed |
| `bug_fingerprints` | findings deduplicated by op-fingerprint hash | ~19--20 per seed |
| `real_bug_fps` | fingerprints surviving fresh-process rerun + tolerance | ~17--20 per seed |

The paper's **2.6 ± 0.9** is the count of FreeFuzz fingerprints that the
post-hoc root-cause matcher attributes to one of the 30 distinct root
causes underlying the 48-bug historical benchmark. Most of FreeFuzz's
~19 surviving fingerprints are kernel-level signatures that do **not**
map to any of the 30 historical root causes (they are FreeFuzz's own
op-level dedup, not the historical benchmark's). The 2.6 represents
exactly the subset that does.

The matcher itself is not bundled with this artifact because it lives
in the FreeFuzz API-model project (anonymized for review). Reviewers
wishing to re-derive 2.6 from the JSONs above can use the same matching
logic as `evaluation/rq1_effectiveness/baseline_replay/attribute_table2.py`,
which maps a fuzzer's fingerprints onto the 30-root-cause set used in
RQ1; the historical benchmark uses an analogous mapping over its own
30-root-cause set.

## Why FreeFuzz lands so low

FreeFuzz mutates **single API calls**, so the kernels it generates are
short pointwise / extern dispatches that rarely exercise the multi-op
fusion / scheduling / compiled-backward paths where the historical
benchmark's bugs live. The "Kernels stressed (≥10 hits)" column in
`tab:budget-split` (24 vs 137 for \tool) makes the same point at the
kernel level.
