# Coverage re-experiment repository manifest

This branch contains the implementation, frozen metric/domain definitions,
formal protocol, per-arm summaries, aggregate analysis, and final independent
audit for `formal_effect_run_v1`.

Included formal evidence:

- `formal_effect_run_v1/protocol.json`, `suite.json`, `analysis.json`, and
  `per_seed.csv`;
- all 24 `seed_*/summary.json` files;
- frozen source snapshots and top-level integrity audits;
- `final_independent_audit_v1.json` and `FINAL_RESULT.md`.

Not committed:

- generated candidate graph files;
- per-request source traces and compiler artifacts;
- Inductor, Triton, and CUDA caches;
- stdout/stderr files below individual seed directories.

Those raw local artifacts contain about 4.3 GB and are reproducible from the
frozen protocol. They remain in the original experiment workspace. The
aggregate analysis can be recomputed from the committed protocol, suite, and
24 arm summaries with:

```bash
python evaluation/route_a/research_20261106/coverage_reexperiment_v1/analyze_experiment.py \
  evaluation/route_a/research_20261106/coverage_reexperiment_v1/formal_effect_run_v1
```

The full 99,497-check topology/source-evidence audit requires the omitted raw
candidate and execution files; its immutable result is committed as
`formal_effect_run_v1/final_independent_audit_v1.json`.
