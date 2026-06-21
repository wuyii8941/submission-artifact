# Fuzzer Coverage Audit (86 bugs)

This audit cross-references each of the 86 confirmed bugs (see
`../data/confirmed_bugs/BUG_MAPPING.md`) against the operators /
templates present in the current fuzzer implementations under
`../kernelfuzz/`. Raw per-bug results are in
[`coverage_audit.csv`](coverage_audit.csv).

## Summary

| Stack | Reachable | Unreachable | Out-of-scope | Total |
|---|---:|---:|---:|---:|
| PyTorch | 30 | 1 (static-analysis) | 0 | 31 |
| TensorFlow / XLA | 28 | 3 (ops not in current pool) | 0 | 31 |
| IREE | 0 | 0 | 7 (separate harness) | 7 |
| TVM | 0 | 0 | 17 (separate harness) | 17 |
| **Total** | **58** | **4** | **24** | **86** |

## Interpretation

### PyTorch (30/31 reachable)

The PyTorch fuzzer's template registry
(`kernelfuzz/templates/registry.py`, 365 templates) and random
generator (`kernelfuzz/random_generator.py`) cover an operator for
every confirmed PT bug except one:

- `bug_appendix_pad_mm_getitem_dead_code` — this is **not a runtime
  bug**. It is a static-analysis finding (an `isinstance` check that
  is unreachable dead code in `pad_mm`); the reproducer in
  `data/confirmed_bugs/pytorch/bug_appendix_pad_mm_getitem_dead_code/`
  is explicitly tagged as a Class C "static demonstration" in
  `BUG_MAPPING.md > Reproducer Provenance Audit`. By design no fuzzer
  reaches it.

### TensorFlow / XLA (28/31 reachable)

The TF graph generator (`kernelfuzz/tensorflow/graph_gen.py`) covers
~126 distinct `tf.*` ops. Three confirmed TF bugs use ops that are
**not in the current op pool**:

- `tf_atan2_subnormal_nan` — `tf.math.atan2` is absent from the pool.
- `tf_bessel_i1_gradient_zero` — `tf.math.bessel_i0e` is present, but
  the sibling `tf.math.bessel_i1` (the buggy op) is not.
- `tf_erfinv_eager_precision` — `tf.math.erfinv` is absent.
- `tf_det_gradient_singular_crash` — `tf.linalg.det` is absent.

These four ops were exercised in the original RQ1 campaign that
produced the bug but were dropped from the trimmed op pool shipped with
the artifact (the original campaign used a larger pool). The
reproducers under `data/confirmed_bugs/tensorflow/` are sufficient to
verify the bugs themselves; re-discovering them from scratch via the
artifact fuzzer would require adding these four ops back to the
`graph_gen` pool.

### IREE (7) and TVM (17) — out of scope for the main fuzzer

The 24 IREE/TVM bugs were not found by `kernelfuzz/fuzzer.py` (which
targets PyTorch) or `kernelfuzz/tensorflow/fuzzer.py`. They were
discovered via:

- **IREE:** an ONNX-frontend harness that compiles randomly-generated
  ONNX graphs through IREE and runs each via ONNX Runtime as the
  reference oracle.
- **TVM:** a Relax-frontend harness plus targeted manual testing of
  DLight schedule rules.

Both harnesses follow the same operation-guided methodology described
in the paper but live in stack-specific drivers (see the per-bug
reproducer headers under `data/confirmed_bugs/{iree,tvm}/` for the
exact entry point used for each bug). The minimized reproducers are
included in the artifact; reviewers can re-verify any IREE/TVM bug
with the stack version listed in the reproducer header.

## Smoke-test status

| Fuzzer | Import | Run | Notes |
|---|---|---|---|
| `kernelfuzz/fuzzer.py` (PyTorch) | OK | OK | Requires `KERNELFUZZ_FREEFUZZ_SRC` (see `INSTALL.md`). Smoke-run with `--v4 --max-iterations 5` populates `<output>/graph-oracle/{success,fail,crash,...}/` and the Inductor cache. |
| `kernelfuzz/tensorflow/fuzzer.py` (TF) | parses cleanly | not runnable on the dev host | All TF installs on the audit host had a broken `tensorflow.python.eager.core` import; the fuzzer module itself parses without errors. A reviewer with a working `tf-nightly` install should be able to run it. |
