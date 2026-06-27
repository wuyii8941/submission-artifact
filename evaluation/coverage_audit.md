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
| TensorFlow / XLA | 31 | 0 | 0 | 31 |
| IREE | 0 | 0 | 7 (separate harness) | 7 |
| TVM | 0 | 0 | 17 (separate harness) | 17 |
| **Total** | **61** | **1** | **24** | **86** |

## Interpretation

### PyTorch (30/31 reachable)

The PyTorch fuzzer's template registry
(`kernelfuzz/templates/registry.py`, 365 templates) and random
generator (`kernelfuzz/random_generator.py`) cover an operator for
every confirmed PT bug except one:

- `bug_030_pad_mm_getitem_dead_code` — this is **not a runtime
  bug**. It is a static-analysis finding (an `isinstance` check that
  is unreachable dead code in `pad_mm`); the reproducer in
  `data/confirmed_bugs/pytorch/bug_030_pad_mm_getitem_dead_code/`
  is explicitly tagged as a Class C "static demonstration" in
  `BUG_MAPPING.md > Reproducer Provenance Audit`. By design no fuzzer
  reaches it.

### TensorFlow / XLA (31/31 reachable)

The TF graph generator (`kernelfuzz/tensorflow/graph_gen.py`) covers
~130 distinct `tf.*` ops, including the four bug-trigger ops that were
originally exercised during the RQ1 campaign but had been dropped from
the trimmed pool: `tf.math.atan2`, `tf.math.bessel_i1`,
`tf.math.erfinv`, and `tf.linalg.det`. These now live alongside the
other bug-specific operators in `BUGSPEC_OPS` (entries 13-16). With
the four ops restored, every confirmed TF bug is reachable by the
released fuzzer.

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
