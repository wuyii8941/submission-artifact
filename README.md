# KernelFuzz — Anonymized Artifact

This is the artifact for the paper *KernelFuzz: Operation-Guided Fuzzing for
Deep Learning Compiler Stacks*, anonymized for double-blind review.

## What's in this repository

```
kernelfuzz/         Core fuzzing framework (generation, oracles, kernel tracing)
evaluation/         Scripts for the experiments reported in the paper
  rq1_bugsindlls/   RQ1 external benchmark (BugsInDLLs)
  rq2_benchmark/    RQ2 historical PyTorch benchmark + RQ3 ablation
  rq4_oracle/       Oracle ablation
  rq5_cross_framework/  IREE / TVM / cross-framework portability
  whitefox_baseline/    Replay scripts for the WhiteFox baseline
  nnsmith_baseline/     Replay scripts for the NNSmith baseline
  ...                   See INSTALL.md and REPRODUCE.md
data/
  confirmed_bugs/   Minimized reproducers for the 77 confirmed bugs reported
                    in the paper, grouped by stack (pytorch, tensorflow,
                    xla, iree, tvm). Upstream issue links are anonymized
                    for double-blind review.
scripts/            Convenience launchers
```

## Getting started

See `INSTALL.md` for environment setup and `REPRODUCE.md` for the mapping
from each paper claim (table/figure/RQ answer) to the script that reproduces
it.

## Notes for reviewers

- All paths referencing `${ARTIFACT_ROOT}` should be set to the location
  of this repository on your machine (or any writable directory used as
  the working root).
- External dependencies (PyTorch, IREE, TVM, BugsInDLLs corpus, WhiteFox
  pre-generated tests) are described in `INSTALL.md`. Download URLs point
  to the official upstream projects.
- Upstream bug-tracker issue links are anonymized for review and will be
  restored in the camera-ready version.
