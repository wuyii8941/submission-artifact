# DeepOpFuzz — Anonymized Artifact

This is the artifact for the paper *DeepOpFuzz: Operation-Guided Fuzzing for
Deep Learning Compiler Stacks*, anonymized for double-blind review.

## What's in this repository

```
kernelfuzz/         Core fuzzing framework (generation, oracles, kernel tracing)
evaluation/         Scripts for the experiments reported in the paper
  rq1_effectiveness/ RQ1 — index pointing to the 86-bug catalog (no script)
  rq2_efficiency/    RQ2 — historical benchmark (5 seeds × 4 modes × 4h)
  rq3_ablation/      RQ3 — reuses rq2_efficiency runs + BugsInDLLs
    bugsindlls/      External API-level benchmark (Table 4)
  rq4_portability/   RQ4 — TensorFlow / IREE / TVM cross-framework
  baselines/         WhiteFox / NNSmith replay helpers for RQ2
data/
  confirmed_bugs/   Minimized reproducers for the 86 confirmed bugs reported
                    in the paper, grouped by stack (pytorch, tensorflow,
                    xla, iree, tvm). Upstream issue links are anonymized
                    for double-blind review.
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
