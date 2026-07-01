# RQ1: Effectiveness

RQ1 reports previously unknown bugs found by DeepOpFuzz's two primary
campaigns (PyTorch/Inductor and TensorFlow/XLA) plus the IREE and TVM
portability campaigns. There is no driver script here — the evidence is
the catalog of minimized reproducers under
`../../data/confirmed_bugs/{pytorch,tensorflow,iree,tvm}/`.

| Stack | Counted bugs |
|---|---|
| PyTorch/Inductor | 31 |
| TensorFlow/XLA | 31 |
| IREE | 7 |
| TVM | 17 |
| **Total** | **86** |

The full mapping (Internal ID → root cause → upstream status) lives in
`../../data/confirmed_bugs/BUG_MAPPING.md`.

To re-verify a bug, install the corresponding stack version (listed in
each reproducer's header comment) and run the script:

```bash
cd ../../data/confirmed_bugs/pytorch
python bug_003_slice_scatter_backward/minimal_repro.py
```

## Table II — 4-hour baseline replay

Table II (`tab:pipeline-gap`) compares DeepOpFuzz against four baselines
(Random, FreeFuzz, NNSmith, WhiteFox) under a 4-hour wall-clock budget
on PyTorch 2.11 with a shared eager-vs-compiled differential oracle.
Each method's saved trigger programs are attributed to the 30
PyTorch/Inductor root causes derived from the campaign bugs, and the
table reports distinct root causes hit per Inductor stage.

The driver, attribution script, cached reference results, and the
generated LaTeX snippet live under [`baseline_replay/`](baseline_replay/).
See `baseline_replay/README.md` for the reproduction recipe and the
methodology footnote about NNSmith's structural ceiling.
