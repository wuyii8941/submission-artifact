# RQ1: Effectiveness

RQ1 reports previously unknown bugs found by DeepOpFuzz's two primary
campaigns (PyTorch/Inductor and TensorFlow/XLA) plus the IREE and TVM
portability campaigns. There is no driver script here — the evidence is
the catalog of minimized reproducers under
`../../data/confirmed_bugs/{pytorch,tensorflow,xla,iree,tvm}/`.

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
python bug_007_slice_scatter_backward/minimal_repro.py
```
