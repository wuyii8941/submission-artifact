# [Bug] ONNX Softmax/LogSoftmax/Hardmax frontend ignores opset <13 semantics

> **Status: CONFIRMED BY DEVELOPER — FIXED in nightly (0.24.dev0)**
> GitHub issue: <upstream issue, anonymized>

## Description

TVM's ONNX frontend only implements version 13 semantics for Softmax, LogSoftmax, and Hardmax, causing silently wrong results when importing models with opset ≤12.

- **Opset < 13:** Input is flattened at the specified axis, softmax computed on the last dimension, then reshaped back. Default axis=1.
- **Opset ≥ 13:** Softmax applied directly along the axis without flattening. Default axis=-1.

For a 3D tensor (2,3,4), opset 11 computes softmax across 12 flattened elements, while opset 13 computes across 4 elements — producing drastically different outputs.

## Reproduction

See `bug_004_softmax_opset11.py` for full reproduction script.

## Environment

- TVM: v0.23.0 (bug present), nightly 0.24.dev0 (fixed)
- Python: 3.11
- OS: Linux
