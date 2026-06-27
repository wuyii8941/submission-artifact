# `torch.compile` crashes on `F.layer_norm(NestedTensor)` — `InternalTorchDynamoError` in guard building

**Issue:** <upstream issue, anonymized for review>
**Confirmation:** <upstream issue, anonymized for review>

## Description

`torch.compile` fails when applying layer normalization to jagged NestedTensors with gradient tracking enabled. The error occurs during the guard-building phase (before kernel execution) and affects all backends (inductor, aot_eager, eager).

**Root Cause:** When tracing operations between a jagged NestedTensor and its keepdim reduction, a symbolic variable is introduced that lacks a corresponding source entry. The guard builder then attempts to access an empty list, causing an `IndexError`.

**Affected Patterns:**
- Binary operations: `nt + nt.mean(-1, keepdim=True)`
- `F.layer_norm(nt, [D])`
- `nn.LayerNorm(D)(nt)`
- Manual RMSNorm implementations

## Status

- Filed: 2026-05-02
- Confirmed: 2026-05-04 (developer comment)
- Labels: bot-triaged, has-workaround, module:crash, module:dynamic-shapes, module:guards, module:nestedtensor, module:norms-and-normalization, oncall:pt2
