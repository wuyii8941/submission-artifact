# [Bug] ReorderPermuteDimsAfterConcat crashes with negative concat axis

**Filed:** [upstream issue #19575 (anonymized)] — Closed / Fixed (PR [#19588](<upstream PR, anonymized for review>))

## Description

The `ReorderPermuteDimsAfterConcat` pass crashes with `IndexError: Index -1 out of bounds 4` when the concat operation uses a negative axis value (e.g., `axis=-1`). The pass reads the axis value directly without normalizing negative indices.

Using a positive axis value (e.g., `axis=3` instead of `axis=-1`) works correctly.

## Reproducer

```python
import tvm
from tvm import relax
import tvm.relax.op as R

bb = relax.BlockBuilder()
x = relax.Var('x', relax.TensorStructInfo((1, 4, 8, 8), 'float32'))
y = relax.Var('y', relax.TensorStructInfo((1, 4, 8, 8), 'float32'))
with bb.function('main', [x, y]):
    with bb.dataflow():
        xt = bb.emit(R.permute_dims(x, [0, 2, 3, 1]))
        yt = bb.emit(R.permute_dims(y, [0, 2, 3, 1]))
        out = bb.emit_output(R.concat([xt, yt], axis=-1))  # negative axis
    bb.emit_func_output(out)
mod = bb.finalize()

# This crashes:
pipeline = tvm.ir.transform.Sequential([
    relax.transform.ReorderPermuteDimsAfterConcat(),
    relax.transform.LegalizeOps()
])
mod_l = pipeline(mod)  # IndexError: Index -1 out of bounds 4
```

## Error

```
IndexError: Index -1 out of bounds 4
```

at `tvm/3rdparty/tvm-ffi/include/tvm/ffi/container/seq_base.h` line 86.

## Expected behavior

The pass should normalize negative axis values (e.g., `axis=-1` → `axis=ndim-1`) before using them as array indices.

## Workaround

Use positive axis values in concat operations.

## Environment

- TVM version: 0.24.dev0 (commit 0b0afd8dd, 2026-04-24)
