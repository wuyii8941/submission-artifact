## [Bug][Relax] `R.isnan`, `R.isinf`, `R.isfinite` crash at build time — CodeGenVM intrinsic not implemented

### Expected behavior

`R.isnan()`, `R.isinf()`, and `R.isfinite()` should compile and produce correct boolean tensor outputs on both CPU and CUDA targets.

### Actual behavior

All three ops crash during `relax.build()` with:

```
InternalError: CodeGenVM cannot handle this intrinsic now:
Op(relax.isnan)
```

This occurs on **both** CPU (llvm) and CUDA targets.

### Environment

- **TVM**: main branch @ commit `0b0afd8dd` (2026-04-25)
- **OS**: Ubuntu 20.04

### Steps to reproduce

```python
import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R

bb = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((4, 8), "float32"))
with bb.function("main", [x]):
    with bb.dataflow():
        out = bb.emit(R.isnan(x))
        gv = bb.emit_output(out)
    bb.emit_func_output(gv)
mod = bb.get()

pipeline = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
mod_l = pipeline(mod)
exe = relax.build(mod_l, target="llvm")  # CRASH
```

Same crash for `R.isinf(x)` and `R.isfinite(x)`.

### Notes

- These ops are registered in the Relax op registry and can be used to construct valid IR modules.
- The PyTorch FX frontend already supports importing `isnan`/`isinf`/`isfinite` (PR <anon>), but the backend code generation was not updated to handle them — so models using these ops can be imported but not compiled.
- Similar CodeGenVM gaps have been reported and fixed before for other ops (e.g., #<anon>, #<anon>).
