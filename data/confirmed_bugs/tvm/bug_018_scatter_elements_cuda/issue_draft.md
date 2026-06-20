## [Bug][Relax] `scatter_elements` and `scatter_nd` fail to compile for CUDA target

### Expected behavior

`R.scatter_elements()` and `R.scatter_nd()` should compile and run correctly on the CUDA target, as they do on CPU (llvm).

### Actual behavior

Both ops crash during `relax.build()` with:

```
RuntimeError: Memory verification failed with the following errors:
    Variable `out_buf` is directly accessed by host memory
    (it is not contained in a thread environment or in the function arguments.
    ...
  Did you forget to bind?
```

The generated TIR for these ops uses `T.parallel` loops without any GPU thread binding (`blockIdx`/`threadIdx`), so the memory verifier correctly rejects them. This means **the TIR legalization for scatter ops does not produce GPU-compatible code**.

CPU (llvm) works correctly for all tested shapes and axes.

### Environment

- **TVM**: main branch @ commit `0b0afd8dd` (2026-04-25)
- **OS**: Ubuntu 20.04, CUDA 12.2
- **GPU**: NVIDIA GPU (sm_75)

### Steps to reproduce

```python
import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R
from tvm.s_tir import dlight


def _make_var(name, shape, dtype="float32"):
    return relax.Var(name, relax.TensorStructInfo(shape, dtype))


shape = (4, 8)
axis = 0

bb = relax.BlockBuilder()
x = _make_var("x", shape, "float32")
idx = _make_var("idx", shape, "int64")
upd = _make_var("upd", shape, "float32")
with bb.function("main", [x, idx, upd]):
    with bb.dataflow():
        out = bb.emit(R.scatter_elements(x, idx, upd, axis=axis))
        gv = bb.emit_output(out)
    bb.emit_func_output(gv)
mod = bb.get()

# CPU works
pipeline_cpu = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
mod_cpu = pipeline_cpu(mod)
exe_cpu = relax.build(mod_cpu, target="llvm")  # OK

# CUDA crashes
pipeline_cuda = tvm.ir.transform.Sequential([
    relax.transform.LegalizeOps(),
    dlight.ApplyDefaultSchedule(dlight.gpu.Fallback()),
])
with tvm.target.Target("cuda"):
    mod_cuda = pipeline_cuda(mod)
exe_cuda = relax.build(mod_cuda, target="cuda")  # CRASH
```

### Notes

- Tested across 10 shape/axis combinations — **100% failure rate on CUDA**, 100% success on CPU.
- `R.scatter_nd()` has the same issue.
- `R.gather_elements()` and `R.gather_nd()` compile and run correctly on CUDA.
- The generated TIR shows the scatter loop uses `T.parallel` without thread binding, which is the root cause. The legalization likely needs a GPU-aware implementation similar to what `gather_elements` has.
