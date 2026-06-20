"""Bug 018: scatter_elements AND scatter_nd CUDA compilation crashes.

Status: CONFIRMED BUG (NEW)
TVM version: latest main (commit ca68bef6b, 2026-04-20)

Description:
  R.scatter_elements() and R.scatter_nd() always fail when compiled for CUDA.
  Works correctly on CPU (llvm). gather_elements and gather_nd work on CUDA.

Error:
  RuntimeError: Memory verification failed with the following errors:
    Variable `out_buf` is directly accessed by host memory
    (it is not contained in a thread environment or in the function arguments)

Root cause:
  The TIR lowering of scatter ops generates a kernel where `out_buf` is
  accessed outside of any GPU thread binding (threadIdx/blockIdx). The memory
  verifier correctly rejects this, but it means scatter ops are completely
  unusable on CUDA.

Impact:
  Any Relax model using scatter_elements or scatter_nd cannot be compiled for
  CUDA. This blocks ONNX models that use ScatterElements/ScatterND ops.

Trigger: ANY shape, ANY axis — 100% reproducible on both scatter ops.
"""
import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R
from tvm.s_tir import dlight


def _make_var(name, shape, dtype="float32"):
    return relax.Var(name, relax.TensorStructInfo(shape, dtype))


# Minimal reproduction
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
print("CPU build...")
pipeline_cpu = tvm.ir.transform.Sequential([relax.transform.LegalizeOps()])
mod_cpu = pipeline_cpu(mod)
exe_cpu = relax.build(mod_cpu, target="llvm")
vm_cpu = relax.VirtualMachine(exe_cpu, device=tvm.cpu())

indices = np.array([[0, 1, 2, 3, 0, 1, 2, 3],
                    [3, 2, 1, 0, 3, 2, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [2, 2, 3, 3, 0, 0, 1, 1]], dtype="int64")
data = np.ones(shape, dtype="float32")
updates = np.full(shape, 2.0, dtype="float32")

cpu_out = vm_cpu["main"](
    tvm.runtime.tensor(data, device=tvm.cpu()),
    tvm.runtime.tensor(indices, device=tvm.cpu()),
    tvm.runtime.tensor(updates, device=tvm.cpu()),
).numpy()
print(f"CPU output:\n{cpu_out}")

# CUDA crashes
print("\nCUDA build...")
try:
    pipeline_cuda = tvm.ir.transform.Sequential([
        relax.transform.LegalizeOps(),
        dlight.ApplyDefaultSchedule(dlight.gpu.Fallback()),
    ])
    with tvm.target.Target("cuda"):
        mod_cuda = pipeline_cuda(mod)
    exe_cuda = relax.build(mod_cuda, target="cuda")
    print("CUDA build succeeded (unexpected)")
except RuntimeError as e:
    print(f"CUDA build CRASHED (expected):")
    print(f"  {e}")
