"""Bug 001: DLight GEMV scheduler crashes with KeyError on auto-detected CUDA target.

TVM version: 0.23.0
GPU: Tesla T4 (sm_75)
Status: CONFIRMED BUG

Description:
  The DLight GEMV schedule rule (tvm/dlight/gpu/gemv.py:161) accesses
  target.max_shared_memory_per_block unconditionally, but the auto-detected
  CUDA target does NOT include this attribute.

Root cause:
  tvm/target/target.py:217 (max_shared_memory_per_block property) does
  self.attrs["max_shared_memory_per_block"] without checking if key exists.

  The auto-detected target has only: thread_warp_size, max_num_threads, arch.
  Missing: max_shared_memory_per_block.

Trigger:
  ANY matmul where N=1 (matrix-vector multiply) when using GEMV schedule rule.
  48/48 tested shapes crash: all M>=4, all K, N=1.
  N>=2 does NOT trigger (GEMV rule doesn't apply).

Impact:
  Users cannot use DLight's full GPU scheduling for any GEMV operation.
  This blocks optimal performance for matrix-vector products.

Fix suggestion:
  Either populate max_shared_memory_per_block in auto-detection,
  or add a fallback in gemv.py like:
    max_smem = getattr(target, 'max_shared_memory_per_block', 49152)  # T4 default
"""

import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as rop
import tvm.dlight

# Minimal reproduction
bb = relax.BlockBuilder()
a = relax.Var("a", relax.TensorStructInfo((257, 512), "float32"))
b = relax.Var("b", relax.TensorStructInfo((512, 1), "float32"))

with bb.function("main", [a, b]):
    with bb.dataflow():
        out = bb.emit(rop.matmul(a, b))
        gv = bb.emit_output(out)
    bb.emit_func_output(gv)

mod = bb.finalize()

# This triggers the bug
pipeline = tvm.ir.transform.Sequential([
    relax.transform.LegalizeOps(),
    tvm.dlight.ApplyDefaultSchedule(
        tvm.dlight.gpu.Matmul(),
        tvm.dlight.gpu.GEMV(),      # <-- this rule triggers the crash
        tvm.dlight.gpu.Fallback(),
    ),
])

try:
    with tvm.target.Target("cuda"):
        mod_sched = pipeline(mod)
    print("BUG NOT REPRODUCED (may be fixed)")
except KeyError as e:
    print(f"BUG REPRODUCED: {e}")
    print(f"Target attrs: {list(tvm.target.Target('cuda').attrs.keys())}")
    print("max_shared_memory_per_block is missing from auto-detected target")
