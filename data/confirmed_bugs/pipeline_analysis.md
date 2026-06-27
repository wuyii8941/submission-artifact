# Compiler-Pipeline Analysis of 27 Bugs

## Official architecture references

### PyTorch compiler pipeline
- [PyTorch 2 paper (ASPLOS 2024) - Figure 1 is the authoritative pipeline diagram](https://docs.pytorch.org/assets/pytorch2-2.pdf)
- [PyTorch 2.x official landing page - pipeline overview](https://pytorch.org/get-started/pytorch-2-x/)
- [Dynamo Overview - guard/graph-break flow](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)
- [Dynamo Deep Dive - internal implementation details](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.compile Programming Model - how the layers connect](https://docs.pytorch.org/docs/stable/compile/programming_model.html)
- [TorchInductor design document](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)

### TensorFlow / XLA compiler pipeline
- [XLA Architecture - compilation-stage diagram (StableHLO -> HLO -> LLVM IR -> binary)](https://openxla.org/xla/architecture)
- [XLA:GPU Architecture Overview - GPU-backend compilation flow](https://openxla.org/xla/gpu_architecture)
- [TF-to-XLA Bridge (tf2xla) - converting a TF graph to HLO](https://openxla.org/xla/tf2xla)
- [Grappler graph-optimisation official guide - description of each pass](https://www.tensorflow.org/guide/graph_optimization)
- [jit_compile=True tutorial](https://openxla.org/xla/tf2xla/tutorials/jit_compile)
- [List of HLO passes](https://openxla.org/xla/hlo_passes)

---

## 1. PyTorch / Inductor (10 bugs)

### Pipeline structure

```
User Python code
    |
    v
+--------------------------------------------------+
|  TorchDynamo                                     |
|  CPython frame hook -> bytecode rewriting -> FX  |
|  Output: forward-only FX graph + guards          |
+----------------------+---------------------------+
                       |
                       v
+--------------------------------------------------+
|  AOTAutograd                                     |
|  Trace backward graph + meta/fake kernel stride  |
|  Output: forward graph + backward graph          |
|          + (shape, stride, dtype) per tensor     |
+----------------------+---------------------------+
                       |
                       v
+--------------------------------------------------+
|  Inductor                                        |
|  +-----------------------------------------------+
|  | Lowering: ATen ops -> IR (Pointwise/Red/Scan) |
|  +-----------------------------------------------+
|  | Fusion: merge IR nodes into a single kernel   |
|  +-----------------------------------------------+
|  | Scheduling: pick the kernel template          |
|  |  (persistent_red / non-persistent / SplitScan)|
|  +-----------------------------------------------+
|  | Codegen: IR -> Triton Python source           |
|  |  (compute stride literals, build iteration    |
|  |   space, emit load/store)                     |
|  +-----------------------------------------------+
+----------------------+---------------------------+
                       |
                       v   Triton .py source
+--------------------------------------------------+
|  Triton Compiler (OpenAI)                        |
|  Triton IR -> MLIR -> LLVM IR -> PTX -> cubin    |
|  Not responsible for code correctness; only      |
|  compilation + execution.                        |
+--------------------------------------------------+
```


### The 10 bugs attributed by pipeline layer

#### L1: Inductor Fusion Codegen (5 bugs) - the emitted Triton code is wrong

| Bug | Issue | Error mechanism | Type |
|-----|-------|---------|------|
| bug_006 | [#<anon>](<upstream issue, anonymized>) | After fusing `adaptive_avg_pool2d + flatten + sum`, Inductor uses `ceil(3185/32)*32 = 3200` as the batch stride literal in the Triton source, but the buffer's real stride is `3185`. Every load with batch >= 1 is offset by 15 elements. | Silent |
| bug_007 | [#<anon>](<upstream issue, anonymized>) | The AOTAutograd meta kernel annotates `slice_scatter` with stride-0 -> Inductor trusts this and collapses the iteration space to 132 elements (should be 1716) -> dim-1 coordinate is hard-coded to 0 -> boundary checks are constant-folded by the Triton compiler -> `x.grad` is all zero. | Silent |
| bug_013 | [#<anon>](<upstream issue, anonymized>) | Same root-cause family as bug_007. Backward of `slice_scatter` exhibits different error patterns for different slice positions. | Silent |
| bug_022 | [#<anon>](<upstream issue, anonymized>) | In the backward of `diagonal_scatter(...).sum()`, Inductor optimises `clone(expand(scalar,[N,N]))` into a 1-element buffer with stride-(0,0). A subsequent `diagonal_scatter` then writes zero into that element, so the entire N x N gradient becomes zero. | Silent |
| bug_004 | [#<anon>](<upstream issue, anonymized>) | The `fp32 -> bf16 -> fp32` round-trip cast is eliminated by fusion as a no-op, so the emitted Triton kernel performs no dtype conversion and the expected precision truncation is missing. | Silent |

**Root-cause patterns:**

1. **Broken stride trust chain** (bug_007/013/022): the AOTAutograd meta kernel returns an incorrect stride -> Inductor trusts it -> the emitted kernel has the wrong iteration space or indexing.
2. **Wrong numerical literal** (bug_006): codegen uses `ceil_to_warp_size` instead of the real value when computing the stride of view + reduction.
3. **Fusion semantics violation** (bug_004): the fusion pass eliminates an op with observable side effects.

#### L2: AOTAutograd (2 bugs) - autograd tracing goes wrong

| Bug | Issue | Error mechanism | Type |
|-----|-------|---------|------|
| bug_019 | [#<anon>](<upstream issue, anonymized>) | Under `autograd.grad(create_graph=True)`, AOTAutograd silently loses `requires_grad=True` and `grad_fn` on ops such as `mm`/`relu` -> higher-order backward fails. Also reproduces under the `aot_eager` backend, so it is not Inductor-specific. | Silent |
| bug_020 | [#<anon>](<upstream issue, anonymized>) | `inference_mode()` + 3D+ matmul + `torch.compile` -> a weakref to `UntypedStorage` in AOTAutograd's functionalisation phase cannot be resolved -> crash. | Crash |

#### L3: Inductor Scheduling (2 bugs) - kernel template selection / initialisation goes wrong

| Bug | Issue | Error mechanism | Type |
|-----|-------|---------|------|
| bug_011 | [#<anon>](<upstream issue, anonymized>) | When scan dim >= 129, Inductor switches to SplitScan codegen -> `tensor_dim=None` is used as a list index -> crash. | Crash |
| bug_015 | (not submitted) | `associative_scan` + `dynamic=True` + ndim >= 2 -> lifted arguments are unsupported -> codegen crash. | Crash |

#### L4: Dynamo (1 bug) - graph capture / guard construction goes wrong

| Bug | Issue | Error mechanism | Type |
|-----|-------|---------|------|
| bug_023 | [#<anon>](<upstream issue, anonymized>) | `F.layer_norm(NestedTensor)` -> the symbolic variable has no source entry -> the guard builder raises `IndexError`. All backends crash. | Crash |

### PyTorch summary diagram

```
+---------------------------------------------+
|  Dynamo (Graph Capture)                     | <- 1 bug (guard crash)
+---------------------------------------------+
|  AOTAutograd (Backward + Meta Kernels)      | <- 2 bugs (metadata/weakref)
+---------------------------------------------+
|  Inductor - Scheduling                      | <- 2 bugs (SplitScan/dynamic)
+---------------------------------------------+
|  Inductor - Fusion + Codegen <- HIGHEST     | <- 5 bugs (stride/index/dtype)
|                              BUG DENSITY    |
+---------------------------------------------+
|  Triton Compiler                            | <- 0 bugs
+---------------------------------------------+

Crux: code-generation errors. Inductor introduces errors when
translating high-level IR semantics into the numerical literals
of Triton source. Triton faithfully compiles and executes that
(incorrect) code.
```

---

## 2. TensorFlow (7 bugs)

### TF's problems differ fundamentally from PyTorch's

PyTorch bugs are all of a **single kind**: code-generation errors. The 7 TF bugs span **three distinct kinds of problems**:

| Problem type | Nature | Present in PyTorch? |
|---------|------|----------------|
| **Eager-kernel numerical defect** | Unrelated to compilation; the underlying C++ kernel uses an incorrect mathematical implementation. | No (PyTorch's eager kernels are correct in our experiments). |
| **Graph-optimisation semantics violation** | A compiler optimisation pass changes program semantics. | Similar to Inductor's fusion bugs, but at a different layer. |
| **Eager <-> XLA semantic contract mismatch** | The same API behaves differently under eager vs compiled execution. | No (PyTorch's eager vs compile semantics are aligned; bugs are implementation errors rather than specification divergences). |

The third class is the most noteworthy - it is unique to TF.

### Pipeline structure

```
                      User Python code
                      tf.nn.selu(x), tf.math.sign(z), ...
                            |
                +-----------v-----------+
         Path A |     Eager Execution   |  Calls Eigen/cuDNN C++ kernels
                |  tensorflow/core/     |  directly. No graph, no opts.
                |  kernels/             |
                +-----------------------+

                            |  tf.function() decorator
                            v
                +-----------------------+
                |    Graph Construction |  Python -> GraphDef (protobuf)
                |    + tf2xla Bridge    |  Identify XLA-compatible subgraph
                +-----------+-----------+
                            |
            +---------------+---------------+
            |               |               |
    +-------v------+ +------v------+ +------v------+
    |   Grappler   | |  jit_compile| |  jit_compile|
    |  (on by      | |  = False    | |  = True     |
    |   default)   | | (Grappler   | | (Grappler   |
    |              | |  only)      | |  + XLA)     |
    |  Optimisation| +-------------+ +------+------+
    |  passes:     |                        |
    |  * Arithm.   |                        v
    |  * Constant  |              +------------------+
    |  * Layout    |              |  XLA Compiler    |
    |  * Memory    |              |  +-------------+ |
    +--------------+              |  | HLO build   | |
                                  |  | (TF op-HLO) | |
                                  |  +-------------+ |
                                  |  | HLO opts    | |
                                  |  | (fusion,CSE,| |
                                  |  |  constant   | |
                                  |  |  folding...)| |
                                  |  +-------------+ |
                                  |  | Emitter     | |
                                  |  | HLO->LLVM IR| |
                                  |  | or HLO->    | |
                                  |  | Triton MLIR | |
                                  |  +-------------+ |
                                  |  | LLVM/PTX/   | |
                                  |  | cubin       | |
                                  |  +-------------+ |
                                  +------------------+
```

Key structural differences from PyTorch:

1. **TF has two fully independent execution engines**: Eager (Eigen C++ kernels) and XLA (HLO -> LLVM compilation). PyTorch's eager and compiled paths share a single ATen kernel set (the compiled path simply fuses multiple ATen ops into a single Triton kernel).
2. **TF's Grappler runs before XLA**, an extra layer. PyTorch has no equivalent layer (Inductor's FX passes play a Grappler-like role but are coupled inside Inductor).
3. **TF's eager and XLA are two independent implementations**, with semantic alignment maintained manually. PyTorch's eager and Inductor share meta-kernel definitions, so semantics are aligned by construction (though the meta kernels themselves can have bugs).

### The 7 bugs analysed by problem type

#### Type A: eager-kernel numerical defects (2 bugs)

These two bugs are entirely independent of the compilation pipeline; they are mathematical implementation issues in TF's lowest-level Eigen C++ kernels.

| Bug | Issue | Error mechanism | Type |
|-----|-------|---------|------|
| bug_021_selu_elu_expm1 | [#<anon>](<upstream issue, anonymized>) | The Eigen SELU/ELU kernel uses `exp(x) - 1` instead of `expm1(x)`. For small negatives with \|x\| <= 1e-8, `exp(x)` first rounds to `1.0`, then `1.0 - 1.0 = 0.0` (catastrophic cancellation), yielding a 100% error. **Ironically, XLA uses `expm1`, so the compiled result is correct.** | Silent |
| bug_005_complex_sign | [#<anon>](<upstream issue, anonymized>) | When `tf.math.sign(z)` computes \|z\|^2 = re^2 + im^2 for complex64, the intermediate \|z\|^2 underflows to zero when \|z\| < 1.084e-19 (= sqrt(float32_tiny)), so the kernel thinks `z == 0` and returns `sign = 0`. XLA and NumPy are correct. | Silent |

**Characteristics:** these are **wrong numerical method choices** in the eager kernel, with no compiler optimisation involved. The fixes are direct (substitute `expm1`, use `hypot` instead of a sum of squares).

#### Type B: graph-optimisation semantics violation (1 bug)

Grappler is TF's graph-level optimiser; it runs by default during `tf.function` compilation. This bug parallels PyTorch Inductor's fusion problems - the optimisation pass changes the program's observable semantics.

| Bug | Issue | Error mechanism | Type |
|-----|-------|---------|------|
| tf_grappler_argmin | [#<anon>](<upstream issue, anonymized>) | The `ArithmeticOptimizer`'s `OptimizeMaxOrMinOfMonotonic` pass rewrites `ArgMin(f(x))` to `ArgMin(x)` when `f` is marked "monotonic". However `IsElementWiseMonotonic()` marks **non-strictly-monotonic** functions such as ReLU/Floor/Ceil/Sign as monotonic too, so the rewrite is not semantics-preserving (argmin under ties differs). | Silent |

**Detailed mechanism:**
```python
x = [1.0, -1.0, -5.0, 2.0]
relu(x) = [1, 0, 0, 2]    -> argmin = 1 (first zero)
x       = [1, -1, -5, 2]   -> argmin = 2 (most negative)

# Grappler optimisation: ArgMin(relu(x)) -> ArgMin(x) -> returns 2 (should be 1)
```

ReLU is monotonic but not **strictly** monotonic (there is a flat region where `f(x)=0`), so `ArgMin(f(x)) -> ArgMin(x)` does not preserve semantics when ties exist. The optimisation is enabled by default.

**Comparison with PyTorch:** PyTorch bug_004 (bf16 cast elision) is a different surface of the same issue - the compiler optimisation removes an op with observable effect. The PyTorch defect lives in the Inductor fusion layer; the TF defect lives in the Grappler graph-optimisation layer.

#### Type C: eager <-> XLA semantic contract mismatch (4 bugs)

**This is the core problem type for TF, and it is unique to TF.** The four bugs share one feature: the same TF Python API exhibits **different observable behaviour** under eager mode vs `jit_compile=True` mode, without the user being told.

| Bug | Issue | Specific inconsistency | Type |
|-----|-------|----------------|------|
| tf_xla_cpu_nan | [#<anon>](<upstream issue, anonymized>) | `reduce_max([nan, 1, 3])`: eager returns `nan` (IEEE 754 NaN propagation); CPU XLA returns `3` (drops NaN). NumPy and JAX propagate NaN. | Silent |
| bug_003_check_numerics | [#<anon>](<upstream issue, anonymized>) | `check_numerics(nan_tensor)`: eager raises `InvalidArgumentError`; XLA passes silently (the check is skipped). Users who rely on `check_numerics` as a numerical guard lose it after switching to XLA. | Silent |
| tf_cumsum_complex | [#<anon>](<upstream issue, anonymized>) | `cumsum(complex_tensor)`: eager computes normally; XLA compilation crashes (HLO lowering does not implement complex scan). | Crash |
| tf_argmin_dead_code | [#<anon>](<upstream issue, anonymized>) | `argmin(empty_dim_tensor)`: eager handles it normally; XLA compilation crashes (the dead-code path is not handled by XLA). | Crash |

**Why does TF suffer this class while PyTorch does not?**

The root cause is architectural:

```
PyTorch:
  eager kernel --(shared meta-kernel definitions)--> Inductor codegen
  Both paths' semantics are constrained by the same meta kernels.

TensorFlow:
  Eigen eager kernel --(independent implementation)--> XLA HLO lowering
  Two independent code bases; alignment is maintained manually.
```

- PyTorch's eager and compiled paths share `aten::` operator definitions and meta kernels. If eager propagates NaN through `reduce_max(nan)`, the Inductor-generated kernel also propagates NaN, because Inductor lowers from the same aten op definition.
- TF's eager kernels (`tensorflow/core/kernels/`) and XLA's HLO lowering (`xla/service/`) are **two independent C++ implementations**. NaN semantics, complex-dtype support, boundary-condition handling, and so on are aligned manually by each implementer. There is no enforced interface contract.

So TF bugs of this class look more like **"missing semantic contracts"** than "implementation errors" - two independent implementations have each made a different reasonable choice for the same mathematical operation (eager chooses to propagate NaN; XLA chooses to drop NaN), yet users expect them to behave identically.

### TensorFlow summary diagram

```
+---------------------------------------------------------+
|  Eager Kernel (Eigen C++)                               |
|  - Independent C++ implementation                       |
|  - 2 bugs: wrong numerical methods (expm1, hypot)       |
|  <- Problem type A: numerical defects unrelated to      |
|     compilation                                         |
+---------------------------------------------------------+
|  Grappler (graph optimisation, on by default for        |
|  tf.function)                                           |
|  - ArithmeticOptimizer, ConstantFolding, ...            |
|  - 1 bug: ArgMin rewrite for non-strictly-monotonic     |
|    functions violates semantics                         |
|  <- Problem type B: optimisation pass changes           |
|     observable semantics                                |
+---------------------------------------------------------+
|  XLA Compiler (jit_compile=True)                        |
|  - Independent HLO-lowering implementation              |
|  - 4 bugs: different NaN semantics / check_numerics     |
|    skipped / complex scan unimplemented / empty-dim     |
|    crash                                                |
|  <- Problem type C: eager <-> XLA semantic contract     |
|     mismatch (unique to TF)                             |
+---------------------------------------------------------+

                         vs.
           +---- PyTorch has neither A nor C ----+
           |  PyTorch's eager kernels are        |
           |  uniformly correct in our tests;    |
           |  eager <-> compiled semantics are   |
           |  guaranteed by shared meta-kernel   |
           |  definitions.                       |
           +-------------------------------------+
```

---

## 3. XLA / OpenXLA (1 bug)

| Bug | Issue | Pipeline location | Error mechanism | Type |
|-----|-------|---------|---------|------|
| erf scalar-vs-vector | [openxla/xla#<anon>](<upstream PR, anonymized>) | XLA LLVM codegen (Emitter layer) | `erf` uses two implementations for the scalar and vector paths: the vector path has a saturation select (\|x\| >= 3.83 -> +/-1.0); the scalar path only clamps the input (`clamp(x, +/-3.74)`, no output saturation). The same input `x=4.5` therefore produces different results, which `floor` amplifies to 0 vs 1. | Silent |

**Location:** XLA's lowest-level LLVM IR emitter. In the [XLA Architecture](https://openxla.org/xla/architecture) diagram this corresponds to the final "Backend-specific code generation" stage. The fix has been merged.

---

## 4. IREE (4 bugs)

Pipeline: `ONNX model -> iree-import-onnx -> torch-mlir -> linalg -> HAL -> target binary`

| Bug | Issue | Pipeline location | Root cause | Type |
|-----|-------|---------|------|------|
| f64_demote | [#<anon>](<upstream issue, anonymized>) | Compiler pass (`DemoteF64ToF32Pass`) | The function body is demoted from f64 to f32 but the ABI stays f64, so buffer sizes mismatch and the second half of the data becomes zero. | Silent |
| reduceprod | [#<anon>](<upstream issue, anonymized>) | torch-mlir lowering | `ReduceProd` + `keepdims=0` emits a `tensor.reshape` with the wrong rank. | Crash |
| reverse_seq | [#<anon>](<upstream issue, anonymized>) | ONNX frontend | Indexing is wrong under non-default `batch_axis` / `time_axis` configurations. | Silent |
| reduce_segfault | [#<anon>](<upstream issue, anonymized>) | torch-mlir lowering | Multi-axis + negative index + `keepdims=0` -> SEGFAULT. | Crash |

---

## 5. TVM (5 bugs)

Pipeline: `ONNX model -> Frontend converter -> Relax IR -> LegalizeOps -> TIR -> CodeGen -> binary`

| Bug | Issue | Pipeline location | Root cause | Type |
|-----|-------|---------|------|------|
| gemv_keyerror | [#<anon>](<upstream issue, anonymized>) | DLight scheduling | KeyError when the GEMV schedule rule auto-detects the CUDA target. | Crash |
| softmax_opset | [#<anon>](<upstream issue, anonymized>) | ONNX frontend | Only opset-13 semantics are implemented, so opset <= 12 silently computes wrong results. | Silent |
| cumsum_axis | [#<anon>](<upstream issue, anonymized>) | ONNX frontend | When axis is a `relax.Var`, `axis=0` is hard-coded. | Silent |
| scatter_cuda | [#<anon>](<upstream issue, anonymized>) | TIR legalization | The emitted TIR uses `T.parallel` instead of a GPU thread binding. | Crash |
| isnan_isinf | [#<anon>](<upstream issue, anonymized>) | CodeGenVM | The op is registered, but CodeGenVM does not implement the intrinsic. | Crash |

---

## 6. Cross-framework summary

### By Silent vs Crash

| | PyTorch | TensorFlow | XLA | IREE | TVM | Total |
|------|---------|-----------|-----|------|-----|------|
| **Silent** | 7 | 5 | 1 | 2 | 2 | **17** |
| **Crash** | 3 | 2 | 0 | 2 | 3 | **10** |
| Total | 10 | 7 | 1 | 4 | 5 | **27** |

### By problem type

| Problem type | Description | Bugs involved | Count |
|---------|------|----------|------|
| **Code-generation error** | Codegen emits wrong target code (stride literals, iteration space, dtype). | PT bug_004/006/007/013/022, XLA erf | 6 |
| **Optimisation-pass semantics violation** | A compiler optimisation changes observable semantics. | PT bug_004 (fusion), TF Grappler argmin | 2 |
| **Metadata / annotation error** | Wrong shape/stride/dtype annotations in the IR. | PT bug_007/013/022 (meta kernel), PT bug_019 | 4 |
| **Eager <-> compiled semantic divergence** | The same API behaves differently across execution modes. | TF NaN / check_numerics / cumsum / argmin | 4 |
| **Eager-kernel numerical defect** | The underlying C++ kernel has an incorrect mathematical implementation. | TF expm1, TF complex sign | 2 |
| **Frontend conversion error** | Wrong op-semantics translation at model import. | TVM softmax/cumsum, IREE reverse_seq | 3 |
| **Lowering missing / crash** | Lowering is missing or crashes on specific input combinations. | PT bug_011/015/020/023, IREE reduceprod/segfault, TVM gemv/scatter/isnan | 9 |

### Why traditional testing misses them

1. **Combinatorial trigger conditions**: each bug requires a specific combination of op + shape + dtype + fusion/optimisation decisions (e.g., channel not a multiple of 32 + non-persistent reduction; scan dim >= 129 + broadcast variable).
2. **63% are silent errors**: no crash, no exception - just wrong result values. A crash-only oracle cannot detect them.
3. **Discrete jumps in codegen path**: a +/-1 dimension can switch the kernel template (persistent / non-persistent / SplitScan), and each path carries its own bugs.
4. **Two independent implementations** (TF-specific): Eigen eager and XLA compiled are independent code bases; semantic mismatches can only be discovered through differential testing.
