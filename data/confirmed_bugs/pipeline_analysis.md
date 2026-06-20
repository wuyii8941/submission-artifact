# 27 Bugs 编译管线层次分析

## 官方架构参考

### PyTorch 编译管线
- [PyTorch 2 论文 (ASPLOS 2024) — Figure 1 是最权威的管线图](https://docs.pytorch.org/assets/pytorch2-2.pdf)
- [PyTorch 2.x 官方介绍页 — 管线全景图](https://pytorch.org/get-started/pytorch-2-x/)
- [Dynamo Overview — guard/graph break 流程图](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)
- [Dynamo Deep Dive — 内部实现细节](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [torch.compile Programming Model — 各层衔接](https://docs.pytorch.org/docs/stable/compile/programming_model.html)
- [TorchInductor 设计文档](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)

### TensorFlow / XLA 编译管线
- [XLA Architecture — 编译阶段图 (StableHLO → HLO → LLVM IR → binary)](https://openxla.org/xla/architecture)
- [XLA:GPU Architecture Overview — GPU 后端编译流程图](https://openxla.org/xla/gpu_architecture)
- [TF-to-XLA Bridge (tf2xla) — TF 图到 HLO 的转换](https://openxla.org/xla/tf2xla)
- [Grappler 图优化官方指南 — 各 pass 说明](https://www.tensorflow.org/guide/graph_optimization)
- [jit_compile=True 使用教程](https://openxla.org/xla/tf2xla/tutorials/jit_compile)
- [HLO Passes 列表](https://openxla.org/xla/hlo_passes)

---

## 一、PyTorch/Inductor (10 bugs)

### 管线结构

```
用户 Python 代码
    │
    ▼
┌──────────────────────────────────────────────────┐
│  TorchDynamo                                     │
│  CPython frame hook → 字节码改写 → FX Graph       │
│  输出: forward-only FX Graph + Guards             │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  AOTAutograd                                     │
│  追踪 backward 图 + meta/fake kernel 推断 stride  │
│  输出: forward graph + backward graph             │
│        + 每个 tensor 的 (shape, stride, dtype)    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Inductor                                        │
│  ┌─────────────────────────────────────────────┐ │
│  │ Lowering: ATen ops → IR (Pointwise/Red/Scan)│ │
│  ├─────────────────────────────────────────────┤ │
│  │ Fusion: 合并 IR 节点为单 kernel              │ │
│  ├─────────────────────────────────────────────┤ │
│  │ Scheduling: 选 kernel 模板                   │ │
│  │  (persistent_red / non-persistent / SplitScan)│ │
│  ├─────────────────────────────────────────────┤ │
│  │ Codegen: IR → Triton Python 源代码           │ │
│  │  (计算 stride 字面量, 构建迭代空间, 生成 load/store)│ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼  Triton .py 源码
┌──────────────────────────────────────────────────┐
│  Triton Compiler (OpenAI)                        │
│  Triton IR → MLIR → LLVM IR → PTX → cubin        │
│  ← 不负责代码正确性，只负责编译 + 执行             │
└──────────────────────────────────────────────────┘
```


### 10 个 bugs 按管线层归因

#### L1: Inductor Fusion Codegen (5 bugs) — 生成的 Triton 代码有错

| Bug | Issue | 错误机制 | 类型 |
|-----|-------|---------|------|
| bug_006 | [#179931](<upstream issue, anonymized>) | `adaptive_avg_pool2d + flatten + sum` 融合后，Inductor 用 `ceil(3185/32)*32 = 3200` 作为 batch stride 字面量写入 Triton 源码，但 buffer 的真实 stride 是 `3185`。batch ≥ 1 的 load 全部偏移 15 个元素 | Silent |
| bug_007 | [#180164](<upstream issue, anonymized>) | AOTAutograd 的 meta kernel 对 `slice_scatter` 返回了 stride-0 标注 → Inductor 信任该标注，将迭代空间坍缩为 132 元素（应为 1716）→ dim-1 坐标硬编码为 0 → 边界检查被 Triton 编译器常量折叠 → `x.grad` 全零 | Silent |
| bug_013 | [#180771](<upstream issue, anonymized>) | 同 bug_007 根因家族。`slice_scatter` backward 对不同 slice 位置产生不同错误模式 | Silent |
| bug_022 | [#182012](<upstream issue, anonymized>) | `diagonal_scatter(...).sum()` backward 中，Inductor 将 `clone(expand(scalar,[N,N]))` 优化为 stride-(0,0) 的 1 元素 buffer → 后续 `diagonal_scatter` 写零到该元素 → 整个 N×N 梯度全零 | Silent |
| bug_004 | [#179561](<upstream issue, anonymized>) | `fp32→bf16→fp32` round-trip cast 被 fusion 当 no-op 消除 → 生成的 Triton kernel 完全没有 dtype 转换 → 缺少预期精度截断 | Silent |

**根因模式**：

1. **Stride 信任链断裂**（bug_007/013/022）：AOTAutograd meta kernel 返回错误 stride → Inductor 信任 → 生成的 kernel 迭代空间/索引错误
2. **数值字面量计算错误**（bug_006）：codegen 在计算 view+reduction 的 stride 时用了 `ceil_to_warp_size` 而非真实值
3. **Fusion 语义违反**（bug_004）：fusion pass 消除了有可观测副作用的 op

#### L2: AOTAutograd (2 bugs) — 自动微分追踪出错

| Bug | Issue | 错误机制 | 类型 |
|-----|-------|---------|------|
| bug_019 | [#181581](<upstream issue, anonymized>) | `autograd.grad(create_graph=True)` 下 AOTAutograd 对 `mm`/`relu` 等 op 静默丢失 `requires_grad=True` 和 `grad_fn` → higher-order backward 失败。`aot_eager` backend 也复现，不是 Inductor 特有 | Silent |
| bug_020 | [#181512](<upstream issue, anonymized>) | `inference_mode()` + 3D+ matmul + `torch.compile` → AOTAutograd 的 functionalization 阶段 weakref 到 `UntypedStorage` 无法 resolve → crash | Crash |

#### L3: Inductor Scheduling (2 bugs) — kernel 模板选择/初始化出错

| Bug | Issue | 错误机制 | 类型 |
|-----|-------|---------|------|
| bug_011 | [#180221](<upstream issue, anonymized>) | scan dim ≥ 129 时 Inductor 切到 SplitScan codegen → `tensor_dim=None` 被用作列表索引 → crash | Crash |
| bug_015 | (未提交) | `associative_scan` + `dynamic=True` + ndim≥2 → lifted arguments 不受支持 → codegen crash | Crash |

#### L4: Dynamo (1 bug) — 图捕获/guard 构建出错

| Bug | Issue | 错误机制 | 类型 |
|-----|-------|---------|------|
| bug_023 | [#182217](<upstream issue, anonymized>) | `F.layer_norm(NestedTensor)` → symbolic variable 无 source entry → guard builder `IndexError`。所有 backend 都 crash | Crash |

### PyTorch 总结图

```
┌─────────────────────────────────────────────┐
│  Dynamo (Graph Capture)                     │ ← 1 bug (guard crash)
├─────────────────────────────────────────────┤
│  AOTAutograd (Backward + Meta Kernels)      │ ← 2 bugs (metadata/weakref)
├─────────────────────────────────────────────┤
│  Inductor — Scheduling                      │ ← 2 bugs (SplitScan/dynamic)
├─────────────────────────────────────────────┤
│  Inductor — Fusion + Codegen ← BUG 密度最高  │ ← 5 bugs (stride/index/dtype)
├─────────────────────────────────────────────┤
│  Triton Compiler                            │ ← 0 bugs
└─────────────────────────────────────────────┘

问题本质: 代码生成错误。Inductor 在将高层 IR 语义翻译为
Triton 源码的数值字面量时引入了错误。Triton 忠实编译并执行了
这些（错误的）代码。
```

---

## 二、TensorFlow (7 bugs)

### TF 的问题和 PyTorch 有本质区别

PyTorch 的 bug 是**单一类型**的问题——代码生成错误。TF 的 7 个 bug 分散在**三种完全不同的问题类型**中：

| 问题类型 | 性质 | PyTorch 中有没有 |
|---------|------|----------------|
| **Eager kernel 数值缺陷** | 和编译无关，是底层 C++ kernel 的数学实现有误 | 没有（PyTorch eager 的 kernel 在我们测试中是正确的） |
| **图优化语义违反** | 编译器优化 pass 改变了程序语义 | 类似 Inductor 的 fusion 问题，但在不同层 |
| **Eager ↔ XLA 语义契约不统一** | 同一个 API 在 eager 和 compiled 下行为不同 | 没有（PyTorch eager vs compile 的语义是对齐的，bug 是实现错误而非规范分歧） |

第三种类型最值得关注——这是 TF 特有的问题。

### 管线结构

```
                      用户 Python 代码
                      tf.nn.selu(x), tf.math.sign(z), ...
                            │
                ┌───────────▼───────────┐
         Path A │     Eager Execution   │  直接调用 Eigen/cuDNN C++ kernel
                │  tensorflow/core/     │  无图构建，无优化
                │  kernels/             │
                └───────────────────────┘

                            │  tf.function() 装饰
                            ▼
                ┌───────────────────────┐
                │    Graph Construction │  Python → GraphDef (protobuf)
                │    + tf2xla Bridge    │  识别 XLA-compatible subgraph
                └───────────┬───────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │   Grappler   │ │  jit_compile │ │  jit_compile │
    │  (默认开启)   │ │  = False     │ │  = True      │
    │              │ │  (仅Grappler)│ │ (Grappler+XLA)│
    │  优化 passes: │ └─────────────┘ └──────┬──────┘
    │  • Arithmetic│                        │
    │  • Constant  │                        ▼
    │  • Layout    │              ┌──────────────────┐
    │  • Memory    │              │  XLA Compiler     │
    └──────────────┘              │  ┌──────────────┐ │
                                  │  │ HLO 构建     │ │
                                  │  │ (TF op→HLO)  │ │
                                  │  ├──────────────┤ │
                                  │  │ HLO 优化     │ │
                                  │  │ (fusion,CSE, │ │
                                  │  │  constant    │ │
                                  │  │  folding...) │ │
                                  │  ├──────────────┤ │
                                  │  │ Emitter      │ │
                                  │  │ HLO→LLVM IR  │ │
                                  │  │ 或 HLO→Triton│ │
                                  │  │ MLIR         │ │
                                  │  ├──────────────┤ │
                                  │  │ LLVM/PTX/    │ │
                                  │  │ cubin        │ │
                                  │  └──────────────┘ │
                                  └──────────────────┘
```

与 PyTorch 的关键结构差异：

1. **TF 有两套完全独立的执行引擎**：Eager (Eigen C++ kernels) 和 XLA (HLO→LLVM 编译)。PyTorch 的 eager 和 compiled 共享同一套 ATen kernel（compiled 路径只是把多个 ATen op fuse 成一个 Triton kernel）。
2. **TF 的 Grappler 在 XLA 之前运行**，是额外的一层。PyTorch 没有对等层（Inductor 的 FX passes 和 Grappler 角色类似但耦合在 Inductor 内部）。
3. **TF eager 和 XLA 是两套独立实现**，语义对齐靠人工维护。PyTorch eager 和 Inductor 共享 meta kernel 定义，理论上语义天然对齐（虽然 meta kernel 本身可能有 bug）。

### 7 个 bugs 按问题类型分析

#### 类型 A: Eager Kernel 数值缺陷 (2 bugs)

这两个 bug 和编译管线完全无关，是 TF 最底层 Eigen C++ kernel 的数学实现问题。

| Bug | Issue | 错误机制 | 类型 |
|-----|-------|---------|------|
| tf_selu_elu_expm1 | [#116943](<upstream issue, anonymized>) | Eigen SELU/ELU kernel 用 `exp(x) - 1` 而非 `expm1(x)` → 对 \|x\| ≤ 1e-8 的小负数，`exp(x)` 先 round 到 `1.0`，然后 `1.0 - 1.0 = 0.0`（catastrophic cancellation）→ 100% 错误。**讽刺的是 XLA 用了 `expm1`，编译后的结果反而正确** | Silent |
| tf_complex_sign | [#116945](<upstream issue, anonymized>) | `tf.math.sign(z)` 对 complex64 计算 `\|z\|² = re² + im²` 时，当 \|z\| < 1.084e-19（= sqrt(float32_tiny)），中间 `\|z\|²` 下溢到 zero → kernel 误判 `z == 0` → 返回 `sign = 0`。XLA 和 NumPy 正确 | Silent |

**特点**：这是 eager kernel 的 **数值方法选择错误**，不涉及任何编译优化。修复方式也很直接（替换为 `expm1`、用 `hypot` 代替平方和）。

#### 类型 B: 图优化语义违反 (1 bug)

Grappler 是 TF 的图级优化器，在 `tf.function` 编译时默认运行。这个 bug 类似 PyTorch Inductor 的 fusion 问题——优化 pass 改变了程序的可观测语义。

| Bug | Issue | 错误机制 | 类型 |
|-----|-------|---------|------|
| tf_grappler_argmin | [#116436](<upstream issue, anonymized>) | `ArithmeticOptimizer` 的 `OptimizeMaxOrMinOfMonotonic` pass 将 `ArgMin(f(x))` 重写为 `ArgMin(x)` 当 `f` 被标记为 "monotonic"。但 `IsElementWiseMonotonic()` 将 ReLU/Floor/Ceil/Sign 等**非严格单调**函数也标记为 monotonic → 重写不保语义（ties 的 argmin 不同） | Silent |

**详细机制**：
```python
x = [1.0, -1.0, -5.0, 2.0]
relu(x) = [1, 0, 0, 2]    → argmin = 1 (第一个零)
x       = [1, -1, -5, 2]   → argmin = 2 (最负的)

# Grappler 优化: ArgMin(relu(x)) → ArgMin(x) → 返回 2 (应该是 1)
```

ReLU 是单调的但不是**严格**单调的（有 `f(x)=0` 的平坦区域），所以 `ArgMin(f(x)) → ArgMin(x)` 这个重写在有 ties 时不保持语义。这个优化默认开启。

**和 PyTorch 的对比**：PyTorch 的 bug_004（bf16 cast elision）是同一类问题的不同表现——编译器优化消除了有可观测效果的 op。但 PyTorch 的问题在 Inductor fusion 层，TF 的在 Grappler 图优化层。

#### 类型 C: Eager ↔ XLA 语义契约不统一 (4 bugs)

**这是 TF 最核心的问题类型，也是 TF 特有的。** 这 4 个 bug 的共同点：同一个 TF Python API 在 eager 模式和 `jit_compile=True` 模式下有**不同的可观测行为**，而用户没有被告知。

| Bug | Issue | 不一致的具体表现 | 类型 |
|-----|-------|----------------|------|
| tf_xla_cpu_nan | [#116992](<upstream issue, anonymized>) | `reduce_max([nan, 1, 3])`: eager 返回 `nan`（IEEE 754 NaN propagation），CPU XLA 返回 `3`（丢弃 NaN）。NumPy 和 JAX 都传播 NaN | Silent |
| tf_check_numerics | [#117319](<upstream issue, anonymized>) | `check_numerics(nan_tensor)`: eager 抛 `InvalidArgumentError`，XLA 静默通过（检查被跳过）。用户用 `check_numerics` 做数值 guard 会在切到 XLA 后失效 | Silent |
| tf_cumsum_complex | [#117546](<upstream issue, anonymized>) | `cumsum(complex_tensor)`: eager 正常计算，XLA 编译崩溃（HLO lowering 未实现 complex scan） | Crash |
| tf_argmin_dead_code | [#117547](<upstream issue, anonymized>) | `argmin(empty_dim_tensor)`: eager 正常处理，XLA 编译崩溃（dead code 路径未被 XLA 处理） | Crash |

**为什么 TF 会有这类问题而 PyTorch 不会？**

根本原因在于架构差异：

```
PyTorch:
  eager kernel ──(共享 meta kernel 定义)──→ Inductor codegen
  两条路径的 semantics 由同一份 meta kernel 定义约束

TensorFlow:
  Eigen eager kernel ──(独立实现)──→ XLA HLO lowering
  两条路径是两套独立代码，语义对齐靠人工维护
```

- PyTorch 的 eager 和 compiled 路径共享 `aten::` 算子定义和 meta kernel。如果 eager 对 `reduce_max(nan)` 传播 NaN，Inductor 生成的 kernel 也会传播 NaN，因为 Inductor 基于相同的 aten op 定义来 lower。
- TF 的 eager kernel（`tensorflow/core/kernels/`）和 XLA 的 HLO lowering（`xla/service/`）是**两套独立的 C++ 实现**。NaN 语义、complex dtype 支持、边界条件处理等全靠各自实现者手动对齐。没有强制的接口契约。

所以 TF 的这类 bug 更接近 **"语义契约缺失"** 而非 "实现错误"——两个独立实现对同一个数学运算做了不同的合理选择（eager 选择传播 NaN，XLA 选择丢弃 NaN），但用户期望它们行为一致。

### TensorFlow 总结图

```
┌─────────────────────────────────────────────────────────┐
│  Eager Kernel (Eigen C++)                               │
│  • 独立的 C++ 实现                                       │
│  • 2 bugs: 数值方法选择错误 (expm1, hypot)               │
│  ← 问题类型 A: 和编译无关的数值缺陷                       │
├─────────────────────────────────────────────────────────┤
│  Grappler (图优化, tf.function 默认开启)                 │
│  • ArithmeticOptimizer, ConstantFolding, ...            │
│  • 1 bug: 非严格单调函数的 ArgMin 重写违反语义             │
│  ← 问题类型 B: 优化 pass 改变可观测语义                   │
├─────────────────────────────────────────────────────────┤
│  XLA Compiler (jit_compile=True)                        │
│  • 独立的 HLO lowering 实现                              │
│  • 4 bugs: NaN 语义不同 / check_numerics 跳过 /          │
│            complex scan 未实现 / empty-dim crash          │
│  ← 问题类型 C: Eager ↔ XLA 语义契约不统一 (TF 特有)      │
└─────────────────────────────────────────────────────────┘

                         ↕ 对比
           ┌──── PyTorch 没有类型 A 和 C ────┐
           │  PyTorch 的 eager kernel 在      │
           │  我们测试中全部正确；             │
           │  eager ↔ compiled 语义由          │
           │  共享的 meta kernel 定义保证      │
           └─────────────────────────────────┘
```

---

## 三、XLA / OpenXLA (1 bug)

| Bug | Issue | 管线位置 | 错误机制 | 类型 |
|-----|-------|---------|---------|------|
| erf scalar-vs-vector | [openxla/xla#41167](<upstream PR, anonymized>) | XLA LLVM codegen (Emitter 层) | `erf` 对 scalar path 和 vector path 使用两套不同实现：vector 有 saturation select（`\|x\|≥3.83 → ±1.0`），scalar 只有 input clamp（`clamp(x,±3.74)`，无 output saturation）→ 同一个输入 `x=4.5` 产生不同结果 → `floor` 放大为 0 vs 1 | Silent |

**位置**：XLA 最底层的 LLVM IR emitter。在 [XLA Architecture](https://openxla.org/xla/architecture) 图中对应最后的 "Backend-specific code generation" 阶段。已合并修复。

---

## 四、IREE (4 bugs)

管线：`ONNX model → iree-import-onnx → torch-mlir → linalg → HAL → target binary`

| Bug | Issue | 管线位置 | 根因 | 类型 |
|-----|-------|---------|------|------|
| f64_demote | [#24180](<upstream issue, anonymized>) | Compiler pass (`DemoteF64ToF32Pass`) | 函数体 f64→f32 但 ABI 保持 f64 → buffer 大小不匹配 → 后半数据为零 | Silent |
| reduceprod | [#24181](<upstream issue, anonymized>) | torch-mlir lowering | `ReduceProd` + `keepdims=0` 生成的 `tensor.reshape` rank 错误 | Crash |
| reverse_seq | [#24229](<upstream issue, anonymized>) | ONNX frontend | `batch_axis`/`time_axis` 非默认配置下索引错误 | Silent |
| reduce_segfault | [#24265](<upstream issue, anonymized>) | torch-mlir lowering | 多轴 + 负索引 + `keepdims=0` → SEGFAULT | Crash |

---

## 五、TVM (5 bugs)

管线：`ONNX model → Frontend converter → Relax IR → LegalizeOps → TIR → CodeGen → binary`

| Bug | Issue | 管线位置 | 根因 | 类型 |
|-----|-------|---------|------|------|
| gemv_keyerror | [#19419](<upstream issue, anonymized>) | DLight scheduling | GEMV schedule rule 自动检测 CUDA target 时 KeyError | Crash |
| softmax_opset | [#19420](<upstream issue, anonymized>) | ONNX frontend | 只实现 opset 13 语义 → opset ≤12 静默算错 | Silent |
| cumsum_axis | [#19437](<upstream issue, anonymized>) | ONNX frontend | axis 为 `relax.Var` 时硬编码 `axis=0` | Silent |
| scatter_cuda | [#19451](<upstream issue, anonymized>) | TIR legalization | 生成的 TIR 用 `T.parallel` 而非 GPU thread binding | Crash |
| isnan_isinf | [#19452](<upstream issue, anonymized>) | CodeGenVM | op 已注册但 CodeGenVM 未实现 intrinsic | Crash |

---

## 六、跨框架总结

### 按 Silent vs Crash

| | PyTorch | TensorFlow | XLA | IREE | TVM | 总计 |
|------|---------|-----------|-----|------|-----|------|
| **Silent** | 7 | 5 | 1 | 2 | 2 | **17** |
| **Crash** | 3 | 2 | 0 | 2 | 3 | **10** |
| 总计 | 10 | 7 | 1 | 4 | 5 | **27** |

### 按问题类型

| 问题类型 | 描述 | 涉及 bugs | 数量 |
|---------|------|----------|------|
| **代码生成错误** | Codegen 生成了错误的目标代码（stride 字面量、迭代空间、dtype） | PT bug_004/006/007/013/022, XLA erf | 6 |
| **优化 pass 语义违反** | 编译器优化改变了程序可观测语义 | PT bug_004 (fusion), TF Grappler argmin | 2 |
| **元数据 / 标注错误** | 中间表示的 shape/stride/dtype 标注不正确 | PT bug_007/013/022 (meta kernel), PT bug_019 | 4 |
| **Eager ↔ Compiled 语义分歧** | 同一 API 在两种执行模式下行为不同 | TF NaN/check_numerics/cumsum/argmin | 4 |
| **Eager kernel 数值缺陷** | 底层 C++ kernel 数学实现错误 | TF expm1, TF complex sign | 2 |
| **Frontend 转换错误** | 模型导入时 op 语义翻译错误 | TVM softmax/cumsum, IREE reverse_seq | 3 |
| **Lowering 未实现 / crash** | 特定输入组合下 lowering 缺失或崩溃 | PT bug_011/015/020/023, IREE reduceprod/segfault, TVM gemv/scatter/isnan | 9 |

### 为什么传统测试抓不住

1. **组合触发条件**：每个 bug 需要 op + shape + dtype + fusion/优化决策 的特定组合（如 channel 非 32 倍数 + non-persistent reduction；scan dim ≥ 129 + broadcast variable）
2. **63% 是 silent 错误**：不 crash、不报错，只是结果值错误。crash-only oracle 无法检测
3. **Codegen path 离散跳变**：维度 ±1 即可切换 kernel 模板（persistent / non-persistent / SplitScan），每条路径有独立的 bug
4. **两套独立实现**（TF 特有）：Eigen eager 和 XLA compiled 是独立代码库，语义不一致只能通过 differential testing 发现
