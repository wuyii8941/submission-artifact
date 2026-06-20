# TensorFlow Parallel Methodology: Formal Definition

完全比照 PyTorch/Inductor 的结构，将同一套理论框架 instantiate 到 TensorFlow。

---

## 1. Problem Formulation (同 PyTorch)

$$
P(\text{find}\; b) = \underbrace{P(\text{reach}\; b)}_{\rho_b(q)} \times \underbrace{P(\text{detect}\; b)}_{\delta_b(O;\, q)}
$$

- $\rho_b(q)$: generator $q$ 能否让程序进入触发 bug $b$ 的 execution pipeline state
- $\delta_b(O;\, q)$: oracle $O$ 能否在该 state 下观测到语义偏差

---

## 2. State Space $\mathcal{S}_{\text{TF}}$

### PyTorch 的 state space 回顾

PyTorch state = kernel family（7个）：由 Inductor 的 `_get_heuristic()` 和 extern dispatch 决定。
程序执行哪个 kernel family 取决于 op type + shape。

### TensorFlow 的 state space

TensorFlow 没有 kernel-level scheduling heuristic。其 "compiler decision" 体现在 **execution pipeline selection**：同一个 TF program 可以走三条不同的 execution pipeline，每条 pipeline 有不同的优化和 lowering 行为。

**定义：TensorFlow Execution Pipeline Space**

$$
\mathcal{S}_{\text{TF}} = \mathcal{P} \times \mathcal{D} \times \mathcal{R}
$$

其中：
- $\mathcal{P} = \{\text{eager},\; \text{Grappler},\; \text{XLA}\}$ — execution pipeline
  - eager: 直接 dispatch 到 Eigen/cuDNN kernels，无图优化
  - Grappler: `tf.function(jit_compile=False)`, 图级优化（constant folding, arithmetic simplification, layout optimization）
  - XLA: `tf.function(jit_compile=True)`, full compilation（fusion, algebraic simplification, buffer assignment, codegen）
- $\mathcal{D} = \{\text{fp32},\; \text{fp64},\; \text{fp16},\; \text{bf16},\; \text{complex64},\; \text{complex128}\}$ — dtype regime
- $\mathcal{R} = \{\text{static},\; \text{dynamic},\; \text{boundary},\; \text{empty}\}$ — shape regime
  - static: 常规固定 shape
  - dynamic: `tf.TensorSpec(shape=[None, ...])` 动态维度
  - boundary: 边界条件（dim=1, dim=0, very large）
  - empty: 空维度 (e.g., shape=[4, 0])

有效 state 总数：$|\mathcal{S}_{\text{TF}}| = 3 \times 6 \times 4 = 72$ 个 pipeline-dtype-shape 组合。

### 类比

| | PyTorch | TensorFlow |
|--|---------|------------|
| State 决定因素 | op type + tensor shape → kernel family | execution mode + dtype + shape regime → pipeline path |
| State 数量 | 7 families (6 on T4) | 72 combinations (3×6×4) |
| State 可观测性 | `_get_heuristic()` hook | execution mode 是显式选择的；dtype/shape 是 input property |
| Bug 与 state 的关系 | bug 集中在特定 family (e.g., split_scan) | bug 集中在特定 pipeline (e.g., XLA) 和 dtype (e.g., complex) |

---

## 3. Template Space $\mathcal{T}_{\text{TF}}$ (类比 K1-K5)

### PyTorch 的 template space 回顾

K1-K5: 5 个 operator-level codegen paths, 通过 `OpsForFamily` mapping 从 Inductor lowering registry 枚举。
340 templates = K1-K5 × 6 composition rules.

### TensorFlow 的 template space

T1-T7: 7 个 graph transformation categories，每个 category targets 一类 compilation seam。

| Category | Target Compilation Seam | Ops/Patterns | Cases |
|----------|------------------------|--------------|-------|
| T1: Rewrite-sensitive | Grappler graph rewrites (arithmetic simplification, dead code elimination) | ArgMin/ArgMax + non-monotonic pre-ops, fused patterns | 22 |
| T2: Numerically unstable | Eager kernel numerical contracts (catastrophic cancellation, overflow) | exp, log, SELU, ELU, erf, erfc, igamma | 24 |
| T3: Dtype-sensitive | Cast chain folding, precision demotion across pipelines | bf16/fp16 cast chains, complex dtype ops, saturating casts | 19 |
| T4: Shape-dependent | Reduction/reshape/broadcast lowering with shape boundary conditions | reduce with empty/boundary dims, reshape, index ops | 16 |
| T5: Data-dependent shape | Dynamic shape ops where output shape depends on values | unique, where, boolean_mask | 2 |
| T6: Control-flow | Graph-mode control flow vs eager semantics | tf.while_loop, tf.cond, stateful RNG | 4 |
| T7: Backend divergence | XLA vs eager/Grappler behavioral differences (NaN propagation, dead code, special values) | NaN semantics, check_numerics, gradient boundaries | 34 |

**总计：121 templates across 19 sub-groups (collapsed to 7 categories)**

### OpsForCategory mapping (类比 OpsForFamily)

```
OpsForCategory(T1) = {argmin, argmax, softmax, log_softmax, ...}  # Grappler-rewrite-sensitive
OpsForCategory(T2) = {exp, log, selu, elu, erf, erfc, igamma, ...}  # numerically fragile
OpsForCategory(T3) = {cast, saturate_cast, complex ops, ...}  # dtype-sensitive
OpsForCategory(T4) = {reduce_*, reshape, gather, scatter, ...}  # shape-dependent  
OpsForCategory(T5) = {unique, where, boolean_mask, ...}  # data-dependent shape
OpsForCategory(T6) = {while_loop, cond, ...}  # control-flow
OpsForCategory(T7) = {check_numerics, NaN-propagating ops, gradient ops, ...}  # backend-divergent
```

### Composition Rules (类比 PyTorch 的 6 种 composition)

TF templates 的 composition patterns:
1. **Single op**: 单 op 测试 pipeline behavior
2. **Chain**: op1 → op2 → ... (fusion boundary 测试)
3. **Dead code**: unused result + live result (dead code elimination 测试)
4. **Cross-pipeline**: 同一 program 在 eager/Grappler/XLA 下比较
5. **Boundary input**: extreme values (NaN, Inf, very small, very large, empty)
6. **Dtype chain**: cast → op → cast (precision preservation 测试)

---

## 4. Diversity Metric $\mathcal{D}_{\text{TF}}$ (类比 Kernel Diversity)

### PyTorch 的 kernel diversity 回顾

$$
\mathcal{D}_{\text{PT}}(\mathcal{P}) = |\bigcup_{i} \{\tau(k) \mid k \in K(P_i)\}|
$$

其中 $\tau(k)$ = kernel template identity (scheduling family × fused op composition).

### TensorFlow 的 graph diversity

**定义：Graph Diversity（正式版，完全 parallel Kernel Diversity）**

PyTorch kernel diversity 回顾：
- τ(k) = (scheduling_family, fused_op_set) — 从 compiler output 提取
- scheduling_family 来自 `_get_heuristic()` — compiler 的 path 选择 observable
- fused_op_set 来自 Triton kernel 的 op 名 — program 的 op 组合 observable

TF graph diversity 的 formal parallel：

给定一组 programs $\mathcal{P} = \{P_1, \ldots, P_n\}$，通过 TF 的三条 pipeline 执行，定义每个 program 的 **graph transformation signature**:

$$
\sigma(P_i) = (\text{op\_sig}(P_i),\; \text{dtype}(P_i),\; \text{outcome}(P_i))
$$

其中：
- $\text{op\_sig}(P_i)$: program 使用的 TF op 类型集合 — **类比 fused_op_set**，是 program 结构的 observable
- $\text{dtype}(P_i)$: input dtype — 影响 compiler 的 precision handling decisions
- $\text{outcome}(P_i) \in \{0, 1\}^3$: 三对 pipeline 的一致性 pattern (eager≈grappler?, eager≈xla?, grappler≈xla?) — **类比 scheduling_family**，是 compiler 行为的 observable

**Graph diversity:**
$$
\mathcal{D}_{\text{TF}}(\mathcal{P}) = |\{\sigma(P_i) \mid P_i \in \mathcal{P}\}|
$$

### Formal parallel 对照表

| | PyTorch τ(k) | TF σ(P) |
|--|-------------|---------|
| Op 组合维度 | fused_op_set (Triton kernel 的 op 名) | op_sig (program 的 TF op 类型集) |
| Compiler 行为维度 | scheduling_family (_get_heuristic()) | outcome (3-way pipeline 一致性 pattern) |
| 来源 | 编译器输出的 kernel 结构 | 编译器对 program 的行为观测 |
| 度量含义 | unique codegen paths exercised | unique compilation outcomes observed |

### 空间大小

上限取决于 op 组合的多样性：
- op_sig: 对 targeted = 121 templates 的 unique op sets, 对 random = C(30, 1..4) 组合 >> 100
- dtype: 6 种
- outcome: 2³ = 8 种 pattern（加上 crash 模式 ~10 种）
- 理论上限 >> 1000，足够产生有意义的增长曲线

---

## 5. Targeted Generation (类比 Algorithm 1)

### PyTorch Algorithm 1 回顾

```
Input: target family f, dtype d, fusion degree n
1. O ← OpsForFamily(f)
2. G ← ComposeGraph(O, n)
3. θ.dtype ← d
4. θ.shape ← BoundaryShape(f, d)
Output: program p = (G, θ)
```

### TensorFlow Algorithm (parallel)

```
Input: target category t ∈ {T1,...,T7}, dtype d, shape regime r
1. O ← OpsForCategory(t)           // category → op pool
2. G ← ComposePattern(O, t)        // category-specific composition
3. θ.dtype ← d
4. θ.shape ← BoundaryCondition(t, r)  // category-specific boundary
5. pipeline ← SelectPipeline(t)    // which pipelines to exercise
Output: program p = (G, θ), pipeline configuration
```

**BoundaryCondition (类比 BoundaryShape)**:
- T1: NaN, Inf, tied values, non-strictly-monotonic sequences (Grappler rewrite boundaries)
- T2: small negative values near 0, very large values (numerical overflow/underflow thresholds)
- T3: values where bf16 truncation loses information
- T4: empty dims (shape=[N,0]), dim=1, very large dims
- T5: all-same values, all-unique values
- T7: NaN in reduction, Inf in normalization

---

## 6. Differential Oracle $O_{\text{TF}}$ (类比 eager-vs-compiled)

### PyTorch oracle 回顾

- Primary: eager vs `torch.compile(backend='inductor')`
- Secondary: aot_eager cross-check, backward-mode amplification, cross-config

### TensorFlow oracle: 3-way pipeline differential

- **Primary**: eager vs Grappler vs XLA (三两两比较)
  - eager vs Grappler: 检测图优化引入的语义变化
  - eager vs XLA: 检测 XLA 编译引入的错误
  - Grappler vs XLA: 检测两种优化路径的分歧
- **Secondary**: CPU vs GPU differential (当 XLA 在不同 device 上 behavior 不同)
- **Filters**: 同 PyTorch — determinism check + perturbation stability

### Oracle contribution (δ) analysis

| Oracle level | What it detects | TF bugs found |
|-------------|-----------------|---------------|
| Crash-only | XLA/Grappler compilation failures | #117546, #117547 (2 bugs) |
| + Grappler differential | Grappler rewrite semantic changes | #116436 (1 bug) |
| + XLA differential | XLA silent numerical divergences | #116992, #117319 (2 bugs) |
| + Boundary values | Eager kernel numerical issues | #116943, #116945 (2 bugs) |

Without the 3-way differential oracle, only 2/7 bugs (the crashes) would be detected. The oracle provides $\delta$ for 5/7 bugs = **71% of TF bugs require the differential oracle** (comparable to PyTorch's 75%).

---

## 7. Bug → State Mapping (类比 Table: Family Reachability)

| Bug | Pipeline State | Category | Random reachable? |
|-----|---------------|----------|-------------------|
| #116436 | Grappler × fp32 × static | T1 | ⚠️ 需要 non-monotonic pre-op + argmin/argmax composition |
| #116943 | eager × fp32 × boundary | T2 | ⚠️ 需要 small negative boundary values |
| #116945 | eager × complex64 × boundary | T3 | ❌ random 很少生成 complex dtype (5% weight) + small magnitude |
| #116992 | XLA × fp32 × boundary | T7 | ⚠️ 需要 NaN input + XLA path + reduce_max |
| #117319 | XLA × fp32 × boundary | T7 | ❌ 需要 check_numerics + NaN + XLA specific combination |
| #117546 | XLA × complex64 × static | T3 | ❌ 需要 complex + cumsum + XLA |
| #117547 | XLA × fp32 × empty | T4 | ❌ 需要 empty-dim tensor (shape=[N,0]) — random 不在采样空间 |

**Random structurally hard to reach: 4/7 bugs** (complex dtype + empty shapes + specific op combinations)

---

## 8. Experimental Validation Plan

完全 parallel PyTorch 的 RQ 结构：

### RQ1 (Effectiveness)
- Table: 7 TF bugs with state, trigger pattern, oracle, status, root cause
- 已有数据，只需写 table

### RQ2 (Reachability)  
- Time-controlled comparison: targeted (121 templates) vs random, 300s, 5 seeds
- Metric: cross-pipeline inconsistency count (diff triggers)
- **实验正在跑**

### RQ3 (Coverage Proxy)
- Graph diversity $\mathcal{D}_{\text{TF}}$ over time
- Plot: $\mathcal{D}_{\text{TF}}(t)$ for targeted vs random
- Metric: unique (group × pipeline_state) pairs reached
- **需要从实验结果中提取**

### RQ4 (Detectability)
- Oracle contribution: crash-only detects 2/7, full 3-way detects 7/7
- 71% of TF bugs are silent → differential oracle is structurally necessary
- **可从已有 bug data 静态分析**

---

## 9. Paper Restructuring Implication

当前结构:
```
RQ1: bugs (PT-focused)
RQ2: ρ (PT-only ablation)
RQ3: D(t) (PT-only coverage)
RQ4: δ (PT-only oracle)
RQ5: transferability (TF/IREE/TVM as secondary)
```

新结构 (parallel):
```
RQ1: bugs (PT 10 + TF 7, both with detailed tables)
RQ2: ρ (PT targeted 2.9× random + TF targeted X× random)
RQ3: D(t) (PT kernel diversity + TF graph diversity)  
RQ4: δ (PT 75% silent + TF 71% silent)
RQ5: 可以变成 "Further Transferability" (IREE/TVM) 或删除
```

这样 Definition 2 ($\mathcal{S}, \mathcal{T}, \mathcal{D}$ triple) 就有了两个 **quantitative instantiations**，不再是装饰性抽象。
