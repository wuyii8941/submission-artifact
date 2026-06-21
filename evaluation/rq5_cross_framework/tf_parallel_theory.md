# TensorFlow Parallel Methodology: Formal Definition

We instantiate the same theoretical framework used for PyTorch/Inductor on TensorFlow, exactly mirroring its structure.

---

## 1. Problem Formulation (same as PyTorch)

$$
P(\text{find}\; b) = \underbrace{P(\text{reach}\; b)}_{\rho_b(q)} \times \underbrace{P(\text{detect}\; b)}_{\delta_b(O;\, q)}
$$

- $\rho_b(q)$: probability that generator $q$ drives the program into the execution-pipeline state that triggers bug $b$.
- $\delta_b(O;\, q)$: probability that oracle $O$ observes a semantic deviation in that state.

---

## 2. State Space $\mathcal{S}_{\text{TF}}$

### Recap of the PyTorch state space

PyTorch state = kernel family (7 in total): determined by Inductor's `_get_heuristic()` and extern dispatch. Which kernel family executes depends on op type + shape.

### TensorFlow state space

TensorFlow has no kernel-level scheduling heuristic. Its "compiler decision" instead manifests as **execution-pipeline selection**: the same TF program can take three different execution pipelines, each with its own optimisations and lowering behaviour.

**Definition: TensorFlow Execution Pipeline Space**

$$
\mathcal{S}_{\text{TF}} = \mathcal{P} \times \mathcal{D} \times \mathcal{R}
$$

where:
- $\mathcal{P} = \{\text{eager},\; \text{Grappler},\; \text{XLA}\}$ - execution pipeline
  - eager: dispatches directly to Eigen/cuDNN kernels, no graph optimisation.
  - Grappler: `tf.function(jit_compile=False)`, graph-level optimisation (constant folding, arithmetic simplification, layout optimisation).
  - XLA: `tf.function(jit_compile=True)`, full compilation (fusion, algebraic simplification, buffer assignment, codegen).
- $\mathcal{D} = \{\text{fp32},\; \text{fp64},\; \text{fp16},\; \text{bf16},\; \text{complex64},\; \text{complex128}\}$ - dtype regime.
- $\mathcal{R} = \{\text{static},\; \text{dynamic},\; \text{boundary},\; \text{empty}\}$ - shape regime.
  - static: ordinary fixed shapes.
  - dynamic: `tf.TensorSpec(shape=[None, ...])` dynamic dimensions.
  - boundary: boundary conditions (dim=1, dim=0, very large).
  - empty: empty dimensions (e.g., shape=[4, 0]).

Total valid state count: $|\mathcal{S}_{\text{TF}}| = 3 \times 6 \times 4 = 72$ pipeline-dtype-shape combinations.

### Analogy

| | PyTorch | TensorFlow |
|--|---------|------------|
| State drivers | op type + tensor shape -> kernel family | execution mode + dtype + shape regime -> pipeline path |
| State count | 7 families (6 on T4) | 72 combinations (3 x 6 x 4) |
| State observability | `_get_heuristic()` hook | execution mode is an explicit choice; dtype/shape are input properties |
| Relation between bugs and state | bugs concentrate in specific families (e.g., split_scan) | bugs concentrate in specific pipelines (e.g., XLA) and dtypes (e.g., complex) |

---

## 3. Template Space $\mathcal{T}_{\text{TF}}$ (analogue of K1-K5)

### Recap of the PyTorch template space

K1-K5: five operator-level codegen paths, enumerated from the Inductor lowering registry via the `OpsForFamily` mapping. 340 templates = K1-K5 x 6 composition rules.

### TensorFlow template space

T1-T7: seven graph-transformation categories, each targeting a class of compilation seam.

| Category | Target Compilation Seam | Ops/Patterns | Cases |
|----------|------------------------|--------------|-------|
| T1: Rewrite-sensitive | Grappler graph rewrites (arithmetic simplification, dead code elimination) | ArgMin/ArgMax + non-monotonic pre-ops, fused patterns | 22 |
| T2: Numerically unstable | Eager kernel numerical contracts (catastrophic cancellation, overflow) | exp, log, SELU, ELU, erf, erfc, igamma | 24 |
| T3: Dtype-sensitive | Cast chain folding, precision demotion across pipelines | bf16/fp16 cast chains, complex dtype ops, saturating casts | 19 |
| T4: Shape-dependent | Reduction/reshape/broadcast lowering with shape boundary conditions | reduce with empty/boundary dims, reshape, index ops | 16 |
| T5: Data-dependent shape | Dynamic shape ops where output shape depends on values | unique, where, boolean_mask | 2 |
| T6: Control-flow | Graph-mode control flow vs eager semantics | tf.while_loop, tf.cond, stateful RNG | 4 |
| T7: Backend divergence | XLA vs eager/Grappler behavioural differences (NaN propagation, dead code, special values) | NaN semantics, check_numerics, gradient boundaries | 34 |

**Total: 121 templates across 19 sub-groups (collapsed to 7 categories).**

### OpsForCategory mapping (analogue of OpsForFamily)

```
OpsForCategory(T1) = {argmin, argmax, softmax, log_softmax, ...}  # Grappler-rewrite-sensitive
OpsForCategory(T2) = {exp, log, selu, elu, erf, erfc, igamma, ...}  # numerically fragile
OpsForCategory(T3) = {cast, saturate_cast, complex ops, ...}  # dtype-sensitive
OpsForCategory(T4) = {reduce_*, reshape, gather, scatter, ...}  # shape-dependent  
OpsForCategory(T5) = {unique, where, boolean_mask, ...}  # data-dependent shape
OpsForCategory(T6) = {while_loop, cond, ...}  # control-flow
OpsForCategory(T7) = {check_numerics, NaN-propagating ops, gradient ops, ...}  # backend-divergent
```

### Composition Rules (analogue of PyTorch's 6 composition rules)

Composition patterns for TF templates:
1. **Single op**: single-op test of pipeline behaviour.
2. **Chain**: op1 -> op2 -> ... (fusion-boundary tests).
3. **Dead code**: unused result + live result (dead-code-elimination tests).
4. **Cross-pipeline**: compare the same program under eager/Grappler/XLA.
5. **Boundary input**: extreme values (NaN, Inf, very small, very large, empty).
6. **Dtype chain**: cast -> op -> cast (precision-preservation tests).

---

## 4. Diversity Metric $\mathcal{D}_{\text{TF}}$ (analogue of Kernel Diversity)

### Recap of PyTorch kernel diversity

$$
\mathcal{D}_{\text{PT}}(\mathcal{P}) = |\bigcup_{i} \{\tau(k) \mid k \in K(P_i)\}|
$$

where $\tau(k)$ = kernel template identity (scheduling family x fused-op composition).

### TensorFlow graph diversity

**Definition: Graph Diversity (formal version, fully parallel to Kernel Diversity).**

Recap of PyTorch kernel diversity:
- $\tau(k) = (\text{scheduling\_family}, \text{fused\_op\_set})$ - extracted from compiler output.
- scheduling_family is from `_get_heuristic()`, an observable of the compiler's path choice.
- fused_op_set is from the op names in the Triton kernel, an observable of the program's op composition.

The formal parallel for TF graph diversity:

Given a set of programs $\mathcal{P} = \{P_1, \ldots, P_n\}$ executed through TF's three pipelines, define each program's **graph transformation signature**:

$$
\sigma(P_i) = (\text{op\_sig}(P_i),\; \text{dtype}(P_i),\; \text{outcome}(P_i))
$$

where:
- $\text{op\_sig}(P_i)$: the set of TF op types used by the program - **analogue of fused_op_set**, an observable of the program's structure.
- $\text{dtype}(P_i)$: input dtype - influences the compiler's precision-handling decisions.
- $\text{outcome}(P_i) \in \{0, 1\}^3$: pairwise consistency pattern across the three pipelines (eager~grappler?, eager~xla?, grappler~xla?) - **analogue of scheduling_family**, an observable of compiler behaviour.

**Graph diversity:**
$$
\mathcal{D}_{\text{TF}}(\mathcal{P}) = |\{\sigma(P_i) \mid P_i \in \mathcal{P}\}|
$$

### Formal parallel: side-by-side comparison

| | PyTorch $\tau(k)$ | TF $\sigma(P)$ |
|--|-------------|---------|
| Op composition axis | fused_op_set (op names in the Triton kernel) | op_sig (program's TF op-type set) |
| Compiler behaviour axis | scheduling_family (_get_heuristic()) | outcome (3-way pipeline consistency pattern) |
| Source | kernel structure emitted by the compiler | compiler-observed behaviour on the program |
| Meaning | unique codegen paths exercised | unique compilation outcomes observed |

### Space size

The upper bound depends on the diversity of op compositions:
- op_sig: for targeted = 121 templates' unique op sets; for random = C(30, 1..4) combinations >> 100.
- dtype: 6.
- outcome: $2^3 = 8$ patterns (plus crash modes, ~10 in total).
- Theoretical upper bound >> 1000, sufficient to produce a meaningful growth curve.

---

## 5. Targeted Generation (analogue of Algorithm 1)

### Recap of PyTorch Algorithm 1

```
Input: target family f, dtype d, fusion degree n
1. O <- OpsForFamily(f)
2. G <- ComposeGraph(O, n)
3. theta.dtype <- d
4. theta.shape <- BoundaryShape(f, d)
Output: program p = (G, theta)
```

### TensorFlow algorithm (parallel)

```
Input: target category t in {T1,...,T7}, dtype d, shape regime r
1. O <- OpsForCategory(t)              // category -> op pool
2. G <- ComposePattern(O, t)           // category-specific composition
3. theta.dtype <- d
4. theta.shape <- BoundaryCondition(t, r)  // category-specific boundary
5. pipeline <- SelectPipeline(t)       // which pipelines to exercise
Output: program p = (G, theta), pipeline configuration
```

**BoundaryCondition (analogue of BoundaryShape):**
- T1: NaN, Inf, tied values, non-strictly-monotonic sequences (Grappler rewrite boundaries).
- T2: small negatives near 0, very large values (numerical overflow/underflow thresholds).
- T3: values where bf16 truncation loses information.
- T4: empty dims (shape=[N,0]), dim=1, very large dims.
- T5: all-same values, all-unique values.
- T7: NaN in reduction, Inf in normalisation.

---

## 6. Differential Oracle $O_{\text{TF}}$ (analogue of eager-vs-compiled)

### Recap of the PyTorch oracle

- Primary: eager vs `torch.compile(backend='inductor')`.
- Secondary: aot_eager cross-check, backward-mode amplification, cross-config.

### TensorFlow oracle: 3-way pipeline differential

- **Primary**: eager vs Grappler vs XLA (three pairwise comparisons).
  - eager vs Grappler: detects semantic changes introduced by graph optimisation.
  - eager vs XLA: detects errors introduced by XLA compilation.
  - Grappler vs XLA: detects divergences between the two optimisation paths.
- **Secondary**: CPU vs GPU differential (when XLA behaves differently across devices).
- **Filters**: same as PyTorch - determinism check + perturbation stability.

### Oracle contribution ($\delta$) analysis

| Oracle level | What it detects | TF bugs found |
|-------------|-----------------|---------------|
| Crash-only | XLA/Grappler compilation failures | #anon-1, #anon-2 (2 bugs) |
| + Grappler differential | Grappler rewrite semantic changes | #anon-3 (1 bug) |
| + XLA differential | XLA silent numerical divergences | #anon-4, #anon-5 (2 bugs) |
| + Boundary values | Eager kernel numerical issues | #anon-6, #anon-7 (2 bugs) |

Without the 3-way differential oracle, only 2/7 bugs (the crashes) would be detected. The oracle provides $\delta$ for 5/7 bugs = **71% of TF bugs require the differential oracle** (comparable to PyTorch's 75%).

---

## 7. Bug -> State Mapping (analogue of the Family Reachability table)

| Bug | Pipeline State | Category | Random reachable? |
|-----|---------------|----------|-------------------|
| #anon-3 | Grappler x fp32 x static | T1 | Partial: requires a non-monotonic pre-op composed with argmin/argmax. |
| #anon-6 | eager x fp32 x boundary | T2 | Partial: requires small-negative boundary values. |
| #anon-7 | eager x complex64 x boundary | T3 | No: random rarely generates complex dtype (5% weight) + small magnitude. |
| #anon-4 | XLA x fp32 x boundary | T7 | Partial: requires NaN input + XLA path + reduce_max. |
| #anon-5 | XLA x fp32 x boundary | T7 | No: requires check_numerics + NaN + XLA-specific combination. |
| #anon-1 | XLA x complex64 x static | T3 | No: requires complex + cumsum + XLA. |
| #anon-2 | XLA x fp32 x empty | T4 | No: requires an empty-dim tensor (shape=[N,0]) - outside random's sampling space. |

**Random structurally hard to reach: 4/7 bugs** (complex dtype + empty shapes + specific op combinations).

---

## 8. Experimental Validation Plan

Fully parallel to PyTorch's RQ structure:

### RQ1 (Effectiveness)
- Table: 7 TF bugs with state, trigger pattern, oracle, status, and root cause.
- Data already collected; only the table needs to be written.

### RQ2 (Reachability)
- Time-controlled comparison: targeted (121 templates) vs random, 300 s, 5 seeds.
- Metric: cross-pipeline inconsistency count (diff triggers).
- **Experiment in progress.**

### RQ3 (Coverage Proxy)
- Graph diversity $\mathcal{D}_{\text{TF}}$ over time.
- Plot: $\mathcal{D}_{\text{TF}}(t)$ for targeted vs random.
- Metric: unique (group x pipeline_state) pairs reached.
- **To be extracted from the experiment results.**

### RQ4 (Detectability)
- Oracle contribution: crash-only detects 2/7; full 3-way detects 7/7.
- 71% of TF bugs are silent -> the differential oracle is structurally necessary.
- **Analysable statically from existing bug data.**

---

## 9. Paper Restructuring Implication

Current structure:
```
RQ1: bugs (PT-focused)
RQ2: rho (PT-only ablation)
RQ3: D(t) (PT-only coverage)
RQ4: delta (PT-only oracle)
RQ5: transferability (TF/IREE/TVM as secondary)
```

New structure (parallel):
```
RQ1: bugs (PT 10 + TF 7, both with detailed tables)
RQ2: rho (PT targeted 2.9x random + TF targeted Xx random)
RQ3: D(t) (PT kernel diversity + TF graph diversity)  
RQ4: delta (PT 75% silent + TF 71% silent)
RQ5: can become "Further Transferability" (IREE/TVM) or be dropped
```

This gives Definition 2 (the $\mathcal{S}, \mathcal{T}, \mathcal{D}$ triple) two **quantitative instantiations**, so it is no longer a decorative abstraction.
