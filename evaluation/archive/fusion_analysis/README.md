# Fusion Interaction Analysis

Experiments supporting the paper's core finding:
**DL compiler bugs disproportionately emerge from under-exercised semantic interaction regions.**

## Semantic Categories

Operator categories are derived from PyTorch Inductor's own IR class hierarchy
(`torch/_inductor/ir.py`), which the compiler uses for all fusion decisions:

| Category | IR Class | Source |
|----------|----------|--------|
| Pointwise | `ir.Pointwise` | ir.py:1126 |
| Reduction | `ir.Reduction` | ir.py:1276 |
| Scan | `ir.Scan` | ir.py:2442 |
| Scatter | `ir.Scatter` | ir.py:1167 |
| ExternKernel | `ir.ExternKernel` | ir.py:6243 |

Kernel name prefixes (`poi`/`red`/`per`/`spl`) are extracted by
`codegen/triton.py:6607` via `get_kernel_category_by_source_code()`.

## Experiments

### 1. Random Baseline (1000 programs)

**Script:** `fusion_baseline_1k.py`
**Output:** `summary_random_1k.json`, `summary_targeted_1k.json`

Generates 1000 random programs, compiles each with `torch.compile(backend='inductor')`,
and extracts fusion patterns from generated Triton kernel names.

Key findings:
- 852/1000 compiled successfully, 57 unique fusion ops
- Op discovery saturates at ~200 programs (saturation curve)
- 829 unique fusion patterns but only 57 unique ops

### 2. Fusion Op Frequency Analysis

**Script:** `fusion_space_analysis.py`
**Output:** `fusion_analysis_results.json`, figures in `tex/figures/`

Maps 17 confirmed Inductor bugs to their fusion ops and classifies by rarity:
- 11/18 bug-relevant fusion ops are completely absent from 1000 random programs
- 9/17 bugs involve absent fusion ops
- 0/13 bugs (with fusion ops) are in common (>=10%) regions

Figures:
- `fusion_op_frequency.pdf` — op frequency distribution (red=bug-relevant)
- `op_saturation_curve.pdf` — op discovery saturation
- `bug_rarity_distribution.pdf` — bug count by op rarity
- `fusion_distribution_shift.pdf` — random top-30 vs absent bug ops

### 3. Cross-Semantic Boundary Analysis

**Script:** `fusion_boundary_analysis.py`
**Output:** `boundary_analysis_results.json`, figures in `tex/figures/`

Classifies each fusion pattern by which Inductor IR categories it spans,
then correlates boundary type with bug incidence.

Key findings:
- pointwise x reduction (392 fusions, 0 bugs) — well-exercised, safe
- scatter x scatter (0 fusions, 4 bugs) — blind spot
- extern x pointwise (8 fusions, 3 bugs) — under-exercised boundary
- 9/17 bugs in boundary types with <1% baseline frequency

Figures:
- `semantic_interaction_heatmap.pdf` — frequency vs bug incidence side-by-side
- `boundary_frequency_vs_bugs.pdf` — boundary types ranked by frequency

### 4. Interaction Coverage vs Bug Yield

**Script:** `interaction_coverage_vs_bugs.py`
**Output:** `interaction_coverage_results.json`, figures in `tex/figures/`

The causal chain closer: shows that random/targeted generation saturates at a
small number of semantic interaction cells, and that a large fraction of bugs
occupy cells NEVER explored by baselines.

Key findings:
- Random (852 programs) and Targeted (1000 programs) both saturate at exactly 9 interaction cells
- Saturation occurs by program #140 (Random) / #241 (Targeted) — 76-84% of programs wasted
- 4/8 bug-relevant cells (50%) have ZERO baseline coverage
- 8/17 bugs (47%) are in permanently unreachable interaction cells
- Unreachable cells: scatter-only (4 bugs), extern×pointwise (2), extern×pointwise×scan (1), layout×reduction (1)

Figures:
- `interaction_growth_curve.pdf` — cell discovery saturation (programs vs cells)
- `bug_reachability_gap.pdf` — bugs split by cell reachability
- `per_bug_interaction_attribution.pdf` — each bug's cell frequency in baseline

## Reproduction

```bash
# Environment
source ${ARTIFACT_ROOT}/pt_nightly_env/bin/activate

# 1. Generate baselines (takes ~45 min per method)
CUDA_VISIBLE_DEVICES=0 python -u fusion_baseline_1k.py -n 1000 -m random
CUDA_VISIBLE_DEVICES=1 python -u fusion_baseline_1k.py -n 1000 -m targeted

# 2. Fusion op analysis
python fusion_space_analysis.py

# 3. Boundary analysis
python fusion_boundary_analysis.py

# 4. Interaction coverage vs bug yield
python interaction_coverage_vs_bugs.py
```
