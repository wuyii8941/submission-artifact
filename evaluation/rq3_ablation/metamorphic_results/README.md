# Reference-Free (Metamorphic) Oracle Results

These JSON files back the appendix's reference-free rediscovery table
(6/6 PyTorch/Inductor formulations + 4/4 TensorFlow/XLA formulations
detected on the test nightly without an eager-vs-compiled comparison).

| File pattern | Oracle family | Stack |
|---|---|---|
| `composed_*.json` | composed metamorphic relations | PyTorch |
| `equivalence_*.json` | equivalence relations | PyTorch |
| `batch_indep_*.json` | batch-independence metamorphic | PyTorch |
| `frontier_*.json`, `wide_*.json`, `exotic_*.json`, `targeted_*.json` | additional metamorphic / property checks | PyTorch |
| `tf_xla_*.json`, `tf_cumsum_deep_*.json`, `tf_rediscovery_*.json` | TF/XLA reference-free rediscovery | TensorFlow |

Each JSON records the formulation name, the bug it targets, and whether
the bug was detected or already fixed on the test nightly. The appendix
table aggregates these into per-oracle counts (`P`, `E`, `M`, `G`).
