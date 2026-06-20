# RQ5: Cross-Framework Transferability

Demonstrates that the oracle design, adversarial shapes, and bug taxonomy
transfer to TensorFlow/XLA, JAX/XLA, IREE, and TVM without a
framework-specific state model.

## Scripts

### TensorFlow/XLA (4 bugs)
| Script | Purpose |
|--------|---------|
| `tf_main.py` | Main TF fuzzer loop (eager vs tf.function vs XLA) |
| `tf_oracles.py` | Differential oracles: eager/grappler/XLA seams |
| `tf_graph_gen.py` | Random TF program generator with adversarial shapes |
| `tf_coverage.py` | HLO/GraphDef coverage tracking |
| `targeted_consistency_fuzzer.py` | Targeted consistency testing for XLA |

### JAX/XLA (1 bug)
| Script | Purpose |
|--------|---------|
| `jax_main.py` | Main JAX fuzzer loop (jit vs eager, grad, vmap) |
| `jax_oracles.py` | Differential oracles for JAX compilation seams |
| `jax_graph_gen.py` | Random JAX program generator |
| `pattern_migration_test.py` | Tests PyTorch pattern transferability to JAX |

### IREE (4 bugs)
| Script | Purpose |
|--------|---------|
| `iree_runner.py` | IREE compilation + ONNX Runtime oracle |

### TVM (5 bugs)
Bug reproducers in `${ARTIFACT_ROOT}/paper/confirmed_bugs/tvm/`
(discovered via manual Relax/DLight schedule rule testing).

## Source

Scripts are symlinked from:
- `${ARTIFACT_ROOT}/tf_cross/fuzzer/`
- `${ARTIFACT_ROOT}/jax_cross/fuzzer/`

## Reproducing

```bash
# TensorFlow/XLA fuzzer (requires TF 2.x + XLA)
cd ${ARTIFACT_ROOT}/tf_cross
bash harness/run_tf.sh fuzzer/main.py --iters 5000 --seed 42 \
    --oracles eager_vs_tfn,tfn_vs_xla,grad,cpu_gpu

# JAX/XLA fuzzer (requires JAX 0.4+)
cd ${ARTIFACT_ROOT}/jax_cross
bash harness/run_jax.sh fuzzer/main.py --iters 5000 --seed 42

# Pattern migration validation
python pattern_migration_test.py

# IREE (requires iree-compiler + onnxruntime)
python iree_runner.py --model /path/to/model.onnx
```

## Paper Claims

- 14 cross-framework bugs total (4 TF, 1 XLA, 4 IREE, 5 TVM)
- Oracle and adversarial-shape methodology transfers without
  the PyTorch-specific kernel-family model
- Bug types: silent numerical divergence, dtype demotion, shape
  canonicalization failures, schedule rule crashes

## Bug Reproducers

All confirmed bugs with minimal reproduction scripts:
- `../../confirmed_bugs/tensorflow/` (12 directories)
- `../../confirmed_bugs/iree/` (9 directories)
- `../../confirmed_bugs/tvm/` (7 directories)
- `../../confirmed_bugs/xla/` (1 directory)
