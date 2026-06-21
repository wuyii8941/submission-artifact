# RQ3: Ablation

RQ3 isolates the contribution of (a) operation-guided generation and
(b) the multi-stage oracle, using two complementary experiments:

## (1) Component ablation on the historical benchmark

Toggle the generation mode (targeted / random / nnsmith / freefuzz /
whitefox) and re-run the **same** benchmark used for RQ2. The driver
is therefore `../rq2_efficiency/run_benchmark.py`; per-mode JSONs are
re-aggregated into Table 3.

```bash
cd ../rq2_efficiency
./launch_template_runs.sh   # 5 seeds × 4 modes × 4h budget
```

Per-kernel hit counts (Table 5: saturation depth at K=1 and K=10) come
from the same JSON checkpoints — see `timeseries` field.

## (2) External BugsInDLLs validation (Table 4)

`bugsindlls/` is the BugsInDLLs API-level benchmark. It checks whether
operation guidance reaches BugsInDLLs ground-truth bug paths that
unguided API-level mutation misses.

```bash
cd bugsindlls
python run_bugsindlls_batch.py
```

See `bugsindlls/README.md` for the per-API breakdown.
