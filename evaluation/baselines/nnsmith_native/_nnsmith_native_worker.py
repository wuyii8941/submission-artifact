#!/usr/bin/env python3
"""
NNSmith native oracle worker: generates models with NNSmith and tests them
using NNSmith's own multi-backend differential oracle.

Oracle: torchjit-CPU (reference) vs torchjit-CUDA (test)
This is NNSmith's standard usage pattern for finding CPU/CUDA divergences.

Also tests: eager-CUDA vs torch.compile/Inductor-CUDA (our oracle) for
comparison with the main experiment.
"""

import argparse
import json
import os
import sys
import time
import random as pyrandom
import traceback
from collections import defaultdict

import numpy as np
import torch

# NNSmith must be importable (pip install nnsmith[torch,onnx])


def run(seed, time_budget):
    from nnsmith.materialize.torch import TorchModelCUDA, TorchModel
    from nnsmith.narrow_spec import auto_opset
    from nnsmith.backends.torchjit import TorchJITFactory
    from nnsmith.graph_gen import model_gen
    from nnsmith.util import set_seed

    cpu_factory = TorchJITFactory(target='cpu')
    cuda_factory = TorchJITFactory(target='cuda')

    opset = auto_opset(TorchModelCUDA, cuda_factory)

    bugs = []
    crashes = 0
    inconsistencies = 0
    gen_failures = 0
    total = 0
    timeseries = []

    t0 = time.time()
    i = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break

        seed_i = seed * 100000 + i
        set_seed(seed_i)
        pyrandom.seed(seed_i)
        max_nodes = pyrandom.choice([3, 5, 8, 10, 15])

        try:
            gen = model_gen(opset=opset, seed=seed_i, max_nodes=max_nodes)
            ir = gen.make_concrete()
            model = TorchModelCUDA.from_gir(ir)
            model.refine_weights()
        except Exception as e:
            gen_failures += 1
            i += 1
            continue

        net = model.torch_model.eval()

        if model.sat_inputs is not None:
            inputs_np = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                         for k, v in model.sat_inputs.items()}
        else:
            try:
                rand_inps = net.get_random_inps()
                inputs_np = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                             for k, v in rand_inps.items()}
            except Exception:
                gen_failures += 1
                i += 1
                continue

        total += 1

        # --- NNSmith native oracle: CPU vs CUDA torchjit ---
        try:
            cpu_model = TorchModel.from_gir(ir)
            cpu_model.torch_model.load_state_dict(
                {k: v.cpu() for k, v in net.state_dict().items()})

            cpu_callable = cpu_factory.make_backend(cpu_model)
            cuda_callable = cuda_factory.make_backend(model)

            cpu_out = cpu_callable(inputs_np)
            cuda_out = cuda_callable(inputs_np)

            max_diff = 0.0
            for key in cpu_out:
                if key in cuda_out:
                    diff = np.abs(cpu_out[key].astype(np.float64)
                                  - cuda_out[key].astype(np.float64))
                    md = float(diff.max()) if diff.size > 0 else 0.0
                    max_diff = max(max_diff, md)

            if max_diff > 1e-2:
                bug_type = 'INCONSISTENCY_cpu_cuda'
                if max_diff > 10.0:
                    bug_type = 'CATASTROPHIC_cpu_cuda'
                inconsistencies += 1
                bugs.append({
                    'idx': i,
                    'time_sec': round(elapsed, 1),
                    'bug_type': bug_type,
                    'max_diff': max_diff,
                    'oracle': 'cpu_vs_cuda',
                    'max_nodes': max_nodes,
                    'seed_i': seed_i,
                })

        except Exception as e:
            err_str = str(e)[:200]
            if 'compile' in err_str.lower() or 'jit' in err_str.lower():
                crashes += 1
                bugs.append({
                    'idx': i,
                    'time_sec': round(elapsed, 1),
                    'bug_type': 'crash_native',
                    'error': err_str,
                    'oracle': 'cpu_vs_cuda',
                    'max_nodes': max_nodes,
                    'seed_i': seed_i,
                })

        # --- Also run our oracle: eager-CUDA vs compiled-CUDA ---
        try:
            inputs_cuda = {k: torch.from_numpy(v).cuda()
                           for k, v in inputs_np.items()}

            with torch.no_grad():
                eager_out = net(**inputs_cuda)

            torch._dynamo.reset()
            compiled_net = torch.compile(net, backend='inductor')
            with torch.no_grad():
                compiled_out = compiled_net(**inputs_cuda)

            if isinstance(eager_out, (tuple, list)):
                eager_list = list(eager_out)
                compiled_list = list(compiled_out)
            else:
                eager_list = [eager_out]
                compiled_list = [compiled_out]

            max_diff_compile = 0.0
            for eo, co in zip(eager_list, compiled_list):
                if isinstance(eo, torch.Tensor) and isinstance(co, torch.Tensor):
                    if eo.shape != co.shape:
                        bugs.append({
                            'idx': i,
                            'time_sec': round(elapsed, 1),
                            'bug_type': 'shape_mismatch_compile',
                            'oracle': 'eager_vs_compiled',
                            'max_nodes': max_nodes,
                            'seed_i': seed_i,
                        })
                        break
                    d = (eo.float() - co.float()).abs().max().item()
                    max_diff_compile = max(max_diff_compile, d)

            if max_diff_compile > 1e-3:
                bt = 'CATASTROPHIC_compile' if max_diff_compile > 10.0 else 'SUSPICIOUS_compile'
                bugs.append({
                    'idx': i,
                    'time_sec': round(elapsed, 1),
                    'bug_type': bt,
                    'max_diff': max_diff_compile,
                    'oracle': 'eager_vs_compiled',
                    'max_nodes': max_nodes,
                    'seed_i': seed_i,
                })

            torch._dynamo.reset()

        except Exception:
            torch._dynamo.reset()

        if i % 10 == 0:
            timeseries.append({
                'iter': i,
                'time_sec': round(elapsed, 1),
                'total_bugs': len(bugs),
                'inconsistencies': inconsistencies,
                'crashes': crashes,
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:5d}] bugs={len(bugs)} incon={inconsistencies} "
                  f"crash={crashes} gen_fail={gen_failures} "
                  f"t={elapsed:.0f}/{time_budget}s", file=sys.stderr)

        i += 1

    elapsed = time.time() - t0
    timeseries.append({
        'iter': i,
        'time_sec': round(elapsed, 1),
        'total_bugs': len(bugs),
        'inconsistencies': inconsistencies,
        'crashes': crashes,
    })

    # Classify bugs by oracle type
    native_bugs = [b for b in bugs if b['oracle'] == 'cpu_vs_cuda']
    compile_bugs = [b for b in bugs if b['oracle'] == 'eager_vs_compiled']

    output = {
        'seed': seed,
        'time_budget': time_budget,
        'iterations': i,
        'elapsed_sec': round(elapsed, 1),
        'total_generated': total,
        'gen_failures': gen_failures,
        'bugs_found': len(bugs),
        'inconsistencies': inconsistencies,
        'crashes': crashes,
        'native_oracle_bugs': len(native_bugs),
        'compile_oracle_bugs': len(compile_bugs),
        'bugs': bugs,
        'timeseries': timeseries,
    }

    print(json.dumps(output, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time-budget', type=int, default=300)
    args = parser.parse_args()

    run(args.seed, args.time_budget)
