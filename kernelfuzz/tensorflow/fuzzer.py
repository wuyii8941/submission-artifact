"""TF graph fuzzer -- main loop.

Two-phase: in-process scan + subprocess verify.

Usage:
    python -m kernelfuzz.tensorflow.fuzzer [--iters N] [--seed S] [--oracles eager,xla,grad,cpu_gpu]

Oracles:
    eager     -- eager vs tf.function (Grappler diff)                    (always on)
    xla       -- tf.function vs tf.function(jit_compile=True)
    grad      -- backward GradientTape: eager vs tf.function
    cpu_gpu   -- tf.function on CPU vs GPU
    determinism -- same run twice, should be byte-exact
    mixed     -- fp32 vs mixed_float16 path
"""
import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import tensorflow as tf
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

from .graph_gen import GraphGenerator, generate_program_with_inputs
from .oracles import (oracle_eager_vs_tfn, oracle_tfn_vs_xla,
                      oracle_grad_eager_vs_tfn, oracle_cpu_vs_gpu,
                      oracle_eager_cpu_gpu, oracle_eager_grad_cpu_gpu,
                      oracle_math_invariant,
                      oracle_determinism_diff, oracle_mixed_precision_diff,
                      oracle_f64_truth,
                      has_bug, bug_severity, report)
from .coverage import (CoverageTracker, BugBasinTracker,
                       get_hlo_text, get_graph_def_hash, extract_hlo_ops)
from .kernel_state import TFKernelStateSpace, EpsilonGreedyScheduler

KNOWN_BASINS = [
    ({"cast_bf16_f32"}, "bf16 cast elision"),
    ({"cast_f16_f32"}, "fp16 cast elision"),
]

_GRAPPLER_ARGMIN_MONOTONIC_OPS = {
    "nn_relu", "k3_relu", "k3_relu6", "raw_relu",
    "amp_floor", "amp_ceil", "amp_round", "amp_rint",
    "k3_floor", "k3_ceil", "m_rint", "m_round", "m_sign", "k3_sign",
    "amp_softmax_floor",
}

def _is_grappler_argmin_basin(op_fingerprint, oracle_results):
    has_argminmax = "amp_argmin" in op_fingerprint or "amp_argmax" in op_fingerprint
    if not has_argminmax:
        return False
    has_monotonic = bool(set(op_fingerprint) & _GRAPPLER_ARGMIN_MONOTONIC_OPS)
    if not has_monotonic:
        return False
    for r in oracle_results:
        if r.get("oracle") == "eager_vs_tfn" and r.get("status") == "DIFF":
            return True
    return False


def build_fn_and_args(prog_info):
    ns = {}
    exec(prog_info["setup"], ns)
    exec(prog_info["fn_code"], ns)
    fn = ns["fn"]
    args = [ns[f"x{i}"] for i in range(prog_info["n_inputs"])]
    return fn, args


def run_single(prog_info, oracle_flags, coverage_tracker=None, basin_tracker=None):
    seed = prog_info["seed"]

    try:
        fn, args = build_fn_and_args(prog_info)
    except Exception as e:
        return {"status": "GEN_ERROR", "error": str(e)[:200], "seed": seed}

    hlo_info = {}
    hlo_ops = frozenset()
    if coverage_tracker is not None and (seed % 5 == 0):
        try:
            hlo_text = get_hlo_text(fn, args)
            graph_hash, _ = get_graph_def_hash(fn, args)
            if hlo_text:
                hlo_ops = extract_hlo_ops(hlo_text)
            is_novel, details = coverage_tracker.check_novelty(
                hlo_text=hlo_text, graph_sig=graph_hash
            )
            hlo_info = {"is_novel": is_novel, "hlo_ops": sorted(hlo_ops), **details}
        except Exception:
            hlo_info = {}

    oracle_results = []

    r = oracle_eager_vs_tfn(fn, args, label=f"seed={seed}")
    oracle_results.append(r)

    if oracle_flags.get("xla", True):
        r = oracle_tfn_vs_xla(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    if oracle_flags.get("grad", True):
        r = oracle_grad_eager_vs_tfn(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    if oracle_flags.get("cpu_gpu", False):
        r = oracle_cpu_vs_gpu(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    if oracle_flags.get("eager_cpu_gpu", False):
        r = oracle_eager_cpu_gpu(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    if oracle_flags.get("eager_grad_cpu_gpu", False):
        r = oracle_eager_grad_cpu_gpu(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    if oracle_flags.get("math_invariant", False):
        r = oracle_math_invariant(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    if oracle_flags.get("determinism", False):
        r = oracle_determinism_diff(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    if oracle_flags.get("mixed", False):
        r = oracle_mixed_precision_diff(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    found_bug = has_bug(oracle_results)
    severity = bug_severity(oracle_results)

    is_known_basin = False
    if found_bug and basin_tracker is not None:
        op_fp = prog_info.get("op_fingerprint", [])
        is_known_basin = basin_tracker.is_known(op_fp, hlo_ops)

    return {
        "status": "BUG" if found_bug else "OK",
        "seed": seed,
        "severity": severity,
        "is_known_basin": is_known_basin,
        "oracle_results": oracle_results,
        "hlo_info": hlo_info,
        "op_fingerprint": prog_info.get("op_fingerprint", []),
        "dtype": prog_info["dtype"],
    }


def verify_in_subprocess(prog_info, python_cmd="python"):
    """Re-run in fresh process to confirm bug."""
    setup = prog_info["setup"]
    fn_code = prog_info["fn_code"]
    call_args = prog_info["call_args"]

    script = f"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import json

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

{setup}
{fn_code}

args = [{call_args}]

eager_tfn_diff = -1.0
eager_ok = False
tfn_crash = None
try:
    out_e = fn(*args); _ = out_e.numpy()
    eager_ok = True
except Exception:
    pass

if eager_ok:
    try:
        fn_t = tf.function(fn, jit_compile=False)
        out_t = fn_t(*args); _ = out_t.numpy()
        e = np.asarray(out_e.numpy(), dtype=np.float64)
        t = np.asarray(out_t.numpy(), dtype=np.float64)
        eager_tfn_diff = float(np.abs(e - t).max())
    except Exception as ex:
        tfn_crash = str(ex)[:200]

tfn_xla_diff = -1.0
xla_crash = None
if eager_ok:
    try:
        fn_t = tf.function(fn, jit_compile=False)
        out_t = fn_t(*args)
        fn_x = tf.function(fn, jit_compile=True)
        out_x = fn_x(*args); _ = out_x.numpy()
        t = np.asarray(out_t.numpy(), dtype=np.float64)
        x = np.asarray(out_x.numpy(), dtype=np.float64)
        tfn_xla_diff = float(np.abs(t - x).max())
    except Exception as ex:
        xla_crash = str(ex)[:200]

grad_diff = -1.0
grad_tfn_crash = None
if eager_ok:
    try:
        def grad_fn(*inputs):
            with tf.GradientTape() as tape:
                tape.watch(inputs[0])
                out = fn(*inputs)
                if out.shape.rank is not None and out.shape.rank > 0:
                    out = tf.reduce_sum(out)
            return tape.gradient(out, inputs[0])
        g_e = grad_fn(*args); _ = g_e.numpy()
        g_t = tf.function(grad_fn)(*args); _ = g_t.numpy()
        ge = np.asarray(g_e.numpy(), dtype=np.float64)
        gt = np.asarray(g_t.numpy(), dtype=np.float64)
        grad_diff = float(np.abs(ge - gt).max())
    except Exception as ex:
        grad_tfn_crash = str(ex)[:200]

cpu_gpu_diff = -1.0
if eager_ok:
    try:
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/CPU:0'):
                fn_c = tf.function(fn)
                out_c = fn_c(*[tf.identity(a) for a in args])
            with tf.device('/GPU:0'):
                fn_g = tf.function(fn)
                out_g = fn_g(*[tf.identity(a) for a in args])
            c = np.asarray(out_c.numpy(), dtype=np.float64)
            g = np.asarray(out_g.numpy(), dtype=np.float64)
            cpu_gpu_diff = float(np.abs(c - g).max())
    except Exception:
        pass

result = {{
    "eager_ok": eager_ok,
    "eager_tfn_diff": eager_tfn_diff,
    "tfn_xla_diff": tfn_xla_diff,
    "grad_diff": grad_diff,
    "cpu_gpu_diff": cpu_gpu_diff,
    "tfn_crash": tfn_crash,
    "xla_crash": xla_crash,
    "grad_tfn_crash": grad_tfn_crash,
}}
print("RESULT:" + json.dumps(result))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        proc = subprocess.run(
            [python_cmd, tmp],
            capture_output=True, text=True, timeout=180
        )
        os.unlink(tmp)
        if proc.returncode != 0:
            return {"verified": False, "reason": "subprocess_error",
                    "stderr": proc.stderr[-500:]}

        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                verified_oracles = []

                if data.get("eager_ok") and data.get("tfn_crash"):
                    verified_oracles.append(("tfn_CRASH", data["tfn_crash"]))
                if data.get("eager_ok") and data.get("xla_crash"):
                    verified_oracles.append(("xla_CRASH", data["xla_crash"]))
                if data.get("eager_ok") and data.get("grad_tfn_crash"):
                    verified_oracles.append(("grad_tfn_CRASH", data["grad_tfn_crash"]))

                if data.get("eager_tfn_diff", -1) > 1e-3:
                    verified_oracles.append(("eager_vs_tfn", data["eager_tfn_diff"]))
                if data.get("tfn_xla_diff", -1) > 1e-2:
                    verified_oracles.append(("tfn_vs_xla", data["tfn_xla_diff"]))
                if data.get("grad_diff", -1) > 1e-2:
                    verified_oracles.append(("grad", data["grad_diff"]))
                if data.get("cpu_gpu_diff", -1) > 1e-2:
                    verified_oracles.append(("cpu_vs_gpu", data["cpu_gpu_diff"]))

                if verified_oracles:
                    return {"verified": True, "oracles": verified_oracles, "raw": data}
                return {"verified": False, "reason": "not_reproduced", "raw": data}

        return {"verified": False, "reason": "no_result_line",
                "stdout": proc.stdout[-500:]}
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return {"verified": False, "reason": "timeout"}
    except Exception as e:
        return {"verified": False, "reason": str(e)[:200]}


def save_bug(prog_info, result, verify_result, bug_id, out_dir):
    seed = prog_info["seed"]
    bug_dir = Path(out_dir) / f"tf_bug_{bug_id:04d}_seed{seed}"
    bug_dir.mkdir(parents=True, exist_ok=True)

    repro = f'''"""DeepOpFuzz TF finding.
seed={seed}
severity={result.get('severity', '?')}
oracles: {[r['oracle'] + ':' + r.get('status','?') for r in result.get('oracle_results', [])]}
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

{prog_info['setup']}

{prog_info['fn_code']}

args = [{prog_info['call_args']}]

out_e = fn(*args)
out_t = tf.function(fn, jit_compile=False)(*args)
out_x = tf.function(fn, jit_compile=True)(*args)
e = np.asarray(out_e.numpy(), dtype=np.float64)
t = np.asarray(out_t.numpy(), dtype=np.float64)
x = np.asarray(out_x.numpy(), dtype=np.float64)
print(f"eager_vs_tfn   max_abs={{np.abs(e-t).max():.6e}}")
print(f"tfn_vs_xla     max_abs={{np.abs(t-x).max():.6e}}")
print(f"eager_vs_xla   max_abs={{np.abs(e-x).max():.6e}}")
'''
    (bug_dir / "minimal_repro.py").write_text(repro)
    (bug_dir / "fn_code.py").write_text(prog_info["fn_code"])
    (bug_dir / "meta.json").write_text(json.dumps({
        "seed": seed,
        "shapes": prog_info.get("shapes"),
        "dtype": prog_info["dtype"],
        "poison_mode": prog_info.get("poison_mode"),
        "op_fingerprint": prog_info.get("op_fingerprint", []),
        "composition_preset": prog_info.get("composition_preset"),
        "severity": result.get("severity"),
        "oracle_results": result.get("oracle_results", []),
        "f64_truth": result.get("f64_truth"),
        "verify": verify_result,
        "hlo_info": result.get("hlo_info", {}),
    }, indent=2, default=str))
    return str(bug_dir)


def main():
    parser = argparse.ArgumentParser(description="DeepOpFuzz TF fuzzer")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracles", type=str, default="eager,xla,grad,cpu_gpu",
                        help="comma-separated oracle set")
    parser.add_argument("--preset", type=str, default="v2",
                        choices=["v2", "v3_tf_only"])
    parser.add_argument("--composition-ratio", type=float, default=0.3)
    parser.add_argument("--batch-report", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--guided", action="store_true", default=True)
    parser.add_argument("--no-guided", dest="guided", action="store_false")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="tf_results")
    parser.add_argument("--python-cmd", type=str, default="python",
                        help="Python command for subprocess verification")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    confirmed_dir = out_dir / "confirmed_bugs"
    confirmed_dir.mkdir(exist_ok=True)
    checkpoint_path = out_dir / "fuzzer_checkpoint.json"

    if args.preset == "v3_tf_only":
        oracle_names = {"eager", "grad", "eager_cpu_gpu", "eager_grad_cpu_gpu",
                        "math_invariant"}
        os.environ["TF_FUZZER_POISON_PRESET"] = "v3"
        os.environ["TF_FUZZER_GRAPH_PRESET"] = "v3_tf_only"
    else:
        oracle_names = set(args.oracles.split(","))

    oracle_flags = {
        "xla": "xla" in oracle_names,
        "grad": "grad" in oracle_names,
        "cpu_gpu": "cpu_gpu" in oracle_names,
        "eager_cpu_gpu": "eager_cpu_gpu" in oracle_names,
        "eager_grad_cpu_gpu": "eager_grad_cpu_gpu" in oracle_names,
        "math_invariant": "math_invariant" in oracle_names,
        "determinism": "determinism" in oracle_names,
        "mixed": "mixed" in oracle_names,
    }

    sys.stdout.reconfigure(line_buffering=True)

    state_space = TFKernelStateSpace()
    scheduler = EpsilonGreedyScheduler(state_space, epsilon=args.epsilon,
                                        seed=args.seed) if args.guided else None

    print(f"DeepOpFuzz TF: {args.iters} iters, base_seed={args.seed}")
    print(f"Oracles: {oracle_names}")
    print(f"Guided: {args.guided} (epsilon={args.epsilon})" if args.guided else "Guided: OFF")
    print(f"Kernel states: {state_space.total_reachable} reachable")
    print(f"Composition ratio: {args.composition_ratio}")
    print(f"TF: {tf.__version__}, GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print()

    coverage = CoverageTracker()
    basin = BugBasinTracker(similarity_threshold=0.7)
    for fp, desc in KNOWN_BASINS:
        basin.add_basin(fp, description=desc)

    start_iter = 0
    stats = {
        "ok": 0, "diff": 0, "verified": 0, "known_basin": 0,
        "near_zero_filtered": 0, "truth_agrees_filtered": 0,
        "gen_err": 0, "tfn_err": 0, "xla_err": 0,
        "grad_diff": 0, "cpu_gpu_diff": 0,
        "suspicious": 0, "crash": 0,
    }
    bug_id = 0
    all_bugs = []

    if args.resume and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        start_iter = ckpt.get("last_iter", 0) + 1
        stats = ckpt.get("stats", stats)
        bug_id = ckpt.get("bug_id", 0)
        print(f"Resuming from iter {start_iter}, bug_id={bug_id}\n")

    t0 = time.time()
    candidates = []

    print("=== Phase 1: In-process scan ===")
    for i in range(start_iter, args.iters):
        seed = args.seed + i
        use_composition = (i % 10) < int(args.composition_ratio * 10)

        target = None
        gen_kwargs = {}
        iter_oracle_flags = oracle_flags
        if scheduler is not None:
            target = scheduler.select_target()
            gen_kwargs = scheduler.target_to_gen_params(target)
            if target is not None:
                iter_oracle_flags = scheduler.oracle_flags_for_target(
                    target, oracle_flags)

        try:
            prog = generate_program_with_inputs(
                seed,
                use_composition=use_composition,
                poison_mode=gen_kwargs.get("poison_mode"),
                dtype_override=gen_kwargs.get("dtype_override"),
                category_weights=gen_kwargs.get("category_weights"),
                force_matmul=gen_kwargs.get("force_matmul", False),
            )
        except Exception:
            stats["gen_err"] += 1
            continue

        result = run_single(prog, iter_oracle_flags,
                           coverage_tracker=coverage,
                           basin_tracker=basin)

        if scheduler is not None:
            active_oracles = scheduler.active_oracle_names(iter_oracle_flags)
            actual_state = state_space.state_from_program(prog, active_oracles)
            state_space.mark_covered(actual_state)

        status = result["status"]
        severity = result.get("severity")

        if status == "OK":
            stats["ok"] += 1
        elif status == "BUG":
            if result.get("is_known_basin"):
                stats["known_basin"] += 1
                continue

            if _is_grappler_argmin_basin(
                    prog.get("op_fingerprint", []),
                    result.get("oracle_results", [])):
                stats["known_basin"] += 1
                continue

            try:
                fn, args_list = build_fn_and_args(prog)
                truth_r = oracle_f64_truth(fn, args_list, label=f"seed={seed}")
                if truth_r.get("status") == "NEAR_ZERO":
                    stats["near_zero_filtered"] += 1
                    continue
                if truth_r.get("status") == "INFO":
                    rels = [truth_r.get(k) for k in ("eager_rel_err", "tfn_rel_err", "xla_rel_err")]
                    rels = [r for r in rels if r is not None]
                    if rels and max(rels) < 1e-3:
                        stats["truth_agrees_filtered"] += 1
                        continue
                    result["f64_truth"] = truth_r
            except Exception:
                pass

            stats["diff"] += 1
            candidates.append((prog, result))

            for r in result.get("oracle_results", []):
                if r.get("status") in ("DIFF", "SUSPICIOUS"):
                    if r["oracle"] == "grad_eager_vs_tfn":
                        stats["grad_diff"] += 1
                    elif r["oracle"] == "cpu_vs_gpu":
                        stats["cpu_gpu_diff"] += 1

            if severity == "SUSPICIOUS":
                stats["suspicious"] += 1
            elif severity == "CRASH":
                stats["crash"] += 1

            prefix = "  !!! CRASH CANDIDATE" if severity == "CRASH" else "  ** CANDIDATE"
            print(f"{prefix} seed={seed} severity={severity} "
                  f"ops={prog.get('op_fingerprint', [])[:5]}")
            for r in result.get("oracle_results", []):
                if r.get("status") not in ("OK", "SKIP_NO_GPU", "INFO", None):
                    report(r)
        elif status == "GEN_ERROR":
            stats["gen_err"] += 1

        if (i + 1) % args.batch_report == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_iter) / elapsed if elapsed > 0 else 0
            cov = coverage.stats()
            kss = state_space.stats()
            print(f"  [{i+1}/{args.iters}] {rate:.1f} iter/s  "
                  f"ok={stats['ok']} diff={stats['diff']} candidates={len(candidates)} "
                  f"basin={stats['known_basin']} "
                  f"nz={stats.get('near_zero_filtered',0)} "
                  f"ta={stats.get('truth_agrees_filtered',0)} "
                  f"gen_err={stats['gen_err']} "
                  f"| KSS={kss['covered']}/{kss['total_reachable']} "
                  f"hlo_ops={cov['hlo_ops']} gsigs={cov['graph_signatures']}")

        if (i + 1) % args.checkpoint_every == 0:
            checkpoint_path.write_text(json.dumps({
                "last_iter": i, "stats": stats, "bug_id": bug_id,
                "coverage": coverage.stats(), "basin": basin.stats(),
                "pending_candidates": len(candidates),
            }, indent=2, default=str))

    phase1_elapsed = time.time() - t0
    total_iters = args.iters - start_iter
    print(f"\n--- Phase 1 done: {total_iters} iters in {phase1_elapsed:.1f}s "
          f"({total_iters/max(phase1_elapsed,0.1):.1f} iter/s) ---")
    print(f"Candidates to verify: {len(candidates)}")
    print()

    if candidates:
        print(f"=== Phase 2: Verifying {len(candidates)} candidates ===")
        t1 = time.time()
        for idx, (prog, result) in enumerate(candidates):
            seed = prog["seed"]
            severity = result.get("severity")
            vr = verify_in_subprocess(prog, python_cmd=args.python_cmd)
            if vr.get("verified"):
                has_crash = any("CRASH" in o[0] for o in vr.get("oracles", []))
                if has_crash:
                    severity = "CRASH"
                stats["verified"] += 1
                bug_id += 1
                path = save_bug(prog, result, vr, bug_id, confirmed_dir)
                basin.add_basin(
                    prog.get("op_fingerprint", []),
                    result.get("hlo_info", {}).get("hlo_ops", []),
                    description=f"seed={seed} severity={severity}"
                )
                bug_marker = "*** CRASH BUG" if has_crash else "*** VERIFIED BUG"
                print(f"  {bug_marker} #{bug_id} seed={seed} "
                      f"severity={severity} -> {path}")
                all_bugs.append({
                    "bug_id": bug_id, "seed": seed, "severity": severity,
                    "oracles": [r["oracle"] + ":" + r.get("status", "?")
                                for r in result.get("oracle_results", [])],
                    "op_fingerprint": prog.get("op_fingerprint", []),
                })
            else:
                if args.verbose:
                    print(f"  [verify {idx+1}/{len(candidates)}] seed={seed} "
                          f"NOT verified: {vr.get('reason', '?')}")
            if (idx + 1) % 10 == 0:
                print(f"  [verified {idx+1}/{len(candidates)}] confirmed={stats['verified']}")
        phase2_elapsed = time.time() - t1
        print(f"--- Phase 2 done in {phase2_elapsed:.1f}s ---\n")

    elapsed = time.time() - t0
    print(f"{'='*70}")
    print(f"DONE: {total_iters} iters in {elapsed:.1f}s")
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print(f"Coverage: {json.dumps(coverage.stats(), indent=2)}")
    print(f"Kernel states: {json.dumps(state_space.stats(), indent=2)}")
    print(f"Basin tracker: {json.dumps(basin.stats(), indent=2)}")
    print(f"Verified NEW bugs: {stats['verified']}")

    if all_bugs:
        print(f"\n--- Bug summary ---")
        for b in all_bugs:
            print(f"  #{b['bug_id']} seed={b['seed']} {b['severity']} "
                  f"ops={b['op_fingerprint'][:5]}")

    run_log = {
        "iters": args.iters, "base_seed": args.seed,
        "start_iter": start_iter, "elapsed_s": elapsed,
        "tf_version": tf.__version__,
        "oracles": sorted(oracle_names),
        "guided": args.guided, "epsilon": args.epsilon,
        "composition_ratio": args.composition_ratio,
        "stats": stats, "coverage": coverage.stats(),
        "kernel_states": state_space.stats(),
        "bugs": all_bugs,
    }
    log_path = out_dir / f"fuzzer_run_{args.seed}_{args.iters}.json"
    log_path.write_text(json.dumps(run_log, indent=2, default=str))
    print(f"Log: {log_path}")

    checkpoint_path.write_text(json.dumps({
        "last_iter": args.iters - 1, "stats": stats, "bug_id": bug_id,
        "coverage": coverage.stats(), "kernel_states": state_space.stats(),
        "basin": basin.stats(), "completed": True,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
