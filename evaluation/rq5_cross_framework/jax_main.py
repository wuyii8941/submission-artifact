"""JAX graph fuzzer — v2 main loop.

Multi-oracle, HLO coverage-guided, with bug basin dedup and checkpoint/resume.

Usage:
    ${ARTIFACT_ROOT}/jax_cross/harness/run_jax.sh ${ARTIFACT_ROOT}/jax_cross/fuzzer/main.py \
        [--iters N] [--seed S] [--oracles jit,grad,vmap,cpu_gpu] [--resume]

Oracles (comma-separated):
    jit      — jit vs eager (always on)
    grad     — backward (grad jit vs eager)
    vmap     — vmap vs manual loop
    cpu_gpu  — CPU vs GPU comparison
    precision— matmul precision modes
"""
import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
import traceback
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from graph_gen import GraphGenerator, generate_program_with_inputs
from oracles import (oracle_jit_vs_eager, oracle_grad_jit_vs_eager,
                     oracle_vmap_vs_loop, oracle_cpu_vs_gpu,
                     oracle_precision_matrix, has_bug, bug_severity, report)
from coverage import CoverageTracker, BugBasinTracker, get_hlo_text, extract_hlo_ops

OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
CONFIRMED_DIR = Path(__file__).parent.parent / "confirmed_bugs"
CONFIRMED_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = OUT_DIR / "fuzzer_checkpoint.json"

# Pre-register known bug basins (from v1 findings)
# Each basin is (op_set, description). A program matches if ANY known basin op
# appears in its fingerprint (subset matching for known root causes).
KNOWN_BASINS = [
    ({"to_bf16_f32"}, "bf16 cast elision — known #32545"),
    ({"to_f16_f32"}, "fp16 cast elision — known #32545"),
]


def build_fn_and_args(prog_info):
    """Execute setup + fn_code, return (fn, args) or raise."""
    ns = {}
    exec(prog_info["setup"], ns)
    exec(prog_info["fn_code"], ns)
    fn = ns["fn"]
    args = [ns[f"x{i}"] for i in range(prog_info["n_inputs"])]
    return fn, args


def run_single(prog_info, oracle_flags, coverage_tracker=None, basin_tracker=None):
    """Run all enabled oracles on a single program.

    Returns dict with status, oracle results, coverage info.
    """
    seed = prog_info["seed"]
    is_f64 = "float64" in prog_info["dtype"]

    # Build fn and args
    try:
        fn, args = build_fn_and_args(prog_info)
    except Exception as e:
        return {"status": "GEN_ERROR", "error": str(e)[:200], "seed": seed}

    # HLO coverage (sample 1 in 5 to save memory/time)
    hlo_info = {}
    hlo_ops = frozenset()
    if coverage_tracker is not None and (seed % 5 == 0):
        try:
            hlo_text = get_hlo_text(fn, args)
            if hlo_text:
                is_novel, novelty_score, details = coverage_tracker.check_novelty(hlo_text)
                hlo_ops = extract_hlo_ops(hlo_text)
                hlo_info = {
                    "is_novel": is_novel,
                    "novelty_score": novelty_score,
                    "hlo_ops": sorted(hlo_ops),
                }
            else:
                hlo_info = {"is_novel": False, "novelty_score": 0}
        except Exception:
            hlo_info = {"is_novel": False, "novelty_score": 0}

    # Run oracles
    oracle_results = []

    # Oracle 1: jit vs eager (always)
    r = oracle_jit_vs_eager(fn, args, label=f"seed={seed}",
                            rtol=1e-8 if is_f64 else 1e-4,
                            atol=1e-10 if is_f64 else 1e-5)
    oracle_results.append(r)

    # Oracle 2: backward
    if oracle_flags.get("grad", True):
        r = oracle_grad_jit_vs_eager(fn, args, label=f"seed={seed}",
                                     rtol=1e-6 if is_f64 else 1e-3,
                                     atol=1e-8 if is_f64 else 1e-4)
        oracle_results.append(r)

    # Oracle 3: vmap
    if oracle_flags.get("vmap", True):
        r = oracle_vmap_vs_loop(fn, args, label=f"seed={seed}",
                                rtol=1e-8 if is_f64 else 1e-4,
                                atol=1e-10 if is_f64 else 1e-5)
        oracle_results.append(r)

    # Oracle 4: CPU vs GPU
    if oracle_flags.get("cpu_gpu", False):
        r = oracle_cpu_vs_gpu(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    # Oracle 5: precision matrix
    if oracle_flags.get("precision", False):
        r = oracle_precision_matrix(fn, args, label=f"seed={seed}")
        oracle_results.append(r)

    # Check for bugs
    found_bug = has_bug(oracle_results)
    severity = bug_severity(oracle_results)

    # Basin dedup
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


def verify_in_subprocess(prog_info, run_jax_sh):
    """Re-run in fresh process to eliminate state pollution."""
    setup = prog_info["setup"]
    fn_code = prog_info["fn_code"]
    call_args = prog_info["call_args"]
    is_f64 = "float64" in prog_info["dtype"]

    script = f"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import json

{setup}
{fn_code}

args = [{call_args}]

# Oracle 1: jit vs eager
try:
    out_e = fn(*args); jax.block_until_ready(out_e)
    jax.clear_caches()
    out_j = jax.jit(fn)(*args); jax.block_until_ready(out_j)
    e = np.asarray(jax.device_get(out_e), dtype=np.float64)
    j = np.asarray(jax.device_get(out_j), dtype=np.float64)
    diff = np.abs(e - j)
    jit_diff = float(diff.max())
except Exception as ex:
    jit_diff = -1.0

# Oracle 2: grad
grad_diff = -1.0
try:
    grad_fn = jax.grad(fn)
    g_e = grad_fn(*args); jax.block_until_ready(g_e)
    jax.clear_caches()
    g_j = jax.jit(grad_fn)(*args); jax.block_until_ready(g_j)
    ge = np.asarray(jax.device_get(g_e), dtype=np.float64)
    gj = np.asarray(jax.device_get(g_j), dtype=np.float64)
    grad_diff = float(np.abs(ge - gj).max())
except Exception:
    pass

# Oracle 3: vmap
vmap_diff = -1.0
try:
    batch_size = args[0].shape[0]
    if batch_size <= 32:
        out_vmap = jax.jit(jax.vmap(fn))(*args)
        jax.block_until_ready(out_vmap)
        outs = []
        for i in range(batch_size):
            sliced = tuple(a[i] for a in args)
            outs.append(jax.jit(fn)(*sliced))
        out_loop = jnp.stack(outs)
        jax.block_until_ready(out_loop)
        vv = np.asarray(jax.device_get(out_vmap), dtype=np.float64)
        vl = np.asarray(jax.device_get(out_loop), dtype=np.float64)
        vmap_diff = float(np.abs(vv - vl).max())
except Exception:
    pass

result = {{"jit_diff": jit_diff, "grad_diff": grad_diff, "vmap_diff": vmap_diff}}
print("RESULT:" + json.dumps(result))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        proc = subprocess.run(
            [run_jax_sh, tmp],
            capture_output=True, text=True, timeout=120
        )
        os.unlink(tmp)
        if proc.returncode != 0:
            return {"verified": False, "reason": "subprocess_error",
                    "stderr": proc.stderr[-500:]}

        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                rtol = 1e-8 if is_f64 else 1e-4
                atol = 1e-10 if is_f64 else 1e-5
                # Check each oracle
                verified_oracles = []
                if data.get("jit_diff", -1) > atol:
                    verified_oracles.append(("jit", data["jit_diff"]))
                if data.get("grad_diff", -1) > 1e-3:
                    verified_oracles.append(("grad", data["grad_diff"]))
                if data.get("vmap_diff", -1) > atol:
                    verified_oracles.append(("vmap", data["vmap_diff"]))

                if verified_oracles:
                    return {"verified": True, "oracles": verified_oracles,
                            "raw": data}
                return {"verified": False, "reason": "not_reproduced",
                        "raw": data}

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


def save_bug(prog_info, result, verify_result, bug_id):
    """Save a verified bug to confirmed_bugs/."""
    seed = prog_info["seed"]
    bug_dir = CONFIRMED_DIR / f"jax_fuzz_{bug_id:04d}_seed{seed}"
    bug_dir.mkdir(exist_ok=True)

    # Minimal repro
    repro = f'''"""JAX fuzzer v2 finding.
seed={seed}
severity={result.get('severity', '?')}
oracles: {[r['oracle'] + ':' + r.get('status','?') for r in result.get('oracle_results', [])]}
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special
import numpy as np

{prog_info['setup']}

{prog_info['fn_code']}

args = [{prog_info['call_args']}]

# Forward: jit vs eager
out_eager = fn(*args)
jax.block_until_ready(out_eager)
jax.clear_caches()
out_jit = jax.jit(fn)(*args)
jax.block_until_ready(out_jit)
e = np.asarray(jax.device_get(out_eager), dtype=np.float64)
j = np.asarray(jax.device_get(out_jit), dtype=np.float64)
print(f"jit_vs_eager max_abs={{np.abs(e-j).max():.6e}}")

# Backward: grad jit vs eager
try:
    grad_fn = jax.grad(fn)
    g_e = grad_fn(*args); jax.block_until_ready(g_e)
    jax.clear_caches()
    g_j = jax.jit(grad_fn)(*args); jax.block_until_ready(g_j)
    ge = np.asarray(jax.device_get(g_e), dtype=np.float64)
    gj = np.asarray(jax.device_get(g_j), dtype=np.float64)
    print(f"grad max_abs={{np.abs(ge-gj).max():.6e}}")
except Exception as ex:
    print(f"grad error: {{ex}}")
'''
    (bug_dir / "minimal_repro.py").write_text(repro)
    (bug_dir / "meta.json").write_text(json.dumps({
        "seed": seed,
        "shapes": prog_info.get("shapes"),
        "dtype": prog_info["dtype"],
        "op_fingerprint": prog_info.get("op_fingerprint", []),
        "composition_preset": prog_info.get("composition_preset"),
        "severity": result.get("severity"),
        "oracle_results": result.get("oracle_results", []),
        "verify": verify_result,
        "hlo_info": result.get("hlo_info", {}),
    }, indent=2, default=str))
    (bug_dir / "fn_code.py").write_text(prog_info["fn_code"])
    return str(bug_dir)


def save_checkpoint(state, path=CHECKPOINT_PATH):
    """Save fuzzer state for resume."""
    path.write_text(json.dumps(state, indent=2, default=str))


def load_checkpoint(path=CHECKPOINT_PATH):
    """Load fuzzer state for resume."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def main():
    parser = argparse.ArgumentParser(description="JAX graph fuzzer v2")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracles", type=str, default="jit,grad,vmap",
                        help="Comma-separated: jit,grad,vmap,cpu_gpu,precision")
    parser.add_argument("--composition-ratio", type=float, default=0.3,
                        help="Fraction of programs using composition generator")
    parser.add_argument("--batch-report", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_jax_sh = str(Path(__file__).parent.parent / "harness" / "run_jax.sh")

    # Parse oracle flags
    oracle_names = set(args.oracles.split(","))
    oracle_flags = {
        "grad": "grad" in oracle_names,
        "vmap": "vmap" in oracle_names,
        "cpu_gpu": "cpu_gpu" in oracle_names,
        "precision": "precision" in oracle_names,
    }

    # Unbuffered output
    sys.stdout.reconfigure(line_buffering=True)

    print(f"JAX fuzzer v2: {args.iters} iters, base_seed={args.seed}")
    print(f"Oracles: {oracle_names}")
    print(f"Composition ratio: {args.composition_ratio}")
    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    print()

    # Initialize trackers
    coverage = CoverageTracker()
    basin = BugBasinTracker(similarity_threshold=0.7)

    # Register known basins
    for fp, desc in KNOWN_BASINS:
        basin.add_basin(fp, description=desc)

    # Resume support
    start_iter = 0
    stats = {
        "ok": 0, "diff": 0, "verified": 0, "known_basin": 0,
        "gen_err": 0, "eager_err": 0, "jit_err": 0,
        "grad_diff": 0, "vmap_diff": 0,
        "suspicious": 0, "crash": 0,
    }
    bug_id = 0
    all_bugs = []

    if args.resume:
        ckpt = load_checkpoint()
        if ckpt:
            start_iter = ckpt.get("last_iter", 0) + 1
            stats = ckpt.get("stats", stats)
            bug_id = ckpt.get("bug_id", 0)
            print(f"Resuming from iter {start_iter}, bug_id={bug_id}")
            print(f"Previous stats: {stats}")
            print()

    t0 = time.time()
    candidates = []  # (prog_info, result) pairs to verify later

    # ── Phase 1: Fast in-process scan ──
    print("=== Phase 1: In-process scan ===")
    for i in range(start_iter, args.iters):
        seed = args.seed + i

        # Decide: composition or plain program
        use_composition = (i % 10) < int(args.composition_ratio * 10)

        try:
            prog = generate_program_with_inputs(seed, use_composition=use_composition)
        except Exception as e:
            stats["gen_err"] += 1
            continue

        # Run all oracles
        result = run_single(prog, oracle_flags,
                           coverage_tracker=coverage,
                           basin_tracker=basin)

        status = result["status"]
        severity = result.get("severity")

        if status == "OK":
            stats["ok"] += 1
        elif status == "BUG":
            # Check basin
            if result.get("is_known_basin"):
                stats["known_basin"] += 1
                if args.verbose:
                    print(f"  [iter {i}] seed={seed} — known basin, skipping")
            else:
                stats["diff"] += 1
                candidates.append((prog, result))

                # Categorize by oracle
                for r in result.get("oracle_results", []):
                    if r.get("status") in ("DIFF", "SUSPICIOUS"):
                        if r["oracle"] == "grad_jit_vs_eager":
                            stats["grad_diff"] += 1
                        elif r["oracle"] == "vmap_vs_loop":
                            stats["vmap_diff"] += 1

                if severity == "SUSPICIOUS":
                    stats["suspicious"] += 1
                elif severity == "CRASH":
                    stats["crash"] += 1

                print(f"  ** CANDIDATE seed={seed} severity={severity} "
                      f"ops={prog.get('op_fingerprint', [])[:5]}")
                for r in result.get("oracle_results", []):
                    if r.get("status") not in ("OK", "SKIP_SHAPE", "SKIP_LARGE_BATCH",
                                                "INFO", None):
                        report(r)
        elif status == "GEN_ERROR":
            stats["gen_err"] += 1

        # Periodic cache clear to prevent OOM
        if (i + 1) % 20 == 0:
            jax.clear_caches()

        # Progress report
        if (i + 1) % args.batch_report == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_iter) / elapsed if elapsed > 0 else 0
            cov = coverage.stats()
            print(f"  [{i+1}/{args.iters}] {rate:.1f} iter/s  "
                  f"ok={stats['ok']} diff={stats['diff']} candidates={len(candidates)} "
                  f"known_basin={stats['known_basin']} "
                  f"gen_err={stats['gen_err']} "
                  f"| hlo_ops={cov['unique_ops']} sigs={cov['unique_signatures']}")

        # Checkpoint
        if (i + 1) % args.checkpoint_every == 0:
            save_checkpoint({
                "last_iter": i,
                "stats": stats,
                "bug_id": bug_id,
                "coverage": coverage.stats(),
                "basin": basin.stats(),
                "pending_candidates": len(candidates),
            })

    phase1_elapsed = time.time() - t0
    total_iters = args.iters - start_iter
    print(f"\n--- Phase 1 done: {total_iters} iters in {phase1_elapsed:.1f}s "
          f"({total_iters/phase1_elapsed:.1f} iter/s) ---")
    print(f"Candidates to verify: {len(candidates)}")
    print()

    # ── Phase 2: Batch subprocess verification ──
    if candidates:
        print(f"=== Phase 2: Verifying {len(candidates)} candidates ===")
        t1 = time.time()
        for idx, (prog, result) in enumerate(candidates):
            seed = prog["seed"]
            severity = result.get("severity")
            vr = verify_in_subprocess(prog, run_jax_sh)
            if vr.get("verified"):
                stats["verified"] += 1
                bug_id += 1
                path = save_bug(prog, result, vr, bug_id)
                basin.add_basin(
                    prog.get("op_fingerprint", []),
                    result.get("hlo_info", {}).get("hlo_ops", []),
                    description=f"seed={seed} severity={severity}"
                )
                print(f"  *** VERIFIED BUG #{bug_id} seed={seed} "
                      f"severity={severity} → {path}")
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
                print(f"  [verified {idx+1}/{len(candidates)}] "
                      f"confirmed={stats['verified']}")
        phase2_elapsed = time.time() - t1
        print(f"--- Phase 2 done in {phase2_elapsed:.1f}s ---")
        print()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE: {total_iters} iters in {elapsed:.1f}s")
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print(f"Coverage: {json.dumps(coverage.stats(), indent=2)}")
    print(f"Basin tracker: {json.dumps(basin.stats(), indent=2)}")
    print(f"Verified NEW bugs: {stats['verified']}")

    if all_bugs:
        print(f"\n--- Bug summary ---")
        for b in all_bugs:
            print(f"  #{b['bug_id']} seed={b['seed']} {b['severity']} "
                  f"ops={b['op_fingerprint'][:5]}...")

    # Save run log
    run_log = {
        "iters": args.iters, "base_seed": args.seed,
        "start_iter": start_iter,
        "elapsed_s": elapsed,
        "jax_version": jax.__version__,
        "oracles": sorted(oracle_names),
        "composition_ratio": args.composition_ratio,
        "stats": stats,
        "coverage": coverage.stats(),
        "bugs": all_bugs,
    }
    log_path = OUT_DIR / f"fuzzer_v2_run_{args.seed}_{args.iters}.json"
    log_path.write_text(json.dumps(run_log, indent=2, default=str))
    print(f"Log: {log_path}")

    # Final checkpoint
    save_checkpoint({
        "last_iter": args.iters - 1,
        "stats": stats,
        "bug_id": bug_id,
        "coverage": coverage.stats(),
        "basin": basin.stats(),
        "completed": True,
    })


if __name__ == "__main__":
    main()
