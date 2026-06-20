#!/usr/bin/env python3
"""
Batch WhiteFox test runner — processes tests in-process for speed.
Handles crashes via signal-based subprocess isolation per batch.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 batch_runner.py \
        --device cuda --batch-size 50 --timeout 10

Processes all 61K pre-generated WhiteFox tests against current Inductor.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

WHITEFOX_DIR = Path("${ARTIFACT_ROOT}/WhiteFox")
TORCH_EXEC_DIR = WHITEFOX_DIR / "torch-exec"
PREGENERATED_DIR = WHITEFOX_DIR / "starcoder-generated" / "torch-inductor-generated"
PYTHON = os.environ.get('KERNELFUZZ_PYTHON', sys.executable)

CDC_OPS = [
    "slice_scatter", "diagonal_scatter", "cumsum",
    "adaptive_avg_pool", "AdaptiveAvgPool",
    "layer_norm", "LayerNorm",
    "inference_mode", "create_graph",
    "NestedTensor", "nested_tensor",
    "bfloat16", "bf16",
]

# Script template for batch processing
BATCH_SCRIPT_TEMPLATE = """
import sys, os, json, time, traceback, signal
sys.path.insert(0, "TORCH_EXEC_DIR_PLACEHOLDER")
os.chdir("TORCH_EXEC_DIR_PLACEHOLDER")

import torch
torch.manual_seed(420)
if torch.cuda.is_available():
    torch.cuda.manual_seed(420)

from torch_utils import test_wrapper
from template_exec import CodeParser
import template_exec

template_exec.DEVICE = "DEVICE_PLACEHOLDER"
parser = CodeParser(lib_name="torch")

test_files = json.loads(os.environ["WHITEFOX_TEST_FILES"])
timeout_per_test = TIMEOUT_PLACEHOLDER
test_ad = TEST_AD_PLACEHOLDER

results = []

for tf in test_files:
    r = {"file": tf, "result": "UNKNOWN", "errors": {}}

    def alarm_handler(signum, frame):
        raise TimeoutError("test timeout")
    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout_per_test)

    try:
        with open(tf) as f:
            raw_code = f.read()

        code = parser.preprocessing(raw_code)
        class_code, inputs, input_init_code = parser.split_func_tensor(code)
        func_def_code = class_code + "\\n" + input_init_code

        result, errors = test_wrapper(func_def_code, 420, inputs, "DEVICE_PLACEHOLDER", test_ad=test_ad)
        rname = result.name if hasattr(result, "name") else str(result)
        r["result"] = rname

        if isinstance(errors, dict):
            r["errors"] = {k: str(v)[:300] for k, v in errors.items()}

    except TimeoutError:
        r["result"] = "TIMEOUT"
    except Exception as e:
        r["result"] = "ERROR_" + type(e).__name__
        r["errors"] = {"msg": str(e)[:300]}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    results.append(r)

    if len(results) % 20 == 0:
        torch._dynamo.reset()

print("BATCH_RESULTS:" + json.dumps(results))
"""


def run_batch(test_files, device, timeout, test_ad, gpu):
    """Run a batch of tests in a single subprocess."""
    test_files_json = json.dumps([str(f) for f in test_files])

    script = (BATCH_SCRIPT_TEMPLATE
              .replace("TORCH_EXEC_DIR_PLACEHOLDER", str(TORCH_EXEC_DIR))
              .replace("DEVICE_PLACEHOLDER", device)
              .replace("TIMEOUT_PLACEHOLDER", str(timeout))
              .replace("TEST_AD_PLACEHOLDER", "True" if test_ad else "False"))

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TORCHINDUCTOR_PERMUTE_FUSION": "1",
        "TORCHINDUCTOR_SHAPE_PADDING": "1",
        "WHITEFOX_TEST_FILES": test_files_json,
    }

    batch_timeout = min(len(test_files) * (timeout + 5) + 30, 600)

    try:
        proc = subprocess.run(
            [PYTHON, "-c", script],
            env=env, capture_output=True, text=True,
            timeout=batch_timeout, cwd=str(TORCH_EXEC_DIR),
        )

        for line in proc.stdout.split("\n"):
            if line.startswith("BATCH_RESULTS:"):
                return json.loads(line[len("BATCH_RESULTS:"):])

        return [{"file": str(f), "result": "NO_OUTPUT", "errors": {}}
                for f in test_files]

    except subprocess.TimeoutExpired:
        return [{"file": str(f), "result": "BATCH_TIMEOUT", "errors": {}}
                for f in test_files]
    except Exception as e:
        return [{"file": str(f), "result": f"BATCH_ERROR:{e}", "errors": {}}
                for f in test_files]


def is_bug(result_name):
    bugs = {"JIT_VALUE", "JIT_CRASH", "JIT_STATUS", "JIT_GRAD",
            "REV_VALUE", "REV_GRAD", "REV_CRASH",
            "FWD_VALUE", "FWD_GRAD", "FWD_CRASH"}
    return result_name in bugs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--test-ad", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-tests", type=int, default=0)
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    tag = f"gpu{args.gpu}_{args.device}"
    if args.test_ad:
        tag += "_ad"
    result_dir = Path(f"${ARTIFACT_ROOT}/paper/evaluation/whitefox_baseline/results_{tag}")
    result_dir.mkdir(parents=True, exist_ok=True)

    logfile = result_dir / "experiment.log"
    bugs_jsonl = result_dir / "bugs.jsonl"
    summary_file = result_dir / "summary.json"
    checkpoint_file = result_dir / "checkpoint.json"

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logfile, "a") as f:
            f.write(line + "\n")

    ver = subprocess.run([PYTHON, "-c", "import torch; print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()

    log(f"WhiteFox Baseline Experiment (batch mode)")
    log(f"PyTorch: {ver}, GPU: {args.gpu}, Device: {args.device}")
    log(f"Timeout: {args.timeout}s/test, Batch: {args.batch_size}, test_ad: {args.test_ad}")

    all_tests = sorted(PREGENERATED_DIR.rglob("*.py"))
    if args.start_from > 0:
        all_tests = all_tests[args.start_from:]
        log(f"Resuming from test {args.start_from}")
    if args.max_tests > 0:
        all_tests = all_tests[:args.max_tests]
    log(f"Tests to run: {len(all_tests)}")

    stats = defaultdict(int)
    bug_list = []
    start_time = time.time()
    tests_done = args.start_from

    for batch_start in range(0, len(all_tests), args.batch_size):
        batch = all_tests[batch_start:batch_start + args.batch_size]
        batch_results = run_batch(batch, args.device, args.timeout, args.test_ad, args.gpu)

        for r in batch_results:
            result_name = r["result"]
            stats[result_name] += 1
            tests_done += 1

            if is_bug(result_name):
                tf = Path(r["file"])
                parts = tf.parts
                opt_name = parts[-2] if len(parts) >= 2 else "unknown"
                step_name = parts[-3] if len(parts) >= 3 else "unknown"

                try:
                    with open(tf) as f:
                        code = f.read()
                    cdc_ops = [op for op in CDC_OPS if op in code]
                except:
                    cdc_ops = []

                entry = {
                    "index": tests_done,
                    "test_file": r["file"],
                    "step": step_name,
                    "optimization": opt_name,
                    "category": result_name,
                    "errors": r.get("errors", {}),
                    "cdc_ops": cdc_ops,
                    "timestamp": datetime.now().isoformat(),
                }
                bug_list.append(entry)
                with open(bugs_jsonl, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                log(f"  BUG #{len(bug_list)}: [{result_name}] {step_name}/{opt_name}/{tf.name}"
                    f"{' CDC:'+str(cdc_ops) if cdc_ops else ''}")

        elapsed = time.time() - start_time
        done = batch_start + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        total_bugs = len(bug_list)
        remaining_tests = len(all_tests) - done
        eta_s = remaining_tests / rate if rate > 0 else 0

        if (done % 2000 < args.batch_size) or done == len(all_tests):
            log(f"Progress: {done}/{len(all_tests)} ({rate:.1f}/s) | "
                f"Bugs: {total_bugs} | ETA: {eta_s/60:.0f}m | "
                f"Stats(top3): {dict(sorted(stats.items(), key=lambda x:-x[1])[:3])}")

            with open(checkpoint_file, "w") as f:
                json.dump({"tests_done": tests_done, "elapsed": elapsed}, f)

    elapsed = time.time() - start_time

    # Deduplicate
    seen = set()
    unique_bugs = []
    for b in bug_list:
        sig = b["category"] + "|" + b["optimization"] + "|" + str(b.get("errors", {}))[:100]
        if sig not in seen:
            seen.add(sig)
            unique_bugs.append(b)

    summary = {
        "experiment": "whitefox_baseline",
        "pytorch_version": ver,
        "config": {
            "gpu": args.gpu, "device": args.device,
            "timeout": args.timeout, "test_ad": args.test_ad,
            "batch_size": args.batch_size,
        },
        "total_tests": len(all_tests) + args.start_from,
        "tests_executed": tests_done,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(elapsed / 3600, 2),
        "rate_per_second": round(len(all_tests) / elapsed, 2) if elapsed > 0 else 0,
        "result_distribution": dict(sorted(stats.items(), key=lambda x: -x[1])),
        "total_raw_bugs": len(bug_list),
        "unique_bugs": len(unique_bugs),
        "bugs_by_category": dict(sorted(
            {cat: sum(1 for b in bug_list if b["category"] == cat)
             for cat in set(b["category"] for b in bug_list)}.items(),
            key=lambda x: -x[1]
        )) if bug_list else {},
        "bugs_by_optimization": dict(sorted(
            defaultdict(int, {b["optimization"]: 0 for b in bug_list}).items()
        )) if bug_list else {},
        "cdc_analysis": {
            "bugs_with_cdc_ops": sum(1 for b in bug_list if b.get("cdc_ops")),
            "unique_cdc_ops_found": sorted(set(
                op for b in bug_list for op in b.get("cdc_ops", [])
            )),
        },
    }

    # Count bugs per optimization properly
    if bug_list:
        opt_counts = defaultdict(int)
        for b in bug_list:
            opt_counts[b["optimization"]] += 1
        summary["bugs_by_optimization"] = dict(sorted(opt_counts.items(), key=lambda x: -x[1]))

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n{'='*60}")
    log(f"EXPERIMENT COMPLETE")
    log(f"Total tests: {len(all_tests)}")
    log(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    log(f"Rate: {len(all_tests)/elapsed:.1f} tests/s")
    log(f"Distribution: {dict(sorted(stats.items(), key=lambda x:-x[1]))}")
    log(f"Raw bugs: {len(bug_list)}, Unique: {len(unique_bugs)}")
    if bug_list:
        log(f"By category: {summary['bugs_by_category']}")
        log(f"By optimization: {summary['bugs_by_optimization']}")
        log(f"CDC analysis: {summary['cdc_analysis']}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
