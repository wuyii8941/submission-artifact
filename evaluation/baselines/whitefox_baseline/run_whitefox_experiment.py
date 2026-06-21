"""
WhiteFox Baseline Experiment on PyTorch 2.13.0 nightly Inductor.

Goal: Run WhiteFox's pre-generated tests + fresh LLM-generated tests
against the same Inductor version as DeepOpFuzz, for 24h.
Measure: total bugs, bug types, CDC bug overlap.

Usage:
    python run_whitefox_experiment.py --phase pregenerated --gpu 0
    python run_whitefox_experiment.py --phase generate --gpu 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

WHITEFOX_DIR = Path(os.environ.get("WHITEFOX_DIR", "${ARTIFACT_ROOT}/WhiteFox"))
TORCH_EXEC_DIR = WHITEFOX_DIR / "torch-exec"
PREGENERATED_DIR = WHITEFOX_DIR / "starcoder-generated" / "torch-inductor-generated"
RESULT_BASE = Path(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.environ.get('KERNELFUZZ_PYTHON', sys.executable)

CDC_BUG_SIGNATURES = {
    "#anon-1": ["bf16", "bfloat16", "cast", "sigmoid", "to(torch.bfloat16)"],
    "#anon-2": ["adaptive_avg_pool", "AdaptiveAvgPool", "stride"],
    "#anon-3": ["slice_scatter", "backward", "gradient"],
    "#anon-4": ["cumsum", "split_scan", "dim=129", "SplitScan"],
    "#anon-5": ["slice_scatter", "backward", "index 0"],
    "#anon-6": ["slice_scatter", "backward"],
    "#anon-7": ["inference_mode", "batched", "matmul", "compile"],
    "#anon-8": ["autograd.grad", "create_graph", "higher-order"],
    "#anon-9": ["diagonal_scatter", "backward", "grad"],
    "#anon-10": ["NestedTensor", "layer_norm", "guard"],
}

CDC_OPS = [
    "slice_scatter", "diagonal_scatter", "cumsum", "adaptive_avg_pool",
    "layer_norm", "inference_mode", "create_graph", "NestedTensor",
    "bfloat16", "bf16",
]


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(logfile, msg):
    line = f"[{timestamp()}] {msg}"
    print(line)
    with open(logfile, "a") as f:
        f.write(line + "\n")


def run_single_test(test_file, device, timeout=30, test_ad=False):
    """Run a single WhiteFox test file and return the result."""
    env = {
        **os.environ,
        "TORCHINDUCTOR_PERMUTE_FUSION": "1",
        "TORCHDYNAMO_VERBOSE": "1",
        "TORCHINDUCTOR_SHAPE_PADDING": "1",
    }

    script = f"""
import sys
sys.path.insert(0, '{TORCH_EXEC_DIR}')
import torch
from torch_utils import test_wrapper
from template_exec import CodeParser

parser = CodeParser(lib_name="torch")
test_file = '{test_file}'

with open(test_file) as f:
    raw_code = f.read()

try:
    code = parser.preprocessing(raw_code)
    class_code, inputs, input_init_code = parser.split_func_tensor(code)
except Exception as e:
    print(f"PARSE_ERROR: {{e}}")
    sys.exit(0)

func_def_code = class_code + "\\n" + input_init_code
tensors = inputs

try:
    result, errors = test_wrapper(func_def_code, 420, tensors, '{device}', test_ad={test_ad})
    print(f"RESULT: {{result}}")
    if hasattr(errors, 'items'):
        for k, v in errors.items():
            v_str = str(v)[:500]
            print(f"ERROR_{{k}}: {{v_str}}")
    else:
        print(f"ERRORS: {{str(errors)[:500]}}")
except Exception as e:
    print(f"EXEC_ERROR: {{e}}")
"""

    try:
        proc = subprocess.run(
            [PYTHON, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(TORCH_EXEC_DIR),
        )
        stdout = proc.stdout
        stderr = proc.stderr

        result = "UNKNOWN"
        errors = {}
        for line in stdout.split("\n"):
            if line.startswith("RESULT: "):
                result = line[len("RESULT: "):]
            elif line.startswith("ERROR_"):
                key, _, val = line.partition(": ")
                key = key[len("ERROR_"):]
                errors[key] = val
            elif line.startswith("PARSE_ERROR:"):
                result = "PARSE_ERROR"
                errors["parse"] = line[len("PARSE_ERROR: "):]
            elif line.startswith("EXEC_ERROR:"):
                result = "EXEC_ERROR"
                errors["exec"] = line[len("EXEC_ERROR: "):]

        return {
            "result": result,
            "errors": errors,
            "stderr": stderr[-1000:] if stderr else "",
            "test_file": str(test_file),
        }
    except subprocess.TimeoutExpired:
        return {
            "result": "TIMEOUT",
            "errors": {"timeout": f"{timeout}s"},
            "stderr": "",
            "test_file": str(test_file),
        }
    except Exception as e:
        return {
            "result": "RUNNER_ERROR",
            "errors": {"runner": str(e)},
            "stderr": "",
            "test_file": str(test_file),
        }


def check_cdc_overlap(test_file, result_info):
    """Check if a bug might overlap with our CDC bugs."""
    overlaps = []
    try:
        with open(test_file) as f:
            code = f.read().lower()
    except:
        code = ""

    error_text = " ".join(str(v) for v in result_info.get("errors", {}).values()).lower()
    combined = code + " " + error_text

    for bug_id, signatures in CDC_BUG_SIGNATURES.items():
        match_count = sum(1 for sig in signatures if sig.lower() in combined)
        if match_count >= 2:
            overlaps.append((bug_id, match_count, len(signatures)))

    cdc_ops_found = [op for op in CDC_OPS if op.lower() in code]
    return overlaps, cdc_ops_found


def collect_pregenerated_tests():
    """Collect all pre-generated test files."""
    tests = []
    for step_dir in sorted(PREGENERATED_DIR.iterdir()):
        if not step_dir.is_dir():
            continue
        for opt_dir in sorted(step_dir.iterdir()):
            if not opt_dir.is_dir():
                continue
            for test_file in sorted(opt_dir.glob("*.py")):
                tests.append(test_file)
    return tests


def run_pregenerated_phase(args):
    """Phase 1: Run all pre-generated WhiteFox tests."""
    result_dir = RESULT_BASE / f"pregenerated_{args.device}_ad{args.test_ad}"
    result_dir.mkdir(parents=True, exist_ok=True)
    logfile = result_dir / "experiment.log"
    bugs_file = result_dir / "bugs.jsonl"
    summary_file = result_dir / "summary.json"

    log(logfile, f"Starting pre-generated test phase on device={args.device}, test_ad={args.test_ad}")
    log(logfile, f"PyTorch version: checking...")

    version_check = subprocess.run(
        [PYTHON, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True
    )
    pt_version = version_check.stdout.strip()
    log(logfile, f"PyTorch version: {pt_version}")

    tests = collect_pregenerated_tests()
    log(logfile, f"Total pre-generated tests: {len(tests)}")

    stats = defaultdict(int)
    bug_details = []
    cdc_candidates = []
    start_time = time.time()

    for i, test_file in enumerate(tests):
        result = run_single_test(
            test_file, args.device, timeout=args.timeout, test_ad=args.test_ad
        )
        result_type = result["result"]
        stats[result_type] += 1

        is_bug = result_type in [
            "ResType.JIT_VALUE", "ResType.JIT_CRASH", "ResType.JIT_STATUS",
            "ResType.JIT_GRAD", "ResType.REV_VALUE", "ResType.REV_GRAD",
            "ResType.REV_CRASH", "ResType.FWD_VALUE", "ResType.FWD_GRAD",
        ]

        if is_bug:
            overlaps, cdc_ops = check_cdc_overlap(test_file, result)
            entry = {
                "index": i,
                "test_file": str(test_file),
                "result_type": result_type,
                "errors": result.get("errors", {}),
                "cdc_overlaps": overlaps,
                "cdc_ops": cdc_ops,
                "timestamp": timestamp(),
            }
            bug_details.append(entry)
            with open(bugs_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            if overlaps:
                cdc_candidates.append(entry)
                log(logfile, f"  *** CDC CANDIDATE: {test_file.name} -> {overlaps}")

        if (i + 1) % 500 == 0 or (i + 1) == len(tests):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            bugs_so_far = sum(1 for r in ["ResType.JIT_VALUE", "ResType.JIT_CRASH",
                "ResType.JIT_STATUS", "ResType.JIT_GRAD",
                "ResType.REV_VALUE", "ResType.REV_GRAD"] if stats.get(r, 0) > 0)
            total_bugs = sum(stats.get(r, 0) for r in [
                "ResType.JIT_VALUE", "ResType.JIT_CRASH", "ResType.JIT_STATUS",
                "ResType.JIT_GRAD", "ResType.REV_VALUE", "ResType.REV_GRAD",
                "ResType.REV_CRASH", "ResType.FWD_VALUE", "ResType.FWD_GRAD",
            ])
            log(logfile, f"Progress: {i+1}/{len(tests)} ({rate:.1f}/s) | "
                f"Bugs: {total_bugs} | CDC candidates: {len(cdc_candidates)} | "
                f"Stats: {dict(stats)}")

    elapsed = time.time() - start_time
    summary = {
        "phase": "pregenerated",
        "pytorch_version": pt_version,
        "device": args.device,
        "test_ad": args.test_ad,
        "total_tests": len(tests),
        "elapsed_seconds": elapsed,
        "stats": dict(stats),
        "total_bugs": len(bug_details),
        "cdc_candidates": len(cdc_candidates),
        "cdc_candidate_details": cdc_candidates,
        "bug_type_breakdown": {
            "jit_value": stats.get("ResType.JIT_VALUE", 0),
            "jit_crash": stats.get("ResType.JIT_CRASH", 0),
            "jit_status": stats.get("ResType.JIT_STATUS", 0),
            "jit_grad": stats.get("ResType.JIT_GRAD", 0),
            "rev_value": stats.get("ResType.REV_VALUE", 0),
            "rev_grad": stats.get("ResType.REV_GRAD", 0),
            "rev_crash": stats.get("ResType.REV_CRASH", 0),
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log(logfile, f"\n{'='*60}")
    log(logfile, f"EXPERIMENT COMPLETE")
    log(logfile, f"Total tests: {len(tests)}")
    log(logfile, f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    log(logfile, f"Total bugs: {len(bug_details)}")
    log(logfile, f"CDC candidates: {len(cdc_candidates)}")
    log(logfile, f"Stats: {dict(stats)}")
    log(logfile, f"Summary saved to {summary_file}")
    log(logfile, f"{'='*60}")


def run_generation_phase(args):
    """Phase 2: Generate fresh tests with local LLM and execute them."""
    result_dir = RESULT_BASE / f"generated_{args.device}_ad{args.test_ad}"
    result_dir.mkdir(parents=True, exist_ok=True)
    logfile = result_dir / "experiment.log"
    bugs_file = result_dir / "bugs.jsonl"
    gen_dir = result_dir / "generated_tests"
    gen_dir.mkdir(exist_ok=True)

    log(logfile, f"Starting generation phase on device={args.device}")
    log(logfile, f"Duration: {args.duration}h")
    log(logfile, f"Generation GPU: {args.gen_gpu}")

    prompts_dir = WHITEFOX_DIR / "Prompts" / "torch-inductor" / "req2test"
    prompt_files = sorted(prompts_dir.glob("*.txt"))
    log(logfile, f"Found {len(prompt_files)} optimization prompts")

    start_time = time.time()
    end_time = start_time + args.duration * 3600
    round_num = 0
    stats = defaultdict(int)
    bug_details = []
    cdc_candidates = []

    while time.time() < end_time:
        round_num += 1
        round_dir = gen_dir / f"round_{round_num:04d}"
        round_dir.mkdir(exist_ok=True)

        log(logfile, f"\n--- Generation round {round_num} ---")

        gen_cmd = [
            PYTHON, str(WHITEFOX_DIR / "starcoder_gen.py"),
            "--prompt-dir", str(prompts_dir),
            "--output-dir", str(round_dir),
            "--num", "5",
            "--hf-home", os.path.expanduser("~/.cache/huggingface"),
            "--hf-cache", os.path.expanduser("~/.cache/huggingface"),
        ]

        gen_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gen_gpu)}

        try:
            log(logfile, f"Generating tests with Magicoder on GPU {args.gen_gpu}...")
            gen_proc = subprocess.run(
                gen_cmd, env=gen_env, capture_output=True, text=True,
                timeout=3600, cwd=str(WHITEFOX_DIR)
            )
            if gen_proc.returncode != 0:
                log(logfile, f"Generation error: {gen_proc.stderr[-500:]}")
                continue
        except subprocess.TimeoutExpired:
            log(logfile, "Generation timed out after 1h, continuing...")
            continue
        except Exception as e:
            log(logfile, f"Generation failed: {e}")
            continue

        test_files = list(round_dir.rglob("*.py"))
        log(logfile, f"Generated {len(test_files)} test files")

        for j, test_file in enumerate(test_files):
            if time.time() >= end_time:
                break

            result = run_single_test(
                test_file, args.device, timeout=args.timeout, test_ad=args.test_ad
            )
            result_type = result["result"]
            stats[result_type] += 1

            is_bug = result_type in [
                "ResType.JIT_VALUE", "ResType.JIT_CRASH", "ResType.JIT_STATUS",
                "ResType.JIT_GRAD", "ResType.REV_VALUE", "ResType.REV_GRAD",
            ]

            if is_bug:
                overlaps, cdc_ops = check_cdc_overlap(test_file, result)
                entry = {
                    "round": round_num,
                    "test_file": str(test_file),
                    "result_type": result_type,
                    "errors": result.get("errors", {}),
                    "cdc_overlaps": overlaps,
                    "cdc_ops": cdc_ops,
                    "timestamp": timestamp(),
                }
                bug_details.append(entry)
                with open(bugs_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                if overlaps:
                    cdc_candidates.append(entry)
                    log(logfile, f"  *** CDC CANDIDATE: {test_file.name} -> {overlaps}")

        elapsed = time.time() - start_time
        remaining = (end_time - time.time()) / 3600
        total_bugs = sum(stats.get(r, 0) for r in [
            "ResType.JIT_VALUE", "ResType.JIT_CRASH", "ResType.JIT_STATUS",
            "ResType.JIT_GRAD", "ResType.REV_VALUE", "ResType.REV_GRAD",
        ])
        log(logfile, f"Round {round_num} done. Elapsed: {elapsed/3600:.1f}h, "
            f"Remaining: {remaining:.1f}h | Bugs: {total_bugs} | "
            f"CDC: {len(cdc_candidates)}")

    summary = {
        "phase": "generated",
        "device": args.device,
        "test_ad": args.test_ad,
        "total_rounds": round_num,
        "elapsed_seconds": time.time() - start_time,
        "stats": dict(stats),
        "total_bugs": len(bug_details),
        "cdc_candidates": len(cdc_candidates),
    }
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(logfile, f"Generation phase complete. Summary saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pregenerated", "generate"], required=True)
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for test execution")
    parser.add_argument("--gen-gpu", type=int, default=1, help="GPU for LLM generation")
    parser.add_argument("--timeout", type=int, default=30, help="per-test timeout")
    parser.add_argument("--test-ad", type=int, default=0, help="test backward (0 or 1)")
    parser.add_argument("--duration", type=float, default=24, help="hours for generate phase")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.phase == "pregenerated":
        run_pregenerated_phase(args)
    elif args.phase == "generate":
        run_generation_phase(args)
