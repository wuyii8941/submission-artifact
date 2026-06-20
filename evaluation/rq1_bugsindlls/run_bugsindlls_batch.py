#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BugsInDLLs Batch Runner v9 — HKG-Fuzz 集成版
==============================================

从 BugsInDLLs CSV 中提取所有可测 API，批量调用 benchmark_full_oracle_triton.py。

--scope 选项：
  fuzzable   : 常规 tensor API
  compile    : torch.compile 类 API
  both       : fuzzable + compile (默认)
  all        : 全部
  high-risk  : 三大雷区 (nested / linalg / distributions)
  graph      : Graph-level fuzzing (GraphFuzzer，独立实验维度，不依赖 api_name)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# AOTI / libstdc++ bootstrap (P0.2, 2026-04-12)
# =============================================================================
# The graph_fuzzer.py P0.2 AOTI oracle compiles programs to .so files via
# torch._inductor.aoti_compile_and_package(). Those .so files dlopen libstdc++
# at runtime and require GLIBCXX_3.4.30+. The system libstdc++ on this machine
# (/lib/x86_64-linux-gnu/libstdc++.so.6) only has GLIBCXX_3.4.28, which is too
# old. The conda env at ${ARTIFACT_ROOT}/.conda/envs/pt211/lib has 6.0.34
# (GLIBCXX_3.4.34) which is new enough.
#
# Setting os.environ['LD_LIBRARY_PATH'] *after* python startup has no effect
# on the running process — the dynamic linker reads it once at process start.
# So we re-exec self with the corrected env BEFORE any torch import. The
# `_AOTI_BOOTSTRAP_DONE` marker prevents infinite re-exec loops if the path
# can't be applied.
#
# Without this bootstrap, AOTI fails on every program with
# "Error in dlopen: libstdc++ ... GLIBCXX_3.4.30 not found". The errors are
# filtered as env errors by graph_fuzzer.py:_is_aoti_env_error so they don't
# pollute the bug reports, but AOTI as an oracle is silently disabled.
_AOTI_CONDA_LIB = "${ARTIFACT_ROOT}/.conda/envs/pt211/lib"
# Skip re-exec if the script is being imported via `python -c "..."` or
# `python -m ...`, where sys.argv doesn't contain a reconstructible command
# line. The bootstrap is only meaningful when run_bugsindlls_batch.py is the
# main script invoked directly.
_AOTI_CAN_REEXEC = bool(sys.argv) and sys.argv[0] not in ("-c", "-m", "")
if (_AOTI_CAN_REEXEC
        and os.path.isdir(_AOTI_CONDA_LIB)
        and _AOTI_CONDA_LIB not in os.environ.get("LD_LIBRARY_PATH", "")
        and not os.environ.get("_AOTI_BOOTSTRAP_DONE")):
    _new_env = os.environ.copy()
    _new_env["LD_LIBRARY_PATH"] = (
        _AOTI_CONDA_LIB + ":" + _new_env.get("LD_LIBRARY_PATH", "")
    )
    _new_env["_AOTI_BOOTSTRAP_DONE"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, _new_env)


# =============================================================================
# Classification helpers
# =============================================================================

_UNFUZZABLE_KEYWORDS = [
    'distributed', '.load(', '.save(', 'optim.lr_scheduler',
    'set_default_dtype', 'set_float32_matmul', 'inference_mode',
    'device_count', 'is_available', 'return_types', 'Generator',
    '_OutOfMemory', '.fx', '.onnx', 'get_device', 'autograd.Function',
    'frombuffer', 'from_numpy',
]

_COMPILE_KEYWORDS = [
    'compile', '_dynamo', 'export.export', 'export.unflatten',
    'jit.freeze', 'jit.script',
]

_TOOL_APIS = {
    'torch.rand', 'torch.randn', 'torch.tensor', 'torch.zeros',
    'torch.ones', 'torch.arange', 'torch.manual_seed',
    'torch.no_grad', 'torch.is_tensor', 'torch.cuda',
}


# =============================================================================
# 高风险雷区 API 列表
# =============================================================================

def _hr(api_name: str, tag: str) -> tuple:
    return (api_name, "high-risk", [{
        "issue_id": f"hr-{tag}", "bug_type": "high-risk",
        "version": "", "device": "cuda", "reproduced": False,
        "title": f"{api_name} high-risk fuzzing target",
    }])


HIGH_RISK_APIS = [
    _hr("torch.nested.nested_tensor",         "nested-001"),
    _hr("torch.linalg.svd",                   "linalg-svd"),
    _hr("torch.linalg.eig",                   "linalg-eig"),
    _hr("torch.linalg.solve",                 "linalg-solve"),
    _hr("torch.linalg.lstsq",                 "linalg-lstsq"),
    _hr("torch.linalg.pinv",                  "linalg-pinv"),
    _hr("torch.linalg.inv",                   "linalg-inv"),
    _hr("torch.distributions.Normal",         "dist-normal"),
    _hr("torch.distributions.Beta",           "dist-beta"),
    _hr("torch.distributions.kl_divergence",  "dist-kl"),
]


# =============================================================================
# BugsInDLLs API Extractor
# =============================================================================

def extract_apis_from_csv(csv_path: str,
                           bugsindlls_dir: Optional[str] = None
                           ) -> Dict[str, List[dict]]:
    raw = Path(csv_path).read_text('utf-8', errors='replace')
    if raw.startswith('\ufeff'):
        raw = raw[1:]

    rows = list(csv.reader(raw.splitlines()))
    if len(rows) < 3:
        print(f"  CSV too short: {len(rows)} rows")
        return {}

    header = rows[1]
    col    = {c.strip().lower(): i for i, c in enumerate(header)}

    def _get(row, name, default=''):
        i = col.get(name.lower())
        return row[i].strip() if i is not None and i < len(row) else default

    api_to_issues: Dict[str, List[dict]] = defaultdict(list)
    total = 0

    for row in rows[2:]:
        if not row or not any(c.strip() for c in row):
            continue
        issue_id = _get(row, 'issue #')
        if not issue_id:
            continue
        total += 1

        title    = _get(row, 'title')
        repro    = _get(row, 'reproduced')
        buggy_api = _get(row, 'buggy api')
        bug_type  = _get(row, 'type')
        version   = _get(row, 'buggy version')
        device    = _get(row, 'device')

        info = {
            'issue_id': issue_id, 'bug_type': bug_type,
            'version': version, 'device': device,
            'reproduced': repro.lower() == 'yes',
            'title': title[:120],
        }

        apis_found = []
        if buggy_api:
            apis_found.append(buggy_api.rstrip('.'))

        for ta in re.findall(r'(torch\.[a-zA-Z_][a-zA-Z0-9_.]*)', title):
            ta_clean = ta.rstrip('.')
            if ta_clean not in apis_found and ta_clean not in _TOOL_APIS:
                apis_found.append(ta_clean)

        lower_title = title.lower()
        for kw, api in {
            'scaled_dot_product_attention':
                'torch.nn.functional.scaled_dot_product_attention',
            'frexp_cuda': 'torch.frexp',
            'lobpcg':     'torch.lobpcg',
        }.items():
            if kw in lower_title and api not in apis_found:
                apis_found.append(api)

        if bugsindlls_dir:
            for sa in _extract_from_test_script(bugsindlls_dir, issue_id):
                if sa not in apis_found:
                    apis_found.append(sa)

        for api in apis_found:
            api_to_issues[api].append(info)

    print(f"  Scanned {total} bugs -> {len(api_to_issues)} unique APIs")
    return dict(api_to_issues)


def _extract_from_test_script(bugsindlls_dir: str, issue_id: str) -> List[str]:
    for p in [
        Path(bugsindlls_dir) / 'pytorch' / f'issue_{issue_id}' / f'test_issue_{issue_id}.py',
        Path(bugsindlls_dir) / f'issue_{issue_id}' / f'test_issue_{issue_id}.py',
    ]:
        if p.exists():
            try:
                return _extract_apis_from_code(p.read_text(errors='replace'))
            except Exception:
                pass
    return []


def _extract_apis_from_code(content: str) -> List[str]:
    found = set()
    for pat in [
        r'(torch\.[a-zA-Z_][a-zA-Z0-9_.]*)\s*\(',
        r'(torch\.nn\.functional\.[a-zA-Z_]\w*)\s*\(',
        r'(torch\.nn\.[A-Z][a-zA-Z0-9_]*)\s*\(',
    ]:
        for m in re.finditer(pat, content):
            api = m.group(1).rstrip('.')
            if api not in _TOOL_APIS and len(api) > 8:
                found.add(api)
    return list(found)


def classify_api(api: str) -> str:
    api_lower = api.lower()
    if any(k in api_lower for k in _COMPILE_KEYWORDS):
        return 'compile'
    if any(k in api_lower for k in _UNFUZZABLE_KEYWORDS):
        return 'unfuzzable'
    return 'fuzzable'


# =============================================================================
# Batch Runner (per-API benchmark)
# =============================================================================

def check_api_in_mongodb(api_name: str, config_path: str) -> Tuple[bool, str]:
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        host    = config.get('database', 'host',     fallback='localhost')
        port    = config.getint('database', 'port',  fallback=27017)
        db_name = config.get('database', 'database', fallback='freefuzz')

        import pymongo
        db = pymongo.MongoClient(
            host=host, port=port, serverSelectionTimeoutMS=3000)[db_name]

        has_sig  = db['signature'].find_one({'api': api_name}) is not None
        has_coll = (api_name in db.list_collection_names()
                    and db[api_name].estimated_document_count() > 0)

        if has_sig and has_coll: return True, "sig+coll"
        if has_sig:              return True, "sig-only(fallback)"

        try:
            import inspect
            obj = eval(api_name)
            if callable(obj):
                sig    = inspect.signature(obj)
                params = [p.name for p in sig.parameters.values()
                          if p.name != 'self']
                if params:
                    return True, f"inspect-only({len(params)} params)"
        except Exception:
            pass

        return False, "no-data"
    except Exception:
        return True, "db-unavailable"


def run_single_api(api_name: str, output_dir: str, max_iterations: int,
                   config_file: str, timeout_minutes: int, diff_bound: float,
                   conda_env: Optional[str] = None,
                   benchmark_script: str = 'benchmark_full_oracle_triton.py') -> dict:
    api_output = os.path.join(output_dir, api_name.replace('.', '_'))
    cmd = [
        sys.executable if not conda_env else 'python',
        benchmark_script,
        '--api', api_name,
        '--max-iterations', str(max_iterations),
        '--output', api_output,
        '--conf', config_file,
        '--diff-bound', str(diff_bound),
        '--timeout', str(timeout_minutes),
    ]
    if conda_env:
        cmd = ['conda', 'run', '-n', conda_env, '--no-capture-output'] + cmd[1:]

    start  = time.time()
    result = {
        'api': api_name, 'output_dir': api_output,
        'command': ' '.join(cmd),
        'start_time': datetime.now().isoformat(),
        'success': False, 'exit_code': None,
        'elapsed_seconds': 0, 'error': None,
    }
    try:
        print(f"    CMD: {' '.join(cmd[:6])}...")
        proc = subprocess.run(
            cmd, timeout=timeout_minutes * 60 + 300,
            capture_output=True, text=True)
        result['exit_code'] = proc.returncode
        result['success']   = proc.returncode == 0
        if proc.returncode != 0:
            result['error'] = '\n'.join(proc.stderr.strip().split('\n')[-5:])
    except subprocess.TimeoutExpired:
        result['error']    = f'Timeout after {timeout_minutes} minutes'
        result['exit_code'] = -1
    except Exception as e:
        result['error']    = str(e)
        result['exit_code'] = -2

    result['elapsed_seconds'] = time.time() - start
    return result


def run_batch(api_list: List[Tuple[str, str, List[dict]]],
              output_dir: str, max_iterations: int,
              config_file: str, timeout_minutes: int, diff_bound: float,
              conda_env: Optional[str], benchmark_script: str,
              resume: bool = False) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)

    completed      = set()
    progress_file  = os.path.join(output_dir, '_batch_progress.json')
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file) as f:
                progress_data = json.load(f)
            completed = {r['api'] for r in progress_data.get('results', [])
                         if r.get('success')}
            print(f"  Resume: {len(completed)} APIs already completed")
        except Exception:
            pass

    results    = []
    total      = len(api_list)
    start_time = time.time()

    for idx, (api_name, category, issues) in enumerate(api_list, 1):
        if api_name in completed:
            print(f"\n[{idx}/{total}] (skip) {api_name}")
            continue

        issue_ids = [f"#{i['issue_id']}" for i in issues]
        print(f"\n[{idx}/{total}] {api_name} ({category})")
        print(f"    Issues: {', '.join(issue_ids)}")

        api_dir = os.path.join(output_dir, api_name.replace('.', '_'))
        if resume and os.path.exists(api_dir):
            if (os.path.exists(os.path.join(api_dir, 'random')) and
                    os.path.exists(os.path.join(api_dir, 'guided'))):
                print(f"    Output exists, skipping")
                results.append({'api': api_name, 'success': True,
                                'elapsed_seconds': 0, 'note': 'resumed'})
                continue

        result             = run_single_api(
            api_name, output_dir, max_iterations, config_file,
            timeout_minutes, diff_bound, conda_env, benchmark_script)
        result['category'] = category
        result['issue_ids'] = [i['issue_id'] for i in issues]
        results.append(result)

        status  = 'OK' if result['success'] else 'FAIL'
        elapsed = result['elapsed_seconds']
        print(f"    [{status}] {elapsed:.0f}s | exit={result['exit_code']}")
        if result.get('error'):
            print(f"    Error: {result['error'][:100]}")

        _save_progress(progress_file, results, api_list, start_time)

    return results


def _save_progress(progress_file: str, results: list,
                   api_list: list, start_time: float):
    try:
        with open(progress_file, 'w') as f:
            json.dump({
                'total_apis': len(api_list),
                'completed':  len(results),
                'elapsed_seconds': time.time() - start_time,
                'last_update':     datetime.now().isoformat(),
                'results':         results,
            }, f, indent=2, default=str)
    except Exception:
        pass


# =============================================================================
# Manifest
# =============================================================================

def generate_manifest(api_to_issues: Dict[str, List[dict]],
                      api_list: List[Tuple[str, str, List[dict]]],
                      output_dir: str):
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'csv_total_apis': len(api_to_issues),
        'fuzzed_apis': [],
        'api_to_issues': {},
        'issue_to_apis': defaultdict(list),
    }
    for api_name, category, issues in api_list:
        manifest['fuzzed_apis'].append({
            'api': api_name, 'category': category,
            'issue_count': len(issues),
            'issue_ids': [i['issue_id'] for i in issues],
        })
        manifest['api_to_issues'][api_name] = [i['issue_id'] for i in issues]
        for i in issues:
            manifest['issue_to_apis'][i['issue_id']].append(api_name)

    manifest['all_api_to_issues'] = {
        api: [i['issue_id'] for i in issues]
        for api, issues in api_to_issues.items()
    }

    manifest_path = os.path.join(output_dir, 'api_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Manifest: {manifest_path}")
    return manifest_path


# =============================================================================
# Graph scope result printer
# =============================================================================

def _print_graph_summary(result: dict, output_dir: str, elapsed: float):
    print(f"\n{'='*70}")
    print(f"  GRAPH FUZZING SUMMARY")
    print(f"{'='*70}")
    status = "COMPLETED" if result.get("success") else "FAILED"
    print(f"  Status      : {status}")
    print(f"  Elapsed     : {elapsed/60:.1f} min")
    print(f"  Programs    : {result.get('total_programs', 0)}")
    print(f"  Kernels     : {result.get('total_kernels', 0)}")
    print(f"  Unique bugs : {result.get('total_bugs', 0)}")
    print(f"  Corpus size : {result.get('corpus_size', 0)}")

    by_type = result.get('bugs_by_type', {})
    if by_type:
        print(f"\n  Bug breakdown:")
        priority_label = {
            "backend_crash":  "P3 backend_crash  (inductor CRASH)",
            "inductor_diff":  "P3 inductor_diff  (inductor-specific numeric diff)",
            "frontend_crash": "P2 frontend_crash (aot_eager CRASH)",
            "frontend_diff":  "P2 frontend_diff  (aot_eager numeric diff)",
            "compile_diff":   "P1 compile_diff   (aot-level diff)",
        }
        for bt, count in sorted(by_type.items(),
                                 key=lambda x: -x[1]):
            label = priority_label.get(bt, bt)
            print(f"    {count:3d}  {label}")

    if result.get('mutation_status'):
        print(f"\n  Mutation : {result['mutation_status']}")
    if result.get('error'):
        print(f"\n  Error    : {result['error'][:150]}")

    output_path = os.path.join(output_dir, "graph", "graph-oracle", "potential-bug")
    if os.path.exists(output_path):
        bug_files = list(Path(output_path).rglob("*.py"))
        print(f"\n  Bug files: {len(bug_files)} in {output_path}")
    print(f"{'='*70}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BugsInDLLs Batch Runner v9 + GraphFuzzer'
    )
    parser.add_argument('--csv', '-c', default=None,
                        help='BugsInDLLs bug_dataset_pytorch.csv (CSV scopes only)')
    parser.add_argument('--bugsindlls-dir', '-d', default=None)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--max-iterations', '-n', type=int, default=10000,
                        help='Per-API iterations (default 10000). '
                             'For --scope graph: total program iterations.')
    parser.add_argument('--timeout', '-t', type=int, default=60)
    parser.add_argument('--conf', default='demo_torch.conf')
    parser.add_argument('--diff-bound', type=float, default=1e-5)
    parser.add_argument('--conda-env', default=None)
    parser.add_argument('--benchmark-script',
                        default='benchmark_full_oracle_triton.py')
    parser.add_argument('--scope',
                        choices=['all', 'fuzzable', 'compile', 'both',
                                 'high-risk', 'graph'],
                        default='both',
                        help=(
                            'fuzzable=常规API, compile=编译类, both=两者, '
                            'all=全部, high-risk=三大雷区, '
                            'graph=Graph-level fuzzing (独立实验，不需要 --csv)'
                        ))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-check-mongodb', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--v3', action='store_true',
                        help='[scope=graph only] 启用 V3 Anti-Exploitation GraphFuzzer: '
                             'bug basin tracking, diversity forcing, '
                             'template composition, structural dedup')
    parser.add_argument('--v4', action='store_true',
                        help='[scope=graph only] 启用 V4 Targeted Exploration GraphFuzzer: '
                             'adversarial shapes (2^n±1/primes), semantic mutation, '
                             'layout disruption stages, no_cudagraph config')
    parser.add_argument('--novelty-guided', action='store_true',
                        help='[scope=graph --v4 only] Generate a novelty-aware seed batch '
                             'and inject those candidates into V4 generation.')
    parser.add_argument('--novelty-batch-size', type=int, default=64,
                        help='[scope=graph --novelty-guided] Number of static novelty seeds.')
    parser.add_argument('--novelty-ratio', type=float, default=0.35,
                        help='[scope=graph --novelty-guided] Probability that V4 samples '
                             'from novelty-selected seeds.')
    parser.add_argument('--novelty-batch', default=None,
                        help='[scope=graph --novelty-guided] Existing novelty_batch.json. '
                             'If omitted, one is generated under --output.')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  BugsInDLLs Batch Runner v9 + GraphFuzzer")
    print(f"{'='*70}")
    print(f"  Output    : {args.output}")
    print(f"  Scope     : {args.scope}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  Timeout   : {args.timeout} min")
    if args.conda_env:
        print(f"  Conda env : {args.conda_env}")
    print(f"{'='*70}")

    # =========================================================================
    # scope: graph — GraphFuzzer 独立实验，不需要 CSV，不依赖 api_name
    # =========================================================================
    if args.scope == 'graph':
        v_flags = sum([args.v3, getattr(args, 'v4', False)])
        if v_flags > 1:
            print("\n  --v3, --v4 are mutually exclusive, please choose one.")
            return
        try:
            from graph_fuzzer import run_graph_phase_isolated_v3
            from graph_fuzzer import run_graph_phase_isolated_v4
        except ImportError as e:
            print(f"\n  graph_fuzzer.py not found: {e}")
            print(f"  Please place graph_fuzzer.py in the same directory as this script.")
            return

        # 解析 config 路径（与 run_full_oracle_experiment 相同逻辑）
        config_path = None
        for p in [join('config', args.conf),
                  join('..', 'config', args.conf),
                  args.conf]:
            if os.path.exists(p):
                config_path = os.path.abspath(p)
                break

        os.makedirs(args.output, exist_ok=True)
        graph_output = os.path.join(args.output, "graph")

        if getattr(args, 'v4', False):
            mode_label = "Graph-Level Fuzzing Mode (V4 Targeted Exploration)"
        elif args.v3:
            mode_label = "Graph-Level Fuzzing Mode (V3 Anti-Exploitation)"
        else:
            mode_label = "Graph-Level Fuzzing Mode (V4 Targeted Exploration, default)"
        print(f"\n  {mode_label}")
        print(f"  graph_output : {graph_output}")
        print(f"  iterations   : {args.max_iterations}")
        print(f"  timeout      : {args.timeout} min")
        print(f"  config       : {config_path or '(not found, OK)'}")
        novelty_batch_path = None
        if args.novelty_guided:
            novelty_batch_path = args.novelty_batch or os.path.join(
                args.output, "novelty_batch.json")
            if args.novelty_batch and os.path.exists(args.novelty_batch):
                print(f"  novelty      : {novelty_batch_path} (existing)")
            else:
                selector = "${ARTIFACT_ROOT}/graph_exp/novelty_batch_selector.py"
                candidate_paths = [
                    "${ARTIFACT_ROOT}/graph_exp/graph/graph-oracle/corpus_checkpoint.json",
                    "${ARTIFACT_ROOT}/graph_exp/realistic_starter.json",
                    "${ARTIFACT_ROOT}/graph_exp/mined_subgraphs.json",
                ]
                observed_paths = [
                    "${ARTIFACT_ROOT}/graph_exp/graph/graph-oracle/potential-bug",
                    os.path.join(graph_output, "graph-oracle", "potential-bug"),
                ]
                cmd = [
                    sys.executable, selector,
                    "--candidates", *candidate_paths,
                    "--observed", *observed_paths,
                    "--batch-size", str(args.novelty_batch_size),
                    "--out", novelty_batch_path,
                ]
                print(f"  novelty      : generating {novelty_batch_path}")
                try:
                    proc = subprocess.run(
                        cmd, text=True, capture_output=True, timeout=180)
                    if proc.returncode != 0:
                        print("  [NoveltyGuided] selector failed; disabling novelty.")
                        print((proc.stderr or proc.stdout)[-2000:])
                        novelty_batch_path = None
                    else:
                        print("  [NoveltyGuided] selector summary:")
                        for line in proc.stdout.splitlines()[-20:]:
                            print(f"    {line}")
                except Exception as e:
                    print(f"  [NoveltyGuided] selector error: {type(e).__name__}: {e}")
                    novelty_batch_path = None
        print(f"\n  What this tests:")
        print(f"    Multi-op programs (2~4 ops) compiled with torch.compile")
        print(f"    Three-way oracle: eager vs aot_eager vs inductor")
        if getattr(args, 'v4', False) or not args.v3:
            print(f"    [V4] All V3 features + targeted exploration mechanisms")
            print(f"    [V4] Adversarial shapes: 2^n+/-1 and primes (50%) to stress Triton tile masking")
            print(f"    [V4] Semantic mutation: .clone/.contiguous/+0/*1/reshape/dtype cast (20%)")
            print(f"    [V4] Layout disruption: permute+contiguous, strided slice+pad stages")
            print(f"    [V4] Config: +no_cudagraph preset to expose dynamic-shape bugs")
            print(f"    [V4] Output: potential-bug/ (CATASTROPHIC) + weak-signal/ (SUSPICIOUS)")
            if novelty_batch_path:
                print(f"    [NoveltyGuided] Seed injection enabled: "
                      f"batch={novelty_batch_path}, ratio={args.novelty_ratio:.0%}")
        elif args.v3:
            print(f"    [V3] All V2 features + anti-exploitation mechanisms")
            print(f"    [V3] Bug basin tracking: suppress redundant bug clusters")
            print(f"    [V3] Diversity forcing: category rotation + adaptive template weights")
            print(f"    [V3] Template composition: novel topologies from composable stages")
            print(f"    [V3] Structural dedup: op-fingerprint-based novelty detection")
            print(f"    [V3] Output: potential-bug/ (CATASTROPHIC) + weak-signal/ (SUSPICIOUS)")

        if args.dry_run:
            print(f"\n  Dry run — not executing")
            return

        batch_start = time.time()

        if args.v3:
            result = run_graph_phase_isolated_v3(
                output_dir=graph_output,
                max_iterations=args.max_iterations,
                timeout_minutes=args.timeout,
                config_path=config_path,
            )
        else:
            # Default to V4
            result = run_graph_phase_isolated_v4(
                output_dir=graph_output,
                max_iterations=args.max_iterations,
                timeout_minutes=args.timeout,
                config_path=config_path,
                novelty_batch_path=novelty_batch_path,
                novelty_ratio=args.novelty_ratio if novelty_batch_path else 0.0,
            )

        elapsed = time.time() - batch_start
        _print_graph_summary(result, args.output, elapsed)

        # V3/V4 额外打印分级统计
        if True:
            tag = "V4" if not args.v3 else "V3"
            print(f"\n  {tag} Divergence Breakdown:")
            print(f"    CATASTROPHIC (high confidence) : "
                  f"{result.get('total_catastrophic', 0)}")
            print(f"    SUSPICIOUS   (weak signal)     : "
                  f"{result.get('total_suspicious', 0)}")
            print(f"    Backward Amplified             : "
                  f"{result.get('total_backward_amplified', 0)}")
            by_div = result.get('bugs_by_divergence', {})
            if by_div:
                print(f"    by_divergence: {by_div}")

        # V3/V4 额外打印 anti-exploitation 统计
        if True:
            print(f"\n  V3 Anti-Exploitation Stats:")
            print(f"    Novel bugs (new fingerprint)   : "
                  f"{result.get('total_novel_bugs', 0)}")
            print(f"    Composed programs generated     : "
                  f"{result.get('total_composed', 0)}")
            print(f"    Basin-suppressed iterations     : "
                  f"{result.get('total_basin_suppressed', 0)}")
            print(f"    Unique fingerprints             : "
                  f"{result.get('unique_fingerprints', 0)}")
            if result.get('basin_status'):
                print(f"    Basin status: {result['basin_status']}")
            if result.get('diversity_status'):
                print(f"    Diversity: {result['diversity_status']}")

        # V4 额外打印 targeted exploration 统计
        if not args.v3:
            print(f"\n  V4 Targeted Exploration Stats:")
            print(f"    Novelty-guided programs         : "
                  f"{result.get('total_novelty_guided', 0)}")
            print(f"    Adversarial shape programs      : "
                  f"{result.get('total_adversarial_shapes', 0)}")
            print(f"    Semantic mutations applied       : "
                  f"{result.get('total_semantic_mutations', 0)}")
            print(f"    CUDA crashes (saved to crash/)   : "
                  f"{result.get('total_cuda_crashes', 0)}")

        # 保存完整结果
        summary_path = os.path.join(args.output, 'graph_batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'scope':           'graph',
                'version':         'v3' if args.v3 else 'v4',
                'v3':              args.v3,
                'v4':              not args.v3,
                'guidance_mode':   'guided',
                'max_iterations':  args.max_iterations,
                'timeout_minutes': args.timeout,
                'elapsed_seconds': elapsed,
                'result':          result,
            }, f, indent=2, default=str)
        print(f"\n  Summary saved: {summary_path}")

        # Save evaluation data (timeseries + hit frequency) for figures
        timeseries = result.get('coverage_timeseries', [])
        if timeseries:
            ts_path = os.path.join(args.output, 'coverage_timeseries_guided.json')
            with open(ts_path, 'w') as f:
                json.dump(timeseries, f)
            print(f"  Timeseries : {ts_path} ({len(timeseries)} points)")

        hit_freq = result.get('kss_hit_frequency', {})
        if hit_freq:
            hf_path = os.path.join(args.output, 'kss_hit_frequency_guided.json')
            with open(hf_path, 'w') as f:
                json.dump(hit_freq, f, indent=2)
            print(f"  Hit freq   : {hf_path}")
        print(f"\n  Next step: inspect bugs in")
        print(f"    {graph_output}/graph-oracle/potential-bug/  (CATASTROPHIC)")
        print(f"    {graph_output}/graph-oracle/weak-signal/    (SUSPICIOUS)")
        return

    # =========================================================================
    # scope: high-risk
    # =========================================================================
    if args.scope == 'high-risk':
        api_list         = HIGH_RISK_APIS
        api_to_issues_hr = {api: issues for api, _, issues in api_list}

        print(f"\n  High-risk mode: {len(api_list)} APIs")
        for api, _, _ in api_list:
            print(f"    {api}")

        os.makedirs(args.output, exist_ok=True)
        generate_manifest(api_to_issues_hr, api_list, args.output)

        if args.dry_run:
            print(f"\n  Dry run — not executing")
            return

        batch_start = time.time()
        results     = run_batch(
            api_list=api_list, output_dir=args.output,
            max_iterations=args.max_iterations, config_file=args.conf,
            timeout_minutes=args.timeout, diff_bound=args.diff_bound,
            conda_env=args.conda_env, benchmark_script=args.benchmark_script,
            resume=args.resume)
        elapsed     = time.time() - batch_start

        succeeded = sum(1 for r in results if r.get('success'))
        print(f"\n{'='*70}")
        print(f"  HIGH-RISK SUMMARY: {succeeded}/{len(results)} OK  "
              f"{elapsed/60:.1f} min")
        if any(not r.get('success') for r in results):
            for r in results:
                if not r.get('success'):
                    print(f"    FAIL {r['api']}: {r.get('error','')[:80]}")

        with open(os.path.join(args.output, 'batch_summary.json'), 'w') as f:
            json.dump({'scope': 'high-risk', 'elapsed_seconds': elapsed,
                       'results': results}, f, indent=2, default=str)
        return

    # =========================================================================
    # scope: fuzzable / compile / both / all  (CSV 模式)
    # =========================================================================
    if not args.csv:
        print(f"\n  --csv is required for scope '{args.scope}'")
        return

    # Step 1: 提取 API
    print(f"\n  Extracting APIs from CSV: {args.csv}")
    api_to_issues = extract_apis_from_csv(args.csv, args.bugsindlls_dir)
    if not api_to_issues:
        print("  No APIs found")
        return

    # Step 2: 分类
    classified: Dict[str, list] = {'fuzzable': [], 'compile': [], 'unfuzzable': []}
    for api, issues in api_to_issues.items():
        cat = classify_api(api)
        classified[cat].append((api, cat, issues))

    print(f"\n  Classification:")
    print(f"    Fuzzable   : {len(classified['fuzzable'])}")
    print(f"    Compile    : {len(classified['compile'])}")
    print(f"    Unfuzzable : {len(classified['unfuzzable'])}")

    # Step 3: 选择范围
    api_list = []
    if args.scope in ('fuzzable', 'both', 'all'):
        api_list.extend(classified['fuzzable'])
    if args.scope in ('compile', 'both', 'all'):
        api_list.extend(classified['compile'])
    if args.scope == 'all':
        api_list.extend(classified['unfuzzable'])

    seen, deduped = set(), []
    for api, cat, issues in api_list:
        if api not in seen:
            seen.add(api)
            deduped.append((api, cat, issues))
    api_list = sorted(deduped, key=lambda x: x[0])
    print(f"\n  Target APIs: {len(api_list)}")

    # MongoDB 检查
    if not args.no_check_mongodb:
        config_candidates = [join('config', args.conf),
                             join('..', 'config', args.conf), args.conf]
        config_path       = next((p for p in config_candidates
                                   if os.path.exists(p)), None)
        if config_path:
            print(f"\n  Checking MongoDB...")
            available, missing = [], []
            for api, cat, issues in api_list:
                can_run, status = check_api_in_mongodb(api, config_path)
                if can_run:
                    available.append((api, cat, issues))
                    if status != "sig+coll":
                        print(f"    {api}: {status}")
                else:
                    missing.append(api)
            if missing:
                print(f"  {len(missing)} APIs not runnable (no MongoDB data)")
                api_list = available
            else:
                print(f"  All {len(api_list)} APIs available")

    # 打印 API 列表
    print(f"\n{'='*70}")
    print(f"  Target API List ({len(api_list)}):")
    print(f"{'='*70}")
    for api, cat, issues in api_list:
        issue_ids = ', '.join(f"#{i['issue_id']}" for i in issues[:3])
        extra     = f" +{len(issues)-3}" if len(issues) > 3 else ""
        tag       = {'fuzzable': 'G', 'compile': 'B', 'unfuzzable': 'R'}.get(cat, '?')
        print(f"  [{tag}] {api:50s} -> {issue_ids}{extra}")

    if args.dry_run:
        print(f"\n  Dry run — not executing")
        os.makedirs(args.output, exist_ok=True)
        generate_manifest(api_to_issues, api_list, args.output)
        return

    os.makedirs(args.output, exist_ok=True)
    generate_manifest(api_to_issues, api_list, args.output)

    print(f"\n{'='*70}")
    print(f"  Starting batch: {len(api_list)} APIs x {args.max_iterations} iters")
    print(f"{'='*70}")

    batch_start = time.time()
    results     = run_batch(
        api_list=api_list, output_dir=args.output,
        max_iterations=args.max_iterations, config_file=args.conf,
        timeout_minutes=args.timeout, diff_bound=args.diff_bound,
        conda_env=args.conda_env, benchmark_script=args.benchmark_script,
        resume=args.resume)
    elapsed     = time.time() - batch_start

    # 汇总
    succeeded   = sum(1 for r in results if r.get('success'))
    failed      = len(results) - succeeded
    print(f"\n{'='*70}")
    print(f"  BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"  Total: {len(results)}  OK: {succeeded}  FAIL: {failed}")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    if failed > 0:
        print(f"\n  Failed:")
        for r in results:
            if not r.get('success'):
                print(f"    FAIL {r['api']}: {r.get('error','')[:80]}")

    cat_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    for r in results:
        cat = r.get('category', 'unknown')
        cat_stats[cat]['total'] += 1
        if r.get('success'):
            cat_stats[cat]['success'] += 1
    print(f"\n  By category:")
    for cat, s in sorted(cat_stats.items()):
        print(f"    {cat}: {s['success']}/{s['total']}")

    summary_path = os.path.join(args.output, 'batch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'batch_config': {
                'csv': args.csv, 'scope': args.scope,
                'max_iterations': args.max_iterations,
                'timeout_minutes': args.timeout,
                'total_apis': len(api_list),
            },
            'results': results,
            'summary': {
                'succeeded': succeeded, 'failed': failed,
                'elapsed_seconds': elapsed,
                'by_category': dict(cat_stats),
            },
        }, f, indent=2, default=str)

    print(f"\n  Summary: {summary_path}")
    print(f"\n  Next step: run validation")
    print(f"    python run_and_match_bugsindlls_v14.py \\")
    print(f"        -g {args.csv} -b {args.output} \\")
    print(f"        -r {args.output}/report.md")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
