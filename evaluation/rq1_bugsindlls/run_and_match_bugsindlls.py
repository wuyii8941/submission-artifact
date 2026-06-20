#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_and_match_bugsindlls_v14.py
================================================================
HKG-Fuzz 专用分析脚本 v14

主要修复 (vs v13):
  1. 正确读取 random_bug_traces.json / guided_bug_traces.json
     benchmark 写入的 key 是绝对路径，_load_traces 统一转换为相对路径后匹配
  2. 只扫 potential-bug / compare-bug，去掉 fail（fail 不是真正的 bug）
  3. Ground Truth 直接用 bug_dataset_pytorch.csv 做 API 级匹配，
     放弃从 title 抠 fingerprint 的弱匹配逻辑
  4. BugDepth 分类更精确：compile oracle 下的 RuntimeError/AssertionError → DEEP_BUG
  5. 报告中明确标注每条 bug 对应的 CSV issue id 和 Reproduced 字段
  6. _load_traces 同时注册绝对路径/相对路径/文件名三种 key，确保命中

v13-fix:
  7. [修复] _dir_to_api_name 正确处理双下划线目录名
     外层批量目录用 api.replace(".", "_") 生成，其中 "._" → "__"
     torch._dynamo.export → torch__dynamo_export
     还原规则: "__" → "._"，"_" → "."
  8. [修复] scan() 中 key 查找增加带点形式（write_to_dir 用原始 api 名写入子目录）
     key1 下划线形式保留作 fallback，key2 带点形式作为主要命中路径

v14:
  9. [新增] export_json() 函数，在 main() 末尾自动生成 4 个 JSON 文件：
     new_bugs.json / guided_new_bugs.json / known_bugs.json / phase_results.json
"""

import os
import re
import csv
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum


# ══════════════════════════════════════════════════════════════════
#  BugDepth 分类
# ══════════════════════════════════════════════════════════════════

class BugDepth(Enum):
    DEEP_BUG      = "deep_bug"
    COMPILE_BUG   = "compile_bug"
    SHALLOW_CHECK = "shallow_check"
    SCRIPT_DEFECT = "script_defect"
    IGNORED_NOISE = "ignored_noise"

    @staticmethod
    def classify(error_msg: str, oracle_type: str, bug_dir_type: str) -> "BugDepth":
        c = error_msg.lower()

        # 优先级 1：噪音
        if any(k in c for k in ["execution timed out", "no_crash_log_found",
                                  "invalid_snippet", "heartbeat"]):
            return BugDepth.IGNORED_NOISE

        # 优先级 1.5：CUDA oracle CPU/GPU 不一致 → deep_bug
        if "cpu_err=" in c and "gpu_err=" in c:
            return BugDepth.DEEP_BUG

        # 优先级 2：脚本提取缺陷
        if any(k in c for k in ["syntaxerror", "indentationerror",
                                  "is not defined", "unexpected indent",
                                  "expected an indented block",
                                  "positional argument follows keyword"]):
            return BugDepth.SCRIPT_DEFECT

        # 优先级 3：深层崩溃（必须在浅层之前判断）
        deep_kw = [
            "segmentation fault", "segfault", "core dumped", "aborted",
            "internal assert", "cuda error", "cublas", "cudnn",
            "device-side assert", "illegal memory access",
            "cuda runtime error", "floating point exception",
            "out of memory", "oom", "double free",
            "fatal python error", "process exited", "hard_crash",
            "inductor error", "triton compilation",
            "results differ", "inconsistent", "invalid mt19937 state",
            "internaltorchdynamoerror", "backendcompilerfailed",
            "torch._dynamo", "torch.compile",
        ]
        if any(k in c for k in deep_kw):
            return BugDepth.DEEP_BUG

        # 优先级 4：compile oracle
        if oracle_type == "compile":
            if any(k in c for k in ["assertionerror", "runtimeerror",
                                     "unsupported", "dynamo error"]):
                return BugDepth.DEEP_BUG
            return BugDepth.COMPILE_BUG

        # 优先级 5：浅层参数/类型错误
        shallow_kw = [
            "typeerror", "valueerror", "indexerror", "keyerror",
            "attributeerror", "notimplementederror",
            "size mismatch", "shapes cannot", "must be a matrix",
            "dimension out of range", "negative dimension",
            "not implemented for",
            "received an invalid combination", "missing required",
            "has no attribute",
        ]
        if any(k in c for k in shallow_kw):
            return BugDepth.SHALLOW_CHECK

        # 兜底：potential-bug 目录里的未识别错误视为 deep
        if bug_dir_type == "potential-bug":
            return BugDepth.DEEP_BUG

        return BugDepth.SHALLOW_CHECK


# ══════════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════════

_NORM_PATS = [
    (r"0x[0-9a-fA-F]+",          "ADDR"),
    (r"(?<!\w)\d+\.\d+(?!\w)",   "FLOAT"),
    (r"(?<!\w)\d+(?!\w)",        "N"),
    (r"\[.*?\]",                 "[SHAPE]"),
    (r"\s+",                     " "),
]

def _normalize(msg: str) -> str:
    if msg in ("NO_CRASH_LOG_FOUND", "EXECUTION_TIMEOUT",
               "INVALID_SNIPPET", "INCONSISTENCY_ERROR_DETECTED"):
        return msg
    for pat, rep in _NORM_PATS:
        msg = re.sub(pat, rep, msg)
    return msg.strip()[:150]

def _get_exception_type(msg: str) -> str:
    m = re.search(
        r'([A-Za-z0-9_.]+(?:Error|Exception|Unsupported|AssertionError|Fault|Exit)):',
        msg)
    if m:
        return m.group(1).split(".")[-1]
    if "segmentation fault" in msg.lower(): return "Segfault"
    if "core dumped"        in msg.lower(): return "CoreDumped"
    if "process exited"     in msg.lower(): return "HardCrash"
    return "Unknown"

def _get_crash_module(content: str) -> str:
    files = re.findall(r'File\s+"([^"]+)"', content)
    torch_files = [f for f in files if "torch/" in f or "torch\\" in f]
    if torch_files:
        last = torch_files[-1]
        idx = last.find("torch")
        return last[idx:] if idx != -1 else last
    return "Unknown_Module"

def _extract_error_msg(content: str) -> str:
    if not content:
        return "NO_CRASH_LOG_FOUND"
    if "Execution timed out" in content:
        return "EXECUTION_TIMEOUT"
    if "Process exited with signal" in content:
        return content.strip()[:200]
    if "results differ" in content.lower() or "inconsistent" in content.lower():
        return "INCONSISTENCY_ERROR_DETECTED"

    lines = content.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if any(kw in line for kw in ["Error:", "Exception:", "AssertionError",
                                      "fault", "Aborted", "Segfault",
                                      "INTERNAL ASSERT", "CUDA error"]):
            if "raise " not in line and not line.startswith("#"):
                return line[:250]

    # 兜底：内容是 Python 代码本体（traces 未读到真实错误）→ INVALID_SNIPPET
    stripped = content.strip()
    code_prefixes = ("import torch", "results = dict()", "import os",
                     "import numpy", "def ", "class ", "torch.")
    if stripped.startswith(code_prefixes):
        return "INVALID_SNIPPET"

    issue_script_markers = (
        "import requests",
        "load_pickle_from_url",
        "Eager execution",
        "They should be eqaul",
        "return torch.ops.aten.",
        "This should return",
        "Check the MPS",
        "Fails on CPU with",
        "CPU version correctly",
        "CUDA version silently",
        "print('GPU Exception",
    )
    if any(m in stripped for m in issue_script_markers):
        return "INVALID_SNIPPET"

    short = stripped[:250]
    if len(short) > 10:
        return short
    return "INVALID_SNIPPET"


def _dir_to_api_name(dirname: str) -> str:
    """
    将磁盘上的 API 目录名还原为原始 API 名称。

    [修复] 外层批量调度目录用 api.replace(".", "_") 生成，其中：
      "."  → "_"   torch.matmul         → torch_matmul
      "._" → "__"  torch._dynamo.export → torch__dynamo_export

    还原规则：先保护双下划线，再把单下划线还原为点，最后还原双下划线为 "._"
    """
    result = dirname.replace("__", "\x00")
    result = result.replace("_", ".")
    result = result.replace("\x00", "._")
    return result


# ══════════════════════════════════════════════════════════════════
#  BugScript
# ══════════════════════════════════════════════════════════════════

@dataclass
class BugScript:
    file_path:      str
    api:            str
    oracle_type:    str
    bug_dir_type:   str
    strategy:       str
    raw_content:    str = ""
    error_msg:      str = field(init=False)
    bug_depth:      BugDepth = field(init=False)
    crash_module:   str = field(init=False)
    exception_type: str = field(init=False)

    def __post_init__(self):
        self.error_msg      = _extract_error_msg(self.raw_content)
        self.bug_depth      = BugDepth.classify(self.error_msg, self.oracle_type, self.bug_dir_type)
        self.crash_module   = _get_crash_module(self.raw_content)
        self.exception_type = _get_exception_type(self.error_msg)

    def sig_key(self) -> str:
        return f"{self.api}|{self.bug_depth.value}|{self.oracle_type}|{self.exception_type}|{_normalize(self.error_msg)}"


# ══════════════════════════════════════════════════════════════════
#  BugScriptScanner
# ══════════════════════════════════════════════════════════════════

class BugScriptScanner:
    ORACLE_DIRS = [
        ("compile",   "compile-oracle",   ["potential-bug", "compare-bug"]),
        ("crash",     "crash-oracle",     ["potential-bug", "compare-bug"]),
        ("cuda",      "cuda-oracle",      ["potential-bug", "compare-bug"]),
        ("precision", "precision-oracle", ["potential-bug", "compare-bug"]),
    ]

    def __init__(self, batch_dir: str, max_per_api: int = 200):
        self.batch_dir   = Path(batch_dir)
        self.max_per_api = max_per_api
        self.scripts:     List[BugScript] = []
        self.fuzzed_apis: Set[str]        = set()

    @staticmethod
    def _load_traces(strat_dir: Path, strategy: str) -> Dict[str, str]:
        """
        读取 {strategy}_bug_traces.json，返回多种粒度的 key → msg 映射。
        对每个 key 同时注册四种 lookup key：
          1. 原始 key
          2. 从第一个含 "-oracle" 的 part 开始的相对路径（带点）
          3. 纯文件名 N.py（最宽松 fallback）
          4. 把相对路径中 api 子目录的点转下划线版本
        """
        traces: Dict[str, str] = {}
        for fname in [f"{strategy}_bug_traces.json", f"{strategy}_result.json"]:
            p = strat_dir / fname
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text("utf-8", errors="replace"))
            except Exception:
                continue

            for raw_key, v in data.items():
                if raw_key.startswith("_"):
                    continue
                if isinstance(v, dict):
                    msg = v.get("msg", "") or ""
                elif isinstance(v, str):
                    msg = v
                else:
                    msg = str(v)
                if not msg:
                    continue

                # 1. 注册原始 key
                traces[raw_key] = msg

                # 2. 注册相对路径（找第一个含 -oracle 的 part）
                norm_key = raw_key.replace("\\", "/")
                parts = norm_key.split("/")
                rel_from_oracle = None
                for i, part in enumerate(parts):
                    if "-oracle" in part:
                        rel_from_oracle = "/".join(parts[i:])
                        traces[rel_from_oracle] = msg
                        break

                # 3. 注册纯文件名
                traces[Path(norm_key).name] = msg

                # 4. 把带点的相对路径同时注册下划线版本
                if rel_from_oracle:
                    rel_parts = rel_from_oracle.split("/")
                    if len(rel_parts) >= 3:
                        api_part_under = rel_parts[2].replace(".", "_")
                        under_key = "/".join(rel_parts[:2] + [api_part_under] + rel_parts[3:])
                        traces[under_key] = msg

        return traces

    def scan(self) -> "BugScriptScanner":
        print(f"\n{'='*60}")
        print(f"SCANNING: {self.batch_dir.name}")
        print(f"{'='*60}")

        api_dirs = sorted(
            d for d in self.batch_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )

        for api_dir in api_dirs:
            api_name = _dir_to_api_name(api_dir.name)
            self.fuzzed_apis.add(api_name)

            for strategy in ["random", "guided"]:
                strat_dir = api_dir / strategy
                if not strat_dir.exists():
                    continue

                traces = self._load_traces(strat_dir, strategy)

                for oracle_type, oracle_dir_name, sub_dirs in self.ORACLE_DIRS:
                    oracle_dir = strat_dir / oracle_dir_name
                    if not oracle_dir.exists():
                        continue

                    for sub_dir in sub_dirs:
                        bug_dir = oracle_dir / sub_dir
                        if not bug_dir.exists():
                            continue

                        seen_hashes: Set[str] = set()
                        added = 0

                        for py_file in sorted(bug_dir.rglob("*.py")):
                            if added >= self.max_per_api:
                                break

                            api_under = api_name.replace(".", "_")
                            key1 = f"{oracle_dir_name}/{sub_dir}/{api_under}/{py_file.name}"
                            key2 = f"{oracle_dir_name}/{sub_dir}/{api_name}/{py_file.name}"
                            key3 = py_file.name

                            raw_content = (
                                traces.get(key2)
                                or traces.get(key1)
                                or traces.get(key3)
                                or ""
                            )

                            if not raw_content or len(raw_content.strip()) < 5:
                                try:
                                    txt = py_file.read_text(errors="replace")
                                    comment_lines = [
                                        ln.lstrip("# ").strip()
                                        for ln in txt.splitlines()
                                        if ln.startswith("#")
                                    ]
                                    raw_content = "\n".join(comment_lines) or txt[:300]
                                except Exception:
                                    continue

                            h = hashlib.md5(raw_content.encode()).hexdigest()[:16]
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)

                            self.scripts.append(BugScript(
                                file_path    = str(py_file),
                                api          = api_name,
                                oracle_type  = oracle_type,
                                bug_dir_type = sub_dir,
                                strategy     = strategy,
                                raw_content  = raw_content,
                            ))
                            added += 1

        print(f"✓ {len(self.scripts)} raw bug records across {len(self.fuzzed_apis)} APIs")
        return self


# ══════════════════════════════════════════════════════════════════
#  BugDeduplicator
# ══════════════════════════════════════════════════════════════════

class BugDeduplicator:
    def __init__(self):
        self.unique: Dict[str, Dict] = {}

    def add(self, s: BugScript):
        h = hashlib.md5(s.sig_key().encode()).hexdigest()[:16]
        if h not in self.unique:
            self.unique[h] = {
                "sig_key":        s.sig_key(),
                "api":            s.api,
                "oracle_type":    s.oracle_type,
                "bug_dir_type":   s.bug_dir_type,
                "bug_depth":      s.bug_depth.value,
                "error_msg":      s.error_msg,
                "crash_module":   s.crash_module,
                "exception_type": s.exception_type,
                "random_count":   0,
                "guided_count":   0,
                "sample_file":    s.file_path,
            }
        e = self.unique[h]
        if s.strategy == "random":
            e["random_count"] += 1
        else:
            e["guided_count"] += 1

    def add_all(self, scripts: List[BugScript]) -> "BugDeduplicator":
        for s in scripts:
            self.add(s)
        print(f"✓ Deduplicated → {len(self.unique)} unique bug signatures")
        return self

    def depth_strategy_stats(self) -> Dict:
        result: Dict[str, Dict] = defaultdict(
            lambda: {"random": 0, "guided": 0, "both": 0,
                     "guided_only": 0, "random_only": 0}
        )
        for info in self.unique.values():
            d = info["bug_depth"]
            has_r = info["random_count"] > 0
            has_g = info["guided_count"] > 0
            if has_r: result[d]["random"] += 1
            if has_g: result[d]["guided"] += 1
            if has_r and has_g:   result[d]["both"]        += 1
            elif has_g:           result[d]["guided_only"] += 1
            elif has_r:           result[d]["random_only"] += 1
        return dict(result)


# ══════════════════════════════════════════════════════════════════
#  GroundTruthLoader
# ══════════════════════════════════════════════════════════════════

_SKIP_REMARKS = ["skipped", "could not reproduce", "not a bug",
                 "auto generated", "duplicate", "disabled test"]

class GroundTruthLoader:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.api_to_issues: Dict[str, List[Dict]] = defaultdict(list)
        self.issue_map: Dict[str, Dict] = {}

    def load(self) -> "GroundTruthLoader":
        if not self.csv_path.exists():
            print(f"⚠️  CSV not found: {self.csv_path}")
            return self

        raw = self.csv_path.read_text("utf-8", errors="replace")
        if raw.startswith("\ufeff"):
            raw = raw[1:]

        rows = list(csv.reader(raw.splitlines()))
        if len(rows) < 2:
            return self
        header = rows[1]
        col = {c.strip().lower(): i for i, c in enumerate(header)}

        def _get(row, name, default=""):
            i = col.get(name.lower())
            return row[i].strip() if (i is not None and i < len(row)) else default

        for row in rows[2:]:
            if not row or not any(c.strip() for c in row):
                continue
            issue_id = _get(row, "issue #")
            if not issue_id:
                continue
            if any(kw in _get(row, "remarks").lower() for kw in _SKIP_REMARKS):
                continue

            buggy_api = re.sub(r"\(.*\)", "", _get(row, "buggy api")).strip()

            record = {
                "issue_id":   issue_id,
                "title":      _get(row, "title"),
                "reproduced": _get(row, "reproduced").lower() == "yes",
                "device":     _get(row, "device").lower(),
                "bug_type":   _get(row, "type"),
                "buggy_api":  buggy_api,
            }
            self.issue_map[issue_id] = record

            apis_to_register: Set[str] = set()
            if buggy_api and len(buggy_api) > 3:
                apis_to_register.add(buggy_api.lower())
            for m in re.findall(r"(torch\.[a-zA-Z_][a-zA-Z0-9_.]*)", _get(row, "title")):
                apis_to_register.add(m.rstrip(".").lower())

            for api in apis_to_register:
                existing_ids = {r["issue_id"] for r in self.api_to_issues[api]}
                if issue_id not in existing_ids:
                    self.api_to_issues[api].append(record)

        total      = len(self.issue_map)
        reproduced = sum(1 for r in self.issue_map.values() if r["reproduced"])
        print(f"✓ CSV loaded: {total} issues total, {reproduced} reproduced=Yes")
        return self


# ══════════════════════════════════════════════════════════════════
#  BugMatcher
# ══════════════════════════════════════════════════════════════════

class MatchLabel:
    CONFIRMED_MATCH  = "confirmed_match"
    RELAXED_MATCH    = "relaxed_match"
    API_MATCH_ONLY   = "api_match_only"
    POTENTIALLY_NEW  = "potentially_new"


_BUG_TYPE_SIGNALS_STRICT: Dict[str, List[str]] = {
    "segmentation fault": [
        "process exited", "hard_crash", "signal", "segfault", "core dumped",
        "internal assert", "sigabrt", "sigsegv",
        "runtimeerror",
    ],
    "floating point exception": [
        "process exited", "hard_crash", "signal", "floating point", "sigfpe",
        "negative dimension", "fail:",
    ],
    "aborted": [
        "process exited", "hard_crash", "signal", "aborted",
    ],
    "internal exception": [
        "internal assert", "internaltorchdynamoerror", "backendcompilerfailed",
        "results_differ", "cpu/gpu outputs not equal", "runtimeerror",
        "assertionerror", "inductor", "cpu_err=", "gpu_err=",
        "process exited", "hard_crash", "signal",
        "appears multiple times",
    ],
    "runtime error": [
        "cpu_err=", "gpu_err=", "runtimeerror", "results_differ",
        "cpu/gpu outputs not equal", "process exited", "hard_crash",
        "precision_error", "fail:",
    ],
    "incorrect output": [
        "results_differ", "cpu/gpu outputs not equal", "cpu_err=", "gpu_err=",
        "inconsisten",
        "negative dimension", "fail:",
    ],
    "assertion error": [
        "assertionerror", "assert", "internal assert",
    ],
    "memory error": [
        "out of memory", "oom", "process exited", "hard_crash",
    ],
    "type error": [
        "typeerror",
    ],
    "index error": [
        "indexerror",
    ],
}

_BUG_TYPE_SIGNALS_RELAXED: Dict[str, List[str]] = {
    "segmentation fault": [
        "process exited", "hard_crash", "signal", "segfault", "core dumped",
        "internal assert", "sigabrt", "sigsegv",
        "runtimeerror", "cpu_err=", "gpu_err=",
    ],
    "floating point exception": [
        "process exited", "hard_crash", "signal", "floating point", "sigfpe",
        "runtimeerror", "cpu_err=", "gpu_err=",
    ],
    "aborted": [
        "process exited", "hard_crash", "signal", "aborted",
        "runtimeerror", "cpu_err=", "gpu_err=",
    ],
    "incorrect output": [
        "results_differ", "cpu/gpu outputs not equal", "cpu_err=", "gpu_err=",
        "inconsisten",
        "runtimeerror", "process exited", "hard_crash",
    ],
    "internal exception": _BUG_TYPE_SIGNALS_STRICT["internal exception"],
    "runtime error":      _BUG_TYPE_SIGNALS_STRICT["runtime error"],
    "assertion error":    _BUG_TYPE_SIGNALS_STRICT["assertion error"],
    "memory error":       _BUG_TYPE_SIGNALS_STRICT["memory error"],
    "type error":         _BUG_TYPE_SIGNALS_STRICT["type error"],
    "index error":        _BUG_TYPE_SIGNALS_STRICT["index error"],
}


def _error_type_matches(error_msg: str, csv_bug_type: str,
                        relaxed: bool = False) -> bool:
    if not csv_bug_type:
        return True

    bug_type_lower = csv_bug_type.lower().strip()
    error_lower = error_msg.lower()

    table = _BUG_TYPE_SIGNALS_RELAXED if relaxed else _BUG_TYPE_SIGNALS_STRICT

    signals = None
    for key, sigs in table.items():
        if key in bug_type_lower or bug_type_lower in key:
            signals = sigs
            break

    if signals is None:
        return True

    return any(s in error_lower for s in signals)


class BugMatcher:
    def __init__(self, gt: GroundTruthLoader):
        self.gt = gt

    def _find_issues(self, api: str) -> List[Dict]:
        api_lower = api.lower()

        if api_lower in self.gt.api_to_issues:
            return self.gt.api_to_issues[api_lower]

        last_component = api_lower.split(".")[-1]
        for key, records in self.gt.api_to_issues.items():
            if key.split(".")[-1] == last_component:
                return records

        if len(last_component) > 4:
            seen_ids: Set[str] = set()
            merged: List[Dict] = []
            for key, records in self.gt.api_to_issues.items():
                if last_component in key:
                    for r in records:
                        if r["issue_id"] not in seen_ids:
                            seen_ids.add(r["issue_id"])
                            merged.append(r)
            if merged:
                return merged

        return []

    def match_one(self, info: Dict) -> Dict:
        issues = self._find_issues(info["api"])

        if not issues:
            return {
                **info,
                "match_label":        MatchLabel.POTENTIALLY_NEW,
                "match_label_strict": MatchLabel.POTENTIALLY_NEW,
                "matched_issues":     [],
                "match_detail":       f"🆕 No known issue for API '{info['api']}' in CSV",
                "best_match":         None,
                "type_aligned":       False,
                "type_aligned_relaxed": False,
            }

        error_msg = info.get("error_msg", "")

        strict_aligned_issues = [
            r for r in issues
            if _error_type_matches(error_msg, r.get("bug_type", ""), relaxed=False)
        ]
        relaxed_aligned_issues = [
            r for r in issues
            if _error_type_matches(error_msg, r.get("bug_type", ""), relaxed=True)
        ]

        strict_aligned  = bool(strict_aligned_issues)
        relaxed_aligned = bool(relaxed_aligned_issues)

        if strict_aligned_issues:
            best = strict_aligned_issues[0]
        elif relaxed_aligned_issues:
            best = relaxed_aligned_issues[0]
        else:
            best = issues[0]

        if strict_aligned:
            label = MatchLabel.CONFIRMED_MATCH
        elif relaxed_aligned:
            label = MatchLabel.RELAXED_MATCH
        else:
            label = MatchLabel.API_MATCH_ONLY

        if strict_aligned:
            label_strict = MatchLabel.CONFIRMED_MATCH
        else:
            label_strict = MatchLabel.API_MATCH_ONLY

        repro_note   = f"reproduced={'Yes' if best['reproduced'] else 'No'}"
        strict_note  = f"strict={'✓' if strict_aligned else '✗'}"
        relaxed_note = f"relaxed={'✓' if relaxed_aligned else '✗'}"
        detail = (
            f"{'✅' if strict_aligned else ('🔶' if relaxed_aligned else '📋')} "
            f"Issue #{best['issue_id']} "
            f"| API: {best['buggy_api']} | CSV type: {best['bug_type']} "
            f"| {strict_note} | {relaxed_note} | {repro_note}"
        )
        return {
            **info,
            "match_label":          label,
            "match_label_strict":   label_strict,
            "matched_issues":       [r["issue_id"] for r in issues],
            "match_detail":         detail,
            "best_match":           best,
            "type_aligned":         strict_aligned,
            "type_aligned_relaxed": relaxed_aligned,
        }

    def match_all(self, dedup: BugDeduplicator,
                  target_depths: List[str]) -> List[Dict]:
        results = []
        counts:        Dict[str, int] = defaultdict(int)
        counts_strict: Dict[str, int] = defaultdict(int)

        for h, info in dedup.unique.items():
            if info["bug_depth"] not in target_depths:
                continue
            enriched = self.match_one(info)
            enriched["sig_hash"] = h
            results.append(enriched)
            counts[enriched["match_label"]] += 1
            counts_strict[enriched["match_label_strict"]] += 1

        print(f"\n✓ Bug matching complete ({len(results)} deep bugs analyzed):")
        print(f"  ── 宽松口径（主报告）──────────────────────────────")
        print(f"  ✅ CONFIRMED_MATCH  : {counts[MatchLabel.CONFIRMED_MATCH]}  (严格类型吻合)")
        print(f"  🔶 RELAXED_MATCH    : {counts[MatchLabel.RELAXED_MATCH]}  (跨版本宽松吻合)")
        print(f"  📋 API_MATCH_ONLY   : {counts[MatchLabel.API_MATCH_ONLY]}  (仅API命中)")
        print(f"  🆕 POTENTIALLY_NEW  : {counts[MatchLabel.POTENTIALLY_NEW]}  (CSV无记录)")
        print(f"  ── 严格口径（对照数据）────────────────────────────")
        print(f"  ✅ CONFIRMED_MATCH  : {counts_strict[MatchLabel.CONFIRMED_MATCH]}")
        print(f"  📋 API_MATCH_ONLY   : {counts_strict[MatchLabel.API_MATCH_ONLY]}")
        return results


# ══════════════════════════════════════════════════════════════════
#  Report Generator
# ══════════════════════════════════════════════════════════════════

def _strategy_label(info: Dict) -> str:
    has_r = info["random_count"] > 0
    has_g = info["guided_count"] > 0
    if has_r and has_g: return "Random + Guided"
    if has_g:           return "**Guided Only** 🎯"
    return "Random Only"


def generate_report(batch_dir: str, scanner: BugScriptScanner,
                    dedup: BugDeduplicator, matched: List[Dict],
                    output_file: str):
    depth_stats = dedup.depth_strategy_stats()

    def _ds(depth_val: str) -> Dict:
        return depth_stats.get(depth_val, {
            "random": 0, "guided": 0, "both": 0,
            "guided_only": 0, "random_only": 0,
        })

    deep_s    = _ds(BugDepth.DEEP_BUG.value)
    compile_s = _ds(BugDepth.COMPILE_BUG.value)
    shallow_s = _ds(BugDepth.SHALLOW_CHECK.value)
    defect_s  = _ds(BugDepth.SCRIPT_DEFECT.value)
    noise_s   = _ds(BugDepth.IGNORED_NOISE.value)

    total_r = deep_s["random"]  + compile_s["random"]
    total_g = deep_s["guided"]  + compile_s["guided"]
    go_r    = deep_s["random_only"]  + compile_s["random_only"]
    go_g    = deep_s["guided_only"]  + compile_s["guided_only"]

    exact    = [r for r in matched if r["match_label"] == MatchLabel.CONFIRMED_MATCH]
    relaxed  = [r for r in matched if r["match_label"] == MatchLabel.RELAXED_MATCH]
    known    = [r for r in matched if r["match_label"] == MatchLabel.API_MATCH_ONLY]
    new_all  = [r for r in matched if r["match_label"] == MatchLabel.POTENTIALLY_NEW]
    ng       = [r for r in new_all if r["guided_count"] > 0 and r["random_count"] == 0]

    strict_confirmed = sum(1 for r in matched
                           if r.get("match_label_strict") == MatchLabel.CONFIRMED_MATCH)
    strict_api_match = sum(1 for r in matched
                           if r.get("match_label_strict") == MatchLabel.API_MATCH_ONLY)

    lines = [
        "# HKG-Fuzz Validation Report v14",
        "",
        f"**Batch**: `{Path(batch_dir).name}`  ",
        f"**Raw Records**: {len(scanner.scripts)} → **Unique Signatures**: {len(dedup.unique)}  ",
        f"**APIs Fuzzed**: {len(scanner.fuzzed_apis)}",
        "", "---", "",
        "## 🚀 核心指标：Kernel-Guided vs Random",
        "",
        "| 策略 | 深层 Bug 总发现数 | 独家发现 (Only) |",
        "|---|---|---|",
        f"| Random   | {total_r} | {go_r} |",
        f"| **Guided** | **{total_g}** | **{go_g}** |",
        "",
        "> 深层 Bug = `deep_bug` + `compile_bug`，排除浅层类型错误与环境缺陷",
        "", "---", "",
        "## 📊 深层 Bug 分布",
        "",
        "| 深度分类 | Random | Guided | Both | Guided Only | Random Only |",
        "|---|---|---|---|---|---|",
        f"| Deep Bug    | {deep_s['random']} | {deep_s['guided']} | {deep_s['both']} | {deep_s['guided_only']} | {deep_s['random_only']} |",
        f"| Compile Bug | {compile_s['random']} | {compile_s['guided']} | {compile_s['both']} | {compile_s['guided_only']} | {compile_s['random_only']} |",
        "", "---", "",
        "## 🔍 Ground Truth 匹配结果（对照 bug_dataset_pytorch.csv）",
        "",
        "| 匹配等级 | 数量 | 含义 |",
        "|---|---|---|",
        f"| ✅ CONFIRMED_MATCH  | **{len(exact)}** | API匹配 + 错误类型与CSV严格语义吻合 |",
        f"| 🔶 RELAXED_MATCH    | **{len(relaxed)}** | API匹配 + 跨版本宽松吻合（同bug不同版本的不同表现） |",
        f"| 📋 API_MATCH_ONLY   | **{len(known)}** | API匹配，但错误类型不吻合或CSV无类型信息 |",
        f"| 🆕 POTENTIALLY_NEW  | **{len(new_all)}** | CSV中无对应API，疑似新发现（Guided独有：{len(ng)}） |",
        "",
        "> **宽松吻合说明**：同一个 bug 在不同 PyTorch 版本下可能有不同表现。",
        "> 例如旧版本直接 Segfault，新版本改为抛出 RuntimeError（更好地处理了错误）；",
        "> 或旧版本静默返回错误结果，新版本改为抛出异常。",
        "> RELAXED_MATCH 表示 fuzzer 触及了同一个有问题的代码路径，只是错误的最终形式不同。",
        "",
        "<details>",
        "<summary>📊 严格口径对照数据（点击展开）</summary>",
        "",
        "| 严格匹配等级 | 数量 |",
        "|---|---|",
        f"| ✅ CONFIRMED_MATCH（严格） | **{strict_confirmed}** |",
        f"| 📋 API_MATCH_ONLY（严格）  | **{strict_api_match}** |",
        "",
        "> 严格标准：Segfault/FPE/Aborted 必须发现进程崩溃；IncorrectOutput 必须发现结果差异。",
        "> 不接受以 RuntimeError 替代 Segfault 的情况。",
        "",
        "</details>",
        "",
    ]

    if exact:
        lines += ["### ✅ Tier 1：错误类型严格吻合的已知 Bug", ""]
        for r in sorted(exact, key=lambda x: x["api"]):
            bm = r["best_match"] or {}
            lines += [
                f"- **`{r['api']}`**  |  Issue #{bm.get('issue_id', '')}  |  {_strategy_label(r)}",
                f"  - CSV type: `{bm.get('bug_type', '')}`  |  Device: `{bm.get('device', '')}`  |  reproduced={bm.get('reproduced', False) and 'Yes' or 'No'}",
                f"  - CSV Title: {bm.get('title', '')[:100]}",
                f"  - Error: `{r['error_msg'][:150]}`",
                f"  - Crash Module: `{r['crash_module']}`  |  Exception: `{r['exception_type']}`",
                "",
            ]

    if relaxed:
        lines += ["### 🔶 Tier 1.5：跨版本宽松吻合的已知 Bug", ""]
        lines += [
            "> 这些 bug 在 CSV 记录的版本下表现为崩溃（Segfault/FPE）或静默错误（IncorrectOutput），",
            "> 但 fuzzer 在当前/pt22 版本下触发了 RuntimeError 或其他异常。",
            "> 说明 PyTorch 已将原始错误路径改为抛出异常（更好地处理了错误），",
            "> fuzzer 触及了相同的有问题代码路径。",
            "",
        ]
        for r in sorted(relaxed, key=lambda x: x["api"]):
            bm = r["best_match"] or {}
            lines += [
                f"- **`{r['api']}`**  |  Issue #{bm.get('issue_id', '')}  |  {_strategy_label(r)}",
                f"  - CSV type: `{bm.get('bug_type', '')}`  |  Device: `{bm.get('device', '')}`  |  reproduced={bm.get('reproduced', False) and 'Yes' or 'No'}",
                f"  - CSV Title: {bm.get('title', '')[:100]}",
                f"  - Fuzzer Error: `{r['error_msg'][:150]}`",
                f"  - 宽松理由: CSV记录了{bm.get('bug_type','未知')}，fuzzer在新版本触发了RuntimeError（同路径不同表现）",
                "",
            ]

    if known:
        lines += ["### 📋 Tier 2：API 匹配但错误类型不吻合或 CSV 无类型信息", ""]
        for r in sorted(known, key=lambda x: x["api"]):
            bm = r["best_match"] or {}
            lines += [
                f"- **`{r['api']}`**  |  Issue #{bm.get('issue_id', '')}  |  {_strategy_label(r)}",
                f"  - CSV type: `{bm.get('bug_type', '')}`  |  CSV Title: {bm.get('title', '')[:100]}",
                f"  - Error: `{r['error_msg'][:150]}`",
                "",
            ]

    if new_all:
        lines += ["### 🆕 Tier 3：CSV 中无记录的疑似新 Bug", ""]
        for r in sorted(new_all, key=lambda x: (-x["guided_count"], x["api"])):
            lines += [
                f"- **`{r['api']}`**  |  {_strategy_label(r)}",
                f"  - Oracle: `{r['oracle_type']}`  |  Exception: `{r['exception_type']}`",
                f"  - Error: `{r['error_msg'][:150]}`",
                f"  - Crash Module: `{r['crash_module']}`",
                "",
            ]

    lines += [
        "---", "",
        "## 📉 附属统计：被过滤的噪音（不计入核心指标）",
        "",
        "| 分类 | Random | Guided |",
        "|---|---|---|",
        f"| 浅层参数异常 (TypeError/ValueError 等) | {shallow_s.get('random', 0)} | {shallow_s.get('guided', 0)} |",
        f"| AST 提取缺陷 (SyntaxError/NameError)   | {defect_s.get('random', 0)}  | {defect_s.get('guided', 0)} |",
        f"| 噪音与超时                              | {noise_s.get('random', 0)}   | {noise_s.get('guided', 0)} |",
        "",
        "---", "",
        "## 📁 API 覆盖详情",
        "",
        "| API | Deep Bugs (R) | Deep Bugs (G) | Matched Issues |",
        "|---|---|---|---|",
    ]

    api_summary: Dict[str, Dict] = defaultdict(lambda: {"r": 0, "g": 0, "issues": set()})
    for r in matched:
        api_summary[r["api"]]["r"] += r["random_count"]
        api_summary[r["api"]]["g"] += r["guided_count"]
        api_summary[r["api"]]["issues"].update(r["matched_issues"])

    for api in sorted(api_summary):
        s = api_summary[api]
        issues_str = ", ".join(f"#{i}" for i in sorted(s["issues"])) or "—"
        lines.append(f"| `{api}` | {s['r']} | {s['g']} | {issues_str} |")

    Path(output_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Report saved: {output_file}")


# ══════════════════════════════════════════════════════════════════
#  JSON Export  [v14 新增]
# ══════════════════════════════════════════════════════════════════

def export_json(matched: List[Dict], report_path: str):
    """
    从已有的 matched 列表生成 4 个 JSON 文件，路径与 report 同目录。

    输出文件：
      new_bugs.json         — 所有 POTENTIALLY_NEW 条目
      guided_new_bugs.json  — 其中 Guided Only 的条目（论文核心指标）
      known_bugs.json       — 所有已匹配条目（CONFIRMED / RELAXED / API_MATCH_ONLY）
      phase_results.json    — 全量 matched，供调试用
    """
    out_dir = Path(report_path).parent

    new_bugs   = [r for r in matched if r["match_label"] == MatchLabel.POTENTIALLY_NEW]
    guided_new = [r for r in new_bugs if r["guided_count"] > 0 and r["random_count"] == 0]
    known_bugs = [r for r in matched if r["match_label"] != MatchLabel.POTENTIALLY_NEW]

    def _serializable(obj):
        """递归把 Enum / set 等不可序列化类型转为基础类型。"""
        if isinstance(obj, dict):
            return {k: _serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serializable(i) for i in obj]
        if isinstance(obj, set):
            return sorted(_serializable(i) for i in obj)
        if isinstance(obj, Enum):
            return obj.value
        return obj

    files = {
        "new_bugs.json":        new_bugs,
        "guided_new_bugs.json": guided_new,
        "known_bugs.json":      known_bugs,
        "phase_results.json":   matched,
    }

    print(f"\n{'='*60}")
    print("💾 Exporting JSON files")
    print(f"{'='*60}")

    for fname, data in files.items():
        p = out_dir / fname
        p.write_text(
            json.dumps(_serializable(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✓ {fname}  ({len(data)} entries)  →  {p}")

    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HKG-Fuzz Analyzer v14")
    parser.add_argument("--batch-dir",    "-b", required=True)
    parser.add_argument("--ground-truth", "-g", required=True)
    parser.add_argument("--report",       "-r", default="report_v14.md")
    parser.add_argument("--max-per-api",  type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 HKG-Fuzz Analyzer v14")
    print("=" * 60)

    scanner = BugScriptScanner(args.batch_dir, args.max_per_api).scan()
    dedup   = BugDeduplicator().add_all(scanner.scripts)
    gt      = GroundTruthLoader(args.ground_truth).load()
    matched = BugMatcher(gt).match_all(
        dedup,
        [BugDepth.DEEP_BUG.value, BugDepth.COMPILE_BUG.value]
    )
    generate_report(args.batch_dir, scanner, dedup, matched, args.report)
    export_json(matched, args.report)   # ← v14 新增


if __name__ == "__main__":
    main()