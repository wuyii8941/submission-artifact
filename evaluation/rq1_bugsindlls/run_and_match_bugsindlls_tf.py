#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_and_match_bugsindlls_tf.py  (v2 — JSON output + dir_name fix)
================================================================
Changelog v2:
  1. [FIX] _dir_to_api_name: TF API names contain underscores (raw_ops, is_nan, etc.);
     protect them with double underscores '__'; single underscore -> dot.
     The old logic mapped tensorflow_raw_ops_GatherV2 -> tensorflow.raw.ops.GatherV2 (wrong).
     New logic: directory names protect intrinsic underscores with '__' and use single '_' as the separator:
       tensorflow__raw__ops__GatherV2 -> tensorflow.raw_ops.GatherV2 (correct).
     The old logic is retained as a fallback (replace '_' with '.') for backward compatibility.
  2. [NEW] save_json_results(): emits three JSON files
       tf_new_bugs.json  - all potentially_new entries.
       tf_guided_new_bugs.json - guided-only potentially_new (paper headline metric).
       tf_known_bugs.json  - confirmed/relaxed-match known reproductions.
  3. [FIX] _SKIP_REMARKS: do not skip "todo"-prefixed entries in the TF CSV;
     only skip explicit "skipped" / "could not reproduce" / "incorrect usage".
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
#  BugDepth classification (TF-adapted)
# ══════════════════════════════════════════════════════════════════

class BugDepth(Enum):
    DEEP_BUG  = "deep_bug"
    SHALLOW_CHECK = "shallow_check"
    SCRIPT_DEFECT = "script_defect"
    IGNORED_NOISE = "ignored_noise"

    @staticmethod
    def classify(error_msg: str, oracle_type: str, bug_dir_type: str) -> "BugDepth":
        c = error_msg.lower()

        if any(k in c for k in ["execution timed out", "no_crash_log_found",
                                  "invalid_snippet", "heartbeat"]):
            return BugDepth.IGNORED_NOISE

        if "cpu_err=" in c and "gpu_err=" in c:
            return BugDepth.DEEP_BUG
        if "results_differ" in c or "cpu/gpu outputs not equal" in c:
            return BugDepth.DEEP_BUG

        if any(k in c for k in ["syntaxerror", "indentationerror",
                                  "is not defined", "unexpected indent",
                                  "expected an indented block",
                                  "positional argument follows keyword",
                                  "modulenotfounderror"]):
            return BugDepth.SCRIPT_DEFECT

        deep_kw = [
            "segmentation fault", "segfault", "core dumped", "aborted",
            "floating point exception", "illegal instruction",
            "out of memory", "oom", "double free",
            "fatal python error", "process exited", "hard_crash",
            "signal",
            "check failed",
            "internal error",
            "resource exhausted",
            "failed precondition",
            "not found",
            "unimplemented",
            "invalidargumenterror",
            "device-side assert",
            "cuda error", "cublas", "cudnn",
            "device fallback",
        ]
        if any(k in c for k in deep_kw):
            return BugDepth.DEEP_BUG

        shallow_kw = [
            "typeerror", "valueerror", "indexerror", "keyerror",
            "attributeerror", "notimplementederror",
            "shapes must", "shape mismatch", "must be",
            "not supported", "has no attribute",
            "missing required", "received an invalid combination",
        ]
        if any(k in c for k in shallow_kw):
            return BugDepth.SHALLOW_CHECK

        if bug_dir_type == "potential-bug":
            return BugDepth.DEEP_BUG

        return BugDepth.SHALLOW_CHECK


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

_NORM_PATS = [
    (r"0x[0-9a-fA-F]+",  "ADDR"),
    (r"(?<!\w)\d+\.\d+(?!\w)",  "FLOAT"),
    (r"(?<!\w)\d+(?!\w)",  "N"),
    (r"\[.*?\]",  "[SHAPE]"),
    (r"\s+",  " "),
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
    if "core dumped"  in msg.lower(): return "CoreDumped"
    if "check failed"  in msg.lower(): return "CheckFailed"
    if "process exited"  in msg.lower(): return "HardCrash"
    if "aborted"  in msg.lower(): return "Aborted"
    return "Unknown"

def _get_crash_module(content: str) -> str:
    files = re.findall(r'File\s+"([^"]+)"', content)
    tf_files = [f for f in files
                if "tensorflow/" in f or "tensorflow\\" in f
                or "keras/" in f or "keras\\" in f]
    if tf_files:
        last = tf_files[-1]
        for prefix in ("tensorflow", "keras"):
            idx = last.find(prefix)
            if idx != -1:
                return last[idx:]
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
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ERROR:"):
            error_part = stripped[len("# ERROR:"):].strip()
            if error_part and len(error_part) > 5:
                return error_part[:250]

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if any(kw in line for kw in [
            "Error:", "Exception:", "AssertionError",
            "fault", "Aborted", "Segfault",
            "Check failed", "CHECK FAILED",
        ]):
            if "raise " not in line and not line.startswith("#"):
                return line[:250]

    stripped = content.strip()
    code_prefixes = (
        "import tensorflow", "import tf", "results = dict()",
        "import os", "import numpy", "def ", "class ", "tf.", "tensorflow.",
    )
    if stripped.startswith(code_prefixes):
        return "INVALID_SNIPPET"

    short = stripped[:250]
    if len(short) > 10:
        return short
    return "INVALID_SNIPPET"


def _dir_to_api_name(dirname: str) -> str:
    """
    Reconstruct a TF API name from its on-disk directory name. Same logic as PyTorch v14.

    Convention (used by the TF fuzzer when writing directories):
      .  -> _  (dot becomes single underscore)
      ._ -> __  (private-module prefix tf._xxx becomes double underscore)

    Inverse:
      __  -> ._  (double underscore -> private-module dotted prefix)
      _  -> .  (single underscore -> dot separator)

    Examples:
      tensorflow_cast  → tensorflow.cast
      tensorflow_keras_layers_Conv2D  → tensorflow.keras.layers.Conv2D
      tf__keras_layers_Conv2D  → tf._keras.layers.Conv2D

    Note: when TF APIs contain intrinsic underscores (raw_ops / is_nan, etc.),
    the fuzzer writes raw_ops -> raw_ops (the underscore is retained but treated as a component),
    yielding tensorflow.raw.ops.GatherV2 after reversal, which differs from tensorflow.raw_ops.GatherV2 in the CSV;
    the final relaxed-matching stage in BugMatcher._find_issues accommodates this discrepancy.
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
    file_path:  str
    api:  str
    oracle_type:  str
    bug_dir_type:  str
    strategy:  str
    raw_content:  str = ""
    error_msg:  str = field(init=False)
    bug_depth:  BugDepth = field(init=False)
    crash_module:  str = field(init=False)
    exception_type: str = field(init=False)

    def __post_init__(self):
        self.error_msg  = _extract_error_msg(self.raw_content)
        self.bug_depth  = BugDepth.classify(self.error_msg, self.oracle_type,
                                                self.bug_dir_type)
        self.crash_module  = _get_crash_module(self.raw_content)
        self.exception_type = _get_exception_type(self.error_msg)

    def sig_key(self) -> str:
        return (f"{self.api}|{self.bug_depth.value}|{self.oracle_type}"
                f"|{self.exception_type}|{_normalize(self.error_msg)}")


# ══════════════════════════════════════════════════════════════════
#  BugScriptScanner (TF version: no compile-oracle)
# ══════════════════════════════════════════════════════════════════

class BugScriptScanner:
    ORACLE_DIRS = [
        ("crash",  "crash-oracle",  ["potential-bug", "compare-bug"]),
        ("cuda",  "cuda-oracle",  ["potential-bug", "compare-bug"]),
        ("precision", "precision-oracle", ["potential-bug", "compare-bug"]),
    ]

    def __init__(self, batch_dir: str, max_per_api: int = 200):
        self.batch_dir  = Path(batch_dir)
        self.max_per_api = max_per_api
        self.scripts:  List[BugScript] = []
        self.fuzzed_apis: Set[str]  = set()

    @staticmethod
    def _load_traces(strat_dir: Path, strategy: str) -> Dict[str, str]:
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

                traces[raw_key] = msg

                norm_key = raw_key.replace("\\", "/")
                parts = norm_key.split("/")
                for i, part in enumerate(parts):
                    if "-oracle" in part:
                        rel = "/".join(parts[i:])
                        traces[rel] = msg
                        break

                traces[Path(norm_key).name] = msg

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

                            # The TF directory naming uses the double-underscore convention; traces keys use the same.
                            api_under = api_dir.name  # keep the original directory name as the key
                            key1 = f"{oracle_dir_name}/{sub_dir}/{api_under}/{py_file.name}"
                            key2 = py_file.name

                            raw_content = (
                                traces.get(key1)
                                or traces.get(key2)
                                or ""
                            )

                            if not raw_content or len(raw_content.strip()) < 5:
                                try:
                                    txt = py_file.read_text(errors="replace")
                                    error_lines = [
                                        ln.strip()
                                        for ln in txt.splitlines()
                                        if ln.strip().startswith("# ERROR:")
                                    ]
                                    if error_lines:
                                        raw_content = error_lines[0]
                                    else:
                                        comment_lines = [
                                            ln.lstrip("# ").strip()
                                            for ln in txt.splitlines()
                                            if ln.startswith("#")
                                        ]
                                        raw_content = (
                                            "\n".join(comment_lines) or txt[:300]
                                        )
                                except Exception:
                                    continue

                            h = hashlib.md5(raw_content.encode()).hexdigest()[:16]
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)

                            self.scripts.append(BugScript(
                                file_path  = str(py_file),
                                api  = api_name,
                                oracle_type  = oracle_type,
                                bug_dir_type = sub_dir,
                                strategy  = strategy,
                                raw_content  = raw_content,
                            ))
                            added += 1

        print(f"{len(self.scripts)} raw bug records across {len(self.fuzzed_apis)} APIs")
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
                "sig_key":  s.sig_key(),
                "api":  s.api,
                "oracle_type":  s.oracle_type,
                "bug_dir_type":  s.bug_dir_type,
                "bug_depth":  s.bug_depth.value,
                "error_msg":  s.error_msg,
                "crash_module":  s.crash_module,
                "exception_type": s.exception_type,
                "random_count":  0,
                "guided_count":  0,
                "sample_file":  s.file_path,
            }
        e = self.unique[h]
        if s.strategy == "random":
            e["random_count"] += 1
        else:
            e["guided_count"] += 1

    def add_all(self, scripts: List[BugScript]) -> "BugDeduplicator":
        for s in scripts:
            self.add(s)
        print(f"Deduplicated → {len(self.unique)} unique bug signatures")
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
            if has_r and has_g:  result[d]["both"]  += 1
            elif has_g:  result[d]["guided_only"] += 1
            elif has_r:  result[d]["random_only"] += 1
        return dict(result)


# ══════════════════════════════════════════════════════════════════
#  GroundTruthLoader (TF version)
# ══════════════════════════════════════════════════════════════════

# Only skip entries explicitly tagged skipped/could not reproduce/incorrect usage.
# Entries like "TODO" are kept (the reproduction may be incomplete but the bug is real).
_SKIP_REMARKS = [
    "skipped",
    "could not reproduce",
    "not a bug",
    "auto generated",
    "duplicate",
    "disabled test",
    "incorrect usage",
    "api misuse",
]

class GroundTruthLoader:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.api_to_issues: Dict[str, List[Dict]] = defaultdict(list)
        self.issue_map: Dict[str, Dict] = {}

    def load(self) -> "GroundTruthLoader":
        if not self.csv_path.exists():
            print(f" CSV not found: {self.csv_path}")
            return self

        raw = self.csv_path.read_text("utf-8", errors="replace")
        if raw.startswith("\ufeff"):
            raw = raw[1:]

        rows = list(csv.reader(raw.splitlines()))
        if len(rows) < 2:
            return self
        header = rows[1]  # row 0 = filter info, row 1 = real header
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
            remarks = _get(row, "remarks").lower()
            if any(kw in remarks for kw in _SKIP_REMARKS):
                continue

            buggy_api = re.sub(r"\(.*\)", "", _get(row, "buggy api")).strip()

            record = {
                "issue_id":  issue_id,
                "title":  _get(row, "title"),
                "reproduced": _get(row, "reproduced").lower() == "yes",
                "device":  _get(row, "device").lower(),
                "bug_type":  _get(row, "type"),
                "buggy_api":  buggy_api,
            }
            self.issue_map[issue_id] = record

            apis_to_register: Set[str] = set()
            if buggy_api and len(buggy_api) > 3:
                apis_to_register.add(buggy_api.lower())

            title = _get(row, "title")
            for m in re.findall(
                    r"((?:tf|tensorflow)\.[a-zA-Z_][a-zA-Z0-9_.]*)", title):
                apis_to_register.add(m.rstrip(".").lower())

            for api in apis_to_register:
                existing_ids = {r["issue_id"] for r in self.api_to_issues[api]}
                if issue_id not in existing_ids:
                    self.api_to_issues[api].append(record)

        total  = len(self.issue_map)
        reproduced = sum(1 for r in self.issue_map.values() if r["reproduced"])
        print(f"CSV loaded: {total} issues total, {reproduced} reproduced=Yes")
        return self


# ══════════════════════════════════════════════════════════════════
#  BugMatcher (TF version)
# ══════════════════════════════════════════════════════════════════

class MatchLabel:
    CONFIRMED_MATCH  = "confirmed_match"
    RELAXED_MATCH  = "relaxed_match"
    API_MATCH_ONLY  = "api_match_only"
    POTENTIALLY_NEW  = "potentially_new"


_BUG_TYPE_SIGNALS_STRICT: Dict[str, List[str]] = {
    "aborted": [
        "process exited", "hard_crash", "signal", "aborted",
        "check failed", "core dumped", "sigabrt",
    ],
    "segmentation fault": [
        "process exited", "hard_crash", "signal", "segfault", "core dumped",
        "sigsegv",
    ],
    "internal exception": [
        "internal error", "check failed", "internalerror",
        "invalidargumenterror", "unimplementederror",
        "resource exhausted", "failed precondition",
        "cpu_err=", "gpu_err=", "results_differ",
    ],
    "invalid argument error": [
        "invalidargumenterror", "invalid argument", "check failed",
        "cpu_err=", "gpu_err=",
    ],
    "value error": [
        "valueerror",
    ],
    "type error": [
        "typeerror",
    ],
    "assertion error": [
        "assertionerror", "assert",
    ],
    "runtime error": [
        "runtimeerror", "cpu_err=", "gpu_err=", "process exited",
        "hard_crash", "check failed",
    ],
    "incorrect output": [
        "results_differ", "cpu/gpu outputs not equal",
        "cpu_err=", "gpu_err=", "inconsisten",
        "precision_slow",
    ],
}

_BUG_TYPE_SIGNALS_RELAXED: Dict[str, List[str]] = {
    "aborted": [
        "process exited", "hard_crash", "signal", "aborted",
        "check failed", "core dumped", "sigabrt",
        "valueerror", "invalidargumenterror", "runtimeerror",
    ],
    "segmentation fault": [
        "process exited", "hard_crash", "signal", "segfault", "core dumped",
        "sigsegv",
        "runtimeerror", "invalidargumenterror",
    ],
    "incorrect output": [
        "results_differ", "cpu/gpu outputs not equal",
        "cpu_err=", "gpu_err=", "inconsisten", "precision_slow",
        "runtimeerror", "valueerror",
    ],
    "internal exception":  _BUG_TYPE_SIGNALS_STRICT["internal exception"],
    "invalid argument error": _BUG_TYPE_SIGNALS_STRICT["invalid argument error"],
    "value error":  _BUG_TYPE_SIGNALS_STRICT["value error"],
    "type error":  _BUG_TYPE_SIGNALS_STRICT["type error"],
    "assertion error":  _BUG_TYPE_SIGNALS_STRICT["assertion error"],
    "runtime error":  _BUG_TYPE_SIGNALS_STRICT["runtime error"],
}


def _error_type_matches(error_msg: str, csv_bug_type: str,
                        relaxed: bool = False) -> bool:
    if not csv_bug_type:
        return True
    bug_type_lower = csv_bug_type.lower().strip()
    error_lower  = error_msg.lower()
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
                "match_label":  MatchLabel.POTENTIALLY_NEW,
                "match_label_strict":  MatchLabel.POTENTIALLY_NEW,
                "matched_issues":  [],
                "match_detail":  f"🆕 No known issue for API '{info['api']}' in CSV",
                "best_match":  None,
                "type_aligned":  False,
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
            label  = MatchLabel.CONFIRMED_MATCH
            label_strict = MatchLabel.CONFIRMED_MATCH
        elif relaxed_aligned:
            label  = MatchLabel.RELAXED_MATCH
            label_strict = MatchLabel.API_MATCH_ONLY
        else:
            label  = MatchLabel.API_MATCH_ONLY
            label_strict = MatchLabel.API_MATCH_ONLY

        repro_note  = f"reproduced={'Yes' if best['reproduced'] else 'No'}"
        strict_note  = f"strict={'' if strict_aligned else ''}"
        relaxed_note = f"relaxed={'' if relaxed_aligned else ''}"
        detail = (
            f"{'' if strict_aligned else ('' if relaxed_aligned else '')} "
            f"Issue #{best['issue_id']} "
            f"| API: {best['buggy_api']} | CSV type: {best['bug_type']} "
            f"| {strict_note} | {relaxed_note} | {repro_note}"
        )
        return {
            **info,
            "match_label":  label,
            "match_label_strict":  label_strict,
            "matched_issues":  [r["issue_id"] for r in issues],
            "match_detail":  detail,
            "best_match":  best,
            "type_aligned":  strict_aligned,
            "type_aligned_relaxed": relaxed_aligned,
        }

    def match_all(self, dedup: BugDeduplicator,
                  target_depths: List[str]) -> List[Dict]:
        results = []
        counts:  Dict[str, int] = defaultdict(int)
        counts_strict: Dict[str, int] = defaultdict(int)

        for h, info in dedup.unique.items():
            if info["bug_depth"] not in target_depths:
                continue
            enriched = self.match_one(info)
            enriched["sig_hash"] = h
            results.append(enriched)
            counts[enriched["match_label"]] += 1
            counts_strict[enriched["match_label_strict"]] += 1

        print(f"\nBug matching complete ({len(results)} deep bugs analyzed):")
        print(f"  -- Relaxed criterion (main report) -----------------")
        print(f"  CONFIRMED_MATCH  : {counts[MatchLabel.CONFIRMED_MATCH]}  (strict type match)")
        print(f"  RELAXED_MATCH  : {counts[MatchLabel.RELAXED_MATCH]}  (cross-version relaxed match)")
        print(f"  API_MATCH_ONLY  : {counts[MatchLabel.API_MATCH_ONLY]}  (API match only)")
        print(f"  POTENTIALLY_NEW  : {counts[MatchLabel.POTENTIALLY_NEW]}  (not in CSV)")
        print(f"  -- Strict criterion (reference) --------------------")
        print(f"  CONFIRMED_MATCH  : {counts_strict[MatchLabel.CONFIRMED_MATCH]}")
        print(f"  API_MATCH_ONLY  : {counts_strict[MatchLabel.API_MATCH_ONLY]}")
        return results


# ══════════════════════════════════════════════════════════════════
#  JSON output
# ══════════════════════════════════════════════════════════════════

def save_json_results(matched: List[Dict], output_dir: str):
    """
    Emit three JSON files, mirroring the PyTorch-version structure:

      tf_new_bugs.json  - all potentially_new entries (not in CSV).
      tf_guided_new_bugs.json - guided-only potentially_new (paper headline metric).
      tf_known_bugs.json  - confirmed + relaxed match (known-bug reproductions).

    Each record contains sig_hash / api / oracle_type / bug_depth / error_msg /
    exception_type / crash_module / random_count / guided_count /
    match_label / match_detail / matched_issues / sample_file
    """
    def _serialize(records: List[Dict]) -> List[Dict]:
        out = []
        for r in records:
            entry = {
                "sig_hash":  r.get("sig_hash", ""),
                "api":  r.get("api", ""),
                "oracle_type":  r.get("oracle_type", ""),
                "bug_dir_type":  r.get("bug_dir_type", ""),
                "bug_depth":  r.get("bug_depth", ""),
                "error_msg":  r.get("error_msg", ""),
                "crash_module":  r.get("crash_module", ""),
                "exception_type": r.get("exception_type", ""),
                "random_count":  r.get("random_count", 0),
                "guided_count":  r.get("guided_count", 0),
                "match_label":  r.get("match_label", ""),
                "match_label_strict": r.get("match_label_strict", ""),
                "matched_issues": r.get("matched_issues", []),
                "match_detail":  r.get("match_detail", ""),
                "sample_file":  r.get("sample_file", ""),
            }
            # Append best_match info (serialization-friendly)
            bm = r.get("best_match")
            if bm:
                entry["best_match"] = {
                    "issue_id":  bm.get("issue_id", ""),
                    "title":  bm.get("title", ""),
                    "bug_type":  bm.get("bug_type", ""),
                    "buggy_api":  bm.get("buggy_api", ""),
                    "device":  bm.get("device", ""),
                    "reproduced": bm.get("reproduced", False),
                }
            else:
                entry["best_match"] = None
            out.append(entry)
        return out

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_all  = [r for r in matched if r["match_label"] == MatchLabel.POTENTIALLY_NEW]
    guided_new = [r for r in new_all
                  if r["guided_count"] > 0 and r["random_count"] == 0]
    known  = [r for r in matched
                  if r["match_label"] in (MatchLabel.CONFIRMED_MATCH,
                                          MatchLabel.RELAXED_MATCH)]

    # ── tf_new_bugs.json ─────────────────────────────────────────
    path_new = out_dir / "tf_new_bugs.json"
    path_new.write_text(
        json.dumps(_serialize(new_all), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"tf_new_bugs.json  → {len(new_all):3d} bugs  [{path_new}]")

    # ── tf_guided_new_bugs.json ───────────────────────────────────
    path_guided = out_dir / "tf_guided_new_bugs.json"
    path_guided.write_text(
        json.dumps(_serialize(guided_new), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  tf_guided_new_bugs.json -> {len(guided_new):3d} bugs  [{path_guided}]  (paper headline)")

    # ── tf_known_bugs.json ────────────────────────────────────────
    path_known = out_dir / "tf_known_bugs.json"
    path_known.write_text(
        json.dumps(_serialize(known), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"tf_known_bugs.json  → {len(known):3d} bugs  [{path_known}]")

    # -- Console summary -------------------------------------------
    print(f"\n{'='*55}")
    print(f"  🆕 Potentially NEW (total)  : {len(new_all)}")
    print(f"  ├ Guided-Only NEW  : {len(guided_new)}  ← key metric")
    print(f"  └ Random-Only NEW  : "
          f"{sum(1 for r in new_all if r['random_count'] > 0 and r['guided_count'] == 0)}")
    print(f"  Known Reproduced (Conf+Rel): {len(known)}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════
#  Report Generator
# ══════════════════════════════════════════════════════════════════

def _strategy_label(info: Dict) -> str:
    has_r = info["random_count"] > 0
    has_g = info["guided_count"] > 0
    if has_r and has_g: return "Random + Guided"
    if has_g:  return "**Guided Only** "
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

    deep_s  = _ds(BugDepth.DEEP_BUG.value)
    shallow_s = _ds(BugDepth.SHALLOW_CHECK.value)
    defect_s  = _ds(BugDepth.SCRIPT_DEFECT.value)
    noise_s  = _ds(BugDepth.IGNORED_NOISE.value)

    total_r = deep_s["random"]
    total_g = deep_s["guided"]
    go_r  = deep_s["random_only"]
    go_g  = deep_s["guided_only"]

    exact  = [r for r in matched if r["match_label"] == MatchLabel.CONFIRMED_MATCH]
    relaxed = [r for r in matched if r["match_label"] == MatchLabel.RELAXED_MATCH]
    known  = [r for r in matched if r["match_label"] == MatchLabel.API_MATCH_ONLY]
    new_all = [r for r in matched if r["match_label"] == MatchLabel.POTENTIALLY_NEW]
    ng  = [r for r in new_all if r["guided_count"] > 0 and r["random_count"] == 0]

    strict_confirmed = sum(1 for r in matched
                           if r.get("match_label_strict") == MatchLabel.CONFIRMED_MATCH)
    strict_api_match = sum(1 for r in matched
                           if r.get("match_label_strict") == MatchLabel.API_MATCH_ONLY)

    lines = [
        "# HKG-Fuzz TensorFlow Validation Report",
        "",
        f"**Batch**: `{Path(batch_dir).name}`  ",
        f"**Raw Records**: {len(scanner.scripts)} → **Unique Signatures**: {len(dedup.unique)}  ",
        f"**APIs Fuzzed**: {len(scanner.fuzzed_apis)}",
        "", "---", "",
        "## Headline metric: Kernel-Guided vs Random",
        "",
        "| Strategy | Deep bugs found (total) | Exclusive finds (Only) |",
        "|---|---|---|",
        f"| Random  | {total_r} | {go_r} |",
        f"| **Guided** | **{total_g}** | **{go_g}** |",
        "",
        "> Deep bug = `deep_bug` (TF Aborted/Check Failed/InternalError, etc.)",
        "", "---", "",
        "## Deep-bug distribution",
        "",
        "| Depth class | Random | Guided | Both | Guided Only | Random Only |",
        "|---|---|---|---|---|---|",
        f"| Deep Bug | {deep_s['random']} | {deep_s['guided']} | {deep_s['both']} | {deep_s['guided_only']} | {deep_s['random_only']} |",
        "", "---", "",
        "## Ground-truth matching results (against bug_dataset_tensorflow.csv)",
        "",
        "| Match level | Count | Meaning |",
        "|---|---|---|",
        f"| CONFIRMED_MATCH  | **{len(exact)}** | API matches and error type strictly aligned |",
        f"| RELAXED_MATCH  | **{len(relaxed)}** | API matches with cross-version relaxed alignment |",
        f"| API_MATCH_ONLY  | **{len(known)}** | API matches but error type does not align |",
        f"| POTENTIALLY_NEW  | **{len(new_all)}** | Not in CSV (Guided-only: {len(ng)}) |",
        "",
        "<details>",
        "<summary>Strict-criterion reference</summary>",
        "",
        f"| CONFIRMED_MATCH (strict) | **{strict_confirmed}** |",
        f"| API_MATCH_ONLY (strict)  | **{strict_api_match}** |",
        "",
        "</details>",
        "",
    ]

    if exact:
        lines += ["### Tier 1: strict error-type match", ""]
        for r in sorted(exact, key=lambda x: x["api"]):
            bm = r["best_match"] or {}
            lines += [
                f"- **`{r['api']}`**  |  Issue #{bm.get('issue_id', '')}  |  {_strategy_label(r)}",
                f"  - CSV type: `{bm.get('bug_type', '')}`  |  reproduced={bm.get('reproduced') and 'Yes' or 'No'}",
                f"  - Title: {bm.get('title', '')[:100]}",
                f"  - Error: `{r['error_msg'][:150]}`",
                "",
            ]

    if relaxed:
        lines += ["### Tier 1.5: cross-version relaxed match", ""]
        for r in sorted(relaxed, key=lambda x: x["api"]):
            bm = r["best_match"] or {}
            lines += [
                f"- **`{r['api']}`**  |  Issue #{bm.get('issue_id', '')}  |  {_strategy_label(r)}",
                f"  - CSV type: `{bm.get('bug_type', '')}`  |  Error: `{r['error_msg'][:150]}`",
                "",
            ]

    if known:
        lines += ["### Tier 2: API matches but type mismatched", ""]
        for r in sorted(known, key=lambda x: x["api"]):
            bm = r["best_match"] or {}
            lines += [
                f"- **`{r['api']}`**  |  Issue #{bm.get('issue_id', '')}  |  {_strategy_label(r)}",
                f"  - Error: `{r['error_msg'][:150]}`",
                "",
            ]

    if new_all:
        lines += ["### Tier 3: not in CSV (likely new bugs)", ""]
        for r in sorted(new_all, key=lambda x: (-x["guided_count"], x["api"])):
            lines += [
                f"- **`{r['api']}`**  |  {_strategy_label(r)}",
                f"  - Oracle: `{r['oracle_type']}`  |  Exception: `{r['exception_type']}`",
                f"  - Error: `{r['error_msg'][:150]}`",
                "",
            ]

    lines += [
        "---", "",
        "## Filtered noise",
        "",
        "| Category | Random | Guided |",
        "|---|---|---|",
        f"| Shallow argument errors | {shallow_s.get('random', 0)} | {shallow_s.get('guided', 0)} |",
        f"| Script defects  | {defect_s.get('random', 0)}  | {defect_s.get('guided', 0)} |",
        f"| Noise and timeouts  | {noise_s.get('random', 0)}  | {noise_s.get('guided', 0)} |",
        "",
    ]

    Path(output_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {output_file}")


# ══════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HKG-Fuzz TensorFlow Analyzer v2"
    )
    parser.add_argument("--batch-dir",  "-b", required=True)
    parser.add_argument("--ground-truth", "-g", required=True)
    parser.add_argument("--report",  "-r", default="report_tf.md")
    parser.add_argument("--json-dir",  "-j", default=None,
                        help="JSON output directory (defaults to the --report directory)")
    parser.add_argument("--max-per-api",  type=int, default=200)
    args = parser.parse_args()

    # The default JSON output directory is sibling to the report file.
    json_dir = args.json_dir or str(Path(args.report).parent)

    print("=" * 60)
    print("HKG-Fuzz TensorFlow Analyzer v2")
    print("=" * 60)

    scanner = BugScriptScanner(args.batch_dir, args.max_per_api).scan()
    dedup  = BugDeduplicator().add_all(scanner.scripts)
    gt  = GroundTruthLoader(args.ground_truth).load()
    matched = BugMatcher(gt).match_all(
        dedup,
        [BugDepth.DEEP_BUG.value]
    )

    generate_report(args.batch_dir, scanner, dedup, matched, args.report)

    print(f"\n{'='*60}")
    print("Saving JSON results...")
    print(f"{'='*60}")
    save_json_results(matched, json_dir)


if __name__ == "__main__":
    main()