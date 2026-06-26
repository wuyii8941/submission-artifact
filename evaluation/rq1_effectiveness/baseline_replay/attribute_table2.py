#!/usr/bin/env python3
"""
Attribute 4-hour replay outputs to 30 PyTorch/Inductor root causes.

Reads:
  - DeepOpFuzz trigger files     under <runs_root>/deepopfuzz/graph-oracle/{potential-bug,crash}/<trigger>/*.py
  - Per-baseline trigger files   under <runs_root>/<baseline>_ckpt/triggers/<baseline>_seed42/*.py
    for baseline in {random, freefuzz, nnsmith, whitefox}.

Writes:
  - results/table2_attribution.json   per-run + union breakdown
  - stdout                            human-readable summary table

Deduplication: distinct root causes hit per run (collapsing bug variants that
share the same root cause); raw trigger counts are reported separately as a
diagnostic, not as the headline metric.

Usage:
  python attribute_table2.py --runs-root /path/to/4h_runs [--output table2.json]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set

# Self-contained: bug_classification is a sibling module in this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bug_classification import BUGS, DECISION_REGIONS  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Op detection patterns (source-level regex)
# ════════════════════════════════════════════════════════════════════════════

OP_PATTERNS: Dict[str, List[str]] = {
    "matmul":            [r"torch\.matmul\b", r"torch\.mm\b", r"torch\.bmm\b",
                          r"F\.linear\b", r"nn\.Linear\b"],
    "layer_norm":        [r"layer_norm", r"LayerNorm"],
    "expand":            [r"\.expand\(", r"\.expand_as\("],
    "abs":               [r"torch\.abs\b", r"\.abs\(\)"],
    "fractional_max_pool": [r"[Ff]ractional[_]?[Mm]ax[_]?[Pp]ool"],
    "div":               [r"torch\.div\b", r"\.div\(", r"torch\.floor_divide"],
    "rounding_mode":     [r"rounding_mode\s*="],
    "foreach":           [r"torch\._foreach_", r"\._foreach_"],
    "alpha":             [r"alpha\s*=\s*[^0)\s]"],
    "cumsum":            [r"\.cumsum\(", r"torch\.cumsum\b"],
    "cumprod":           [r"\.cumprod\(", r"torch\.cumprod\b"],
    "adaptive_avg_pool": [r"[Aa]daptive[_]?[Aa]vg[_]?[Pp]ool"],
    "flatten":           [r"\.flatten\(", r"torch\.flatten\b"],
    "sum":               [r"\.sum\(", r"torch\.sum\b"],
    "addcmul":           [r"addcmul", r"addcdiv"],
    "inplace":           [r"inplace\s*=\s*True"],
    "randperm":          [r"torch\.randperm\b"],
    "index":             [r"\.index_select\(", r"\.index_put", r"\.index_add",
                          r"\.index_copy"],
    "slice_scatter":     [r"slice_scatter"],
    "diagonal_scatter":  [r"diagonal_scatter"],
    "index_put":         [r"index_put"],
    "flip":              [r"torch\.flip\b", r"\.flip\("],
    "cat":               [r"torch\.cat\("],
    "while_loop":        [r"while_loop"],
    "associative_scan":  [r"associative_scan"],
    "backward":          [r"requires_grad\s*[=_(]", r"\.backward\("],
    "create_graph":      [r"create_graph\s*=\s*True"],
    "inference_mode":    [r"inference_mode"],
    "nested_tensor":     [r"nested_tensor", r"torch\.nested"],
    "deterministic":     [r"use_deterministic_algorithms"],
    "dynamic":           [r"dynamic\s*=\s*True", r"mark_dynamic"],
    "complex":           [r"complex64", r"complex128", r"dtype\s*=\s*torch\.complex"],
    "float8":            [r"float8_e[45]m[23]"],
    "bf16":              [r"bfloat16"],
    "0d_tensor":         [r"torch\.tensor\(\s*[\d\.\-]", r"torch\.scalar_tensor"],
}

# Kernel-name fragments that imply specific ops (from `kernel_templates:` comments).
KERNEL_OP_MAP: Dict[str, List[str]] = {
    "cumsum":             ["cumsum"],
    "cumprod":            ["cumprod"],
    "slice_scatter":      ["slice_scatter"],
    "diagonal_scatter":   ["diagonal_scatter"],
    "index_put":          ["index_put"],
    "flip":               ["flip"],
    "cat":                ["cat"],
    "abs":                ["abs"],
    "sum":                ["sum"],
    "flatten":            ["flatten"],
    "addcmul":            ["addcmul", "addcdiv"],
    "adaptive_avg_pool":  ["adaptive_avg_pool", "adaptive_pool"],
    "expand":             ["expand"],
    "div":                ["div"],
    "matmul":             ["mm", "bmm", "addmm"],
}


def extract_features(code: str) -> Set[str]:
    """Extract structural features from program code via regex."""
    features: Set[str] = set()

    # 1. Op-pattern regexes against source code
    for feat, patterns in OP_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, code):
                features.add(feat)
                break

    # 2. matmul via Python `@` operator (not in OP_PATTERNS)
    if re.search(r"[a-zA-Z0-9_\)\]]\s*@\s*[a-zA-Z_]", code):
        features.add("matmul")

    # 3. Large dimension (>= 2048) in tensor constructors → split-scan path
    for m in re.finditer(
        r"(?:torch\.randn|torch\.zeros|torch\.ones|torch\.empty|"
        r"torch\.rand|torch\.full)\s*\(([\d,\s]+)", code
    ):
        dims = [int(d.strip()) for d in m.group(1).split(",") if d.strip().isdigit()]
        if any(d > 2048 for d in dims):
            features.add("large_dim")

    # 4. Ops inferred from kernel_templates comment (per trigger file convention)
    for m in re.finditer(r"kernel_templates:\s*\[(.*?)\]", code, re.DOTALL):
        templates_str = m.group(1)
        for feat, kops in KERNEL_OP_MAP.items():
            if any(kop in templates_str for kop in kops):
                features.add(feat)

    return features


# ════════════════════════════════════════════════════════════════════════════
# Root-cause attribution
# ════════════════════════════════════════════════════════════════════════════

# Override structural_reqs for bugs whose original taxonomy used
# Inductor-internal flags that cannot appear in user source code (e.g. split_scan).
# These overrides keep attribution honest: only conditions a test program can
# control are required.
OVERRIDE_REQS: Dict[str, List[str]] = {
    # bug_011 (#180221) cumsum + dim>=129 → Inductor picks SplitScan internally
    "bug_011":    ["cumsum", "large_dim"],
    # bug_031 (#183250) cumprod backward + dim>=8193 → Inductor picks SplitDim
    "bug_031":    ["cumprod", "backward", "large_dim"],
    # bug_051 (#184222) scheduler prologue typo → code-level bug, not user-triggerable
    "bug_051":    [],
    # bug_059 (#184416) pick_deterministic_choice → needs determinism + extern dispatch
    "bug_059":    ["deterministic", "matmul"],
    # paper-appendix-only bugs with thin structural specs (avoid over-counting)
    "bug_182200": [],
    "bug_183015": [],
}

# Collapse duplicate root-cause IDs (paper's 30-distinct count merges variants).
BUG_ID_COLLAPSE: Dict[str, str] = {
    "bug_013v": "bug_013",
}

REGION_TO_COL = {
    "Frontend/context":    "FE/ctx",
    "AOT/backward":        "AOT/bwd",
    "Lowering/IR":         "Low/IR",
    "Fusion":              "Fusion",
    "Scheduling/template": "Sched.",
    "Triton codegen":      "Triton",
}
COL_ORDER = ["FE/ctx", "AOT/bwd", "Low/IR", "Fusion", "Sched.", "Triton"]
COL_DENOM = {"FE/ctx": 3, "AOT/bwd": 5, "Low/IR": 6, "Fusion": 3, "Sched.": 4, "Triton": 9}


def attribute_program(code: str) -> List[str]:
    """Return list of bug_ids whose required features are all present in `code`."""
    features = extract_features(code)
    hits: List[str] = []
    for bug in BUGS:
        reqs = set(OVERRIDE_REQS.get(bug.bug_id, bug.structural_reqs))
        if reqs and reqs.issubset(features):
            hits.append(bug.bug_id)
    return hits


# ════════════════════════════════════════════════════════════════════════════
# Input collection (two on-disk layouts)
# ════════════════════════════════════════════════════════════════════════════

def collect_baseline_triggers(triggers_dir: str) -> List[tuple]:
    """Return [(filename, code)] for one baseline's saved trigger programs."""
    out = []
    if not os.path.isdir(triggers_dir):
        return out
    for py in sorted(glob.glob(os.path.join(triggers_dir, "trigger_*.py"))):
        try:
            with open(py) as f:
                out.append((os.path.basename(py), f.read()))
        except Exception:
            pass
    return out


def collect_deepopfuzz_findings(graph_oracle_dir: str) -> List[tuple]:
    """Return [(relname, code)] for DeepOpFuzz potential-bug + crash findings."""
    out = []
    for sub in ("potential-bug", "crash"):
        d = os.path.join(graph_oracle_dir, sub)
        if not os.path.isdir(d):
            continue
        for py in sorted(glob.glob(os.path.join(d, "*", "*.py"))):
            try:
                with open(py) as f:
                    out.append((os.path.relpath(py, graph_oracle_dir), f.read()))
            except Exception:
                pass
    return out


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-root", required=True,
                    help="Directory containing deepopfuzz/ and <baseline>_ckpt/ trees")
    ap.add_argument("--output",
                    default=os.path.join(os.path.dirname(__file__), "results", "table2_attribution.json"),
                    help="Output JSON file (default: results/table2_attribution.json under script dir)")
    args = ap.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    runs = [
        ("DeepOpFuzz", collect_deepopfuzz_findings(
            os.path.join(runs_root, "deepopfuzz", "graph-oracle"))),
        ("Random",   collect_baseline_triggers(
            os.path.join(runs_root, "random_ckpt", "triggers", "random_seed42"))),
        ("FreeFuzz", collect_baseline_triggers(
            os.path.join(runs_root, "freefuzz_ckpt", "triggers", "freefuzz_seed42"))),
        ("NNSmith",  collect_baseline_triggers(
            os.path.join(runs_root, "nnsmith_ckpt", "triggers", "nnsmith_seed42"))),
        ("WhiteFox", collect_baseline_triggers(
            os.path.join(runs_root, "whitefox_ckpt", "triggers", "whitefox_seed42"))),
    ]

    per_run_results = {}
    print(f"\n{'='*78}\nAttribution to 30 PyTorch/Inductor root causes\n{'='*78}")

    for label, files in runs:
        bug_to_files: Dict[str, Set[str]] = defaultdict(set)
        for fname, code in files:
            for bug_id in attribute_program(code):
                bug_id = BUG_ID_COLLAPSE.get(bug_id, bug_id)
                bug_to_files[bug_id].add(fname)

        region_bugs: Dict[str, Set[str]] = defaultdict(set)
        for bug in BUGS:
            bid = BUG_ID_COLLAPSE.get(bug.bug_id, bug.bug_id)
            if bug_to_files.get(bid):
                region_bugs[bug.decision_region].add(bid)

        row = {col: len(region_bugs[region]) for region, col in REGION_TO_COL.items()}
        row["Total"]            = sum(row[c] for c in COL_ORDER)
        row["TotalFiles"]       = len(files)
        row["DistinctBugIds"]   = sorted(bug_to_files.keys())

        per_run_results[label] = {
            "row": row,
            "bug_to_files_count": {b: len(s) for b, s in bug_to_files.items()},
        }

        print(f"\n--- {label} ({len(files)} trigger files) ---")
        for col in COL_ORDER:
            print(f"  {col:8s}: {row[col]}/{COL_DENOM[col]}")
        print(f"  {'Total':8s}: {row['Total']}/30")
        print(f"  Distinct root causes hit: {row['DistinctBugIds']}")

    # "Any baseline" union (excluding DeepOpFuzz)
    baseline_union: Set[str] = set()
    for label, _ in runs[1:]:
        baseline_union.update(per_run_results[label]["row"]["DistinctBugIds"])
    union_region: Dict[str, Set[str]] = defaultdict(set)
    for bug in BUGS:
        bid = BUG_ID_COLLAPSE.get(bug.bug_id, bug.bug_id)
        if bid in baseline_union:
            union_region[bug.decision_region].add(bid)
    union_row = {col: len(union_region[region]) for region, col in REGION_TO_COL.items()}
    union_row["Total"] = sum(union_row[c] for c in COL_ORDER)

    print("\n--- Any baseline (union) ---")
    for col in COL_ORDER:
        print(f"  {col:8s}: {union_row[col]}/{COL_DENOM[col]}")
    print(f"  {'Total':8s}: {union_row['Total']}/30")

    # Final summary table
    print(f"\n{'='*78}\nTable II summary (4-hour replay, attributed to 30 root causes)\n{'='*78}")
    header = f"{'Method':14s}" + "".join(f"{c:>9s}" for c in COL_ORDER) + f"{'Total':>9s}"
    print(header)
    print(f"{'Root causes':14s}" + "".join(f"{COL_DENOM[c]:>9d}" for c in COL_ORDER) + f"{'30':>9s}")
    for label, _ in runs:
        r = per_run_results[label]["row"]
        print(f"{label:14s}" + "".join(f"{r[c]:>9d}" for c in COL_ORDER) + f"{r['Total']:>9d}")
    print(f"{'Any baseline':14s}" + "".join(f"{union_row[c]:>9d}" for c in COL_ORDER) + f"{union_row['Total']:>9d}")

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "per_run": per_run_results,
            "any_baseline_union": {
                "row": union_row,
                "distinct_bug_ids": sorted(baseline_union),
            },
        }, f, indent=2, default=lambda o: list(o) if isinstance(o, set) else o)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
