#!/usr/bin/env python3
"""P1.0 Retrospective Enrichment Analysis
==========================================

Replays ~300 programs (bug .py + corpus non-bug) from historical runs with a
monkey-patched PatternMatcherPass to extract which inductor passes fire for
each program. Then computes enrichment ratios to find pass pairs (or single
passes) that are statistically associated with bug-producing programs.

This is the **gating experiment** for P1: if enrichment ≥ 3× is found for ≥3
pass pairs, the constraint-guided generator (P1.1) has concrete anchor targets
and P1 proceeds. Otherwise, the search space design needs rethinking.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python retrospective_enrichment.py

Output:
    ${ARTIFACT_ROOT}/graph_exp/retrospective_p1_data.json  (raw per-program data)
    ${ARTIFACT_ROOT}/graph_exp/retrospective_p1_report.md  (analysis report)
"""

import gc
import glob
import hashlib
import json
import os
import re
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from itertools import combinations

import torch
import torch._dynamo
import torch._inductor.config as ic

# ---------------------------------------------------------------------------
# 1. Monkey-patch PatternMatcherPass to capture fired passes
# ---------------------------------------------------------------------------
from torch._inductor.pattern_matcher import PatternMatcherPass as PMP

_orig_pmp_apply = PMP.apply
_fired_passes_global = []
_pattern_trace_ok = False

try:
    from torch._dynamo.utils import counters as _dyn_counters

    def _traced_pmp_apply(_self, _gm):
        cnt_before = _dyn_counters.get("inductor", {}).get("pattern_matcher_count", 0)
        result = _orig_pmp_apply(_self, _gm)
        cnt_after = _dyn_counters.get("inductor", {}).get("pattern_matcher_count", 0)
        fired = cnt_after - cnt_before
        pid = _self.pass_name or f"pass_k{len(_self.patterns)}"
        if fired > 0:
            _fired_passes_global.append((pid, fired))
        return result

    PMP.apply = _traced_pmp_apply
    _pattern_trace_ok = True
except Exception as e:
    print(f"[WARN] Cannot monkey-patch PatternMatcherPass: {e}")

# Disable caches so pattern matcher always runs fresh
try:
    ic.fx_graph_cache = False
    ic.fx_graph_remote_cache = False
except AttributeError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 2. Timeout helper (alarm-based, Unix only)
# ---------------------------------------------------------------------------
class CompileTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise CompileTimeout("compile timed out")


COMPILE_TIMEOUT_SEC = 60


# ---------------------------------------------------------------------------
# 3. Discover programs from all runs
# ---------------------------------------------------------------------------
RUNS = {
    "run7_v211_p0": "${ARTIFACT_ROOT}/graph_exp_v211_p0_run",
    "run6_v211_0411": "${ARTIFACT_ROOT}/graph_exp_v211_0411",
    "run_v211": "${ARTIFACT_ROOT}/graph_exp_v211",
    "run_v27": "${ARTIFACT_ROOT}/graph_exp_v27",
    "run_phase3": "${ARTIFACT_ROOT}/graph_exp_phase3",
}


def _parse_bug_py_metadata(filepath):
    """Extract metadata from a bug .py file's header comments."""
    meta = {"filepath": filepath, "is_bug": True}
    try:
        with open(filepath) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if "config_preset=" in line:
                    m = re.search(r"config_preset=(\S+)", line)
                    if m:
                        meta["config_preset"] = m.group(1)
                if "type=" in line:
                    m = re.search(r"type=(\S+)", line)
                    if m:
                        meta["bug_type"] = m.group(1)
                if "program:" in line:
                    m = re.search(r"program:\s*(\S+)", line)
                    if m:
                        meta["program_name"] = m.group(1)
        with open(filepath) as f:
            meta["body_code"] = f.read()
    except Exception as e:
        meta["error"] = str(e)
    return meta


def discover_programs():
    """Discover all programs from historical runs. Dedup by body_code hash."""
    programs = []  # list of {body_code, is_bug, source_run, source_file, ...}
    seen_hashes = set()

    for run_label, base_dir in RUNS.items():
        if not os.path.isdir(base_dir):
            continue

        # Bug programs (.py files in potential-bug/)
        bug_dir = os.path.join(base_dir, "graph/graph-oracle/potential-bug")
        if os.path.isdir(bug_dir):
            for pyf in sorted(glob.glob(os.path.join(bug_dir, "**/*.py"), recursive=True)):
                meta = _parse_bug_py_metadata(pyf)
                if "body_code" not in meta:
                    continue
                h = hashlib.md5(meta["body_code"].encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                meta["source_run"] = run_label
                meta["source_file"] = pyf
                meta["hash"] = h
                programs.append(meta)

        # Corpus programs (non-bug seeds from corpus_checkpoint.json)
        cp = os.path.join(base_dir, "graph/graph-oracle/corpus_checkpoint.json")
        if os.path.exists(cp):
            try:
                data = json.load(open(cp))
                for entry in data.get("entries", []):
                    body = entry.get("body_code", "")
                    if not body.strip():
                        continue
                    # Reconstruct full body with imports
                    full_body = "import torch\n\n" + body if "import torch" not in body else body
                    h = hashlib.md5(full_body.encode()).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    programs.append({
                        "body_code": full_body,
                        "is_bug": False,
                        "source_run": run_label,
                        "source_file": cp,
                        "program_name": entry.get("name", "corpus"),
                        "config_preset": entry.get("config_preset", "default"),
                        "hash": h,
                    })
            except Exception as e:
                print(f"  [WARN] cannot read corpus: {cp}: {e}")

    return programs


# ---------------------------------------------------------------------------
# 4. Compile a single program and extract fired_passes
# ---------------------------------------------------------------------------
def compile_and_extract(program_info):
    """Compile one program with torch.compile(backend='inductor') and return
    the set of fired passes, or None on failure."""
    body = program_info.get("body_code", "")
    if not body.strip():
        return None, "empty_body"

    _fired_passes_global.clear()
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reset inductor config to defaults for fair comparison
    _reset_ic_defaults()

    ns = {"torch": torch}
    try:
        exec(body, ns)
    except Exception as e:
        return None, f"exec_fail:{type(e).__name__}"

    program_fn = ns.get("program")
    inputs = ns.get("_program_inputs")
    if program_fn is None or inputs is None:
        return None, "no_program_or_inputs"

    # Move to CUDA
    try:
        inputs = tuple(
            t.cuda() if isinstance(t, torch.Tensor) else t for t in inputs
        )
    except Exception as e:
        return None, f"cuda_move_fail:{type(e).__name__}"

    # Compile + run (triggers compilation which fires pattern matcher)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(COMPILE_TIMEOUT_SEC)
    try:
        torch._dynamo.reset()
        fn = torch.compile(program_fn, backend="inductor")
        _ = fn(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except CompileTimeout:
        return None, "compile_timeout"
    except Exception as e:
        # Compilation or runtime error — we may still have partial fired_passes
        # from the lowering phase before the crash
        err_type = type(e).__name__
        if not _fired_passes_global:
            return None, f"compile_fail:{err_type}"
        # Partial: we got some passes before the crash
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        torch._dynamo.reset()

    if not _fired_passes_global:
        return set(), "compiled_but_no_passes"

    # Return frozenset of (pass_name, fire_count) tuples
    return frozenset(_fired_passes_global), "ok"


def _reset_ic_defaults():
    """Best-effort reset of inductor config to defaults."""
    defaults = {
        "epilogue_fusion": True,
        "split_reductions": True,
        "pattern_matcher": True,
        "freezing": False,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
        "combo_kernels": False,
        "combo_kernels_pointwise_only": False,
        "deterministic": False,
        "max_fusion_buffer_group_pairwise_attempts": 64,
        "min_num_split": 1,
    }
    for k, v in defaults.items():
        try:
            setattr(ic, k, v)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# 5. Enrichment computation
# ---------------------------------------------------------------------------
def compute_enrichment(programs_with_passes):
    """Compute per-pass and per-pair enrichment ratios."""
    bug_programs = [p for p in programs_with_passes if p["is_bug"] and p["passes"] is not None]
    non_programs = [p for p in programs_with_passes if not p["is_bug"] and p["passes"] is not None]

    if not bug_programs or not non_programs:
        return None, "insufficient_data"

    # Extract pass NAME sets (ignore fire counts for enrichment)
    def pass_names(passes_frozenset):
        return frozenset(name for name, _ in passes_frozenset)

    bug_pass_sets = [pass_names(p["passes"]) for p in bug_programs]
    non_pass_sets = [pass_names(p["passes"]) for p in non_programs]
    n_bug = len(bug_pass_sets)
    n_non = len(non_pass_sets)

    # All unique pass names
    all_passes = set()
    for ps in bug_pass_sets + non_pass_sets:
        all_passes.update(ps)

    # --- Single-pass enrichment ---
    single_enrichment = {}
    for p in sorted(all_passes):
        p_bug = sum(1 for ps in bug_pass_sets if p in ps) / n_bug
        p_non = sum(1 for ps in non_pass_sets if p in ps) / n_non
        enrichment = p_bug / p_non if p_non > 0 else (float("inf") if p_bug > 0 else 1.0)
        single_enrichment[p] = {
            "p_bug": p_bug,
            "p_non": p_non,
            "enrichment": enrichment,
            "bug_count": sum(1 for ps in bug_pass_sets if p in ps),
            "non_count": sum(1 for ps in non_pass_sets if p in ps),
        }

    # --- Pair enrichment ---
    # Only consider passes that appear in ≥2 programs total
    frequent_passes = [p for p in all_passes
                       if single_enrichment[p]["bug_count"] + single_enrichment[p]["non_count"] >= 3]

    pair_enrichment = {}
    for a, b in combinations(sorted(frequent_passes), 2):
        pair = (a, b)
        p_bug = sum(1 for ps in bug_pass_sets if a in ps and b in ps) / n_bug
        p_non = sum(1 for ps in non_pass_sets if a in ps and b in ps) / n_non
        enrichment = p_bug / p_non if p_non > 0 else (float("inf") if p_bug > 0 else 1.0)
        if p_bug > 0:  # only record pairs that appear in at least 1 bug
            pair_enrichment[pair] = {
                "p_bug": p_bug,
                "p_non": p_non,
                "enrichment": enrichment,
                "bug_count": sum(1 for ps in bug_pass_sets if a in ps and b in ps),
                "non_count": sum(1 for ps in non_pass_sets if a in ps and b in ps),
            }

    return {
        "n_bug": n_bug,
        "n_non": n_non,
        "n_passes": len(all_passes),
        "all_passes": sorted(all_passes),
        "single_enrichment": single_enrichment,
        "pair_enrichment": {str(k): v for k, v in pair_enrichment.items()},
        "top_single": sorted(single_enrichment.items(),
                             key=lambda x: x[1]["enrichment"], reverse=True)[:15],
        "top_pairs": sorted(pair_enrichment.items(),
                            key=lambda x: x[1]["enrichment"], reverse=True)[:20],
    }, "ok"


# ---------------------------------------------------------------------------
# 6. Report generation
# ---------------------------------------------------------------------------
def generate_report(enrichment_data, programs_with_passes, elapsed_sec):
    """Generate the retrospective report markdown."""
    lines = [
        "# P1.0 Retrospective Enrichment Report",
        "",
        f"> Generated {time.strftime('%Y-%m-%d %H:%M')} | Elapsed: {elapsed_sec:.0f}s",
        f"> Hypothesis: specific pass pairs co-fire in bug programs more than non-bug programs",
        "",
        "## Summary",
        "",
    ]

    if enrichment_data is None:
        lines.append("**INSUFFICIENT DATA** — cannot compute enrichment.")
        return "\n".join(lines)

    n_bug = enrichment_data["n_bug"]
    n_non = enrichment_data["n_non"]
    n_passes = enrichment_data["n_passes"]

    total_programs = len(programs_with_passes)
    n_ok = sum(1 for p in programs_with_passes if p.get("passes") is not None)
    n_fail = total_programs - n_ok

    lines.extend([
        f"- **Programs discovered**: {total_programs} ({n_bug + n_non} compiled successfully, {n_fail} failed)",
        f"- **Bug programs**: {n_bug}",
        f"- **Non-bug programs**: {n_non}",
        f"- **Unique passes observed**: {n_passes}",
        "",
    ])

    # Gating decision
    top_pairs = enrichment_data["top_pairs"]
    high_enrichment_pairs = [(k, v) for k, v in top_pairs if v["enrichment"] >= 3.0 and v["bug_count"] >= 2]

    if len(high_enrichment_pairs) >= 3:
        lines.extend([
            "### GATING DECISION: **PASS** ",
            "",
            f"Found **{len(high_enrichment_pairs)} pass pairs** with enrichment ≥ 3× and ≥2 bug hits.",
            "P1.1 constraint-guided generator has concrete anchor targets. Proceed to P1.1.",
            "",
        ])
    elif len(high_enrichment_pairs) >= 1:
        lines.extend([
            "### GATING DECISION: **WEAK PASS** ",
            "",
            f"Found **{len(high_enrichment_pairs)} pass pairs** with enrichment ≥ 3× and ≥2 bug hits.",
            "Signal exists but is weak. Consider expanding the dataset or relaxing criteria before P1.1.",
            "",
        ])
    else:
        lines.extend([
            "### GATING DECISION: **FAIL** ",
            "",
            "No pass pairs found with enrichment ≥ 3× and ≥2 bug hits.",
            "The hypothesis 'pass pairs enrich in bug programs' is not supported by this data.",
            "Consider alternative search space designs: pass × dtype, single rare pass × shape, etc.",
            "",
        ])

    # Top single passes
    lines.extend([
        "## Top Enriched Single Passes",
        "",
        "| Pass | P(fire|bug) | P(fire|non) | Enrichment | bug_N | non_N |",
        "|---|---|---|---|---|---|",
    ])
    for name, data in enrichment_data["top_single"]:
        e = data["enrichment"]
        e_str = f"{e:.2f}" if e < 100 else "∞" if e == float("inf") else f"{e:.0f}"
        lines.append(
            f"| `{name}` | {data['p_bug']:.3f} | {data['p_non']:.3f} | "
            f"**{e_str}** | {data['bug_count']} | {data['non_count']} |"
        )

    # Top pairs
    lines.extend([
        "",
        "## Top Enriched Pass Pairs",
        "",
        "| Pass A | Pass B | P(A∧B|bug) | P(A∧B|non) | Enrichment | bug_N | non_N |",
        "|---|---|---|---|---|---|---|",
    ])
    for pair, data in top_pairs[:20]:
        e = data["enrichment"]
        e_str = f"{e:.2f}" if e < 100 else "∞" if e == float("inf") else f"{e:.0f}"
        a, b = pair if isinstance(pair, tuple) else eval(pair)
        lines.append(
            f"| `{a}` | `{b}` | {data['p_bug']:.3f} | {data['p_non']:.3f} | "
            f"**{e_str}** | {data['bug_count']} | {data['non_count']} |"
        )

    # All passes frequency
    lines.extend([
        "",
        "## All Passes (frequency)",
        "",
        "| Pass | total fire count (bug+non) |",
        "|---|---|",
    ])
    for name in enrichment_data["all_passes"]:
        s = enrichment_data["single_enrichment"][name]
        total = s["bug_count"] + s["non_count"]
        lines.append(f"| `{name}` | {total} |")

    # Failure summary
    failed_programs = [p for p in programs_with_passes if p.get("passes") is None]
    if failed_programs:
        lines.extend([
            "",
            "## Compile Failures",
            "",
            f"**{len(failed_programs)}** programs failed to compile (skipped from analysis):",
            "",
        ])
        fail_reasons = Counter(p.get("status", "unknown") for p in failed_programs)
        for reason, count in fail_reasons.most_common(10):
            lines.append(f"- `{reason}`: {count}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("P1.0 Retrospective Enrichment Analysis")
    print("=" * 70)
    print(f"torch={torch.__version__} device={DEVICE}")
    print(f"pattern_trace_ok={_pattern_trace_ok}")
    print()

    # Discover
    t0 = time.time()
    programs = discover_programs()
    n_bug = sum(1 for p in programs if p["is_bug"])
    n_non = sum(1 for p in programs if not p["is_bug"])
    print(f"Discovered {len(programs)} unique programs: {n_bug} bug + {n_non} non-bug")
    print()

    # Compile and extract
    results = []
    for i, prog in enumerate(programs):
        label = "BUG" if prog["is_bug"] else "corpus"
        name = prog.get("program_name", "?")[:40]
        run = prog.get("source_run", "?")

        passes, status = compile_and_extract(prog)
        prog["passes"] = passes
        prog["status"] = status
        n_passes = len(passes) if passes else 0

        if (i + 1) % 20 == 0 or status != "ok":
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i+1:3d}/{len(programs)}] {label:6s} {name:40s} "
                f"passes={n_passes:2d} status={status:30s} "
                f"({speed:.1f}/s, ~{(len(programs)-i-1)/max(speed,0.01)/60:.0f}m left)"
            )

        results.append({
            "name": name,
            "is_bug": prog["is_bug"],
            "source_run": run,
            "passes": list(passes) if passes else None,
            "status": status,
            "hash": prog.get("hash", ""),
            "bug_type": prog.get("bug_type", ""),
        })

    elapsed = time.time() - t0
    print(f"\nCompiled {len(programs)} programs in {elapsed:.0f}s")

    # Save raw data
    raw_path = "${ARTIFACT_ROOT}/graph_exp/retrospective_p1_data.json"
    with open(raw_path, "w") as f:
        json.dump({"programs": results, "elapsed_sec": elapsed}, f, indent=2, default=str)
    print(f"Saved raw data: {raw_path}")

    # Compute enrichment
    enrichment_data, enrich_status = compute_enrichment(programs)
    print(f"Enrichment status: {enrich_status}")

    # Generate report
    report = generate_report(enrichment_data, programs, elapsed)
    report_path = "${ARTIFACT_ROOT}/graph_exp/retrospective_p1_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report: {report_path}")
    print()

    # Print quick summary
    if enrichment_data:
        print("=== QUICK SUMMARY ===")
        top_pairs = enrichment_data["top_pairs"]
        high = [(k, v) for k, v in top_pairs if v["enrichment"] >= 3.0 and v["bug_count"] >= 2]
        print(f"Pass pairs with enrichment ≥ 3× and ≥2 bug hits: {len(high)}")
        for pair, data in high[:5]:
            print(f"  {pair}: enrichment={data['enrichment']:.2f} bug_N={data['bug_count']} non_N={data['non_count']}")

    # Restore PatternMatcherPass
    PMP.apply = _orig_pmp_apply


if __name__ == "__main__":
    main()
