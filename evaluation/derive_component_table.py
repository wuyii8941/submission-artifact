#!/usr/bin/env python3
"""
Derive the BugsInDLLs component ablation table (tab:components) from raw data.

Data sources:
  PyTorch: bugsindlls_bugs_0316/phase_results.json (v9_20260316 batch)
  TensorFlow: tf_validated_bugs/tf_validation_report_matched.json

Output: per-config hit counts for the paper table.
"""

import json
import os
from collections import defaultdict

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PT_PHASE = os.environ.get("PT_PHASE_JSON", os.path.join(_EVAL_DIR, "data", "phase_results.json"))
TF_MATCHED = os.environ.get("TF_MATCHED_JSON", os.path.join(_EVAL_DIR, "data", "tf_validation_report_matched.json"))
TF_CSV_REPRO_YES = 28
PT_CSV_REPRO_YES = 39


def load_pt_issues():
    """Load PT issue-level data from phase_results.json."""
    with open(PT_PHASE) as f:
        data = json.load(f)

    issues = {}
    for r in data:
        if not r.get("matched_issues"):
            continue
        bm = r.get("best_match", {})
        if not bm.get("reproduced"):
            continue

        has_r = r.get("random_count", 0) > 0
        has_g = r.get("guided_count", 0) > 0
        oracle = r.get("oracle_type", "unknown")
        strict = r.get("match_label_strict", "") == "confirmed_match"
        relaxed = r.get("match_label", "") in ("confirmed_match", "relaxed_match")

        for iss in r["matched_issues"]:
            if iss not in issues:
                issues[iss] = {"random": False, "guided": False,
                               "strict": False, "relaxed": False,
                               "oracles": set()}
            if has_r:
                issues[iss]["random"] = True
            if has_g:
                issues[iss]["guided"] = True
            if strict:
                issues[iss]["strict"] = True
            if relaxed:
                issues[iss]["relaxed"] = True
            issues[iss]["oracles"].add(oracle)

    return issues


def load_tf_issues():
    """Load TF issue-level data from validation report."""
    with open(TF_MATCHED) as f:
        data = json.load(f)

    # Map issue_id -> {random, guided, reproduced}
    # TF report has reproduced info in the matched issue data
    # We also need to cross-reference with the CSV for reproduced status
    tf_csv = os.environ.get("TF_CSV", os.path.join(_EVAL_DIR, "data", "bug_dataset_tensorflow.csv"))
    repro_set = set()
    if os.path.exists(tf_csv):
        import csv
        with open(tf_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                repro = row.get("Reproduced", row.get("reproduced", "")).strip().lower()
                if repro == "yes":
                    repro_set.add(row.get("Issue Number", row.get("issue_number", "")).strip("#").strip())

    issues = {}
    for bug in data.get("bugs", []):
        triggered = bug.get("triggered_by", {})
        has_r = triggered.get("random", 0) > 0
        has_g = triggered.get("guided", 0) > 0

        for match in bug.get("tensorflow_matches", []):
            iss = str(match.get("issue_id", ""))
            if not iss:
                continue
            if repro_set and iss not in repro_set:
                continue

            if iss not in issues:
                issues[iss] = {"random": False, "guided": False}
            if has_r:
                issues[iss]["random"] = True
            if has_g:
                issues[iss]["guided"] = True

    return issues


def main():
    print("=" * 70)
    print("BugsInDLLs Component Ablation Table Derivation")
    print("=" * 70)

    # ── PyTorch ──
    pt = load_pt_issues()
    print(f"\nPyTorch: {len(pt)} reproduced=Yes issues matched")

    pt_random = {i for i, v in pt.items() if v["random"]}
    pt_guided = {i for i, v in pt.items() if v["guided"]}
    pt_strict = {i for i, v in pt.items() if v["strict"]}
    pt_relaxed = {i for i, v in pt.items() if v["relaxed"]}
    pt_union = set(pt.keys())

    # Issues by strategy under equivalent matching
    pt_random_eq = {i for i in pt_random if pt[i]["relaxed"]}
    pt_guided_eq = {i for i in pt_guided if pt[i]["relaxed"]}
    pt_both_eq = pt_random_eq & pt_guided_eq
    pt_r_only_eq = pt_random_eq - pt_guided_eq
    pt_g_only_eq = pt_guided_eq - pt_random_eq

    print(f"  Random (equivalent):  {len(pt_random_eq)} issues")
    print(f"  Guided (equivalent):  {len(pt_guided_eq)} issues")
    print(f"  Both (overlap):       {len(pt_both_eq)} issues")
    print(f"  Random-only:          {len(pt_r_only_eq)} → {sorted(pt_r_only_eq)}")
    print(f"  Guided-only:          {len(pt_g_only_eq)} → {sorted(pt_g_only_eq)}")
    print(f"  Strict match total:   {len(pt_strict)}")
    print(f"  Equivalent match:     {len(pt_relaxed)}")

    # ── TensorFlow ──
    tf = load_tf_issues()
    print(f"\nTensorFlow: {len(tf)} reproduced=Yes issues matched")

    tf_random = {i for i, v in tf.items() if v["random"]}
    tf_guided = {i for i, v in tf.items() if v["guided"]}
    tf_union = set(tf.keys())

    print(f"  Random: {len(tf_random)} issues")
    print(f"  Guided: {len(tf_guided)} issues")
    print(f"  Both:   {len(tf_random & tf_guided)} issues")
    print(f"  R-only: {len(tf_random - tf_guided)}")
    print(f"  G-only: {len(tf_guided - tf_random)}")
    print(f"  Union:  {len(tf_union)}")

    # ── Table ──
    print(f"\n{'=' * 70}")
    print("PAPER TABLE: tab:components")
    print("Known-bug reachability sanity check on BugsInDLLs")
    print(f"{'=' * 70}")
    print()

    # Config definitions (consistent equivalent matching):
    # A = Random mutation only (equivalent matching)
    # B = Guided state-aware targeting only (equivalent matching)
    # C = Union of strategies, strict matching
    # D = Union of strategies, equivalent matching

    pt_A = len(pt_random_eq)
    pt_B = len(pt_guided_eq)
    pt_C = len(pt_strict)
    pt_D = len(pt_relaxed)

    tf_A = len(tf_random)
    tf_B = len(tf_guided)
    tf_D = len(tf_union)

    print(f"Config | Added factor                | PT (/{PT_CSV_REPRO_YES})       | TF (/{TF_CSV_REPRO_YES})")
    print(f"{'─'*70}")
    print(f"A      | random mutation             | {pt_A:2d} ({pt_A/PT_CSV_REPRO_YES*100:4.1f}%)       | {tf_A:2d} ({tf_A/TF_CSV_REPRO_YES*100:4.1f}%)")
    print(f"B      | guided state targeting      | {pt_B:2d} ({pt_B/PT_CSV_REPRO_YES*100:4.1f}%)       | {tf_B:2d} ({tf_B/TF_CSV_REPRO_YES*100:4.1f}%)")
    print(f"C      | strict error match          | {pt_C:2d} ({pt_C/PT_CSV_REPRO_YES*100:4.1f}%)       | —")
    print(f"D      | state + equivalent matching | {pt_D:2d} ({pt_D/PT_CSV_REPRO_YES*100:4.1f}%)       | {tf_D:2d} ({tf_D/TF_CSV_REPRO_YES*100:4.1f}%)")
    print()

    # Verification
    print(f"Complementarity:")
    print(f"  PT: A∪B = {len(pt_random_eq | pt_guided_eq)} (of {pt_D} total equivalent)")
    print(f"  PT: A∩B = {len(pt_both_eq)} (shared issues)")
    pt_unique_from_state = pt_D - pt_A
    print(f"  PT: D - A = {pt_unique_from_state} issues added by guided targeting")
    print(f"  TF: A∪B = {len(tf_random | tf_guided)}")
    print(f"  TF: A∩B = {len(tf_random & tf_guided)}")

    # Save results
    results = {
        "pytorch": {
            "total_reproduced_yes": PT_CSV_REPRO_YES,
            "matched_issues": len(pt),
            "random_equivalent": pt_A,
            "guided_equivalent": pt_B,
            "strict_total": pt_C,
            "equivalent_total": pt_D,
            "random_only": sorted(pt_r_only_eq),
            "guided_only": sorted(pt_g_only_eq),
            "overlap": sorted(pt_both_eq),
        },
        "tensorflow": {
            "total_reproduced_yes": TF_CSV_REPRO_YES,
            "matched_issues": len(tf),
            "random": tf_A,
            "guided": tf_B,
            "union": tf_D,
            "random_only": sorted(tf_random - tf_guided),
            "guided_only": sorted(tf_guided - tf_random),
            "overlap": sorted(tf_random & tf_guided),
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "component_table_derived.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
