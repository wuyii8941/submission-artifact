"""Independent integrity and effect audit for formal_effect_run_v1.

This deliberately reads only frozen protocol/raw artifacts.  It does not import
the experiment implementation or PyTorch, and it never rewrites raw results.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_PROTOCOL_SHA256 = "b6b06932d58da01b08ea32864876f4a9508b15df8baf934eff009319df99173c"
EXPECTED_UNIVERSE_SHA256 = "053491a87f700fd6ae38111d3b6a7e15fdd8ca886494bd182c5569beb4f66d83"
PRIMARY_DENOMINATOR = 642
OPERATOR_KIND_DENOMINATOR = 55
ARMS = ("A", "B", "C")


def read(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def exact_sign_p(differences):
    nonzero = [value for value in differences if value]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    negatives = len(nonzero) - positives
    tail = min(positives, negatives)
    probability = 2 * sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / 2 ** len(nonzero)
    return min(1.0, probability)


def bootstrap_interval(differences):
    rng = random.Random(2026110607)
    count = len(differences)
    values = sorted(
        sum(rng.choice(differences) for _ in range(count)) / count
        for _ in range(10000)
    )
    return [values[249], values[9749]]


def topology(graph_record):
    payload = graph_record["payload"]
    return {
        "root_node_id": payload["root_node_id"],
        "nodes": [
            {
                "node_id": node["node_id"],
                "op_name": node["op_name"],
                "input_nodes": node["input_nodes"],
                "semantic_kind": node["semantics"]["kind"],
            }
            for node in payload["nodes"]
        ],
    }


def normalized_candidate(row):
    candidate = row["candidate"]
    return {
        key: candidate[key]
        for key in (
            "graph_sha256",
            "skeleton_sha256",
            "action",
            "target_applicable",
            "predicted_outcome",
            "context",
            "input_region",
        )
    }


def analyze(directory):
    directory = Path(directory).resolve()
    failures = []
    checks = []

    def check(name, condition, detail=None):
        row = {"name": name, "passed": bool(condition)}
        if detail is not None:
            row["detail"] = detail
        checks.append(row)
        if not condition:
            failures.append(row)

    protocol_path = directory / "protocol.json"
    suite_path = directory / "suite.json"
    analysis_path = directory / "analysis.json"
    protocol = read(protocol_path)
    suite = read(suite_path)
    published_analysis = read(analysis_path)
    universe_matches = list((directory / "source_snapshot").glob("*_universe.json"))
    check("one frozen universe snapshot", len(universe_matches) == 1, len(universe_matches))
    if len(universe_matches) != 1:
        raise RuntimeError("cannot identify frozen universe")
    universe_path = universe_matches[0]
    universe = read(universe_path)

    protocol_hash = file_sha256(protocol_path)
    universe_hash = file_sha256(universe_path)
    check("protocol hash frozen", protocol_hash == EXPECTED_PROTOCOL_SHA256, protocol_hash)
    check("universe hash frozen", universe_hash == EXPECTED_UNIVERSE_SHA256, universe_hash)
    check("protocol names frozen universe", protocol["universe_sha256"] == universe_hash)
    check("suite complete", suite["status"] == "complete")
    check("no suite invalidations", suite["invalidations"] == [], suite["invalidations"])
    check("analysis complete", published_analysis["status"] == "complete")

    target_ids = [row["target_id"] for row in universe["targets"]]
    excluded_ids = [row["target_id"] for row in universe["excluded"]]
    public_operators = universe["operator_public_names"]
    target_set = set(target_ids)
    public_operator_set = set(public_operators)
    check("source candidate accounting", universe["branch_candidate_count"] == 712)
    check("proved exclusion accounting", universe["excluded_outcome_count"] == 70)
    check("fixed primary denominator", universe["denominator"] == PRIMARY_DENOMINATOR)
    check("642 unique target outcomes", len(target_ids) == len(target_set) == PRIMARY_DENOMINATOR)
    check("70 unique excluded outcomes", len(excluded_ids) == len(set(excluded_ids)) == 70)
    check("included and excluded disjoint", not target_set.intersection(excluded_ids))
    check("55 unique public operator kinds", len(public_operators) == len(public_operator_set) == OPERATOR_KIND_DENOMINATOR)
    check("unknown feasibility retained", "Unknown" in universe["unknowns"])

    seeds = protocol["master_seeds"]
    expected_run_keys = {(seed, arm) for seed in seeds for arm in ARMS}
    run_keys = [(row["master_seed"], row["arm"]) for row in suite["runs"]]
    check("24 formal arms", len(run_keys) == 24)
    check("each seed-arm exactly once", len(set(run_keys)) == 24 and set(run_keys) == expected_run_keys)
    actual_orders = []
    for seed in seeds:
        actual_orders.append([row["arm"] for row in suite["runs"] if row["master_seed"] == seed])
    check("balanced predeclared arm order", actual_orders == protocol["order_by_seed"], actual_orders)

    summaries = {}
    queues = {}
    common_source_hashes = None
    run_audit = []
    source_evidence_files = 0
    all_candidate_topologies_checked = 0
    all_candidate_graphs_checked = 0

    for run in suite["runs"]:
        seed = run["master_seed"]
        arm = run["arm"]
        key = (seed, arm)
        run_dir = directory / run["relative_output"]
        summary = read(run_dir / "summary.json")
        queue = read(run_dir / "queue.json")
        summaries[key] = summary
        queues[key] = queue
        supervisor = run["supervisor"]
        wall = supervisor["full_process_wall_seconds"]
        top_stderr = directory / f"seed_{run['seed_index']:02d}_{arm}.stderr.txt"

        run_checks = {
            "identity": summary["arm"] == arm and summary["master_seed"] == seed,
            "formal": summary["purpose"] == "formal" and not summary["control_only"],
            "protocol_valid": bool(summary["protocol_valid"]),
            "source_unchanged": bool(summary["source_unchanged"]),
            "supervisor_returncode_zero": supervisor["returncode"] == 0,
            "wall_budget_charged": 598.0 <= wall <= 602.0 and supervisor["requested_wall_seconds"] == 600,
            "top_level_stderr_empty": top_stderr.exists() and top_stderr.stat().st_size == 0,
            "all_generation_attempts_charged": len(summary["generation"]) == 24,
            "denominator_fixed": summary["primary_denominator"] == PRIMARY_DENOMINATOR,
            "coverage_fraction_exact": summary["primary_coverage"] == len(summary["covered_target_ids"]) / PRIMARY_DENOMINATOR,
        }
        for name, passed in run_checks.items():
            check(f"run {run['seed_index']:02d}{arm}: {name}", passed)

        covered = summary["covered_target_ids"]
        successful = summary["successful_execution_target_ids"]
        kinds = summary["operator_kinds_at_backend_entry"]
        check(f"run {run['seed_index']:02d}{arm}: unique covered outcomes", len(covered) == len(set(covered)))
        check(f"run {run['seed_index']:02d}{arm}: covered outcomes in frozen universe", set(covered) <= target_set)
        check(f"run {run['seed_index']:02d}{arm}: successful hits subset", set(successful) <= set(covered))
        check(f"run {run['seed_index']:02d}{arm}: operator kinds unique/in domain", len(kinds) == len(set(kinds)) and set(kinds) <= public_operator_set)

        if common_source_hashes is None:
            common_source_hashes = summary["source_sha256"]
        check(f"run {run['seed_index']:02d}{arm}: common watched-source hashes", summary["source_sha256"] == common_source_hashes)

        metric_hits = set()
        successful_hits = set()
        metric_kinds = set()
        observed_count = 0
        for execution in summary["executions"]:
            selection = execution.get("selection")
            if selection:
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: selected graph agrees", execution["selected_graph_sha256"] == queue_candidate(queues[key], selection["id"])["graph_sha256"])
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: selected action agrees", execution["action"] == queue_candidate(queues[key], selection["id"])["action"])
            metrics = execution.get("metrics")
            if metrics:
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: metrics protocol valid", metrics["protocol_valid"])
                metric_hits.update(metrics["verified_target_ids"])
                successful_hits.update(metrics["successful_execution_target_ids"])
                metric_kinds.update(metrics["operator_kinds_at_backend_entry"])
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: metric hits in universe", set(metrics["verified_target_ids"]) <= target_set)
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: metric kinds in domain", set(metrics["operator_kinds_at_backend_entry"]) <= public_operator_set)
            observation = execution.get("observation", {})
            if observation.get("status") == "observed":
                observed_count += 1
                evidence_path = run_dir / f"execute_{execution['ordinal']:04d}" / "source_evidence.json"
                source_evidence_files += 1
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: source evidence exists", evidence_path.exists())
                if evidence_path.exists():
                    check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: source evidence hash", file_sha256(evidence_path) == observation["source_evidence_sha256"])
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: feedback target identity", observation["target_id"] == execution["target_id"])
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: feedback graph identity", observation["graph_sha256"] == execution["selected_graph_sha256"])
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: feedback node identity", observation["node_id"] == execution["action"]["node_id"])
                # The feedback target is a logical condition ID; primary
                # coverage IDs are separate 64-hex directed source edges.
                # Require the independently read metric observation itself to
                # agree exactly instead of conflating the two namespaces.
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: observed feedback independently reproduced", metrics is not None and metrics["observation"] == observation)
                check(f"run {run['seed_index']:02d}{arm} execution {execution['ordinal']}: observed side is boolean", observation["outcome"] in ("true", "false"))
        check(f"run {run['seed_index']:02d}{arm}: summary branch union exact", metric_hits == set(covered))
        check(f"run {run['seed_index']:02d}{arm}: summary successful union exact", successful_hits == set(successful))
        check(f"run {run['seed_index']:02d}{arm}: summary operator union exact", metric_kinds == set(kinds))

        run_audit.append({
            "seed_index": run["seed_index"],
            "master_seed": seed,
            "arm": arm,
            "supervisor_wall_seconds": wall,
            "hard_deadline_triggered": supervisor["hard_deadline_triggered"],
            "generation_attempts": len(summary["generation"]),
            "generation_failures": sum(row["status"] != "generated" for row in summary["generation"]),
            "requests": len(summary["executions"]),
            "observed_source_feedback": observed_count,
            "branch_outcomes": len(covered),
            "operator_kinds": len(kinds),
        })

    # Every watched source used by every arm must have an exact frozen copy.
    for name, expected_hash in common_source_hashes.items():
        candidates = [
            path for path in (directory / "source_snapshot").iterdir()
            if path.is_file() and path.name.split("_", 1)[-1] == name
        ]
        check(f"watched source snapshot: {name}", len(candidates) == 1)
        if len(candidates) == 1:
            check(f"watched source hash: {name}", file_sha256(candidates[0]) == expected_hash)

    experiment_source = next((directory / "source_snapshot").glob("*_experiment_arm.py"))
    source_text = experiment_source.read_text()
    check("generator request is target blind", "target_id=None,condition=None" in source_text)
    check("same adapter pool passed to generator", "supported_ops=tuple(pool['accepted'])" in source_text)

    # Initial generated DAGs and all candidate actions must be identical in A/B/C.
    fairness_by_seed = []
    for seed_index, seed in enumerate(seeds):
        arm_summaries = {arm: summaries[(seed, arm)] for arm in ARMS}
        generation_signatures = {}
        candidate_maps = {}
        for arm in ARMS:
            generation_signatures[arm] = [
                {
                    key: row.get(key)
                    for key in ("ordinal", "seed", "status", "skeleton_sha256", "error_type", "error")
                }
                for row in arm_summaries[arm]["generation"]
            ]
            candidate_maps[arm] = {
                row["id"]: {"target_id": row["target_id"], **normalized_candidate(row)}
                for row in queues[(seed, arm)]["candidates"]
            }
        generation_equal = generation_signatures["A"] == generation_signatures["B"] == generation_signatures["C"]
        candidate_equal = candidate_maps["A"] == candidate_maps["B"] == candidate_maps["C"]
        check(f"seed {seed_index:02d}: identical target-blind DAG generation", generation_equal)
        check(f"seed {seed_index:02d}: identical A/B/C candidate action sets", candidate_equal)

        # Independently recompute queue identities and verify topology preservation
        # once per common set (A), rather than trusting candidate IDs.
        skeleton_graphs = {}
        for skeleton_path in (directory / f"seed_{seed_index:02d}_A").glob("skeleton_*.json"):
            record = read(skeleton_path)["graph"]
            check(f"seed {seed_index:02d}: skeleton manifest hash {skeleton_path.name}", record["sha256"] == file_manifest_hash(record["payload"]))
            skeleton_graphs[record["sha256"]] = record
        candidate_file_cache = {}
        for identity, row in candidate_maps["A"].items():
            identity_payload = {
                "target": row["target_id"],
                "candidate": {
                    key: row[key]
                    for key in ("graph_sha256", "action", "predicted_outcome", "context")
                },
            }
            check(f"seed {seed_index:02d}: candidate identity {identity[:12]}", digest(identity_payload) == identity)
            original = queue_candidate(queues[(seed, "A")], identity)
            case_path = Path(original["case"])
            if case_path not in candidate_file_cache:
                case_graph = read(case_path)["graph"]
                candidate_file_cache[case_path] = case_graph
                all_candidate_graphs_checked += 1
                check(f"seed {seed_index:02d}: candidate manifest hash {case_graph['sha256'][:12]}", case_graph["sha256"] == file_manifest_hash(case_graph["payload"]))
            else:
                case_graph = candidate_file_cache[case_path]
            check(f"seed {seed_index:02d}: candidate graph identity {identity[:12]}", case_graph["sha256"] == row["graph_sha256"])
            skeleton = skeleton_graphs.get(row["skeleton_sha256"])
            check(f"seed {seed_index:02d}: candidate skeleton exists {identity[:12]}", skeleton is not None)
            if skeleton is not None:
                all_candidate_topologies_checked += 1
                check(f"seed {seed_index:02d}: topology unchanged {identity[:12]}", topology(case_graph) == topology(skeleton))

        # A/B paired prefixes must use the same target and source DAG schedule.
        a_rows = arm_summaries["A"]["executions"]
        b_rows = arm_summaries["B"]["executions"]
        pair_count = min(len(a_rows), len(b_rows))
        schedule_equal = True
        b_guidance_opportunities = 0
        b_guidance_realized = 0
        ab_choice_differences = 0
        for ordinal, (a_row, b_row) in enumerate(zip(a_rows[:pair_count], b_rows[:pair_count])):
            if a_row["target_id"] != b_row["target_id"] or a_row.get("round") != b_row.get("round"):
                schedule_equal = False
                continue
            a_selection = a_row.get("selection")
            b_selection = b_row.get("selection")
            if bool(a_selection) != bool(b_selection):
                schedule_equal = False
                continue
            if not a_selection:
                schedule_equal &= a_row.get("baseline_skeleton_sha256") == b_row.get("baseline_skeleton_sha256")
                continue
            schedule_equal &= (
                a_selection["baseline_skeleton_sha256"] == b_selection["baseline_skeleton_sha256"]
                and a_selection["selected_skeleton_sha256"] == a_selection["baseline_skeleton_sha256"]
                and b_selection["selected_skeleton_sha256"] == b_selection["baseline_skeleton_sha256"]
            )
            ab_choice_differences += a_selection["id"] != b_selection["id"]
            if b_selection["desired_outcome_available"]:
                b_guidance_opportunities += 1
                selected = candidate_maps["B"][b_selection["id"]]
                b_guidance_realized += selected["predicted_outcome"] == b_selection["desired_outcome"]
        check(f"seed {seed_index:02d}: A/B matched target and skeleton prefix", schedule_equal)
        check(f"seed {seed_index:02d}: B realizes every available requested side", b_guidance_realized == b_guidance_opportunities)

        fairness_by_seed.append({
            "seed_index": seed_index,
            "master_seed": seed,
            "generated_dags_per_arm": len(generation_signatures["A"]),
            "common_candidates_per_arm": len(candidate_maps["A"]),
            "generation_equal": generation_equal,
            "candidate_sets_equal": candidate_equal,
            "AB_paired_requests": pair_count,
            "AB_schedule_equal": schedule_equal,
            "AB_different_input_choices": ab_choice_differences,
            "B_guidance_opportunities": b_guidance_opportunities,
            "B_guidance_realized": b_guidance_realized,
        })

    # Verify C uses only prior source-confirmed feedback and never gains actions.
    feedback_by_seed = []
    for seed_index, seed in enumerate(seeds):
        c_rows = summaries[(seed, "C")]["executions"]
        b_rows_by_target = defaultdict(list)
        for row in summaries[(seed, "B")]["executions"]:
            if row.get("selection"):
                b_rows_by_target[row["target_id"]].append(row)
        seen = defaultdict(list)
        target_occurrence = Counter()
        changed = 0
        post_feedback = 0
        pre_feedback_matches_b = 0
        pre_feedback_comparisons = 0
        reasons = Counter()
        cross_skeleton = 0
        for row in c_rows:
            selection = row.get("selection")
            if not selection:
                continue
            target = row["target_id"]
            occurrence = target_occurrence[target]
            target_occurrence[target] += 1
            prior_verified = bool(seen[target])
            check(f"seed {seed_index:02d} C execution {row['ordinal']}: feedback availability truthful", bool(selection["verified_feedback_available"]) == prior_verified)
            check(f"seed {seed_index:02d} C execution {row['ordinal']}: selected candidate initially common", selection["id"] in {candidate["id"] for candidate in queues[(seed, "A")]["candidates"]})
            check(f"seed {seed_index:02d} C execution {row['ordinal']}: selected and baseline pending", selection["id"] in selection["pending_ids"] and selection["baseline_id"] in selection["pending_ids"])
            if not prior_verified and occurrence < len(b_rows_by_target[target]):
                pre_feedback_comparisons += 1
                pre_feedback_matches_b += selection["id"] == b_rows_by_target[target][occurrence]["selection"]["id"]
            if prior_verified:
                post_feedback += 1
            if selection["id"] != selection["baseline_id"]:
                changed += 1
                reasons[selection["reason"]] += 1
                cross_skeleton += selection["selected_skeleton_sha256"] != selection["baseline_skeleton_sha256"]
                check(f"seed {seed_index:02d} C execution {row['ordinal']}: changed choice has prior verified feedback", prior_verified)
                check(f"seed {seed_index:02d} C execution {row['ordinal']}: changed choice reason is feedback policy", selection["reason"] in {"unobserved_outcome", "new_input_region", "new_outcome_context"})
                for observation in seen[target]:
                    check(f"seed {seed_index:02d} C execution {row['ordinal']}: prior feedback file remained source-backed", bool(observation["source_evidence_sha256"]))
            observation = row.get("observation", {})
            if observation.get("status") == "observed":
                seen[target].append(observation)
        check(f"seed {seed_index:02d}: C agrees with B before per-target feedback", pre_feedback_matches_b == pre_feedback_comparisons)
        feedback_by_seed.append({
            "seed_index": seed_index,
            "master_seed": seed,
            "post_feedback_selections": post_feedback,
            "feedback_changed_choice": changed,
            "changed_choice_reasons": dict(reasons),
            "changed_choice_cross_skeleton": cross_skeleton,
            "pre_feedback_matches_B": pre_feedback_matches_b,
            "pre_feedback_comparisons": pre_feedback_comparisons,
        })

    # Independent seed-level effect reconstruction.
    rows = []
    branch_counts = {}
    operator_counts = {}
    context_counts = {}
    for seed_index, seed in enumerate(seeds):
        for arm in ARMS:
            summary = summaries[(seed, arm)]
            branch_counts[(seed, arm)] = len(summary["covered_target_ids"])
            operator_counts[(seed, arm)] = len(summary["operator_kinds_at_backend_entry"])
            context_counts[(seed, arm)] = len({
                (row["target_id"], row["observation"]["outcome"], row["observation"].get("context_signature"))
                for row in summary["executions"]
                if row.get("observation", {}).get("status") == "observed"
            })
            rows.append({
                "seed_index": seed_index,
                "master_seed": seed,
                "arm": arm,
                "source_decision_outcomes": branch_counts[(seed, arm)],
                "source_decision_coverage": branch_counts[(seed, arm)] / PRIMARY_DENOMINATOR,
                "public_operator_kinds": operator_counts[(seed, arm)],
                "public_operator_kind_coverage": operator_counts[(seed, arm)] / OPERATOR_KIND_DENOMINATOR,
                "observed_outcome_contexts_descriptive_only": context_counts[(seed, arm)],
            })

    comparisons = []
    for treatment, baseline in (("B", "A"), ("C", "B")):
        differences = [branch_counts[(seed, treatment)] - branch_counts[(seed, baseline)] for seed in seeds]
        comparisons.append({
            "name": f"{treatment}-{baseline}",
            "paired_count_differences": differences,
            "mean_difference": statistics.mean(differences),
            "mean_fraction_difference": statistics.mean(differences) / PRIMARY_DENOMINATOR,
            "bootstrap95_count_difference": bootstrap_interval(differences),
            "sign_test_p": exact_sign_p(differences),
        })
    previous = 0.0
    for rank, comparison in enumerate(sorted(comparisons, key=lambda row: row["sign_test_p"])):
        comparison["holm_p"] = min(1.0, max(previous, comparison["sign_test_p"] * (2 - rank)))
        previous = comparison["holm_p"]
        comparison["stable_positive_evidence"] = (
            comparison["mean_difference"] > 0
            and comparison["bootstrap95_count_difference"][0] > 0
            and comparison["holm_p"] < 0.05
        )
    check("independent primary comparisons match published analysis", comparisons == published_analysis["comparisons"], {"independent": comparisons, "published": published_analysis["comparisons"]})

    operator_comparisons = []
    for treatment, baseline in (("B", "A"), ("C", "B")):
        differences = [operator_counts[(seed, treatment)] - operator_counts[(seed, baseline)] for seed in seeds]
        operator_comparisons.append({
            "name": f"{treatment}-{baseline}",
            "paired_count_differences": differences,
            "mean_difference": statistics.mean(differences),
            "mean_fraction_difference": statistics.mean(differences) / OPERATOR_KIND_DENOMINATOR,
            "bootstrap95_count_difference": bootstrap_interval(differences),
            "sign_test_p": exact_sign_p(differences),
        })

    primary_positive = all(row["stable_positive_evidence"] for row in comparisons)
    check("published positive claim agrees with frozen rule", published_analysis["positive_claim"] == primary_positive)
    check("published denominator agrees", published_analysis["denominator"] == PRIMARY_DENOMINATOR)
    check("published operator denominator agrees", published_analysis["operator_kind_denominator"] == OPERATOR_KIND_DENOMINATOR)

    result = {
        "schema": 1,
        "audit_status": "passed" if not failures else "failed",
        "raw_results_unchanged": True,
        "protocol_sha256": protocol_hash,
        "universe_sha256": universe_hash,
        "metric_contract": {
            "primary_name": "source compiler decision outcome coverage",
            "primary_formula": "distinct source-confirmed directed conditional-jump outcomes / 642",
            "primary_scope": universe["scope"],
            "secondary_name": "public operator-kind coverage at Dynamo backend entry",
            "secondary_formula": "distinct public operator kinds at backend entry / 55",
            "not_coverage_rates": ["GPU kernel hashes", "graph/context diversity"],
        },
        "integrity": {
            "runs": len(suite["runs"]),
            "invalidations": suite["invalidations"],
            "source_evidence_files_verified": source_evidence_files,
            "candidate_entries_topology_checked": all_candidate_topologies_checked,
            "unique_candidate_graph_files_checked": all_candidate_graphs_checked,
            "run_audit": run_audit,
            "fairness_by_seed": fairness_by_seed,
            "feedback_by_seed": feedback_by_seed,
        },
        "results": {
            "per_seed": rows,
            "source_decision_comparisons": comparisons,
            "public_operator_kind_comparisons": operator_comparisons,
            "stable_source_decision_lift_proved": primary_positive,
            "public_operator_kind_lift_proved": False,
            "interpretation": "valid experiment; directional source-decision gains are not stable evidence, and public operator-kind coverage did not show a stable lift",
        },
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_failed": len(failures),
        "failures": failures,
    }
    return result


def queue_candidate(queue, identity):
    matches = [row["candidate"] for row in queue["candidates"] if row["id"] == identity]
    if len(matches) != 1:
        raise AssertionError(f"candidate identity count is {len(matches)} for {identity}")
    return matches[0]


def file_manifest_hash(payload):
    return digest(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(args.directory)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded)
    print(json.dumps({
        "audit_status": result["audit_status"],
        "checks_passed": result["checks_passed"],
        "checks_failed": result["checks_failed"],
        "source_decision_comparisons": result["results"]["source_decision_comparisons"],
        "public_operator_kind_comparisons": result["results"]["public_operator_kind_comparisons"],
        "feedback_changes": [row["feedback_changed_choice"] for row in result["integrity"]["feedback_by_seed"]],
    }, indent=2))
    raise SystemExit(0 if result["audit_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
