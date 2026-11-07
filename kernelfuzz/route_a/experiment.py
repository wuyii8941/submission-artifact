"""Bounded Route A campaign driver.

All arms use the same generation, execution, observation, and storage path.
Only target/condition selection is changed by the arm configuration.  This
module intentionally supports short qualification runs first; a formal
campaign should use a frozen catalog and a reviewed run configuration.
"""

import argparse
import json
import os
import signal
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .compat import qualification_snapshot, supported_ops_default, supported_ops_broad
from .model import ConditionRequest, GraphRequest, GenerationResult, Observation
from .policy import (
    ArtifactFeedbackTargetPolicy,
    AttemptBalancedTargetPolicy,
    ConditionPolicy,
    ExplorationHistory,
    UniformTargetPolicy,
)
from .runner import RouteAExecutor
from .storage import EventStore
from .targets import (
    TargetDefinition,
    all_condition_cells,
    condition_cells,
    target_map,
)
from .torchfuzz_adapter import TorchFuzzAdapter
from .accounting import classify_trial, CATEGORIES


class _GenerationDeadline(BaseException):
    pass


def _generate_with_deadline(adapter, request, seconds):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return adapter.generate(request)
    previous_handler = signal.getsignal(signal.SIGALRM)

    def alarm_handler(_signum, _frame):
        raise _GenerationDeadline()

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return adapter.generate(request)
    except _GenerationDeadline:
        return GenerationResult(
            status="invalid", elapsed_seconds=float(seconds), attempts=0,
            failure_reason="generation_timeout",
            failure_detail="generation exceeded per-trial deadline",
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _choose_dtype(value: object):
    import torch

    if not isinstance(value, str):
        return value
    name = value.replace("torch.", "")
    if not hasattr(torch, name):
        raise ValueError("unsupported dtype cell: %s" % value)
    return getattr(torch, name)


def _compatible_prior_cells(
    target: TargetDefinition,
    prior_cells: Iterable[Tuple[str, str, object, str]],
) -> List[Tuple[str, str, object, str]]:
    """Filter the fixed A1 prior only by target-level type applicability.

    A1 must not use target hit history or target-specific boundary cells.  It
    may, however, avoid asking a pointwise entry to accept a dtype that the
    target catalog explicitly marks unsupported.  Shape values remain from
    the global prior and are checked again by the graph legalizer.
    """
    allowed_attributes = set(target.condition_attributes)
    allowed_dtypes = set(target.dtype_cells)
    result = []
    for attribute, cell, value, binding in prior_cells:
        if attribute not in allowed_attributes:
            continue
        if attribute == "dtype" and str(value) not in allowed_dtypes:
            continue
        result.append((attribute, cell, value, binding))
    return result


def _make_condition(
    target_id: str,
    selected: Tuple[str, str, object, str],
    *,
    directed: bool,
) -> ConditionRequest:
    attribute, cell, value, _binding = selected
    return ConditionRequest(
        target_id=target_id,
        attribute=attribute,
        axis=-1 if attribute == "shape" else None,
        value=_choose_dtype(value) if attribute == "dtype" else value,
        cell=cell,
        provenance=(
            "route-a-target-condition"
            if directed
            else "route-a-fixed-nondirected-prior"
        ),
        binding_scope="entry_lineage" if directed else "random_node",
    )


def _target_policy(arm: str, target_ids: Tuple[str, ...], seed: int):
    if arm in ("a3", "a4"):
        return ArtifactFeedbackTargetPolicy(target_ids, seed)
    if arm in ("a1", "a2"):
        return AttemptBalancedTargetPolicy(target_ids, seed)
    return UniformTargetPolicy(target_ids, seed)


def run(args) -> Dict[str, object]:
    if not args.supported_ops:
        args.supported_ops = (supported_ops_broad() if getattr(args, 'operator_pool', 'legacy') == 'broad'
                              else supported_ops_default())
    # Keep every isolated trial and any accidental parent-side compile bounded.
    # PyTorch reads this setting while importing Inductor's config module.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if Path(args.out, 'events.jsonl').exists():
        raise ValueError('Run directory already has events; choose a new --out to preserve run identity.')
    target_defs = target_map()
    target_ids = tuple(target_defs.keys())
    seed = int(args.seed)
    store = EventStore(args.out)
    qualification = qualification_snapshot()
    if not args.generate_only and not qualification['cuda_available']:
        raise RuntimeError('CUDA unavailable in this process. GPU runs require device access; use generate-only for static checks.')
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*')):
        if path.suffix in ('.py', '.json'):
            code_hash.update(str(path.relative_to(Path(__file__).parent)).encode())
            code_hash.update(path.read_bytes())
    store.append(
        {
            "event": "run_start",
            "arm": args.arm,
            "seed": seed,
            "qualification": qualification,
            'config': vars(args),
            'implementation_sha256': code_hash.hexdigest(),
            'policy_version': 'route_a_v3_stable_behavior_and_operation_hit',
            'generator_revision': (Path(__file__).parents[2] / 'third_party/torchfuzz/REVISION').read_text().strip(),
            "target_catalog": [target.to_dict() for target in target_defs.values()],
            "condition_prior": all_condition_cells(),
            "result_scope": {
                "catalog_scope": "route_a_qualification_targets",
                "target_count": len(target_defs),
                "claim_limit": "Only this frozen, version-scoped qualification catalog is measured; this is not full compiler operation coverage.",
                "hit_rule": "A target hit requires a completed observer and a matched source origin, original/lowered ATen operation, kernel family, and computation-specific artifact evidence. Source Nodes alone and auxiliary view/copy kernels do not count.",
                "bug_rule": "Oracle candidates are reported as candidates and deduplicated by generated program SHA-256 only. No candidate is counted as a verified unique bug without independent validation and root-cause mapping.",
            },
        },
        sync=True,
    )

    adapter = TorchFuzzAdapter(
        supported_ops=args.supported_ops or supported_ops_default()
    )
    history = ExplorationHistory(target_ids)
    target_policy = _target_policy(args.arm, target_ids, seed)
    condition_policy = ConditionPolicy(seed + 1009)
    fixed_prior = all_condition_cells()
    executor = None if args.generate_only else RouteAExecutor(
        python_executable=args.python_executable,
        timeout_seconds=args.timeout,
        work_dir=args.work_dir,
        runtime_root=str(Path(args.out, 'runtime').resolve()),
    )

    started = time.monotonic()
    stop_requested = {"value": False}
    previous_signals = {}

    def _request_stop(_signum, _frame):
        stop_requested["value"] = True

    for _signal in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if _signal is not None:
            previous_signals[_signal] = signal.getsignal(_signal)
            signal.signal(_signal, _request_stop)
    counts: Dict[str, int] = {
        "trials": 0,
        "generation_attempts": 0,
        "valid": 0,
        "observations": 0,
        "hit": 0,
        "miss": 0,
        "unknown": 0,
        "compile_failure": 0,
        "generation_failures": 0,
        "target_node_hits": 0,
        "target_operation_hits": 0,
        "family_only_target_matches": 0,
        "oracle_pass": 0,
        "oracle_candidate": 0,
        "measurement_unknown": 0,
    }
    failure_reasons: Dict[str, int] = {}
    outcome_counts = {category: 0 for category in CATEGORIES}
    seen_target_behavior_signatures = set()
    candidate_program_hashes = set()
    observed_target_counts = {tid: 0 for tid in target_ids}
    stop_reason = "attempt_limit"

    while True:
        if stop_requested["value"]:
            stop_reason = "signal"
            break
        if args.attempts is not None and counts["trials"] >= args.attempts:
            stop_reason = "attempt_limit"
            break
        if shutil.disk_usage(args.out).free < 1024**3:
            store.append({'event': 'resource_stop', 'reason': 'free_disk_below_1GiB'}, sync=True)
            stop_reason = "free_disk_below_1GiB"
            break
        if args.seconds > 0 and time.monotonic() - started >= args.seconds:
            stop_reason = "time_budget"
            break
        iteration = counts["trials"]
        target_id: Optional[str] = None
        condition: Optional[ConditionRequest] = None

        if args.arm != "a0":
            target_id = target_policy.select(history)
            target = target_defs[target_id]
            if args.arm == "a1":
                cells = _compatible_prior_cells(target, fixed_prior)
                selected = condition_policy.select(target_id, cells, history, False)
                directed = False
            else:
                cells = condition_cells(target)
                selected = condition_policy.select(
                    target_id, cells, history, args.arm == "a4"
                )
                directed = True
            if selected is not None:
                condition = _make_condition(target_id, selected, directed=directed)

        request = GraphRequest(
            seed=seed * 100000 + iteration,
            max_depth=args.max_depth,
            template=args.template,
            supported_ops=tuple(args.supported_ops or supported_ops_default()),
            target_id=target_id,
            condition=condition,
            max_entry_retries=args.max_entry_retries,
            runtime_device=args.runtime_device,
            entry_strategy=args.entry_strategy,
            target_weight=getattr(args, 'target_weight', 4.0),
        )
        generation = _generate_with_deadline(adapter, request, args.generation_timeout)
        counts["trials"] += 1
        counts["generation_attempts"] += generation.attempts
        row: Dict[str, object] = {
            "event": "trial",
            "elapsed_seconds": time.monotonic() - started,
            "iteration": iteration,
            "request": request.to_dict(),
            "generation": generation.to_dict(),
            "condition_policy_branch": condition_policy.last_branch if condition else "not_selected",
        }

        if not generation.valid:
            counts["generation_failures"] += 1
            reason = generation.failure_reason or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            row['outcome_accounting'] = classify_trial(request.to_dict(), generation.to_dict())
            outcome_counts[row['outcome_accounting']['category']] += 1
            history.record(
                target_id,
                condition.cell if condition else None,
                valid=False,
                observable=False,
                selected_hit=False,
                novel=False,
                cost=generation.elapsed_seconds,
            )
            store.append(row)
            store.checkpoint({'counts': counts, 'target_stats': {tid: vars(stat) for tid, stat in history.target_stats.items()}})
            continue

        counts["valid"] += 1
        testcase = generation.test_case
        assert testcase is not None
        row["testcase"] = testcase.to_dict()
        row["program_blob"] = store.put_text(testcase.code)
        # The manifest and replayable program are durable before execution.
        # Full stdout/source events are kept in the content-addressed evidence
        # file; JSONL stays compact enough for a multi-hour campaign.
        store.append({
            "event": "trial_start",
            "iteration": iteration,
            "elapsed_seconds": time.monotonic() - started,
            "request": request.to_dict(),
            "generation": generation.to_dict(),
            "testcase": testcase.to_dict(),
            "program_blob": row["program_blob"],
        }, sync=True)

        observation: Optional[Observation] = None
        selected_hit = False
        novel = False
        if executor is not None:
            target_tokens = target_defs[target_id].operation_tokens if target_id else ()
            observation = executor.run(
                testcase,
                target_tokens,
                selected_target=target_id,
            )
            counts["observations"] += 1
            counts[observation.status] = counts.get(observation.status, 0) + 1
            if observation.measurement_status == "unknown":
                counts["measurement_unknown"] += 1
            for observed_target in observation.observed_targets:
                if observed_target in observed_target_counts:
                    observed_target_counts[observed_target] += 1
            selected_hit = (
                observation.target_node_hit
                and observation.measurement_status == "complete"
            )
            if selected_hit:
                counts["target_node_hits"] += 1
            if observation.target_operation_hit:
                counts["target_operation_hits"] += 1
            elif observation.target_match_level == "family_only":
                counts["family_only_target_matches"] += 1
            oracle_status = observation.oracle.get("status")
            if oracle_status in ("pass", "candidate"):
                counts["oracle_" + oracle_status] += 1
            if oracle_status == "candidate":
                program_hash = row["program_blob"]["sha256"]
                candidate_program_hashes.add(program_hash)
                row["oracle_candidate_program_sha256"] = program_hash
            novel = observation.target_node_hit and any(
                signature not in seen_target_behavior_signatures
                for signature in observation.target_behavior_signatures
            )
            seen_target_behavior_signatures.update(observation.target_behavior_signatures)
            evidence = store.put_json(observation.to_dict(), prefix="observation")
            observation.evidence_path = evidence["path"]
            compact_observation = observation.to_dict()
            for field in ("stdout", "stderr", "raw_events"):
                compact_observation.pop(field, None)
            row["observation"] = compact_observation

        history.record(
            target_id,
            condition.cell if condition else None,
            valid=True,
            observable=observation is not None
            and observation.measurement_status == "complete",
            selected_hit=selected_hit,
            novel=novel,
            cost=generation.elapsed_seconds
            + (observation.elapsed_seconds if observation else 0.0),
        )
        row["feedback_accounting"] = {
            "target_node_hit": bool(observation and observation.target_node_hit),
            "target_node_evidence": observation.target_node_evidence if observation else [],
            "target_behavior_signatures": observation.target_behavior_signatures if observation else [],
            "novel_target_behavior": bool(novel),
        }
        row["generation_finished_seconds"] = row["elapsed_seconds"]
        row['outcome_accounting'] = classify_trial(request.to_dict(), generation.to_dict(),
            observation.to_dict() if observation else None)
        outcome_counts[row['outcome_accounting']['category']] += 1
        row["elapsed_seconds"] = time.monotonic() - started
        store.append(row)
        store.checkpoint({'counts': counts, 'target_stats': {tid: vars(stat) for tid, stat in history.target_stats.items()}})

    for _signal, _handler in previous_signals.items():
        signal.signal(_signal, _handler)

    summary = {
        "arm": args.arm,
        "seed": seed,
        "trials": counts["trials"],
        "generation_attempts": counts["generation_attempts"],
        "valid": counts["valid"],
        "elapsed_seconds": time.monotonic() - started,
        "counts": counts,
        "failure_reasons": failure_reasons,
        "outcome_counts": outcome_counts,
        "stop_reason": stop_reason,
        "target_stats": {tid: vars(stat) for tid, stat in history.target_stats.items()},
        "observed_target_test_hits": observed_target_counts,
        "cell_stats": {
            "%s/%s" % key: vars(stat) for key, stat in history.cell_stats.items()
        },
        "qualification": qualification,
        'implementation_sha256': code_hash.hexdigest(),
        "result_scope": {
            "catalog_scope": "route_a_qualification_targets",
            "target_count": len(target_defs),
            "claim_limit": "Only the version-scoped qualification catalog was measured; results do not estimate full compiler operation coverage.",
            "hit_rule": "Policy rewards only computation-confirmed matches from the common scorer; source-only, family-only, and incomplete-hook observations are excluded.",
            "bug_rule": "Oracle candidates are not verified bugs. Program-SHA deduplication is not root-cause deduplication.",
        },
        "unique_candidate_programs": len(candidate_program_hashes),
        "candidate_program_sha256": sorted(candidate_program_hashes),
        "verified_unique_bug_count": None,
        "note": "Route A qualification/short run; candidate signals are not verified bugs.",
    }
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    store.checkpoint(summary, name="summary.json")
    store.append({"event": "run_end", "summary": summary}, sync=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Route A TorchFuzz experiment")
    parser.add_argument("--arm", choices=("a0", "a1", "a2", "a3", "a4"), default="a0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attempts", type=int, default=None,
        help="Optional additional trial cap. Omit it for a time-budget-only run.",
    )
    parser.add_argument(
        "--seconds", type=int, default=14400,
        help="Wall-clock budget in seconds (default: four hours).",
    )
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-entry-retries", type=int, default=16)
    parser.add_argument("--template", default="default")
    parser.add_argument("--supported-ops", nargs="*", default=None)
    parser.add_argument('--operator-pool', choices=('legacy', 'broad'), default='legacy')
    parser.add_argument('--target-weight', type=float, default=4.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--generation-timeout", type=int, default=30,
        help="Per-trial graph-generation deadline in seconds; zero disables it.",
    )
    parser.add_argument(
        "--runtime-device",
        choices=("cuda",),
        default="cuda",
        help="TorchFuzz's pinned template currently emits CUDA programs.",
    )
    parser.add_argument(
        "--entry-strategy",
        choices=("bounded_filter", "target_biased", "weighted"),
        default="target_biased",
        help="Entry selection mode; freeze it across A1--A4 before formal use.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
