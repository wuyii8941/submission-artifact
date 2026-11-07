"""Parse worker output and conservatively score the frozen target catalog."""

import json
import re
from typing import Iterable, List, Optional

from .model import Observation
from .scoring import REQUIRED_HOOKS, score_calls

_EVENT_PREFIX = "ROUTE_A_KERNEL_EVENTS="
_HOOK_PREFIX = "ROUTE_A_HOOK_STATUS="


def normalize_kernel_event(event: str):
    if not event or event == "extern_out:None":
        return None
    if event.startswith("call:"):
        try:
            payload = json.loads(event[len("call:"):])
            return str(payload.get("family", "unknown")) + ":" + str(payload.get("kernel", "unknown"))
        except Exception:
            return None
    if event.startswith("artifact:"):
        try:
            artifact = json.loads(event[len("artifact:"):])
            return str(artifact["family"]) + ":" + str(artifact["sha256"])
        except Exception:
            return None
    if event.startswith("extern_") or event.startswith("extern:"):
        return event[:160]
    return re.sub(r"_\d+$", "", event)


def behavior_signatures(raw_events: Iterable[str]) -> List[str]:
    result = set()
    for event in raw_events:
        if event.startswith("kernel_metadata:") or event.startswith("observer_error:"):
            continue
        if event.startswith("call:"):
            try:
                payload = json.loads(event[len("call:"):])
                origins = payload.get("origins") or []
                result.add(json.dumps({
                    "family": payload.get("family"),
                    "original_aten": sorted(str(x.get("original_aten")) for x in origins),
                    "lowered_aten": sorted(str(x.get("lowered_aten")) for x in origins),
                }, sort_keys=True, separators=(",", ":")))
            except Exception:
                continue
        elif event.startswith("artifact:"):
            try:
                result.add("family:" + str(json.loads(event[len("artifact:"):]).get("family", "unknown")))
            except Exception:
                continue
        elif (normalized := normalize_kernel_event(event)):
            result.add("kernel:" + normalized)
    return sorted(result)


def parse_event_lines(stdout: str) -> List[str]:
    events: List[str] = []
    for line in stdout.splitlines():
        if not line.startswith(_EVENT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(_EVENT_PREFIX):])
            if isinstance(payload, list):
                events.extend(str(item) for item in payload)
        except Exception:
            continue
    return events


def parse_hook_status(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith(_HOOK_PREFIX):
            return line[len(_HOOK_PREFIX):].strip() or "none"
    return "unknown"


def _event_calls(raw_events: Iterable[str]):
    calls = []
    for event in raw_events:
        if not event.startswith("call:"):
            continue
        try:
            payload = json.loads(event[len("call:"):])
        except Exception:
            continue
        if isinstance(payload, dict):
            calls.append(payload)
    return calls


def classify_observation(returncode: int, timeout: bool, stdout: str, stderr: str,
                         target_tokens: Iterable[str] = (),
                         hook_status: Optional[str] = None,
                         graph_manifest: Optional[dict] = None,
                         selected_target: Optional[str] = None) -> Observation:
    del target_tokens
    raw_events = parse_event_lines(stdout)
    hook_status = hook_status or parse_hook_status(stdout)
    hook_set = {item for item in hook_status.split(",") if item and item != "none"}
    stage = next((line.split("=", 1)[1] for line in reversed(stdout.splitlines())
                  if line.startswith("ROUTE_A_STAGE=")), "unknown")
    signatures = sorted({sig for event in raw_events
                         if not event.startswith("kernel_metadata:")
                         if (sig := normalize_kernel_event(event))})
    errors = []
    for event in raw_events:
        if event.startswith("observer_error:"):
            try:
                errors.append(json.loads(event[len("observer_error:"):]).get("where", "observer"))
            except Exception:
                errors.append("observer")

    matches = []
    rejected = []
    observed_targets = []
    selected_matches = []
    if graph_manifest is not None and "per_node_origins" in hook_set:
        scored = score_calls(_event_calls(raw_events), graph_manifest)
        matches = scored["matches"]
        rejected = scored["rejected"]
        observed_targets = scored["observed_targets"]
        selected_matches = scored["selected_matches"]

    measurement_complete = graph_manifest is not None and REQUIRED_HOOKS.issubset(hook_set) and not errors
    measurement_status = (
        "unknown" if graph_manifest is not None and not measurement_complete
        else "complete" if graph_manifest is not None else "legacy"
    )
    selected_hit = bool(selected_matches)
    target_hits = ["target:" + item["target_id"] for item in selected_matches]

    oracle = {}
    for line in stdout.splitlines():
        if line.startswith("ROUTE_A_ORACLE="):
            try:
                oracle = json.loads(line.split("=", 1)[1])
            except Exception:
                oracle = {"status": "malformed"}

    if timeout:
        status, reason = "timeout", "timeout"
    elif returncode != 0:
        status = {"inputs": "input_failure", "eager": "eager_failure",
                  "compile": "compile_failure", "oracle": "oracle_failure"}.get(
                      stage, "execution_failure")
        reason = "failure_during_" + stage
    elif graph_manifest is not None and measurement_status == "unknown":
        status, reason = "unknown", "observer_measurement_incomplete"
    elif graph_manifest is not None and selected_hit:
        status, reason = "hit", "selected_target_computation_confirmed"
    elif graph_manifest is not None:
        status, reason = "miss", "completed_with_no_confirmed_target_computation"
    elif not raw_events:
        status, reason = "unknown", "legacy_observation_without_events"
    else:
        status, reason = "unknown", "legacy_observation_not_scored"

    selected_behaviors = sorted({item["behavior"] for item in selected_matches})
    return Observation(
        status=status,
        returncode=returncode,
        elapsed_seconds=0.0,
        raw_events=raw_events,
        signatures=signatures,
        behavior_signatures=behavior_signatures(raw_events),
        target_hits=target_hits,
        target_operation_hit=selected_hit,
        target_node_hit=selected_hit,
        target_node_evidence=sorted({item["kernel"] for item in selected_matches}),
        target_behavior_signatures=selected_behaviors,
        target_match_level="computation_confirmed" if selected_hit else "none",
        side_hits=sorted(set(observed_targets) - ({selected_target} if selected_target else set())),
        stdout=stdout,
        stderr=stderr,
        hook_status=hook_status,
        status_reason=reason,
        selected_target=selected_target,
        failure_stage=stage if returncode != 0 else None,
        observed_targets=observed_targets,
        match_evidence=matches,
        rejected_evidence=rejected,
        measurement_status=measurement_status,
        measurement_errors=errors,
        oracle=oracle,
    )
