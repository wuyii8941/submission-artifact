"""Stable Route A records.

The records deliberately contain both a graph manifest and the rendered
program.  The manifest makes node/condition bindings auditable, while the
program is sufficient for independent replay.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


def _jsonable(value: Any) -> Any:
    """Convert torch/numpy-like scalar objects to stable JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    return str(value)


@dataclass(frozen=True)
class ConditionRequest:
    target_id: str
    attribute: str
    axis: Optional[int] = None
    value: Any = None
    cell: str = "unspecified"
    provenance: str = "route-a-config"
    binding_scope: str = "entry_lineage"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "attribute": self.attribute,
            "axis": self.axis,
            "value": _jsonable(self.value),
            "cell": self.cell,
            "provenance": self.provenance,
            "binding_scope": self.binding_scope,
        }


@dataclass(frozen=True)
class GraphRequest:
    seed: int
    max_depth: int = 3
    template: str = "default"
    supported_ops: Tuple[str, ...] = ()
    target_id: Optional[str] = None
    condition: Optional[ConditionRequest] = None
    max_entry_retries: int = 8
    runtime_device: str = "cuda"
    entry_strategy: str = "bounded_filter"
    target_weight: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "max_depth": self.max_depth,
            "template": self.template,
            "supported_ops": list(self.supported_ops),
            "target_id": self.target_id,
            "condition": self.condition.to_dict() if self.condition else None,
            "max_entry_retries": self.max_entry_retries,
            "runtime_device": self.runtime_device,
            "entry_strategy": self.entry_strategy,
            "target_weight": self.target_weight,
        }


@dataclass
class TestCase:
    test_id: str
    code: str
    graph_manifest: Dict[str, Any]
    conditions: List[Dict[str, Any]]
    execution: Dict[str, Any]
    provenance: Dict[str, Any]
    graph: Any = field(repr=False, default=None)

    def to_dict(self, include_code: bool = False) -> Dict[str, Any]:
        data = {
            "test_id": self.test_id,
            "graph_manifest": self.graph_manifest,
            "conditions": self.conditions,
            "execution": self.execution,
            "provenance": self.provenance,
        }
        if include_code:
            data["code"] = self.code
        return data


@dataclass
class GenerationResult:
    status: str
    elapsed_seconds: float
    attempts: int
    test_case: Optional[TestCase] = None
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.test_case is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "attempts": self.attempts,
            "failure_reason": self.failure_reason,
            "failure_detail": self.failure_detail,
            "test_id": self.test_case.test_id if self.test_case else None,
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class Observation:
    status: str
    returncode: Optional[int]
    elapsed_seconds: float
    raw_events: List[str] = field(default_factory=list)
    signatures: List[str] = field(default_factory=list)
    behavior_signatures: List[str] = field(default_factory=list)
    target_hits: List[str] = field(default_factory=list)
    target_operation_hit: bool = False
    target_node_hit: bool = False
    target_node_evidence: List[str] = field(default_factory=list)
    target_behavior_signatures: List[str] = field(default_factory=list)
    target_match_level: str = "none"
    side_hits: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    evidence_path: Optional[str] = None
    hook_status: str = "unknown"
    status_reason: Optional[str] = None
    selected_target: Optional[str] = None
    failure_stage: Optional[str] = None
    observed_targets: List[str] = field(default_factory=list)
    match_evidence: List[Dict[str, Any]] = field(default_factory=list)
    rejected_evidence: List[Dict[str, Any]] = field(default_factory=list)
    measurement_status: str = "unknown"
    measurement_errors: List[str] = field(default_factory=list)
    oracle: Dict[str, Any] = field(default_factory=dict)
    worker_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "elapsed_seconds": self.elapsed_seconds,
            "raw_events": self.raw_events,
            "signatures": self.signatures,
            "behavior_signatures": self.behavior_signatures,
            "target_hits": self.target_hits,
            "target_operation_hit": self.target_operation_hit,
            "target_node_hit": self.target_node_hit,
            "target_node_evidence": self.target_node_evidence,
            "target_behavior_signatures": self.target_behavior_signatures,
            "target_match_level": self.target_match_level,
            "side_hits": self.side_hits,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "evidence_path": self.evidence_path,
            "hook_status": self.hook_status,
            "status_reason": self.status_reason,
            "selected_target": self.selected_target,
            "failure_stage": self.failure_stage,
            "observed_targets": self.observed_targets,
            "match_evidence": self.match_evidence,
            "rejected_evidence": self.rejected_evidence,
            "measurement_status": self.measurement_status,
            "measurement_errors": self.measurement_errors,
            "oracle": self.oracle,
            "worker_metadata": self.worker_metadata,
        }
