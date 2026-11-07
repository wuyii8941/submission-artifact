"""Small, version-scoped Route A target and condition registry.

Targets are operation-entry descriptors, not claims about the complete
Inductor operation space.  The list is intentionally easy to audit and must
be regenerated when the compiler/observer version changes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TargetDefinition:
    target_id: str
    operation_tokens: Tuple[str, ...]
    operation_names: Tuple[str, ...]
    condition_attributes: Tuple[str, ...]
    shape_cells: Tuple[int, ...]
    dtype_cells: Tuple[str, ...]
    aten_ops: Tuple[str, ...] = ()
    lowered_ops: Tuple[str, ...] = ()
    kernel_families: Tuple[str, ...] = ()
    reduction_marker: str = ""
    source_reference: str = "route_a_v1_catalog"
    matcher_version: str = "kernel-event-v1"

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_id": self.target_id,
            "operation_tokens": list(self.operation_tokens),
            "operation_names": list(self.operation_names),
            "condition_attributes": list(self.condition_attributes),
            "shape_cells": list(self.shape_cells),
            "dtype_cells": list(self.dtype_cells),
            "aten_ops": list(self.aten_ops),
            "lowered_ops": list(self.lowered_ops),
            "kernel_families": list(self.kernel_families),
            "reduction_marker": self.reduction_marker,
            "source_reference": self.source_reference,
            "matcher_version": self.matcher_version,
        }


def _load_catalog() -> Tuple[TargetDefinition, ...]:
    path = Path(__file__).with_name("catalogs") / "inductor_torch220.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    observer_version = str(data.get("observer_version", "unknown"))
    source_reference = str(path.relative_to(Path(__file__).parent))
    return tuple(
        TargetDefinition(
            target_id=str(item["target_id"]),
            operation_tokens=tuple(str(value) for value in item["operation_tokens"]),
            operation_names=tuple(
                str(value) for value in item.get("operation_names", [])
            ),
            condition_attributes=tuple(
                str(value) for value in item["condition_attributes"]
            ),
            shape_cells=tuple(int(value) for value in item["shape_cells"]),
            dtype_cells=tuple(str(value) for value in item["dtype_cells"]),
            aten_ops=tuple(str(value) for value in item.get("aten_ops", item.get("operation_tokens", []))),
            lowered_ops=tuple(str(value) for value in item.get("lowered_ops", item.get("operation_tokens", []))),
            kernel_families=tuple(str(value) for value in item.get("kernel_families", ())),
            reduction_marker=str(item.get("reduction_marker", "")),
            source_reference=source_reference,
            matcher_version=observer_version,
        )
        for item in data["targets"]
    )


DEFAULT_TARGETS: Tuple[TargetDefinition, ...] = _load_catalog()


def target_map() -> Dict[str, TargetDefinition]:
    return {target.target_id: target for target in DEFAULT_TARGETS}


def condition_cells(target: TargetDefinition) -> List[Tuple[str, str, object, str]]:
    cells: List[Tuple[str, str, object, str]] = []
    for attribute in target.condition_attributes:
        if attribute == "shape":
            for value in target.shape_cells:
                cells.append((attribute, "shape_%s" % value, int(value), "axis_last"))
        elif attribute == "dtype":
            for value in target.dtype_cells:
                cells.append((attribute, "dtype_%s" % value.replace("torch.", ""), value, "all"))
    return cells


def all_condition_cells() -> List[Tuple[str, str, object, str]]:
    """Return the fixed non-directed condition prior used by arm A1.

    The prior is deliberately independent of the selected target.  A1 still
    binds a sampled cell only when it is legal for the selected entry; an
    incompatible binding is recorded as a generation failure rather than
    silently turning A1 into target-aware condition sampling.
    """
    cells: List[Tuple[str, str, object, str]] = []
    seen = set()
    for target in DEFAULT_TARGETS:
        for attribute, cell, value, binding in condition_cells(target):
            key = (attribute, cell, str(value), binding)
            if key not in seen:
                cells.append((attribute, cell, value, binding))
                seen.add(key)
    return cells
