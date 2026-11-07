"""Adapter around the pinned upstream TorchFuzz API."""

import hashlib
import os
import random
import sys
import time
from typing import Any, Dict, Optional

from .compat import install_torch220_import_shims, supported_ops_default, torch_version
from .graph_utils import (
    find_matching_nodes,
    graph_manifest,
    mutate_graph_condition,
    validate_graph_specs,
)
from .model import ConditionRequest, GenerationResult, GraphRequest, TestCase
from .targets import target_map
from .legalizer import bind_condition
from .render import render_graph
from .sampling import target_weights


_THIRD_PARTY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)


class TorchFuzzAdapter:
    """Generate, optionally bind a condition, and render a replayable test."""

    def __init__(self, supported_ops=None):
        install_torch220_import_shims()
        self.supported_ops = tuple(supported_ops or supported_ops_default())
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torchfuzz  # noqa: F401

        self._loaded = True

    def _supported_ops_for_request(self, request: GraphRequest):
        """Optionally bias TorchFuzz toward a requested graph entry.

        ``bounded_filter`` preserves the native TorchFuzz distribution and is
        the default used for controlled comparisons.  ``target_biased`` is a
        separate feasibility mode: it keeps the target operation aliases and
        a small set of tensor-safe helpers.  Because this changes the graph
        prior, it must not be mixed into a formal ablation without being
        frozen as a common configuration.
        """
        supported = tuple(request.supported_ops or self.supported_ops)
        if request.entry_strategy != "target_biased" or not request.target_id:
            return supported
        target = target_map().get(request.target_id)
        if target is None:
            return supported
        normalized_supported = {name.replace("torch.", "") for name in supported}
        target_ops = [
            name
            for name in target.operation_names
            if name in supported or name.replace("torch.", "") in normalized_supported
        ]
        auxiliaries = [
            name
            for name in (
                "torch.add",
                "torch.mul",
                "torch.relu",
                "torch.sigmoid",
                "torch.tanh",
            )
            if name in supported
        ]
        return tuple(dict.fromkeys(target_ops + auxiliaries)) or supported

    def _native_graph(self, request: GraphRequest):
        self._load()
        import torch
        from torchfuzz.ops_fuzzer import fuzz_operation_graph, fuzz_spec

        random.seed(request.seed)
        torch.manual_seed(request.seed)
        spec = fuzz_spec(request.template)
        target = target_map().get(request.target_id)
        weighted = request.entry_strategy == 'weighted' and target is not None
        with target_weights(target.operation_names if weighted else (),
                            request.target_weight if weighted else 1.0):
            return fuzz_operation_graph(
                spec, max_depth=request.max_depth, seed=request.seed,
                template=request.template,
                supported_ops=list(self._supported_ops_for_request(request)),
            )

    @staticmethod
    def _validate_rendered_code(code: str) -> None:
        """Run syntax validation without executing generated code."""
        compile(code, "<route-a-generated>", "exec")

    @staticmethod
    def _select_condition_node(graph: Any, entry_node: str,
                               condition: ConditionRequest, seed: int) -> str:
        """Select a legal non-directed binding node for arm A1.

        A1 still selects a target entry so that target-selection cost is held
        constant, but its condition is attached to an independently sampled
        tensor node.  This prevents A1 from silently becoming the same as A2.
        The node choice is deterministic for the trial seed and is recorded in
        the binding diff.
        """
        if condition.binding_scope != "random_node":
            return entry_node
        candidates = []
        for node_id in graph.get_topological_order():
            spec = graph.nodes[node_id].output_spec
            if not hasattr(spec, "size"):
                continue
            if condition.attribute == "shape" and len(spec.size) == 0:
                continue
            candidates.append(node_id)
        non_entry = [node_id for node_id in candidates if node_id != entry_node]
        if non_entry:
            candidates = non_entry
        if not candidates:
            raise ValueError("no_legal_random_condition_node")
        return random.Random(seed).choice(candidates)

    def generate(self, request: GraphRequest) -> GenerationResult:
        started = time.monotonic()
        if request.runtime_device != "cuda":
            return GenerationResult(
                status="invalid",
                elapsed_seconds=0.0,
                attempts=0,
                failure_reason="unsupported_runtime_device",
                failure_detail=(
                    "The pinned TorchFuzz template emits CUDA programs; "
                    "use generate-only for CPU-side qualification."
                ),
            )
        if request.entry_strategy not in ("bounded_filter", "target_biased", "weighted"):
            return GenerationResult(
                status="invalid",
                elapsed_seconds=0.0,
                attempts=0,
                failure_reason="unsupported_entry_strategy",
                failure_detail=request.entry_strategy,
            )
        last_reason = "unknown"
        last_detail = ""
        target_defs = target_map()
        attempt_trace = []
        for attempt in range(1, max(1, request.max_entry_retries) + 1):
            attempt_valid = False
            attempt_started = time.monotonic()
            attempt_ops = []
            target_present = None
            try:
                graph = self._native_graph(
                    GraphRequest(
                        seed=request.seed * 1009 + attempt - 1,
                        max_depth=request.max_depth,
                        template=request.template,
                        supported_ops=request.supported_ops,
                        max_entry_retries=request.max_entry_retries,
                        runtime_device=request.runtime_device,
                        target_id=request.target_id,
                        entry_strategy=request.entry_strategy,
                        target_weight=request.target_weight,
                    )
                )
                attempt_ops = sorted({str(n.op_name) for n in graph.nodes.values()})
                requested_target = target_defs.get(request.target_id)
                if requested_target is not None:
                    target_present = bool(find_matching_nodes(graph, requested_target.operation_names, exact=True))
                graph_ok, graph_reason = validate_graph_specs(graph)
                if not graph_ok:
                    last_reason = "native_graph_invalid"
                    last_detail = graph_reason
                    continue
                original_manifest = graph_manifest(graph)
                target = target_defs.get(request.target_id) if request.target_id else None
                entry_node = None
                if target is not None:
                    operation_names = target.operation_names or target.operation_tokens
                    candidates = find_matching_nodes(
                        graph,
                        operation_names,
                        exact=bool(target.operation_names),
                    )
                    if not candidates:
                        last_reason = "entry_not_found"
                        available = sorted(
                            {
                                str(graph.nodes[node_id].op_name)
                                for node_id in graph.get_topological_order()
                            }
                        )
                        last_detail = "target=%s;available_ops=%s" % (
                            target.target_id,
                            ",".join(available[:32]),
                        )
                        continue
                    entry_node = candidates[-1]
                condition_diff: Optional[Dict[str, Any]] = None
                if request.condition is not None and entry_node is not None:
                    condition = request.condition
                    try:
                        binding_node = self._select_condition_node(
                            graph, entry_node, condition,
                            request.seed + attempt - 1,
                        )
                        graph, condition_diff = bind_condition(
                            graph, binding_node, condition.attribute,
                            condition.axis, condition.value,
                            binding_scope=condition.binding_scope,
                        )
                        graph_ok, graph_reason = validate_graph_specs(graph)
                        if not graph_ok:
                            last_reason = "condition_graph_invalid"
                            last_detail = graph_reason
                            continue
                    except Exception as exc:
                        last_reason = "condition_binding_failed"
                        last_detail = "%s: %s" % (type(exc).__name__, exc)
                        continue

                # Static-shape compilation keeps the target observation tied to
                # the concrete graph/condition cell and avoids the known
                # scalar-reduction stalls of PyTorch 2.2 dynamic=True.
                code = render_graph(
                    graph, seed=request.seed * 1009 + attempt - 1, dynamic=False
                )
                self._validate_rendered_code(code)
                manifest = graph_manifest(graph)
                manifest["original_structure_signature"] = original_manifest[
                    "structure_signature"
                ]
                manifest["original_full_signature"] = original_manifest[
                    "full_signature"
                ]
                if entry_node is not None:
                    manifest["entry_node"] = entry_node
                    manifest["entry_op"] = graph.nodes[entry_node].op_name
                if condition_diff is not None:
                    manifest["condition_selected_node"] = condition_diff["entry_node"]
                    manifest["condition_binding_node"] = condition_diff["binding_node"]
                    condition_diff['before_specs'] = original_manifest['nodes']
                    condition_diff['after_specs'] = manifest['nodes']
                manifest['generation_scope'] = {
                    'entry_strategy': request.entry_strategy,
                    'target_weight': request.target_weight,
                    'operator_pool': list(self._supported_ops_for_request(request)),
                    'topology_generator': 'torchfuzz.fuzz_operation_graph',
                }
                test_id = hashlib.sha256(
                    (manifest["full_signature"] + str(request.to_dict())).encode()
                ).hexdigest()[:24]
                conditions = []
                if request.condition is not None:
                    item = request.condition.to_dict()
                    item["entry_node"] = entry_node
                    item["binding_node"] = (
                        condition_diff.get("binding_node") if condition_diff else None
                    )
                    item["binding_diff"] = condition_diff
                    conditions.append(item)
                testcase = TestCase(
                    test_id=test_id,
                    code=code,
                    graph_manifest=manifest,
                    conditions=conditions,
                    execution={
                        "seed": request.seed * 1009 + attempt - 1,
                        "torch_version": torch_version(),
                        "template": request.template,
                        "runtime_device": request.runtime_device,
                        "dynamic_shapes": False,
                    },
                    provenance={
                        "generator": "torchfuzz",
                        "adapter": "route_a_v1",
                        "request": request.to_dict(),
                        "entry_retry": attempt,
                    },
                    graph=graph,
                )
                attempt_valid = True
                return GenerationResult(
                    status="valid",
                    elapsed_seconds=time.monotonic() - started,
                    attempts=attempt,
                    test_case=testcase,
                    metadata={
                        "attempt_trace": attempt_trace,
                        "entry_node": entry_node,
                        "entry_op": manifest.get("entry_op"),
                        "condition_binding_node": manifest.get("condition_binding_node"),
                        "condition_diff": condition_diff,
                        "entry_strategy": request.entry_strategy,
                    },
                )
            except Exception as exc:
                last_reason = "generation_error"
                last_detail = "%s: %s" % (type(exc).__name__, exc)
            finally:
                attempt_trace.append({
                    'attempt': attempt,
                    'seed': request.seed * 1009 + attempt - 1,
                    'status': 'meta_validated' if attempt_valid else 'rejected',
                    'reason': None if attempt_valid else last_reason,
                    'detail': None if attempt_valid else last_detail,
                    'operators': attempt_ops,
                    'target_present': target_present,
                    'elapsed_seconds': time.monotonic() - attempt_started,
                })
        return GenerationResult(
            status="invalid",
            elapsed_seconds=time.monotonic() - started,
            attempts=max(1, request.max_entry_retries),
            failure_reason=last_reason,
            failure_detail=last_detail,
            metadata={'attempt_trace': attempt_trace},
        )
