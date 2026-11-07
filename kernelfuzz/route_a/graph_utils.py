"""TorchFuzz graph manifests and conservative condition mutations."""

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _spec_dict(spec: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {"type": type(spec).__name__}
    if hasattr(spec, "size"):
        data["size"] = list(spec.size)
        data["stride"] = list(spec.stride)
        data["dtype"] = str(spec.dtype)
    else:
        data["dtype"] = str(getattr(spec, "dtype", "unknown"))
        data["constant"] = getattr(spec, "constant", None)
    return data


def _node_dict(node: Any) -> Dict[str, Any]:
    return {
        "node_id": node.node_id,
        "op_name": node.op_name,
        "input_nodes": list(node.input_nodes),
        "depth": node.depth,
        "input_specs": [_spec_dict(s) for s in node.input_specs],
        "output_spec": _spec_dict(node.output_spec),
    }


def graph_manifest(graph: Any) -> Dict[str, Any]:
    nodes = [_node_dict(graph.nodes[nid]) for nid in graph.get_topological_order()]
    structural = [
        {
            "node_id": n["node_id"],
            "op_name": n["op_name"],
            "input_nodes": n["input_nodes"],
        }
        for n in nodes
    ]
    structure_payload = json.dumps(structural, sort_keys=True, separators=(",", ":"))
    full_payload = json.dumps(nodes, sort_keys=True, separators=(",", ":"))
    return {
        "root_node_id": graph.root_node_id,
        "node_count": len(nodes),
        "nodes": nodes,
        "structure_signature": hashlib.sha256(structure_payload.encode()).hexdigest(),
        "full_signature": hashlib.sha256(full_payload.encode()).hexdigest(),
    }


def find_matching_nodes(graph: Any, operation_tokens: Iterable[str], *, exact: bool = False) -> List[str]:
    tokens = tuple(t.lower() for t in operation_tokens)
    result: List[str] = []
    for node_id in graph.get_topological_order():
        op_name = str(graph.nodes[node_id].op_name).lower()
        matched = op_name in tokens if exact else any(token in op_name for token in tokens)
        if matched:
            result.append(node_id)
    return result


def _replace_tensor_spec(spec: Any, attribute: str, axis: Optional[int], value: Any) -> Any:
    """Return a changed spec, or the original spec when it is not a tensor."""
    from torchfuzz.tensor_fuzzer import TensorSpec

    if not isinstance(spec, TensorSpec):
        return spec
    size = list(spec.size)
    stride = list(spec.stride)
    dtype = spec.dtype
    if attribute == "shape":
        if axis is None:
            raise ValueError("shape condition requires a valid axis")
        resolved_axis = len(size) - 1 if axis == -1 else axis
        if resolved_axis < 0 or resolved_axis >= len(size):
            raise ValueError("shape condition axis is out of range")
        new_size = int(value)
        if new_size <= 0:
            raise ValueError("shape size must be positive")
        old_size = tuple(size)
        size[resolved_axis] = new_size
        # Keep a legal dense layout for the propagated condition.  Existing
        # non-contiguous layouts are retained by the layout-specific path.
        if tuple(stride) == _contiguous_stride(old_size):
            stride = list(_contiguous_stride(tuple(size)))
    elif attribute == "dtype":
        dtype = value
    elif attribute == "stride":
        if axis is None:
            raise ValueError("stride condition requires a valid axis")
        resolved_axis = len(stride) - 1 if axis == -1 else axis
        if resolved_axis < 0 or resolved_axis >= len(stride):
            raise ValueError("stride condition axis is out of range")
        stride[resolved_axis] = int(value)
    else:
        raise ValueError("unsupported condition attribute: %s" % attribute)
    return TensorSpec(size=tuple(size), stride=tuple(stride), dtype=dtype)


def _contiguous_stride(size: Tuple[int, ...]) -> Tuple[int, ...]:
    stride: List[int] = [1] * len(size)
    running = 1
    for index in range(len(size) - 1, -1, -1):
        stride[index] = running
        running *= max(1, int(size[index]))
    return tuple(stride)


def mutate_graph_condition(graph: Any, node_id: str, attribute: str,
                           axis: Optional[int], value: Any,
                           binding_scope: str = "entry_lineage") -> Tuple[Any, Dict[str, Any]]:
    """Compatibility entry point for the semantic dimension legalizer."""
    from .legalizer import bind_condition
    return bind_condition(graph, node_id, attribute, axis, value, binding_scope)


def validate_graph_specs(graph: Any) -> Tuple[bool, str]:
    """Check graph invariants before invoking TorchFuzz code generation.

    TorchFuzz stores both an operation's input requirements and the output
    specification of its producers.  A condition binder must not silently
    create a graph whose edge specifications disagree.  Operator-specific
    semantic checks remain the responsibility of code generation and runtime
    validation; this function only checks the invariant that can be checked
    generically.
    """
    if not graph.nodes or graph.root_node_id not in graph.nodes:
        return False, "empty_or_missing_root"
    for node_id in graph.get_topological_order():
        node = graph.nodes[node_id]
        output_spec = node.output_spec
        if hasattr(output_spec, "size") and any(int(dimension) <= 0 for dimension in output_spec.size):
            return False, "%s:nonpositive_output_dimension" % node_id
        for input_spec in node.input_specs:
            if hasattr(input_spec, "size") and any(int(dimension) <= 0 for dimension in input_spec.size):
                return False, "%s:nonpositive_input_dimension" % node_id
        if node.op_name in ("matmul", "mm", "bmm", "torch.matmul", "torch.mm", "torch.bmm"):
            if len(node.input_specs) != 2:
                return False, "%s:matmul_arity" % node_id
            left, right = node.input_specs
            if not hasattr(left, "size") or not hasattr(right, "size"):
                return False, "%s:matmul_non_tensor" % node_id
            a, b = tuple(left.size), tuple(right.size)
            if not a or not b or a[-1] != b[-2 if len(b) > 1 else 0]:
                return False, "%s:matmul_contract_dimension" % node_id
            if left.dtype != right.dtype:
                return False, "%s:matmul_dtype" % node_id
            try:
                import torch
                batch = tuple(torch.broadcast_shapes(a[:-2], b[:-2]))
            except RuntimeError:
                return False, "%s:matmul_batch_broadcast" % node_id
            expected = batch + (a[-2:-1] if len(a) > 1 else ()) + (b[-1:] if len(b) > 1 else ())
            if tuple(node.output_spec.size) != expected:
                return False, "%s:matmul_output_shape" % node_id
        if len(node.input_nodes) != len(node.input_specs):
            return False, "%s:input_arity_mismatch" % node_id
        for parent_id, expected in zip(node.input_nodes, node.input_specs):
            if parent_id not in graph.nodes:
                return False, "%s:missing_parent:%s" % (node_id, parent_id)
            actual = graph.nodes[parent_id].output_spec
            if actual != expected:
                return False, "%s:edge_spec_mismatch:%s" % (node_id, parent_id)
    if graph.nodes[graph.root_node_id].output_spec != graph.target_spec:
        return False, "root_target_spec_mismatch"
    return True, "ok"
