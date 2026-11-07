"""Generic binding actions with frozen topology/semantics and meta checking.

No compiler decision is consulted here. Random and guided policies use the
same actions. Unimplemented relations or illegal proposals are failures,
never exclusions from the source coverage universe.
"""
import copy
from dataclasses import replace
import time
from graph_semantics import manifest, topology, transfer_semantics
from dag_render import function, validate, semantic_signature
from relations import bind_shape
from common_domain import validate_dtype, validate_graph, SHAPE_VALUES
import torch
from torchfuzz.tensor_fuzzer import TensorSpec, ScalarSpec
from torchfuzz.ops_fuzzer import OperationGraph


def propose(graph, node_id, attribute, value, *, input_index=0, axis=-1):
    if attribute == 'shape':
        if type(value) is not int or value not in SHAPE_VALUES:raise ValueError('shape_action_outside_common_domain')
        # Existing generic dimension-equality rules are conservative. Zero
        # extents remain valid generated inputs, but this rule currently
        # reports them unsupported rather than removing them from the domain.
        updated, diff = bind_shape(graph, node_id, input_index, axis, value)
        return transfer_semantics(graph, updated), diff
    if attribute == 'dtype':
        dtype = getattr(torch, value.removeprefix('torch.')) if isinstance(value, str) else value
        if not isinstance(dtype, torch.dtype):
            raise ValueError('invalid_dtype')
        validate_dtype(dtype)
        nodes = {}
        for nid in graph.get_topological_order():
            old = graph.nodes[nid]; spec = old.output_spec
            if isinstance(spec, TensorSpec):
                spec = TensorSpec(spec.size, spec.stride, dtype)
            elif isinstance(spec, ScalarSpec):
                spec = ScalarSpec(dtype, spec.constant)
            else:
                raise ValueError('unsupported_spec_type')
            nodes[nid] = replace(old, input_specs=[nodes[p].output_spec for p in old.input_nodes], output_spec=spec)
        updated = OperationGraph(nodes, graph.root_node_id, nodes[graph.root_node_id].output_spec)
        return transfer_semantics(graph, updated), {'scope': 'whole_graph_dtype', 'value': str(dtype)}
    if attribute == 'layout':
        leaf_id = graph.nodes[node_id].input_nodes[input_index]; leaf = graph.nodes[leaf_id]
        if leaf.semantics['kind'] != 'argument' or not isinstance(leaf.output_spec, TensorSpec):
            raise ValueError('layout_requires_direct_tensor_argument')
        size = leaf.output_spec.size; stride = [1] * len(size); running = 1
        if value not in ('contiguous', 'column_major'):
            raise ValueError('unsupported_layout')
        indices = range(len(size) - 1, -1, -1) if value == 'contiguous' else range(len(size))
        for i in indices:
            stride[i] = running; running *= max(1, size[i])
        updated = copy.deepcopy(graph)
        updated.nodes[leaf_id].output_spec = TensorSpec(size, tuple(stride), leaf.output_spec.dtype)
        for nid in updated.get_topological_order():
            node = updated.nodes[nid]
            node.input_specs = [updated.nodes[p].output_spec for p in node.input_nodes]
        updated.target_spec = updated.nodes[updated.root_node_id].output_spec
        return updated, {'scope': 'one_direct_tensor_argument', 'leaf_id': leaf_id, 'value': value}
    raise ValueError('unsupported_binding_attribute:' + str(attribute))


def bind(graph, node_id, attribute, value, *, input_index=0, axis=-1):
    started = time.monotonic()
    original = manifest(graph)
    record = {'status': 'binding_failed', 'node_id': node_id, 'attribute': attribute,
              'value': str(value) if isinstance(value, torch.dtype) else value,
              'input_index': input_index, 'axis': axis, 'original_graph_sha256': original['sha256']}
    try:
        validate_graph(graph)
        old_code, _ = function(graph)
        updated, diff = propose(graph, node_id, attribute, value, input_index=input_index, axis=axis)
        if topology(graph) != topology(updated):
            raise ValueError('topology_changed')
        for nid in graph.nodes:
            if graph.nodes[nid].semantics != updated.nodes[nid].semantics:
                raise ValueError('stored_node_semantics_changed')
        validate_graph(updated)
        new_code, _ = function(updated)
        if semantic_signature(old_code, attribute) != semantic_signature(new_code, attribute):
            raise ValueError('unrequested_operation_parameters_changed')
        meta = validate(updated, new_code)
        record.update(status='bound', propagation=diff, meta=meta, bound_graph=manifest(updated))
        return updated, record
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc)[-2000:])
        return None, record
    finally:
        if manifest(graph) != original:
            raise RuntimeError('binding_mutated_original_graph')
        record['elapsed_seconds'] = time.monotonic() - started
