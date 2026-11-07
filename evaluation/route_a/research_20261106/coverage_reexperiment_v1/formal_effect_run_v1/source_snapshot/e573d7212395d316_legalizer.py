"""Condition propagation using tensor dimension equalities, independent of kernels."""

from dataclasses import replace
from .graph_utils import _contiguous_stride

REDUCTIONS = {'torch.sum', 'torch.mean', 'torch.amax', 'torch.amin'}
MATMULS = {'matmul', 'mm', 'bmm', 'torch.matmul', 'torch.mm', 'torch.bmm'}


def reduction_axes(input_shape, output_shape):
    if not output_shape:
        return tuple(range(len(input_shape))), False
    for axis in range(len(input_shape)):
        if input_shape[:axis] + input_shape[axis + 1:] == output_shape:
            return (axis,), False
        if input_shape[:axis] + (1,) + input_shape[axis + 1:] == output_shape:
            return (axis,), True
    raise ValueError('unsupported_reduction_shape_relation')


def bind_condition(graph, node_id, attribute, axis, value, binding_scope='entry_lineage'):
    from torchfuzz.tensor_fuzzer import TensorSpec, ScalarSpec
    from torchfuzz.ops_fuzzer import OperationGraph
    nodes = graph.nodes
    parent = {}
    for nid, node in nodes.items():
        if isinstance(node.output_spec, TensorSpec):
            for dim in range(len(node.output_spec.size)):
                parent[(nid, dim)] = (nid, dim)

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        if a in parent and b in parent:
            parent[root(a)] = root(b)

    def shape(nid):
        return tuple(getattr(nodes[nid].output_spec, 'size', ()))

    for nid in graph.get_topological_order():
        node = nodes[nid]
        out = shape(nid)
        if node.op_name in MATMULS:
            left, right = node.input_nodes
            a, b = shape(left), shape(right)
            union((left, len(a)-1), (right, len(b)-2 if len(b)>1 else 0))
            if len(a)>1:
                union((left, len(a)-2), (nid, len(out)-(2 if len(b)>1 else 1)))
            if len(b)>1:
                union((right, len(b)-1), (nid, len(out)-1))
            for source in (left, right):
                s = shape(source)
                for i in range(max(0, len(s)-2)):
                    j = len(out) - (int(len(a)>1)+int(len(b)>1)) - (len(s)-2) + i
                    if s[i] != 1:
                        union((source, i), (nid, j))
        elif node.op_name in REDUCTIONS:
            source = node.input_nodes[0]
            reduced, keepdim = reduction_axes(shape(source), out)
            j = 0
            for i in range(len(shape(source))):
                if i not in reduced:
                    union((source, i), (nid, i if keepdim else j))
                    j += 1
        else:
            for source in node.input_nodes:
                s = shape(source)
                for i, extent in enumerate(s):
                    j = len(out)-len(s)+i
                    if j >= 0 and extent == out[j] and extent != 1:
                        union((source, i), (nid, j))

    binding_node = node_id
    bound_axis = axis
    node = nodes[node_id]
    if attribute == 'shape':
        if node.op_name in REDUCTIONS:
            binding_node = node.input_nodes[0]
            reduced, _ = reduction_axes(shape(binding_node), shape(node_id))
            bound_axis = reduced[-1] if axis == -1 else axis
            if bound_axis not in reduced:
                raise ValueError('requested_axis_is_not_reduced')
        elif node.op_name in MATMULS:
            binding_node = node.input_nodes[0]
            contracted_axis = len(shape(binding_node))-1
            bound_axis = contracted_axis if axis == -1 else axis
            if bound_axis != contracted_axis:
                raise ValueError('requested_axis_is_not_contracted')
        else:
            bound_axis = len(shape(binding_node))-1 if axis == -1 else axis
        key = (binding_node, bound_axis)
        if key not in parent or int(value) <= 0:
            raise ValueError('invalid_shape_binding')
        selected_root = root(key)
    elif attribute != 'dtype':
        raise ValueError('unsupported_condition_attribute')

    updated = {}
    changed = []
    for nid in graph.get_topological_order():
        node = nodes[nid]
        spec = node.output_spec
        if isinstance(spec, TensorSpec):
            size = tuple(int(value) if attribute == 'shape' and root((nid, i)) == selected_root else d
                         for i, d in enumerate(spec.size))
            stride = _contiguous_stride(size) if size != spec.size else spec.stride
            spec = TensorSpec(size, stride, value if attribute == 'dtype' else spec.dtype)
            storage = 1 + sum(max(0, d-1)*abs(s) for d, s in zip(size, stride))
            if storage > 4_000_000:
                raise ValueError('tensor_storage_budget_exceeded')
        elif isinstance(spec, ScalarSpec) and attribute == 'dtype':
            spec = ScalarSpec(value, spec.constant)
        inputs = [updated[p].output_spec for p in node.input_nodes]
        updated[nid] = replace(node, output_spec=spec, input_specs=inputs)
        if spec != node.output_spec or inputs != node.input_specs:
            changed.append(nid)
    return OperationGraph(updated, graph.root_node_id, updated[graph.root_node_id].output_spec), {
        'entry_node': node_id, 'binding_node': binding_node, 'axis': bound_axis,
        'attribute': attribute, 'value': str(value) if attribute == 'dtype' else value,
        'binding_scope': binding_scope, 'changed_nodes': changed,
        'requested_axis': axis,
        'effective_scope': 'whole_graph_dtype' if attribute == 'dtype' else 'dimension_equalities',
        'dimension_members': [list(key) for key in sorted(parent)
                              if attribute == 'shape' and root(key) == selected_root],
        'already_satisfied': not changed,
        'layout_policy': 'changed_shapes_contiguous; unchanged_shapes_preserve_stride',
    }
