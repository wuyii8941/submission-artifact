"""Conservative dimension-equality propagation for common DAG operators.

Unsupported operations fail explicitly. Operator semantics are independent of
which compiler decision is being evaluated.
"""
from dataclasses import replace
from kernelfuzz.route_a.graph_utils import _contiguous_stride
from kernelfuzz.route_a.legalizer import reduction_axes

MATRIX = {'mm', 'bmm', 'matmul', 'addmm'}
REDUCTIONS = {'torch.sum', 'torch.mean', 'torch.amax', 'torch.amin'}
POINTWISE = {'add', 'sub', 'mul', 'div', 'torch.add', 'torch.sub', 'torch.mul', 'torch.div',
             'torch.relu', 'torch.sigmoid', 'torch.tanh', 'torch.exp', 'torch.sqrt',
             'torch.abs', 'torch.sin', 'torch.cos', 'torch.neg', 'torch.square',
             'torch.reciprocal', 'torch.rsqrt', 'torch.exp2', 'torch.expm1',
             'torch.log', 'torch.log2', 'torch.log10', 'torch.log1p', 'torch.erf',
             'torch.erfc', 'torch.floor', 'torch.ceil', 'torch.round', 'torch.trunc',
             'torch.sign', 'torch.sinh', 'torch.cosh', 'torch.atan', 'torch.asinh',
             'torch.maximum', 'torch.minimum', 'torch.pow', 'torch.remainder',
             'torch.fmod', 'torch.hypot', 'torch.logaddexp',
             'torch.nn.functional.gelu', 'torch.nn.functional.silu',
             'torch.nn.functional.softmax',
             'torch.nn.functional.elu', 'torch.nn.functional.leaky_relu',
             'torch.nn.functional.softplus', 'torch.nn.functional.hardswish',
             'torch.nn.functional.mish'}


def bind_shape(graph, node_id, input_index, axis, value):
    from torchfuzz.tensor_fuzzer import TensorSpec
    from torchfuzz.ops_fuzzer import OperationGraph
    nodes = graph.nodes
    parent = {(nid, i): (nid, i) for nid, node in nodes.items()
              if isinstance(node.output_spec, TensorSpec) for i in range(len(node.output_spec.size))}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k

    def union(a, b):
        if a not in parent or b not in parent:
            raise ValueError('unsupported_dimension_relation')
        parent[root(a)] = root(b)

    def shape(nid):
        return tuple(getattr(nodes[nid].output_spec, 'size', ()))

    def broadcast(source, output):
        s, o = shape(source), shape(output)
        if len(s) > len(o):
            raise ValueError('unsupported_broadcast_rank')
        for i, size in enumerate(s):
            j = len(o) - len(s) + i
            if size != 1:
                union((source, i), (output, j))

    def matrix(left, right, output):
        a, b, out = shape(left), shape(right), shape(output)
        if not a or not b:
            raise ValueError('invalid_scalar_matmul_relation')
        union((left, len(a)-1), (right, len(b)-2 if len(b)>1 else 0))
        if len(a) > 1:
            union((left, len(a)-2), (output, len(out)-(2 if len(b) > 1 else 1)))
        if len(b) > 1:
            union((right, len(b)-1), (output, len(out)-1))
        for source in (left, right):
            s = shape(source)
            for i in range(max(0, len(s)-2)):
                j = len(out)-int(len(a)>1)-int(len(b)>1)-max(0,len(s)-2)+i
                if s[i] != 1:
                    union((source, i), (output, j))

    for nid in graph.get_topological_order():
        node = nodes[nid]
        op = str(node.op_name)
        if not node.input_nodes:
            continue
        if not isinstance(node.output_spec, TensorSpec) and all(
                not isinstance(nodes[p].output_spec, TensorSpec) for p in node.input_nodes):
            continue
        if op in MATRIX:
            if op == 'addmm':
                if len(node.input_nodes) != 3:
                    raise ValueError('invalid_addmm_arity')
                bias, left, right = node.input_nodes
                matrix(left, right, nid); broadcast(bias, nid)
            else:
                if len(node.input_nodes) != 2:
                    raise ValueError('invalid_matrix_arity')
                matrix(node.input_nodes[0], node.input_nodes[1], nid)
        elif op in REDUCTIONS:
            source = node.input_nodes[0]
            reduced, keep = reduction_axes(shape(source), shape(nid))
            j = 0
            for i in range(len(shape(source))):
                if i not in reduced:
                    union((source, i), (nid, i if keep else j)); j += 1
        elif op in POINTWISE:
            for source in node.input_nodes:
                if isinstance(nodes[source].output_spec, TensorSpec):
                    broadcast(source, nid)
        else:
            raise ValueError('unsupported_operator_relation:' + op)
    target = nodes[node_id]
    if input_index < 0 or input_index >= len(target.input_nodes):
        raise ValueError('input_index_out_of_range')
    bound_node = target.input_nodes[input_index]
    dims = shape(bound_node)
    resolved_axis = axis if axis >= 0 else len(dims) + axis
    key = (bound_node, resolved_axis)
    if key not in parent or not isinstance(value, int) or value <= 0:
        raise ValueError('unsupported_shape_value_or_axis')
    selected = root(key)
    updated = {}; changed = []
    for nid in graph.get_topological_order():
        node = nodes[nid]; spec = node.output_spec
        if isinstance(spec, TensorSpec):
            size = tuple(value if root((nid, i)) == selected else int(d) for i, d in enumerate(spec.size))
            stride = _contiguous_stride(size) if size != spec.size else spec.stride
            spec = TensorSpec(size, stride, spec.dtype)
            if 1 + sum(max(0, d-1)*abs(s) for d, s in zip(size, stride)) > 4_000_000:
                raise ValueError('tensor_storage_budget_exceeded')
        inputs = [updated[p].output_spec for p in node.input_nodes]
        updated[nid] = replace(node, output_spec=spec, input_specs=inputs)
        if spec != node.output_spec or inputs != node.input_specs:
            changed.append(nid)
    return OperationGraph(updated, graph.root_node_id, updated[graph.root_node_id].output_spec), {
        'entry_node': node_id, 'binding_node': bound_node, 'axis': resolved_axis,
        'attribute': 'shape', 'value': value, 'changed_nodes': changed,
        'dimension_members': [list(k) for k in sorted(parent) if root(k) == selected],
        'already_satisfied': not changed}
