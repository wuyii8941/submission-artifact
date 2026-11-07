"""V4 graph/compiler facts, extending the frozen declarative evaluator."""
from constraints import Unknown, evaluate, node_facts as base_node_facts


def _concrete_int(value):
    if type(value) is int: return value
    if isinstance(value, str):
        try: return int(value)
        except ValueError: pass
    return None


def contiguous(shape, stride):
    if len(shape) != len(stride): return None
    if any(type(value) is not int or value < 0 for value in shape): return None
    if any(type(value) is not int or value < 0 for value in stride): return None
    if any(value == 0 for value in shape): return True
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size != 1:
            if step != expected: return False
            expected *= size
    return True


def _add_tensor_facts(facts, prefix, shape, stride, dtype):
    if isinstance(dtype, str): facts[prefix + '.dtype'] = dtype
    if not isinstance(shape, (tuple, list)): return
    facts[prefix + '.rank'] = len(shape)
    sizes = [_concrete_int(value) for value in shape]
    for axis, value in enumerate(sizes):
        if value is not None: facts[f'{prefix}.size.{axis}'] = value
    if not isinstance(stride, (tuple, list)): return
    strides = [_concrete_int(value) for value in stride]
    for axis, value in enumerate(strides):
        if value is not None: facts[f'{prefix}.stride.{axis}'] = value
    flag = contiguous(sizes, strides)
    if flag is not None: facts[prefix + '.is_contiguous'] = flag


def node_facts(graph, node_id):
    """Facts from declared graph specs are predictions, not runtime observation.

    In particular, an intermediate tensor's actual stride may differ from its
    declared spec. Coverage must use compiler or runtime receipts instead.
    """
    facts = base_node_facts(graph, node_id)
    for index, parent in enumerate(graph.nodes[node_id].input_nodes):
        spec = graph.nodes[parent].output_spec
        if hasattr(spec, 'size'):
            _add_tensor_facts(facts, f'input.{index}', spec.size, spec.stride, str(spec.dtype))
    return facts


def compiler_input_facts(names, inputs):
    facts = {}
    for index, name in enumerate(names):
        tensor = inputs.get(name, {})
        if isinstance(tensor, dict):
            _add_tensor_facts(facts, f'input.{index}', tensor.get('shape'), tensor.get('stride'), tensor.get('dtype'))
    return facts


def targets(graph, spec):
    found = []; rejected = []
    for node_id in graph.get_topological_order():
        if str(graph.nodes[node_id].op_name) not in spec['target_ops']: continue
        try:
            if evaluate(spec['applicability'], node_facts(graph, node_id)): found.append(node_id)
            else: rejected.append({'node_id': node_id, 'reason': 'applicability_false'})
        except Unknown as exc:
            rejected.append({'node_id': node_id, 'reason': 'applicability_unknown', 'detail': str(exc)})
    return found, rejected
