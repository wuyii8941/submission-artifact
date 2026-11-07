"""Small declarative predicate evaluator over explicit graph/tensor facts.

This interpreter never evaluates arbitrary Python. Unknown properties and
unsupported operations are explicit outcomes, not a negative branch hit.
"""

class Unknown(Exception):
    pass


def evaluate(expression, facts):
    if isinstance(expression, (bool, int, float, str)) or expression is None:
        return expression
    if not isinstance(expression, dict) or set(expression) != {'op', 'args'}:
        raise Unknown('invalid_expression')
    op, args = expression['op'], expression['args']
    if op == 'fact':
        if len(args) != 1 or args[0] not in facts:
            raise Unknown('missing_fact:' + str(args))
        return facts[args[0]]
    values = [evaluate(x, facts) for x in args]
    try:
        if op == 'eq': return values[0] == values[1]
        if op == 'ne': return values[0] != values[1]
        if op == 'lt': return values[0] < values[1]
        if op == 'le': return values[0] <= values[1]
        if op == 'gt': return values[0] > values[1]
        if op == 'ge': return values[0] >= values[1]
        if op == 'divisible': return values[0] % values[1] == 0
        if op == 'in': return values[0] in values[1:]
        if op == 'and': return all(values)
        if op == 'or': return any(values)
        if op == 'not': return not values[0]
    except (TypeError, ValueError, ZeroDivisionError, IndexError) as exc:
        raise Unknown('invalid_operands:' + str(exc)) from exc
    raise Unknown('unsupported_operator:' + str(op))


def node_facts(graph, node_id):
    node = graph.nodes[node_id]
    facts = {'node.op': str(node.op_name),
             'node.num_users': sum(node_id in x.input_nodes for x in graph.nodes.values())}
    for index, parent in enumerate(node.input_nodes):
        spec = graph.nodes[parent].output_spec
        prefix = 'input.' + str(index)
        if hasattr(spec, 'size'):
            facts[prefix + '.rank'] = len(spec.size)
            facts[prefix + '.dtype'] = str(spec.dtype)
            for axis, value in enumerate(spec.size):
                facts[prefix + '.size.' + str(axis)] = int(value)
            for axis, value in enumerate(spec.stride):
                facts[prefix + '.stride.' + str(axis)] = int(value)
    return facts
