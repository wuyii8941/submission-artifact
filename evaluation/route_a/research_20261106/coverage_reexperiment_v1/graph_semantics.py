"""Lossless node semantics for the common, target-independent DAG domain.

The current template domain is explicitly default (single-device forward).
Constant expressions are sampled once by upstream, then frozen as literals.
Bindings may change tensor shape/dtype, never the sampled constant value.
"""
import ast
import copy
import hashlib
import inspect
import random
from pathlib import Path
from frozen_codegen import encode, decode, digest, freeze, render as render_operator
import torch
from torchfuzz.operators import get_operator
from torchfuzz.ops_fuzzer import OperationNode, OperationGraph
from torchfuzz.tensor_fuzzer import TensorSpec


def capture(op_name, node_id, output_spec, render_seed, template):
    if template != 'default':
        raise ValueError('unqualified_template:' + template)
    if op_name == 'arg' or op_name.startswith('arg_'):
        return {'kind': 'argument'}
    operator = get_operator(op_name)
    if operator is None:
        raise ValueError('missing_operator:' + op_name)
    if op_name != 'constant':
        return {'kind': 'operator', 'snapshot': freeze(operator, op_name, render_seed)}
    # codegen uses actual CPU data to choose a literal; do not let its RNG
    # consumption change the subsequent upstream graph sampling sequence.
    state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            random.seed(render_seed)
            code = copy.deepcopy(operator).codegen('var_' + node_id, [], output_spec)
    finally:
        random.setstate(state)
    statement = ast.parse(code).body
    if len(statement) != 1 or not isinstance(statement[0], ast.Assign):
        raise ValueError('unsupported_constant_statement')
    value = statement[0].value
    if isinstance(output_spec, TensorSpec):
        if not (isinstance(value, ast.Call) and ast.unparse(value.func) == 'torch.full'
                and len(value.args) == 2):
            raise ValueError('unsupported_constant_tensor_expression')
        fill = ast.literal_eval(value.args[1])
        kind = 'tensor_constant'
    else:
        # complex(a,b) is upstream's canonical scalar complex spelling.
        if isinstance(value, ast.Call) and ast.unparse(value.func) == 'complex':
            fill = complex(*(ast.literal_eval(arg) for arg in value.args))
        else:
            fill = ast.literal_eval(value)
        kind = 'scalar_constant'
    source = Path(inspect.getsourcefile(operator.codegen.__func__))
    payload = {'kind': kind, 'value': encode(fill), 'template': template,
               'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'original_code': code}
    return {'kind': 'constant', 'payload': payload, 'sha256': digest(payload)}


def render_node(semantics, node_id, input_nodes, output_spec):
    name = 'var_' + node_id
    if semantics['kind'] == 'argument':
        return name + ' = arg_' + node_id
    if semantics['kind'] == 'operator':
        return render_operator(semantics['snapshot'], name,
                               ['var_' + nid for nid in input_nodes], output_spec)
    if semantics['kind'] != 'constant':
        raise ValueError('unknown_node_semantics')
    payload = semantics['payload']
    if digest(payload) != semantics['sha256']:
        raise ValueError('constant_snapshot_identity_changed')
    if hashlib.sha256(Path(payload['source']).read_bytes()).hexdigest() != payload['source_sha256']:
        raise ValueError('constant_source_changed')
    value = decode(payload['value'])
    if payload['kind'] == 'scalar_constant':
        if isinstance(output_spec, TensorSpec):
            raise ValueError('constant_kind_changed')
        return name + ' = ' + repr(value)
    if not isinstance(output_spec, TensorSpec):
        raise ValueError('constant_kind_changed')
    # Dtype binding cannot silently round a sampled value. An exactly
    # representable constant is allowed, otherwise report binding failure.
    converted = torch.tensor(value, dtype=output_spec.dtype, device='cpu').item()
    if converted != value or type(converted) is bool and type(value) is not bool:
        raise ValueError('constant_not_exactly_representable')
    return f'{name} = torch.full({output_spec.size!r}, {value!r}, dtype={output_spec.dtype})'


def manifest(graph):
    nodes = []
    for nid in graph.get_topological_order():
        node = graph.nodes[nid]
        nodes.append({'node_id': nid, 'op_name': node.op_name, 'input_nodes': list(node.input_nodes),
                      'depth': node.depth, 'input_specs': encode(node.input_specs),
                      'output_spec': encode(node.output_spec), 'semantics': copy.deepcopy(node.semantics)})
    payload = {'schema': 'frozen-general-dag-v1', 'root_node_id': graph.root_node_id,
               'target_spec': encode(graph.target_spec), 'nodes': nodes}
    return {'payload': payload, 'sha256': digest(payload)}


def restore(record):
    payload = record['payload']
    if digest(payload) != record['sha256'] or payload['schema'] != 'frozen-general-dag-v1':
        raise ValueError('graph_identity_changed')
    nodes = {}
    for row in payload['nodes']:
        node = OperationNode(row['node_id'], row['op_name'], decode(row['input_specs']),
                             decode(row['output_spec']), list(row['input_nodes']), row['depth'])
        node.semantics = copy.deepcopy(row['semantics'])
        nodes[node.node_id] = node
    return OperationGraph(nodes, payload['root_node_id'], decode(payload['target_spec']))


def topology(graph):
    return (graph.root_node_id, tuple((nid, graph.nodes[nid].op_name, tuple(graph.nodes[nid].input_nodes))
                                    for nid in graph.get_topological_order()))


def transfer_semantics(original, updated):
    if topology(original) != topology(updated):
        raise ValueError('binding_changed_topology')
    for nid in original.nodes:
        updated.nodes[nid].semantics = copy.deepcopy(original.nodes[nid].semantics)
    return updated
