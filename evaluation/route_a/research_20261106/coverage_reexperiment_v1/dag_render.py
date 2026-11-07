"""Common frozen-operator rendering and exact shape/dtype legality gate.

Intermediate strides are observed, not inferred from an output proposal.
TensorSpec strides materialize argument leaves only; predicates must use
observed intermediate layouts. No BF16/device exclusion is inferred here.
"""
import ast
from functional_domain import validate_functional_source
from graph_semantics import render_node
import torch
from torchfuzz.tensor_fuzzer import TensorSpec, ScalarSpec


def scalar_value(spec):
    if spec.constant is not None:
        return spec.constant
    if spec.dtype.is_complex:
        return complex(1, 0)
    if spec.dtype.is_floating_point:
        return 1.0
    return True if spec.dtype == torch.bool else 1


def storage_size(spec):
    if len(spec.size) != len(spec.stride) or any(type(d) is not int or d < 0 for d in spec.size):
        raise ValueError('invalid_tensor_shape')
    if any(type(s) is not int or s < 0 for s in spec.stride):
        raise ValueError('invalid_tensor_stride')
    storage = 1 + sum(max(0, d - 1) * s for d, s in zip(spec.size, spec.stride))
    if storage > 4_000_000:
        raise ValueError('tensor_storage_budget_exceeded')
    return storage


def function(graph):
    body = []; leaves = []
    if graph.target_spec != graph.nodes[graph.root_node_id].output_spec:
        raise ValueError('target_spec_mismatch')
    for nid in graph.get_topological_order():
        node = graph.nodes[nid]
        if len(node.input_nodes) != len(node.input_specs):
            raise ValueError('edge_arity_mismatch:' + nid)
        for source, expected in zip(node.input_nodes, node.input_specs):
            if graph.nodes[source].output_spec != expected:
                raise ValueError('edge_spec_mismatch:' + nid)
        if isinstance(node.output_spec, TensorSpec):
            storage_size(node.output_spec)
        if node.semantics['kind'] == 'argument':
            if node.input_nodes:
                raise ValueError('argument_has_dependencies')
            leaves.append(nid)
        body.extend(render_node(node.semantics, nid, node.input_nodes, node.output_spec).splitlines())
    source = 'def fuzzed_program(' + ', '.join('arg_' + nid for nid in leaves) + '):\n'
    source += '\n'.join('    ' + line for line in body)
    source += '\n    return var_' + graph.root_node_id + '\n'
    validate_functional_source(source)
    compile(source, '<frozen-general-dag>', 'exec')
    return source, leaves


def validate(graph, source=None, *, device='meta'):
    emitted, leaves = function(graph)
    if source is not None and source != emitted:
        raise ValueError('validation_source_mismatch')
    probe = emitted.rsplit('    return ', 1)[0] + '    return locals()\n'
    namespace = {'torch': torch}
    exec(compile(probe, '<frozen-dag-meta>', 'exec'), namespace)
    args = []
    for nid in leaves:
        spec = graph.nodes[nid].output_spec
        if isinstance(spec, TensorSpec):
            tensor = torch.empty_strided(spec.size, spec.stride, dtype=spec.dtype, device=device)
            if device == 'cpu':
                tensor.fill_(1)
            args.append(tensor)
        else:
            args.append(scalar_value(spec))
    try:
        with torch.device(device):
            actual = namespace['fuzzed_program'](*args)
    except RuntimeError as exc:
        if device == 'meta' and str(exc) == 'Tensor.item() cannot be called on meta tensors':
            # Data-dependent scalar utilities cannot execute on meta. Keep
            # them and use a real CPU fixture, explicitly labelled. Runtime
            # screening on the eventual device remains mandatory.
            return validate(graph, source, device='cpu')
        raise
    facts = {}
    for nid in graph.get_topological_order():
        spec = graph.nodes[nid].output_spec; value = actual['var_' + nid]
        if isinstance(spec, TensorSpec):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != spec.size or value.dtype != spec.dtype:
                raise ValueError('emitted_spec_mismatch:' + nid + ':expected=' + repr(spec) + ':actual=' +
                                 repr((tuple(value.shape), str(value.dtype)) if isinstance(value, torch.Tensor) else type(value).__name__))
            facts[nid] = {'kind': 'tensor', 'size': list(value.shape), 'dtype': str(value.dtype),
                          'stride': list(value.stride()), 'contiguous': value.is_contiguous(),
                          'layout_evidence': device + '_execution_not_output_spec'}
        elif isinstance(spec, ScalarSpec):
            expected = complex if spec.dtype.is_complex else float if spec.dtype.is_floating_point else bool if spec.dtype == torch.bool else int
            if type(value) is not expected:
                raise ValueError('emitted_scalar_type_mismatch:' + nid)
            facts[nid] = {'kind': 'python_scalar', 'python_type': type(value).__name__}
    return {'status': device + '_valid', 'node_facts': facts, 'argument_nodes': leaves,
            'scalar_fixture': 'typed_unit_values_not_full_value_domain',
            'intermediate_layouts_require_runtime_confirmation': True}


def semantic_signature(source, attribute):
    """Erase only explicitly bound literals; retain alpha/dims/flags/fills.

    Generic AST guard: even a codegen choice depending on a bound dtype
    cannot alter an unrelated keyword or operation. Unsupported transforms
    fail conservatively instead of silently repairing the graph.
    """
    tree = ast.parse(source)
    dtype_names = {str(v).removeprefix('torch.') for v in vars(torch).values() if isinstance(v, torch.dtype)}
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if attribute == 'dtype' and isinstance(node.value, ast.Name) and node.value.id == 'torch' and node.attr in dtype_names:
                return ast.copy_location(ast.Name(id='__bound_dtype__', ctx=ast.Load()), node)
            return node
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if attribute == 'shape' and ast.unparse(node.func) == 'torch.full' and node.args:
                if not isinstance(node.args[0], (ast.Tuple, ast.List)):
                    raise ValueError('unsupported_shape_expression')
                node.args[0].elts = [ast.Constant(value='__bound_extent__') for _ in node.args[0].elts]
            return node
    return ast.dump(Normalize().visit(tree), include_attributes=False)
