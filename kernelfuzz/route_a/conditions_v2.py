"""Opt-in typed conditions. No compiler or scheduler state is implicit here.

Only one dimension is active per request. New dimensions must register a binder
and a domain policy before they can enter the default safe pool.
"""
from dataclasses import asdict, dataclass, field, replace
import ast
from enum import Enum
import hashlib
import json
import time
from typing import Any, Optional, Tuple

from .graph_utils import _contiguous_stride, graph_manifest, find_matching_nodes, validate_graph_specs
from .model import GenerationResult, TestCase
from .targets import target_map

SCHEMA = 'route-a-condition-v2.1'


class Dimension(str, Enum):
    LAYOUT = 'layout'
    DYNAMIC = 'dynamic_shape'
    AUTOGRAD = 'autograd'
    ALIAS = 'alias_mutation'
    CONTROL_FLOW = 'control_flow'
    HIGHER_ORDER = 'higher_order'
    COMPLEX = 'complex'
    LOW_PRECISION = 'low_precision'
    EMPTY = 'empty'
    SCALAR = 'zero_dimensional'
    POWER_TWO = 'power_two_boundary'
    LARGE_REDUCTION = 'large_reduction'


class Applicability(str, Enum):
    APPLICABLE = 'applicable'
    INAPPLICABLE = 'inapplicable'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class Gate:
    applicability: Applicability = Applicability.APPLICABLE
    legal: bool = True
    category: str = 'admitted'
    reasons: Tuple[str, ...] = ()

    def to_dict(self):
        return dict(applicability=self.applicability.value, legal=self.legal,
                    category=self.category, reasons=list(self.reasons))


@dataclass(frozen=True)
class ConditionSpec:
    dimension: Dimension
    value: str
    axis: int = -1
    family: Tuple[int, ...] = ()
    schema: str = SCHEMA

    def __post_init__(self):
        if not isinstance(self.dimension, Dimension) or self.schema != SCHEMA:
            raise ValueError('invalid_condition_schema_or_dimension')
        if len(set(self.family)) != len(self.family):
            raise ValueError('duplicate_family_member')

    def to_dict(self):
        return dict(schema=self.schema, dimension=self.dimension.value, value=self.value,
                    axis=self.axis, family=list(self.family))

    @classmethod
    def from_dict(cls, data):
        return cls(Dimension(data['dimension']), data['value'], data.get('axis', -1),
                   tuple(data.get('family', ())), data['schema'])


@dataclass(frozen=True)
class Capability:
    device: str
    name: str = 'unknown'
    compute_capability: Tuple[int, int] = (0, 0)
    native_bf16: bool = False

    @classmethod
    def detect(cls, device='cuda'):
        import torch
        if device == 'cpu':
            return cls('cpu', 'cpu')
        if not torch.cuda.is_available():
            raise RuntimeError('requested_cuda_unavailable')
        cc = torch.cuda.get_device_capability()
        return cls('cuda', torch.cuda.get_device_name(), tuple(cc), cc[0] >= 8)


@dataclass(frozen=True)
class TensorRecord:
    node: str
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: str
    storage_offset: int = 0
    alias_group: Optional[str] = None
    requires_grad: bool = False

    def to_dict(self):
        return asdict(self)


def tensor_records(graph):
    return [TensorRecord(nid, tuple(n.output_spec.size), tuple(n.output_spec.stride),
                         str(n.output_spec.dtype), alias_group='input:' + nid if not n.input_nodes and n.op_name != 'constant' else None).to_dict()
            for nid, n in sorted(graph.nodes.items()) if hasattr(n.output_spec, 'size')]


@dataclass
class BindingEvidence:
    condition: dict
    gate: dict
    selected_node: str
    binding_nodes: list = field(default_factory=list)
    original_specs: list = field(default_factory=list)
    transformed_specs: list = field(default_factory=list)
    propagation: list = field(default_factory=list)
    changed_nodes: list = field(default_factory=list)
    alias_groups: dict = field(default_factory=dict)
    family_specs: list = field(default_factory=list)
    autograd_mode: str = 'none'
    compiler_stages: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class ConditionRejected(ValueError):
    def __init__(self, gate, evidence=None):
        super().__init__(';'.join(gate.reasons))
        self.gate, self.evidence = gate, evidence


SUPPORTED = {
    Dimension.LAYOUT: ('transpose_dense', 'stride2'),
    Dimension.DYNAMIC: ('family',),
    Dimension.AUTOGRAD: ('first_order', 'second_order', 'aot_autograd'),
    Dimension.ALIAS: ('view', 'inplace', 'shared_inputs'),
}
# Explicit registration, not arbitrary eval or implicitly enabled combinations.
EXTENSIONS = {d.value: {'enabled': False, 'requires': ['binder', 'domain_policy', 'oracle', 'qualification']}
              for d in Dimension if d not in SUPPORTED}


def safe_pool():
    return tuple(ConditionSpec(d, v, family=(8, 16, 24) if d == Dimension.DYNAMIC else ())
                 for d, values in SUPPORTED.items() for v in values)


def reject(reason, category='semantic_inapplicable', legal=True):
    raise ConditionRejected(Gate(Applicability.INAPPLICABLE, legal, category, (reason,)))


def check_storage(shape, stride, limit=4_000_000):
    """Sufficient non-overlap proof, conservatively rejecting exotic layouts."""
    if len(shape) != len(stride) or any(d < 0 for d in shape) or any(s < 0 for s in stride):
        return Gate(Applicability.INAPPLICABLE, False, 'illegal_generation', ('negative_or_malformed_layout',))
    span = 1
    for s, d in sorted(zip(stride, shape)):
        if d > 1:
            if s < span:
                return Gate(Applicability.INAPPLICABLE, False, 'illegal_generation', ('undefined_or_unproven_overlap',))
            span += (d - 1) * s
    if span > limit:
        return Gate(Applicability.INAPPLICABLE, False, 'illegal_generation', ('storage_budget',))
    return Gate()


def static_gate(graph, condition, capability):
    if condition.dimension not in SUPPORTED:
        return Gate(Applicability.INAPPLICABLE, True, 'disabled_extension', ('dimension_not_qualified',))
    if condition.value not in SUPPORTED[condition.dimension]:
        return Gate(Applicability.INAPPLICABLE, False, 'illegal_generation', ('unknown_condition_value',))
    for n in graph.nodes.values():
        s = n.output_spec
        if not hasattr(s, 'size'):
            continue
        layout = check_storage(s.size, s.stride)
        if not layout.legal:
            return layout
        if any(d == 0 for d in s.size):
            return Gate(Applicability.INAPPLICABLE, True, 'disabled_extension', ('empty_tensor_not_qualified',))
        if not s.size and not n.input_nodes and n.op_name != 'constant':
            return Gate(Applicability.INAPPLICABLE, True, 'disabled_extension', ('zero_dimensional_input_not_qualified',))
        if str(s.dtype) == 'torch.bfloat16' and capability.device == 'cuda' and not capability.native_bf16:
            return Gate(Applicability.INAPPLICABLE, True, 'expected_capability_rejection', ('native_bf16_unavailable',))
        if s.dtype.is_complex:
            return Gate(Applicability.INAPPLICABLE, True, 'disabled_extension', ('complex_not_qualified',))
        if condition.dimension == Dimension.AUTOGRAD and str(s.dtype) not in ('torch.float32', 'torch.float64'):
            return Gate(Applicability.INAPPLICABLE, True, 'semantic_inapplicable', ('autograd_requires_fp32_or_fp64',))
    return Gate()


def _ancestors(graph, node):
    result = set()
    def visit(n):
        if n in result:
            return
        result.add(n)
        for p in graph.nodes[n].input_nodes:
            visit(p)
    visit(node)
    return result


def emit_function(graph, condition=None, binding_nodes=(), shape_symbols=None):
    """Emit a pure DAG first; mutations are explicit, local input operations."""
    from torchfuzz.codegen import generate_simple_operation_code
    from torchfuzz.operators import get_operator
    from .legalizer import REDUCTIONS, reduction_axes
    leaves, body = [], []
    for nid in graph.get_topological_order():
        n = graph.nodes[nid]
        var = 'var_' + nid
        if not n.input_nodes and n.op_name != 'constant':
            leaves.append(nid)
            body.append(var + ' = arg_' + nid)
            if condition and condition.dimension == Dimension.ALIAS and nid in binding_nodes:
                body.append(var + ' = ' + var + '.view_as(' + var + ')')
                if condition.value == 'inplace':
                    body.append(var + '.add_(0.25)')
        elif n.op_name in REDUCTIONS:
            axes, keep = reduction_axes(tuple(graph.nodes[n.input_nodes[0]].output_spec.size), tuple(n.output_spec.size))
            dtype = ', dtype=' + str(n.output_spec.dtype) if n.op_name in ('torch.sum', 'torch.mean') else ''
            body.append(f'{var} = {n.op_name}(var_{n.input_nodes[0]}, dim={axes!r}, keepdim={keep!r}{dtype})')
        else:
            op = n.op_name
            if get_operator(op) is None and op.startswith('torch.') and get_operator(op[6:]) is not None:
                op = op[6:]
            if get_operator(op) is None:
                reject('unregistered_operation:' + op, 'illegal_generation', False)
            lines = generate_simple_operation_code(var, ['var_' + p for p in n.input_nodes], op, n.output_spec)
            if shape_symbols and nid in shape_symbols and op == 'constant':
                tree = ast.parse('\n'.join(lines))
                symbols = shape_symbols[nid]
                for call in ast.walk(tree):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'full':
                        call.args[0] = ast.Tuple(elts=[ast.parse(symbols.get(i, repr(d)), mode='eval').body
                                                     for i, d in enumerate(n.output_spec.size)], ctx=ast.Load())
                lines = ast.unparse(ast.fix_missing_locations(tree)).splitlines()
            body.extend(lines)
    ret = 'var_' + graph.root_node_id
    if condition and condition.dimension == Dimension.ALIAS:
        ret = '(' + ret + ', var_' + binding_nodes[0] + ')'
    code = 'def fuzzed_program(' + ', '.join('arg_' + n for n in leaves) + '):\n'
    code += '\n'.join('    ' + line for line in body) + '\n    return ' + ret + '\n'
    return code, leaves


def propagate_meta(graph):
    """Real meta execution defines every transformed edge, including strides."""
    import torch
    from torchfuzz.tensor_fuzzer import TensorSpec
    from torchfuzz.ops_fuzzer import OperationGraph
    code, leaves = emit_function(graph)
    code = code.rsplit('    return ', 1)[0] + '    return locals()\n'
    ns = {'torch': torch}
    exec(compile(code, '<v2-meta>', 'exec'), ns)
    args = []
    for nid in leaves:
        spec = graph.nodes[nid].output_spec
        args.append(torch.empty_strided(spec.size, spec.stride, dtype=spec.dtype, device='meta')
                    if hasattr(spec, 'size') else (spec.constant if spec.constant is not None else 1))
    with torch.device('meta'):
        values = ns['fuzzed_program'](*args)
    nodes, propagation = {}, []
    for nid in graph.get_topological_order():
        n = graph.nodes[nid]
        actual = values['var_' + nid]
        spec = n.output_spec
        if isinstance(spec, TensorSpec):
            if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != tuple(spec.size) or actual.dtype != spec.dtype:
                reject('meta_output_shape_or_dtype:' + nid, 'illegal_generation', False)
            spec = TensorSpec(tuple(actual.shape), tuple(actual.stride()), actual.dtype)
        inputs = [nodes[p].output_spec for p in n.input_nodes]
        nodes[nid] = replace(n, output_spec=spec, input_specs=inputs)
        propagation.append({'node': nid, 'inputs': list(n.input_nodes), 'rule': 'meta_execution',
                            'changed': spec != n.output_spec or inputs != n.input_specs})
    return OperationGraph(nodes, graph.root_node_id, nodes[graph.root_node_id].output_spec), propagation


def bind(graph, selected_node, condition, capability):
    from torchfuzz.tensor_fuzzer import TensorSpec
    from torchfuzz.ops_fuzzer import OperationGraph
    from .legalizer import bind_condition
    gate = static_gate(graph, condition, capability)
    ev = BindingEvidence(condition.to_dict(), gate.to_dict(), selected_node,
                         original_specs=tensor_records(graph))
    if gate.applicability != Applicability.APPLICABLE:
        raise ConditionRejected(gate, ev)
    ancestors = _ancestors(graph, selected_node)
    leaves = sorted(nid for nid in ancestors if not graph.nodes[nid].input_nodes and graph.nodes[nid].op_name != 'constant'
                    and hasattr(graph.nodes[nid].output_spec, 'size'))
    if not leaves:
        reject('no_tensor_input_lineage')
    bound = [leaves[0]]
    if condition.dimension == Dimension.DYNAMIC:
        if len(condition.family) < 2 or any(type(x) is not int or x <= 1 for x in condition.family):
            reject('dynamic_family_requires_distinct_positive_nonunit_sizes', 'illegal_generation', False)
        families = []
        for extent in condition.family:
            variant, diff = bind_condition(graph, selected_node, 'shape', condition.axis, extent)
            variant, propagation = propagate_meta(variant)
            families.append((variant, diff, propagation))
        graph, diff, ev.propagation = families[0]
        ev.family_specs = [tensor_records(g) for g, _, _ in families]
        ev.propagation.append({'rule': 'dimension_equalities', 'members': diff['dimension_members']})
        bound = sorted({n for n, _ in diff['dimension_members']})
    elif condition.dimension == Dimension.LAYOUT:
        n = graph.nodes[bound[0]]
        s = n.output_spec
        if not s.size or max(s.size) <= 1:
            reject('layout_requires_nontrivial_tensor')
        if condition.value == 'transpose_dense':
            if len(s.size) < 2 or sum(d > 1 for d in s.size) < 2:
                reject('transpose_requires_two_nonunit_dimensions')
            stride = tuple(reversed(_contiguous_stride(tuple(reversed(s.size)))))
        else:
            stride = tuple(2 * x for x in _contiguous_stride(s.size))
        layout = check_storage(s.size, stride)
        if not layout.legal:
            raise ConditionRejected(layout, ev)
        nodes = dict(graph.nodes)
        nodes[bound[0]] = replace(n, output_spec=TensorSpec(s.size, stride, s.dtype))
        graph = OperationGraph(nodes, graph.root_node_id, nodes[graph.root_node_id].output_spec)
        graph, ev.propagation = propagate_meta(graph)
    elif condition.dimension == Dimension.AUTOGRAD:
        # Restrict the first qualification to smooth operations with reliable gradgrad.
        smooth = {'arg', 'torch.add', 'torch.sub', 'torch.mul', 'torch.sin', 'torch.cos',
                  'torch.tanh', 'torch.sigmoid', 'torch.sum', 'torch.mean', 'matmul', 'mm',
                  'bmm', 'torch.matmul', 'torch.mm', 'torch.bmm', 'constant'}
        smooth.update(x.removeprefix('torch.') for x in tuple(smooth))
        if any(n.op_name not in smooth for n in graph.nodes.values()):
            reject('autograd_operation_not_qualified')
        if any(not hasattr(graph.nodes[n].output_spec, 'size') for n in leaves):
            reject('autograd_requires_tensor_inputs')
        graph, ev.propagation = propagate_meta(graph)
        ev.autograd_mode = condition.value
        bound = sorted(nid for nid, n in graph.nodes.items() if not n.input_nodes and n.op_name != 'constant'
                       and hasattr(n.output_spec, 'size'))
    else:
        if condition.value == 'inplace' and not graph.nodes[bound[0]].output_spec.dtype.is_floating_point:
            reject('inplace_requires_float_input')
        if condition.value == 'shared_inputs':
            first = graph.nodes[bound[0]].output_spec
            compatible = [n for n in leaves[1:] if graph.nodes[n].output_spec == first]
            if not compatible:
                reject('shared_inputs_requires_matching_leaf_specs')
            bound.append(compatible[0])
        graph, ev.propagation = propagate_meta(graph)
        ev.alias_groups = {'alias:0': bound}
    ev.binding_nodes = bound
    ev.transformed_specs = tensor_records(graph)
    for spec in ev.transformed_specs:
        if condition.dimension == Dimension.AUTOGRAD and spec['node'] in bound:
            spec['requires_grad'] = True
        if condition.dimension == Dimension.ALIAS and spec['node'] in bound:
            spec['alias_group'] = 'alias:0'
    before = {x['node']: x for x in ev.original_specs}
    ev.changed_nodes = sorted({x['node'] for x in ev.transformed_specs if before.get(x['node']) != x}
                              | {x['node'] for x in ev.propagation if x.get('changed')})
    if condition.dimension in (Dimension.AUTOGRAD, Dimension.ALIAS):
        ev.changed_nodes = sorted(set(ev.changed_nodes) | set(bound))
    for specs in [ev.transformed_specs] + ev.family_specs:
        for s in specs:
            g = check_storage(s['shape'], s['stride'])
            if not g.legal:
                raise ConditionRejected(g, ev)
    ok, reason = validate_graph_specs(graph)
    if not ok:
        reject(reason, 'illegal_generation', False)
    return graph, ev


def generate_case(adapter, request):
    """Same native graph generator and target weights, explicit v2 admission."""
    from .execution_v2 import render_case
    started = time.monotonic()
    if request.condition is not None or not isinstance(request.condition_v2, ConditionSpec):
        raise ValueError('v2_requires_one_typed_condition')
    if request.runtime_device not in ('cpu', 'cuda'):
        raise ValueError('unsupported_runtime_device')
    capability = Capability.detect(request.runtime_device)
    trace = []
    last = None
    for attempt in range(1, max(1, request.max_entry_retries) + 1):
        try:
            native_request = replace(request, seed=request.seed * 1009 + attempt - 1,
                                     condition_system='legacy', condition_v2=None)
            graph = adapter._native_graph(native_request)
            original = graph_manifest(graph)
            target = target_map().get(request.target_id)
            matches = find_matching_nodes(graph, target.operation_names, exact=True) if target else [graph.root_node_id]
            if not matches:
                reject('entry_not_found', 'illegal_generation', False)
            selected = matches[-1]
            graph, evidence = bind(graph, selected, request.condition_v2, capability)
            code = render_case(graph, request.seed, request.condition_v2, evidence, request.runtime_device)
            manifest = graph_manifest(graph)
            manifest.update(entry_node=selected, original_full_signature=original['full_signature'],
                            original_structure_signature=original['structure_signature'],
                            condition_v2=evidence.to_dict())
            test_id = hashlib.sha256((code + json.dumps(request.to_dict(), sort_keys=True)).encode()).hexdigest()[:24]
            case = TestCase(test_id, code, manifest, [evidence.to_dict()],
                            {'seed': request.seed, 'condition_system': 'v2', 'runtime_device': request.runtime_device},
                            {'generator': 'torchfuzz', 'request': request.to_dict(), 'attempt': attempt}, graph)
            trace.append({'attempt': attempt, 'gate': Gate().to_dict()})
            return GenerationResult('valid', time.monotonic() - started, attempt, case, metadata={'attempt_trace': trace})
        except ConditionRejected as exc:
            last = exc.gate
            trace.append({'attempt': attempt, 'gate': exc.gate.to_dict(),
                          'evidence': exc.evidence.to_dict() if exc.evidence else None})
        except Exception as exc:
            last = Gate(Applicability.INAPPLICABLE, False, 'illegal_generation', (type(exc).__name__ + ':' + str(exc),))
            trace.append({'attempt': attempt, 'gate': last.to_dict()})
    return GenerationResult('invalid', time.monotonic() - started, len(trace), failure_reason=last.category,
                            failure_detail=';'.join(last.reasons), metadata={'gate': last.to_dict(), 'attempt_trace': trace})
