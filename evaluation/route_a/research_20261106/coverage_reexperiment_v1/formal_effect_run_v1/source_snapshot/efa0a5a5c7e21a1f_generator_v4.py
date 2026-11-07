"""Target-blind TorchFuzz generation with explicit structural/input strata.

This is intentionally a generator policy, not a target policy: it never reads
compiler decisions, target operations, predicates or feedback state.  Every
experiment arm receives the same deterministic profile schedule.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import random


class GenerationExhausted(RuntimeError):
    def __init__(self, profile, trace):
        super().__init__('profile_unmet:' + profile)
        self.provenance = {'generator_version': 'v4_target_blind_stratified', 'profile': profile,
                           'attempts': len(trace), 'attempt_trace': trace, 'status': 'profile_unmet'}


@dataclass(frozen=True)
class Profile:
    name: str
    min_nodes: int
    min_multiuser: int = 0
    min_reconvergence: int = 0
    require_noncontiguous_tensor: bool = False
    max_depth: int = 6
    reuse_probability: float = 0.30


PROFILES = (
    Profile('deep_general', min_nodes=6, max_depth=7, reuse_probability=0.25),
    Profile('shared_fanout', min_nodes=6, min_multiuser=1, max_depth=7, reuse_probability=0.80),
    Profile('reconvergent', min_nodes=7, min_reconvergence=1, max_depth=8, reuse_probability=0.90),
    Profile('noncontiguous_tensor', min_nodes=5, require_noncontiguous_tensor=True,
            max_depth=7, reuse_probability=0.35),
)


def profile_for_seed(seed):
    return PROFILES[(seed ^ 0x53445241) % len(PROFILES)]


@contextmanager
def bounded_registry_weights(cap=10.0):
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('invalid_weight_cap')
    from torchfuzz.operators import list_operators
    changed = []
    try:
        for operator in list_operators().values():
            previous = operator.__dict__.get('get_weight')
            method = operator.get_weight
            operator.get_weight = lambda _method=method, **kwargs: min(cap, _method(**kwargs))
            changed.append((operator, previous))
        yield
    finally:
        for operator, previous in changed:
            if previous is None:
                del operator.get_weight
            else:
                operator.get_weight = previous


def _topology_facts(graph):
    users = {node_id: set() for node_id in graph.nodes}
    ancestors = {}
    reconvergence = 0
    for node_id in graph.get_topological_order():
        parents = graph.nodes[node_id].input_nodes
        parent_ancestors = []
        for parent in parents:
            users[parent].add(node_id)
            parent_ancestors.append(ancestors.get(parent, set()) | {parent})
        if any(parents[i] != parents[j] and parent_ancestors[i] & parent_ancestors[j]
               for i in range(len(parents)) for j in range(i + 1, len(parents))):
            reconvergence += 1
        ancestors[node_id] = set().union(*parent_ancestors) if parent_ancestors else set()
    leaves = [node.output_spec for node in graph.nodes.values() if not node.input_nodes and hasattr(node.output_spec, 'size')]
    tensor_specs = [node.output_spec for node in graph.nodes.values() if hasattr(node.output_spec, 'size')]
    noncontiguous = sum(tuple(spec.stride) != _contiguous_stride(tuple(spec.size)) for spec in tensor_specs)
    return {'node_count': len(graph.nodes), 'multiuser_nodes': sum(len(value) > 1 for value in users.values()),
            'reconvergent_nodes': reconvergence, 'tensor_leaf_count': len(leaves),
            'noncontiguous_tensor_nodes': noncontiguous,
            'layout_scope': 'any_tensor_node'}


def _contiguous_stride(size):
    running = 1; result = [1] * len(size)
    for index in range(len(size) - 1, -1, -1):
        result[index] = running
        running *= max(1, int(size[index]))
    return tuple(result)


def _accept(facts, profile):
    return (facts['node_count'] >= profile.min_nodes
            and facts['multiuser_nodes'] >= profile.min_multiuser
            and facts['reconvergent_nodes'] >= profile.min_reconvergence
            and (not profile.require_noncontiguous_tensor or facts['noncontiguous_tensor_nodes'] > 0))


def generate(adapter, request, *, max_attempts=24):
    """Return a graph and complete target-blind rejection provenance.

    Failed quality attempts are part of generation cost and must be retained by
    callers; no target is inserted or queried during selection.
    """
    if request.target_id is not None or request.condition is not None or request.entry_strategy != 'bounded_filter':
        raise ValueError('generation_must_be_target_independent')
    if max_attempts <= 0:
        raise ValueError('positive_max_attempts_required')
    adapter._load()
    import torch
    import torchfuzz.ops_fuzzer as upstream
    from torchfuzz.ops_fuzzer import OperationGraph, OperationNode, fuzz_op, fuzz_spec
    from kernelfuzz.route_a.graph_utils import validate_graph_specs
    profile = profile_for_seed(request.seed)
    trace = []

    def sample_once(sample_seed):
        """One TorchFuzz sample; failures are retryable, never silently repaired."""
        nonlocal profile
        random.seed(sample_seed); torch.manual_seed(sample_seed); upstream._next_arg_id = 0
        target_spec = fuzz_spec(request.template)
        nodes = {}; next_id = 0; reuse_rng = random.Random(sample_seed ^ 0x64616731)

        def node_for(spec, depth, stack_size=0):
            nonlocal next_id
            compatible = [node_id for node_id, node in nodes.items() if node.output_spec == spec]
            if compatible and reuse_rng.random() < profile.reuse_probability:
                return reuse_rng.choice(compatible)
            op_name, input_specs = fuzz_op(spec, depth, stack_size, request.template,
                                           list(request.supported_ops or adapter.supported_ops))
            node_id = f'node_{next_id}'; next_id += 1
            inputs = [node_for(child, max(0, depth - 1), stack_size + index + 1)
                      for index, child in enumerate(input_specs)]
            nodes[node_id] = OperationNode(node_id, op_name, input_specs, spec, inputs, depth)
            return node_id

        with bounded_registry_weights():
            root = node_for(target_spec, profile.max_depth)
        graph = OperationGraph(nodes, root, target_spec)
        facts = _topology_facts(graph)
        static_valid, static_reason = validate_graph_specs(graph)
        if static_valid:
            # This invokes TorchFuzz's own meta/render legality gate.  It is a
            # target-blind generation cost, not a compiler execution.
            from common_render import render
            code = render(graph, sample_seed)
            compile(code, '<v4-generation>', 'exec')
        return graph, facts, static_valid, static_reason

    for attempt in range(1, max_attempts + 1):
        sample_seed = request.seed * 1009 + attempt - 1
        try:
            graph, facts, static_valid, static_reason = sample_once(sample_seed)
        except Exception as exc:
            trace.append({'attempt': attempt, 'sample_seed': sample_seed, 'accepted': False,
                          'sample_error_type': type(exc).__name__, 'sample_error': str(exc)[-300:]})
            continue
        accepted = static_valid and _accept(facts, profile)
        trace.append({'attempt': attempt, 'sample_seed': sample_seed, **facts,
                      'static_valid': static_valid, 'static_reason': static_reason, 'accepted': accepted})
        if accepted:
            return graph, {'generator_version': 'v4_target_blind_stratified', 'profile': profile.name,
                           'profile_spec': profile.__dict__, 'attempts': attempt, 'attempt_trace': trace,
                           'accepted_topology': facts, 'template': request.template}
    raise GenerationExhausted(profile.name, trace)
