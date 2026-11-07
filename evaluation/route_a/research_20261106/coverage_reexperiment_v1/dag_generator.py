"""One general TorchFuzz DAG sampler, shared by every experiment arm.

Uses upstream fuzz_spec/fuzz_op and the existing target-independent structural
profiles. Captures node semantics before child recursion. No target parameter,
operator injection, branch filtering, or target-specific template is allowed.
"""
import random
import time
from inventory import ROOT
from generator_v4 import (GenerationExhausted, profile_for_seed, bounded_registry_weights,
                          _topology_facts, _accept)
from graph_semantics import capture
from dag_render import function, validate
from common_domain import validate_graph
import torch
from torchfuzz.operators import list_operators
from torchfuzz.ops_fuzzer import OperationNode, OperationGraph, fuzz_spec, fuzz_op
from torchfuzz.tensor_fuzzer import TensorSpec
import torchfuzz.ops_fuzzer as upstream


def generate(adapter, request, *, max_attempts=24):
    if request.target_id is not None or request.condition is not None or request.entry_strategy != 'bounded_filter':
        raise ValueError('generation_must_be_target_independent')
    if request.template != 'default':
        raise ValueError('unqualified_template:' + request.template)
    if max_attempts <= 0:
        raise ValueError('positive_attempt_count_required')
    adapter._load()
    profile = profile_for_seed(request.seed); trace = []; started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        sample_seed = request.seed * 1009 + attempt - 1
        entry = {'attempt': attempt, 'sample_seed': sample_seed, 'accepted': False}
        tick = time.monotonic()
        try:
            random.seed(sample_seed); torch.manual_seed(sample_seed); upstream._next_arg_id = 0
            target = fuzz_spec(request.template); nodes = {}; next_id = 0
            reuse_rng = random.Random(sample_seed ^ 0x64616731)
            layout_rng = random.Random(sample_seed ^ 0x6c61796f7574)
            layout_choices = []
            def node_for(spec, depth, stack_size=0):
                nonlocal next_id
                compatible = [nid for nid, node in nodes.items() if node.output_spec == spec]
                if compatible and reuse_rng.random() < profile.reuse_probability:
                    return reuse_rng.choice(compatible)
                op_name, input_specs = fuzz_op(spec, depth, stack_size, request.template,
                                               list(request.supported_ops or adapter.supported_ops))
                nid = f'node_{next_id}'; next_id += 1
                if (op_name == 'arg' or op_name.startswith('arg_')) and isinstance(spec, TensorSpec):
                    # Layout belongs to argument construction. Upstream
                    # operator output proposals do not materialize strides.
                    # This independent legal-layout sampler is common to all
                    # profiles/arms, never selected by a compiler condition.
                    layout = layout_rng.choice(('upstream', 'contiguous', 'column_major'))
                    if layout != 'upstream':
                        stride = [1] * len(spec.size); running = 1
                        indices = range(len(spec.size)) if layout == 'column_major' else range(len(spec.size)-1,-1,-1)
                        for i in indices:
                            stride[i] = running; running *= max(1, spec.size[i])
                        spec = TensorSpec(spec.size, tuple(stride), spec.dtype)
                    layout_choices.append({'node_id': nid, 'layout': layout})
                semantics = capture(op_name, nid, spec, (sample_seed << 20) + next_id, request.template)
                inputs = [node_for(child, max(0, depth - 1), stack_size + i + 1)
                          for i, child in enumerate(input_specs)]
                node = OperationNode(nid, op_name, [nodes[p].output_spec for p in inputs], spec, inputs, depth)
                node.semantics = semantics; nodes[nid] = node
                return nid
            with bounded_registry_weights():
                root = node_for(target, profile.max_depth)
            graph = OperationGraph(nodes, root, nodes[root].output_spec)
            validate_graph(graph)
            facts = _topology_facts(graph)
            code, _ = function(graph)
            meta = validate(graph, code)
            # A stratum requiring a noncontiguous layout uses actual meta
            # evidence, never arbitrary intermediate proposal strides.
            facts['noncontiguous_tensor_nodes'] = sum(not f['contiguous'] for f in meta['node_facts'].values() if f['kind'] == 'tensor')
            facts['layout_scope'] = 'meta_observed_any_tensor_node'
            accepted = _accept(facts, profile)
            entry.update(facts=facts, semantic_validation=meta['status'], argument_layout_choices=layout_choices, accepted=accepted)
        except Exception as exc:
            entry.update(error_type=type(exc).__name__, error=str(exc)[-1800:])
        entry['elapsed_seconds'] = time.monotonic() - tick; trace.append(entry)
        if entry['accepted']:
            registry = list_operators()
            return graph, {'generator_version': 'frozen_general_dag_v1', 'profile': profile.name,
                           'profile_spec': profile.__dict__, 'attempts': attempt, 'attempt_trace': trace,
                           'accepted_topology': facts, 'meta': meta, 'template': request.template,
                           'registry_pool': [{'key': key, 'public_operator': op.torch_op_name} for key, op in registry.items()
                                             if op.torch_op_name in (request.supported_ops or adapter.supported_ops) or op.torch_op_name is None],
                           'elapsed_seconds': time.monotonic() - started}
    raise GenerationExhausted(profile.name, trace)
