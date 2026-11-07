"""Random compatible typed DAGs, with manifests before and after meta validation.

No post-rejection target insertion or dtype/shape substitution is performed.
An invalid candidate is retained and rejected. A future repair must be submitted
as a new request/program with an explicit parent receipt, never as equivalence.
"""
import hashlib
import json
from pathlib import Path
import random
import time

from .conditions_v3 import SCHEMA, decision, storage_proof
from .graph_utils import _contiguous_stride
from .model import TestCase

TARGETS = {
    'pointwise': ('torch.sin(z)', 'torch.tanh(z)', 'torch.cos(z)'),
    'reduction': ('torch.sum(z, dim=-1)', 'torch.mean(z, dim=-1)', 'torch.amax(z, dim=-1)'),
    'linear': ('torch.mm(z, w)', 'torch.matmul(z, w)', 'torch.bmm(z.unsqueeze(0), w.unsqueeze(0)).squeeze(0)'),
    'normalization': ('torch.softmax(z, dim=-1)', 'torch.log_softmax(z, dim=-1)',
                      'torch.nn.functional.layer_norm(z, (z.shape[-1],))'),
    'indexing': ('torch.gather(z, -1, idx.expand_as(z))', 'torch.index_select(z, -1, idx)', 'z[..., idx]'),
    'view': ('z.reshape(-1).reshape(z.shape)', 'z.permute(1, 0)', 'z.flatten().view(z.shape)'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def input_specs(cell, rng, extent=None):
    m, k = rng.choice((3, 5, 7)), rng.choice((7, 11, 13, 17))
    if cell.shape == 'scalar':
        shape = ()
    elif cell.shape == 'empty':
        shape = (0, k)
    elif cell.shape == 'singleton':
        shape = (1, k)
    elif cell.shape == 'large_reduction':
        shape = (2, rng.choice((2053, 3079, 4099)))
    else:
        shape = (m, k)
    if extent is not None:
        shape = (extent, shape[-1]) if cell.mode == 'dynamic_batch' else (shape[0], extent)
    def spec(name, size, dtype=None, layout=None):
        layout = layout or cell.layout
        stride = _contiguous_stride(size)
        offset = 0
        if layout == 'transpose':
            stride = tuple(reversed(_contiguous_stride(tuple(reversed(size)))))
        elif layout in ('slice', 'as_strided'):
            stride = tuple(2 * x for x in stride)
            offset = int(layout == 'as_strided')
        return dict(node=name, shape=list(size), stride=list(stride), dtype='torch.' + (dtype or cell.dtype),
                    storage_offset=offset, storage=storage_proof(size, stride, offset), layout=layout)
    specs = [spec('x', shape)]
    yshape = (1, shape[-1]) if cell.shape == 'broadcast' else shape
    specs.append(spec('y', yshape, layout='contiguous' if cell.shape == 'broadcast' else cell.layout))
    if cell.family == 'linear':
        specs.append(spec('w', (shape[-1], rng.choice((3, 5, 11))), layout='contiguous'))
    if cell.family == 'indexing':
        # Bounded random indices are an explicit legal integer auxiliary domain.
        size = shape[-1]
        s = spec('idx', (size,), 'int64', 'contiguous')
        s.update(index_upper_bound=size, index_rule='0 <= idx < last_input_extent')
        specs.append(s)
    return specs


def build(cell, seed, capability, *, semantic_system=None, device='cuda', backend='inductor'):
    if semantic_system != 'v3':
        raise ValueError('semantic_domain_v3_requires_explicit_enablement')
    import torch
    started = time.monotonic()
    gate = decision(cell, capability)
    record = {'schema': SCHEMA, 'seed': seed, 'requested_cell': cell.to_dict(), 'capability_decision': gate,
              'pre_manifest': None, 'post_manifest': None, 'target_before_repair': None,
              'repairs': [], 'repair_parent': None, 'actual_executed_domain': None,
              'rejection': None, 'phase': 'capability' if gate['phase'] == 'capability' else 'domain'}
    if not gate['admitted']:
        record['rejection'] = {'phase': gate['phase'], 'reason': gate['reason'], 'category': 'expected_rejection'}
        record['generation_seconds'] = time.monotonic() - started
        return None, record
    rng = random.Random(int(digest([SCHEMA, seed, cell.to_dict()]), 16))
    specs = input_specs(cell, rng)
    nodes = []
    def add(name, expression, parents, target=False):
        op = ('torch.' + expression.split('torch.', 1)[1].split('(', 1)[0]) if 'torch.' in expression else 'v3_arithmetic'
        nodes.append(dict(node=name, node_id=name, op_name=op, expression=expression,
                          inputs=list(parents), selected_target=target))
        return name
    # A variable branching/depth context, sampled using type-compatible smooth ops.
    z = add('mix', 'x * y * 0.25', ('x', 'y'))
    prior = [z]
    for i in range(rng.randint(1, 4)):
        other = rng.choice(prior)
        choices = [(f'torch.sin({z})', (z,)), (f'torch.cos({z})', (z,)),
                   (f'{z} * 0.5 + {other} * 0.125', (z, other))]
        if cell.dtype != 'complex64':
            choices.append((f'torch.tanh({z})', (z,)))
        if cell.context == 'decomposition' and cell.dtype != 'complex64':
            choices.append((f'torch.nn.functional.gelu({z}, approximate="tanh")', (z,)))
        expr, parents = rng.choice(choices)
        z = add('pre_' + str(i), expr, parents)
        prior.append(z)
    add('z', z, (z,))
    target = TARGETS[cell.family][cell.variant]
    parents = ('z', 'w') if cell.family == 'linear' else ('z', 'idx') if cell.family == 'indexing' else ('z',)
    z = add('selected_target', target, parents, True)
    for i in range(rng.randint(1, 3)):
        expr = rng.choice((f'{z} * 0.75', f'torch.sin({z})', f'{z} + {z} * 0.125'))
        if cell.context == 'branched':
            expr = f'{z} * 0.5 + torch.cos({z}) * 0.25'
        z = add('post_' + str(i), expr, (z,))
    args = ', '.join(s['node'] for s in specs)
    lines = [f'def fuzzed_program({args}):']
    if cell.layout == 'mutation':
        lines.append('    x.add_(0.125)')
    if cell.layout == 'view_chain':
        lines.append('    x = x.reshape(-1).view(x.shape)')
    lines += ['    ' + n['node'] + ' = ' + n['expression'] for n in nodes]
    ret = '(' + z + ', x)' if cell.layout in ('shared', 'mutation', 'view_chain') else z
    lines.append('    return ' + ret)
    function = '\n'.join(lines) + '\n'
    manifest = {'inputs': specs, 'nodes': nodes, 'function': function, 'output_node': z,
                'structure_signature': digest([{k: n[k] for k in ('node', 'expression', 'inputs')} for n in nodes]),
                'target': {'node': 'selected_target', 'expression': target, 'family': cell.family, 'variant': cell.variant}}
    record['pre_manifest'] = json.loads(json.dumps(manifest))
    record['target_before_repair'] = {'present': any(n['selected_target'] for n in nodes), 'node': 'selected_target',
                                      'expression': target, 'checked_phase': 'before_meta_and_any_repair'}
    phase = 'meta_validation'
    try:
        ns = {'torch': torch}
        exec(function, ns)
        meta = {}
        propagation = []
        for s in specs:
            meta[s['node']] = torch.empty_strided(s['shape'], s['stride'], dtype=getattr(torch, s['dtype'].split('.')[-1]), device='meta')
        for n in nodes:
            value = eval(n['expression'], ns, meta)
            meta[n['node']] = value
            propagation.append({'node': n['node'], 'shape': list(value.shape), 'stride': list(value.stride()),
                                'dtype': str(value.dtype), 'device_contract': device,
                                'requires_grad_contract': cell.mode in ('first_order', 'second_order', 'aot_autograd')})
        # Validate each dynamic family's entire meta graph, using synchronized RNG
        # for all invariant dimensions (matrix inner equality is explicit).
        families = [specs]
        if cell.mode.startswith('dynamic'):
            families = []
            state = rng.getstate()
            for extent in ((3, 5, 11) if cell.mode == 'dynamic_batch' else (7, 11, 17)):
                rng.setstate(state)
                family = input_specs(cell, rng, extent)
                ns['fuzzed_program'](*[torch.empty_strided(s['shape'], s['stride'], dtype=getattr(torch, s['dtype'].split('.')[-1]), device='meta') for s in family])
                families.append(family)
        manifest['propagation'] = propagation
        manifest['input_families'] = families
        record['post_manifest'] = manifest
        if time.monotonic() - started > 5:
            raise TimeoutError('generation_deadline_5s')
        plan = {'schema': SCHEMA, 'seed': seed, 'cell': cell.to_dict(), 'device': device, 'backend': backend,
                'condition': {'dimension': 'alias_mutation' if cell.layout in ('shared', 'mutation', 'view_chain') else 'semantic_v3',
                              'value': cell.layout, 'schema': SCHEMA},
                'input_families': families, 'alias_groups': {'shared:0': ['x', 'y']} if cell.layout == 'shared' else {},
                'autograd_mode': cell.mode if cell.mode in ('first_order', 'second_order', 'aot_autograd') else 'none',
                'dynamic': cell.mode.startswith('dynamic'), 'binding_nodes': ['selected_target']}
        root = Path(__file__).resolve().parents[2]
        code = (root / 'evaluation/route_a/unified_oracle.py').read_text() + '\n'
        code += Path(__file__).with_name('runtime_v2.py').read_text() + '\n'
        code += Path(__file__).with_name('runtime_v3.py').read_text() + '\n' + function
        code += '\nrun_v3(fuzzed_program, ' + repr(plan) + ')\n'
        compile(code, '<semantic-v3-worker>', 'exec')
        case = TestCase(digest([code, seed]), code, manifest, [cell.to_dict()], plan,
                        {'generator': SCHEMA, 'seed': seed, 'new_program_equivalence_claim': False})
        record['phase'] = 'admitted'
        record['generation_seconds'] = time.monotonic() - started
        return case, record
    except Exception as exc:
        record['phase'] = phase
        record['rejection'] = {'phase': phase, 'category': 'construction_rejection', 'reason': type(exc).__name__ + ':' + str(exc)}
        record['generation_seconds'] = time.monotonic() - started
        return None, record
