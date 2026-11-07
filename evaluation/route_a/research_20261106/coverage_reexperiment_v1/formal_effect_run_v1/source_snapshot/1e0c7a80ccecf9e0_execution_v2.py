"""Render a standalone v2 worker; the legacy forward harness is unchanged."""
import json
from pathlib import Path
from .conditions_v2 import Dimension, emit_function, tensor_records


def render_case(graph, seed, condition, evidence, device='cuda', backend='inductor'):
    function, leaves = emit_function(graph, condition, evidence.binding_nodes)
    if condition.dimension == Dimension.DYNAMIC:
        variants = [{s['node']: s for s in family} for family in evidence.family_specs]
        symbols = {}
        for nid, node in graph.nodes.items():
            if node.op_name != 'constant' or not hasattr(node.output_spec, 'size'):
                continue
            for axis in range(len(node.output_spec.size)):
                sizes = tuple(v[nid]['shape'][axis] for v in variants)
                if len(set(sizes)) <= 1:
                    continue
                match = next(((leaf, dim) for leaf in leaves if leaf in variants[0]
                              for dim in range(len(variants[0][leaf]['shape']))
                              if tuple(v[leaf]['shape'][dim] for v in variants) == sizes), None)
                if match is None:
                    from .conditions_v2 import reject
                    reject('dynamic_factory_has_no_input_symbol')
                leaf, dim = match
                expression = 'arg_' + leaf + '.shape[' + str(dim) + ']'
                symbols.setdefault(nid, {})[axis] = expression
                evidence.propagation.append({'node': nid, 'axis': axis, 'rule': 'dynamic_factory_input_symbol', 'expression': expression})
        function, leaves = emit_function(graph, condition, evidence.binding_nodes, symbols)
    specs = {x['node']: x for x in tensor_records(graph)}
    families = evidence.family_specs or [list(specs.values())]
    inputs = []
    for family in families:
        by_node = {x['node']: x for x in family}
        inputs.append([by_node[n] if n in by_node else {'node': n, 'constant': getattr(graph.nodes[n].output_spec, 'constant', None)
                        if getattr(graph.nodes[n].output_spec, 'constant', None) is not None else 1} for n in leaves])
    plan = {'schema': 'route-a-execution-v2.1', 'seed': seed, 'device': device, 'backend': backend,
            'condition': condition.to_dict(), 'input_families': inputs,
            'alias_groups': evidence.alias_groups, 'autograd_mode': evidence.autograd_mode,
            'dynamic': condition.dimension == Dimension.DYNAMIC, 'binding_nodes': evidence.binding_nodes}
    root = Path(__file__).resolve().parents[2]
    oracle = (root / 'evaluation/route_a/unified_oracle.py').read_text()
    runtime = Path(__file__).with_name('runtime_v2.py').read_text()
    code = oracle + '\n' + runtime + '\n' + function + '\nrun_v2(fuzzed_program, ' + repr(plan) + ')\n'
    compile(code, '<condition-v2-worker>', 'exec')
    return code
