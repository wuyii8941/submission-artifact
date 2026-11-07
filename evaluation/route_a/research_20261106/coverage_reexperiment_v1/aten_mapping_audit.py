"""Actual dispatch/schema controls for every common operator.

These isolated controls qualify mappings, not automatic generation capability
or complete reachability. Nominal public names are not treated as ATen IDs.
"""
import copy
import hashlib
import json
import random
from inventory import ROOT, describe
from frozen_codegen import encode, freeze, render
from dag_render import scalar_value
import torch
import torch._inductor.decomposition as decomp
import torch._inductor.lowering as lowering
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode
from torchfuzz.operators import list_operators
from torchfuzz.tensor_fuzzer import TensorSpec


def fixture(entry, index):
    key = entry['registry_matches'][0]['key']
    op = copy.deepcopy(list_operators()[key]); random.seed(2026110600 + index)
    output = None
    for dtype in (torch.float32, torch.int64):
        for shape in ((), (3,), (2,3), (2,3,4)):
            stride = []; running = 1
            for n in reversed(shape):
                stride.insert(0, running); running *= max(1,n)
            spec = TensorSpec(shape, tuple(stride), dtype)
            if op.can_produce(spec): output = spec; break
        if output is not None: break
    if output is None: raise ValueError('no_control_output_spec')
    inputs = op.fuzz_inputs_specs(output)
    state = freeze(op, key, 202611066000 + index)
    names = [f'input_{i}' for i in range(len(inputs))]
    emitted = render(state, 'probe_output', names, output)
    code = 'def probe(' + ', '.join(names) + '):\n'
    code += '\n'.join('    ' + line for line in emitted.splitlines()) + '\n    return probe_output\n'
    namespace = {'torch': torch}; exec(compile(code, '<aten-schema-control>', 'exec'), namespace)
    return namespace['probe'], inputs, output, code, state


def arguments(specs, device='cpu'):
    result = []
    for spec in specs:
        if isinstance(spec, TensorSpec):
            t = torch.empty_strided(spec.size, spec.stride, dtype=spec.dtype, device=device)
            t.fill_(1); result.append(t)
        else: result.append(scalar_value(spec))
    return result


def providers(op):
    result = []
    for stage, table in [('inductor_decomposition', decomp.decompositions), ('inductor_lowering', lowering.lowerings)]:
        key = op if op in table else op.overloadpacket if op.overloadpacket in table else None
        if key is not None:
            result.append({'stage': stage, 'registration_key': str(key), 'callable_chain': describe(table[key])})
    return result


class Dispatch(TorchDispatchMode):
    def __init__(self): super().__init__(); self.calls = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls.append({'aten': str(func), 'schema': str(func._schema)})
        return func(*args, **(kwargs or {}))


def main():
    out = ROOT / 'aten_mapping_controls_v1'; out.mkdir(exist_ok=False)
    inventory = json.loads((ROOT / 'source_inventory_v2/inventory.json').read_text())
    rows = []
    for index, entry in enumerate(inventory['operators']):
        row = {'operator': entry['public_operator'], 'control_only': True,
               'complete_mapping_proven': False, 'compiled_cuda_mapping_proven': False}
        try:
            fn, specs, output, code, state = fixture(entry, index)
            args = arguments(specs); dispatch = Dispatch()
            with dispatch: actual = fn(*args)
            graph = make_fx(fn)(*args)
            aten_nodes = []
            for node in graph.graph.nodes:
                if node.op == 'call_function' and hasattr(node.target, '_schema'):
                    aten_nodes.append({'node': node.name, 'aten': str(node.target),
                                       'schema': str(node.target._schema), 'providers': providers(node.target)})
            nominal = {r['aten_overload'] for r in entry['aten_candidates']}
            row.update(status='observed', code=code, operator_snapshot=state,
                       input_specs=encode(specs), output_spec=encode(output),
                       dispatch_calls=dispatch.calls, aten_graph=aten_nodes,
                       aten_graph_code=graph.code,
                       observed_aten_outside_nominal_inventory=sorted({r['aten'] for r in aten_nodes} - nominal),
                       output_matches_proposal=isinstance(actual,torch.Tensor) and actual.shape==output.size and actual.dtype==output.dtype)
        except Exception as exc:
            row.update(status='error', error_type=type(exc).__name__, error=str(exc)[-2000:])
        rows.append(row)
    report = {'schema': 'actual-aten-mapping-controls-v1', 'rows': rows,
              'scope': 'one CPU isolated fixture per public operator; dispatch plus make_fx, not compiled CUDA coverage',
              'all55retained': len(rows)==55, 'denominator_committed': False, 'effect_experiment_started': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = {'operators': len(rows), 'observed': sum(r['status']=='observed' for r in rows),
               'errors': [r['operator'] for r in rows if r['status']=='error'],
               'nominal_mapping_mismatches': {r['operator']: r['observed_aten_outside_nominal_inventory'] for r in rows if r.get('observed_aten_outside_nominal_inventory')},
               'invalid_fixture_proposals': [r['operator'] for r in rows if r.get('output_matches_proposal') is False],
               'denominator_committed': False, 'effect_experiment_started': False}
    (out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


from pathlib import Path
if __name__ == '__main__': main()
