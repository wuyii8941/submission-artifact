"""v3 input domains and evidence around byte-identical embedded v2 oracles.

This source follows unified_oracle.py and runtime_v2.py in a standalone worker.
Only input materialization is overridden: complex input, offset storage proof,
bounded integer indices, and grad only on qualified floating leaf arguments.
"""
import contextlib
import io
import sys
import time


def _make_args(plan, variant):
    torch.manual_seed(plan['seed'] + variant * 104729)
    groups = {n: key for key, names in plan['alias_groups'].items() for n in names}
    shared, args = {}, []
    for s in plan['input_families'][variant]:
        key = groups.get(s['node'], s['node'])
        dtype = getattr(torch, s['dtype'].split('.')[-1])
        if key not in shared:
            base = torch.empty(s['storage']['allocation_elements'], dtype=dtype, device=plan['device'])
            if 'index_upper_bound' in s:
                base.random_(0, s['index_upper_bound'])
            else:
                base.normal_(mean=0.125, std=0.25)
            shared[key] = base
        x = shared[key].as_strided(s['shape'], s['stride'], s['storage_offset'])
        if plan['autograd_mode'] != 'none' and dtype.is_floating_point:
            x = x.detach().requires_grad_(True)
        args.append(x)
    _v3_materializations.append({'variant': variant, 'last_reported_stage': _v3_stage,
        'inputs': [{'node': s['node'], 'value': value_digest(x), 'storage_offset': x.storage_offset(),
                    'storage_elements': x.untyped_storage().nbytes() // x.element_size(),
                    'requires_grad': x.requires_grad, 'alias_group': groups.get(s['node']),
                    'version': x._version} for s, x in zip(plan['input_families'][variant], args)]})
    return tuple(args)


def run_v3(fn, plan):
    global _v3_materializations, _v3_stage
    _v3_materializations, _v3_stage = [], 'inputs'
    started = time.monotonic()
    # Fixed local reference precision; compiler/eager compare same dtype truth.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from torch._dynamo.utils import counters
    counters.clear()
    terminal = sys.stdout
    class Capture(io.StringIO):
        def write(self, value):
            global _v3_stage
            if value.startswith('ROUTE_A_STAGE='):
                _v3_stage = value.split('=', 1)[1].strip()
            terminal.write(value)
            terminal.flush()
            return super().write(value)
    output = Capture()
    with contextlib.redirect_stdout(output):
        run_v2(fn, plan)
    raw = output.getvalue()
    ev = next(json.loads(line.split('=', 1)[1]) for line in raw.splitlines() if line.startswith('ROUTE_A_V2_EVIDENCE='))
    breaks = dict(counters.get('graph_break', {}))
    failure = ev.get('failure', {})
    message = failure.get('message', '')
    graph_break = bool(breaks) or 'graph_break' in message or 'Graph break' in message
    ev['graph_break'] = {'classification': 'fullgraph_rejection' if graph_break else 'none_observed',
                         'counters': breaks, 'fullgraph_required': True}
    ev['actual_executed_domain'] = [
        {'input_specs': f['before'], 'eager_input_specs': f['eager_before'], 'family': plan['cell']['family'],
         'mode': plan['cell']['mode'], 'layout': plan['cell']['layout'], 'phase': 'compiled_and_oracle',
         'storage_contract': next(x['inputs'] for x in _v3_materializations if x['variant'] == f['index'])}
        for f in ev['families']]
    if not ev['families']:
        ev['actual_executed_domain'] = [{'phase': 'reference_qualified_before_compile' if failure.get('stage') == 'compile' else 'inputs_materialized',
             'variant': i, 'compiled': False, 'input_specs': next(x['inputs'] for x in _v3_materializations if x['variant'] == i)}
             for i in sorted({x['variant'] for x in _v3_materializations})]
    ev['input_materializations'] = _v3_materializations
    ev['requested_cell'] = plan['cell']
    ev['input_plan'] = plan['input_families']
    ev['cost_seconds'] = time.monotonic() - started
    ev['oracle_sources'] = ['unchanged_unified_oracle', 'unchanged_v2_run_v2']
    print('ROUTE_A_V3_EVIDENCE=' + json.dumps(ev), flush=True)
