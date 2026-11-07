"""Embedded v2 runtime. Keep this file independent of coordinator imports.

Native compiled double backward is deliberately not assumed. second_order
compiles explicit torch.func derivative graphs and compares them to eager
create_graph autograd. AOT callbacks qualify both forward and backward stages.
"""
import json
import re
import traceback
import torch


class DomainRejected(RuntimeError):
    pass


class SemanticDomain(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if any(x in name for x in ('aten.div.', 'aten.floor_divide.', 'aten.remainder.', 'aten.fmod.')) and len(args) >= 2:
            a, b = args[:2]
            if isinstance(a, torch.Tensor) and not (a.dtype.is_floating_point or a.dtype.is_complex):
                zero = bool((b == 0).any().item()) if isinstance(b, torch.Tensor) else b == 0
                if zero:
                    raise DomainRejected('integer_zero_divisor:' + name)
                if a.dtype != torch.bool:
                    minimum = torch.iinfo(a.dtype).min
                    if bool(((a == minimum) & (b == -1)).any().item()):
                        raise DomainRejected('integer_division_overflow:' + name)
        return func(*args, **(kwargs or {}))


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def _finite(value):
    from torch.utils._pytree import tree_leaves
    return all(bool(torch.isfinite(x).all().item()) for x in tree_leaves(value) if isinstance(x, torch.Tensor))


def _loss(value):
    from torch.utils._pytree import tree_leaves
    leaves = [x for x in tree_leaves(value) if isinstance(x, torch.Tensor) and x.dtype.is_floating_point]
    if not leaves:
        raise DomainRejected('scalarized_loss_requires_float_output')
    # A smooth scalarization that exercises nonlinear first and second derivatives.
    return sum((x.square().mean() for x in leaves))


def _make_args(plan, variant):
    torch.manual_seed(plan['seed'] + variant * 104729)
    groups = {n: key for key, names in plan['alias_groups'].items() for n in names}
    shared, args = {}, []
    for s in plan['input_families'][variant]:
        if 'shape' not in s:
            args.append(s.get('constant', 1)); continue
        key = groups.get(s['node'], s['node'])
        dtype = getattr(torch, s['dtype'].split('.')[-1])
        if key not in shared:
            storage = 1 + sum(max(0, d - 1) * st for d, st in zip(s['shape'], s['stride']))
            base = torch.empty(storage, dtype=dtype, device=plan['device'])
            if dtype.is_floating_point:
                base.normal_(mean=0.25, std=0.5)
            elif dtype == torch.bool:
                base.random_(0, 2)
            else:
                base.random_(0, 3)
            shared[key] = base
        x = shared[key].as_strided(s['shape'], s['stride'], s.get('storage_offset', 0))
        if plan['autograd_mode'] != 'none':
            x = x.detach().requires_grad_(True)
        args.append(x)
    return tuple(args)


def _snapshot(args):
    return [{'value': value_digest(x), 'version': x._version} if isinstance(x, torch.Tensor)
            else {'value': value_digest(x)} for x in args]


def _alias_signature(args, output):
    from torch.utils._pytree import tree_leaves
    tensors = [x for x in args if isinstance(x, torch.Tensor)]
    tensors += [x for x in tree_leaves(output) if isinstance(x, torch.Tensor)]
    # Pointer identities are used only for equality, never as novelty signatures.
    return [[i, j] for i, a in enumerate(tensors) for j, b in enumerate(tensors[:i])
            if a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()]


def _autograd_reference(fn, args, mode):
    y = fn(*args)
    tensors = tuple(x for x in args if isinstance(x, torch.Tensor) and x.requires_grad)
    loss = _loss(y)
    if not loss.requires_grad:
        raise DomainRejected('loss_disconnected_from_inputs')
    grads = torch.autograd.grad(loss, tensors, create_graph=mode == 'second_order', allow_unused=True)
    grads = tuple(torch.zeros_like(x) if g is None else g for x, g in zip(tensors, grads))
    result = {'output': y, 'grad': grads}
    if mode == 'second_order':
        total = sum(g.sum() for g in grads) + sum(x.sum() * 0 for x in tensors)
        second = torch.autograd.grad(total, tensors, allow_unused=True)
        result['gradgrad'] = tuple(torch.zeros_like(x) if g is None else g for x, g in zip(tensors, second))
    return result


def _normal_guard(text):
    text = re.sub(r'0x[0-9a-fA-F]+', '<address>', text)
    text = re.sub(r'(check_(?:obj|type)_id\([^\n]*, )\d+(\))', r'\1<object_id>\2', text)
    return re.sub(r'___[a-z_]+_\d+', '<object>', text)


def run_v2(fn, plan):
    evidence = {'schema': plan['schema'], 'condition': plan['condition'], 'autograd_mode': plan['autograd_mode'],
                'oracle_applicability': 'unknown', 'rejection_reasons': [], 'compiler_stages': [],
                'guards': [], 'guard_signatures': [], 'recompile_signature': None, 'backend_invocations': 0,
                'compiler_decisions': [], 'families': [], 'functionalization': [], 'alias_groups': plan['alias_groups'],
                'second_order_route': 'explicit_torch_func_derivative_graph' if plan['autograd_mode'] == 'second_order' else None}
    oracle = {'status': 'inapplicable', 'inapplicable_reasons': []}
    stage = 'inputs'
    def mark(value):
        nonlocal stage
        stage = value
        print('ROUTE_A_STAGE=' + value, flush=True)
    def guards(guard_set):
        rows = []
        for g in guard_set:
            rows.append({'name': _normal_guard(str(g.name)), 'source': str(g.source),
                         'types': sorted(g.guard_types or []), 'code': sorted(_normal_guard(x) for x in (g.code_list or []))})
        rows.sort(key=lambda x: json.dumps(x, sort_keys=True))
        evidence['guards'].append(rows)
        evidence['guard_signatures'].append(hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    def graph_record(gm, compiler_stage):
        nodes = [{'op': n.op, 'target': str(n.target)} for n in gm.graph.nodes]
        payload = {'stage': compiler_stage, 'nodes': nodes}
        evidence['compiler_decisions'].append(dict(payload, signature=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()))
        evidence['compiler_stages'].append(compiler_stage)
    def lower(gm, examples, compiler_stage):
        graph_record(gm, compiler_stage)
        if plan['backend'] == 'eager':
            return gm.forward
        return torch._inductor.compile(gm, examples)
    def backend(gm, examples):
        evidence['backend_invocations'] += 1
        graph_record(gm, 'dynamo')
        if plan['autograd_mode'] in ('first_order', 'aot_autograd'):
            from torch._dynamo.backends.common import aot_autograd
            from functorch.compile import make_boxed_func
            return aot_autograd(
                fw_compiler=lambda g, e: make_boxed_func(lower(g, e, 'aot_forward')),
                bw_compiler=lambda g, e: make_boxed_func(lower(g, e, 'aot_backward')))(gm, examples)
        return lower(gm, examples, 'inductor' if plan['backend'] == 'inductor' else 'test_backend')
    try:
        mark('inputs')
        if plan['device'] == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('requested_cuda_unavailable')
        # Native TorchFuzz emits factories without a device argument. Match its
        # legacy template's default-device contract for constants and helpers.
        torch.set_default_device(plan['device'])
        # Fail before compilation; every reference uses the same fresh input bits,
        # alias graph and strides, so intentional mutation remains applicable.
        references, repeats = [], []
        for i in range(len(plan['input_families'])):
            mark('eager')
            args = _make_args(plan, i)
            before = _snapshot(args)
            with SemanticDomain():
                ref = _autograd_reference(fn, args, plan['autograd_mode']) if plan['autograd_mode'] != 'none' else fn(*args)
            _sync(plan['device'])
            after = _snapshot(args)
            repeat_args = _make_args(plan, i)
            with SemanticDomain():
                repeat = _autograd_reference(fn, repeat_args, plan['autograd_mode']) if plan['autograd_mode'] != 'none' else fn(*repeat_args)
            if not _finite(ref) or not _finite(repeat):
                raise DomainRejected('nonfinite_reference_or_derivative')
            if compare_outputs(ref, repeat)['status'] != 'pass':
                raise DomainRejected('eager_reference_not_repeatable')
            references.append((ref, args, before, after))
            repeats.append(repeat)
        compiled_fn = fn
        if plan['autograd_mode'] == 'second_order':
            argnums = tuple(i for i, x in enumerate(_make_args(plan, 0)) if isinstance(x, torch.Tensor))
            first = torch.func.grad(lambda *a: _loss(fn(*a)), argnums=argnums)
            second = torch.func.grad(lambda *a: sum(g.sum() for g in first(*a)), argnums=argnums)
            def derivatives(*a):
                return {'output': fn(*a), 'grad': first(*a), 'gradgrad': second(*a)}
            compiled_fn = derivatives
        mark('compile')
        # Exactly one compilation wrapper per family, reused for every shape.
        compiled = torch._dynamo.optimize(backend, nopython=True, dynamic=plan['dynamic'], guard_export_fn=guards)(compiled_fn)
        comparisons = []
        for i, (ref, eager_args, eager_before, eager_after) in enumerate(references):
            args = _make_args(plan, i)
            before = _snapshot(args)
            count_before = evidence['backend_invocations']
            result = _autograd_reference(compiled, args, plan['autograd_mode']) if plan['autograd_mode'] in ('first_order', 'aot_autograd') else compiled(*args)
            _sync(plan['device'])
            after = _snapshot(args)
            mark('oracle')
            comparison = compare_outputs(ref, result)
            comparison['input_state'] = compare_outputs(eager_args, args)
            comparison['aliases_eager'] = _alias_signature(eager_args, ref)
            comparison['aliases_compiled'] = _alias_signature(args, result)
            comparison['version_deltas_eager'] = [a.get('version', 0) - b.get('version', 0) for a, b in zip(eager_after, eager_before)]
            comparison['version_deltas_compiled'] = [a.get('version', 0) - b.get('version', 0) for a, b in zip(after, before)]
            # Functionalization may coalesce mutation increments. Check the
            # observable version invariant, not an implementation-specific count.
            comparison['version_invariant'] = [v > 0 for v in comparison['version_deltas_eager']] == [v > 0 for v in comparison['version_deltas_compiled']]
            comparison['alias_invariant'] = comparison['aliases_eager'] == comparison['aliases_compiled']
            names = [s['node'] for s in plan['input_families'][i]]
            comparison['alias_group_versions'] = {
                key: {'eager': [eager_args[names.index(n)]._version for n in members],
                      'compiled': [args[names.index(n)]._version for n in members]}
                for key, members in plan['alias_groups'].items()}
            comparison['shared_version_invariant'] = all(
                len(set(v['eager'])) == len(set(v['compiled'])) == 1
                for v in comparison['alias_group_versions'].values())
            comparison['version_invariant'] &= comparison['shared_version_invariant']
            if comparison['input_state']['status'] != 'pass' or not comparison['version_invariant'] or not comparison['alias_invariant']:
                comparison['status'] = 'candidate'
            if plan['condition']['dimension'] == 'alias_mutation':
                functional_args = _make_args(plan, i)
                functional_before = _snapshot(functional_args)
                functional = torch.func.functionalize(fn)(*functional_args)
                fr = {'output': compare_outputs(ref, functional), 'input_state': compare_outputs(eager_args, functional_args),
                      'before': functional_before, 'after': _snapshot(functional_args),
                      'alias_signature': _alias_signature(functional_args, functional)}
                evidence['functionalization'].append(fr)
                if fr['output']['status'] != 'pass' or fr['input_state']['status'] != 'pass':
                    comparison['status'] = 'candidate'
            evidence['families'].append({'index': i, 'before': before, 'after': after,
                                        'eager_before': eager_before, 'eager_after': eager_after,
                                        'new_compilations': evidence['backend_invocations'] - count_before,
                                        'comparison': comparison})
            comparisons.append(comparison)
        if plan['autograd_mode'] in ('first_order', 'aot_autograd') and not {'aot_forward', 'aot_backward'} <= set(evidence['compiler_stages']):
            raise DomainRejected('aot_forward_backward_not_both_observed')
        evidence['oracle_applicability'] = 'applicable'
        oracle = {'status': 'pass' if all(c['status'] == 'pass' for c in comparisons) else 'candidate',
                  'version': 'route-a-condition-v2.1', 'applicability': 'applicable', 'family_count': len(comparisons)}
        mark('done')
    except DomainRejected as exc:
        evidence['oracle_applicability'] = 'inapplicable'
        evidence['rejection_reasons'].append(str(exc))
        oracle = {'status': 'inapplicable', 'applicability': 'inapplicable', 'inapplicable_reasons': [str(exc)]}
    except Exception as exc:
        evidence['failure'] = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        # An unexpected compiler rejection is an observation, never a semantic
        # anomaly reward. Eager errors are failed generation/domain qualification.
        evidence['rejection_reasons'].append('unexpected_' + stage + '_failure')
        oracle = {'status': 'not_reached', 'applicability': 'unknown', 'reason': 'unexpected_' + stage + '_failure'}
    evidence['recompiles'] = max(0, evidence['backend_invocations'] - 1)
    evidence['recompile_signature'] = json.dumps({'count': evidence['backend_invocations'],
        'per_shape': [x['new_compilations'] for x in evidence['families']],
        'guards': sorted(set(evidence['guard_signatures']))}, sort_keys=True)
    print('ROUTE_A_ORACLE=' + json.dumps(oracle), flush=True)
    print('ROUTE_A_V2_EVIDENCE=' + json.dumps(evidence), flush=True)
