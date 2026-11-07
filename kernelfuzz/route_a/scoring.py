"""Common, selection-independent scoring of observed compiler calls.

Scope: the frozen entry/behavior catalog, not all compiler operations. A
source origin alone never establishes that a requested computation survived.
"""
import json
import re
from .targets import target_map

VERSION = 'origins-and-computation-v4'
REQUIRED_HOOKS = {'per_node_origins', 'kernel_definition', 'kernel_call', 'extern_call'}


def aten_name(value):
    text = str(value)
    if text.startswith('aten.'):
        return text.split('.')[1]
    if text.startswith('torch.'):
        return text.split('.')[-1]
    return text


def behavior_signature(call):
    origins = call.get('origins') or []
    features = {
        'family': call['family'],
        'original_aten': sorted({aten_name(x['original_aten']) for x in origins}),
        'lowered_aten': sorted({aten_name(x['lowered_aten']) for x in origins}),
        'pointer_dtypes': sorted(set(re.findall(r'\*(?:fp\d+|bf16|i\d+|u\d+)', call.get('source', '')))),
    }
    return json.dumps(features, sort_keys=True, separators=(',', ':'))


def score_calls(calls, manifest):
    targets = target_map()
    node_targets = {}
    for node in manifest.get('nodes', []):
        for tid, target in targets.items():
            if node['op_name'] in target.operation_names:
                node_targets['var_' + node['node_id']] = (node, tid, target)
    matches = []
    rejected = []
    for call_index, call in enumerate(calls):
        for origin in call.get('origins') or []:
            context = node_targets.get(origin['source_node'])
            if context is None:
                continue
            node, tid, target = context
            original = aten_name(origin['original_aten'])
            lowered = aten_name(origin['lowered_aten'])
            reason = None
            if call.get('family') not in target.kernel_families:
                reason = 'kernel_family_mismatch'
            elif original not in target.aten_ops:
                reason = 'original_operation_mismatch'
            elif tid == 'pointwise' and original != aten_name(node['op_name']):
                reason = 'different_pointwise_origin'
            elif lowered not in target.lowered_ops:
                reason = 'no_requested_computation_in_lowering'
            elif target.reduction_marker and not re.search(target.reduction_marker, call.get('source', '')):
                reason = 'missing_reduction_instruction'
            record = {'target_id': tid, 'node_id': node['node_id'],
                      'call_index': call_index, 'kernel': call['kernel'],
                      'family': call.get('family'), 'original_aten': original,
                      'lowered_aten': lowered}
            if reason:
                rejected.append(dict(record, reason=reason))
            else:
                matches.append(dict(record, behavior=behavior_signature(call)))
    # One test contributes at most one hit per target, regardless of nodes/kernels.
    observed = sorted({x['target_id'] for x in matches})
    entry = manifest.get('entry_node')
    # Evidence records use the graph's node_id; origin keys use var_<node_id>.
    selected_node = str(entry).removeprefix('var_') if entry is not None else None
    selected = [x for x in matches if x['node_id'] == selected_node]
    return {'matches': matches, 'rejected': rejected, 'observed_targets': observed,
            'selected_matches': selected,
            'behaviors': sorted({x['behavior'] for x in matches}),
            'selected_behaviors': sorted({x['behavior'] for x in selected})}
