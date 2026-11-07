"""Evidence-based outcomes; missing observations never prove optimization."""


CATEGORIES = ('target_not_generated', 'condition_unmet', 'graph_illegal',
              'optimized_away', 'normal_lowering_unrecorded',
              'auxiliary_computation_rejected', 'unknown_observation',
              'target_confirmed', 'execution_failure', 'unselected')


def classify_trial(request, generation, observation=None):
    reason = generation.get('failure_reason')
    trace = generation.get('metadata', {}).get('attempt_trace', [])
    flags = {'native_target_missing': sum(t.get('target_present') is False for t in trace),
             'auxiliary_computations_rejected': 0}
    if generation.get('status') != 'valid':
        category = ('target_not_generated' if reason == 'entry_not_found' else
                    'condition_unmet' if reason in ('condition_binding_failed', 'condition_graph_invalid') else
                    'graph_illegal' if reason in ('native_graph_invalid', 'generation_error') else
                    'unknown_observation')
    elif observation is None:
        category = 'unknown_observation'
    else:
        rejected = observation.get('rejected_evidence', [])
        flags['auxiliary_computations_rejected'] = sum(
            r.get('lowered_aten') in ('view', 'copy', 'clone', 'full', 'expand', 'permute') for r in rejected)
        # These two outcomes require independent compiler/FX audit evidence.
        audit = observation.get('lowering_audit', {})
        if observation.get('returncode') != 0:
            category = 'execution_failure'
        elif observation.get('measurement_status') != 'complete':
            category = 'unknown_observation'
        elif not request.get('target_id'):
            category = 'unselected'
        elif observation.get('target_node_hit'):
            category = 'target_confirmed'
        elif audit.get('optimized_away') and audit.get('evidence'):
            category = 'optimized_away'
        elif audit.get('normal_lowering') and audit.get('evidence'):
            category = 'normal_lowering_unrecorded'
        elif flags['auxiliary_computations_rejected']:
            category = 'auxiliary_computation_rejected'
        else:
            category = 'unknown_observation'
    return {'category': category, 'flags': flags,
            'rule': 'No inference of elimination or normal lowering from a missing matcher hit.'}
