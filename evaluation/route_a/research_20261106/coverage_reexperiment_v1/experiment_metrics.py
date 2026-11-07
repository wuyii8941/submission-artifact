"""Read verified source reports against the immutable642 target universe."""
import hashlib,json
from pathlib import Path
from inventory import ROOT
from qualify_denominator_calls import validate_report
from input_regions import region,actual_facts
from graph_semantics import restore
from graph_context import context


def read(directory,decision,node_id):
    directory=Path(directory);receipt=json.loads((directory/'receipt.json').read_text())
    from attribute_source_audit import attribute
    report=attribute(directory);audit=json.loads((ROOT/'branch_domain_audit_v3/results.json').read_text())
    validation=validate_report(report,audit)
    universe=json.loads((ROOT/'coverage_universe_v1/universe.json').read_text())
    expected_device=json.loads((ROOT/'device_domain_audit_v1/results.json').read_text())
    scope_failures=[]
    for field,expected in [('torch',universe['torch']),('commit',universe['commit']),('device',expected_device['device']),('capability',expected_device['capability']),('common_domain',universe['domain'])]:
        if receipt.get(field)!=expected:scope_failures.append(field)
    if receipt.get('effect_experiment_started') and receipt.get('source_sha256')!=hashlib.sha256((ROOT/'experiment_runtime.py').read_bytes()).hexdigest():scope_failures.append('runtime_implementation')
    targets={(tuple(r['source_identity']),*r['edge']):r['target_id'] for r in universe['targets']}
    excluded={(tuple(r['source_identity']),*r['edge']) for r in universe['excluded']}
    hits=set();outside=[];sides=set();regions=[]
    edge_sides={tuple(r['edge']):r['outcome'] for r in decision['observation']['edges']}
    formal_inventory=json.loads((ROOT/'forward_source_inventory_v1/inventory.json').read_text())
    source_entry=next(r for r in formal_inventory['sources'] if [r['source'][k] for k in ('source','first_line','code_sha256')]==decision['source_identity'])
    for call in report.get('attributed_calls',[]):
        identity=tuple(call['source_identity'])
        for edge in call['hit_outcomes']:
            key=(identity,*edge)
            if key in targets:hits.add(targets[key])
            elif key in excluded:outside.append({'source_identity':list(identity),'edge':edge})
        if call['source_identity']!=decision['source_identity'] or call['dag_node']!=node_id or call['original_aten'] not in decision['actual_aten_candidates']:continue
        exits={edge_sides[tuple(e)] for e in call['hit_outcomes'] if tuple(e) in edge_sides}
        sides|=exits
        if exits:regions.append(call.get('entry_argument_facts'))
    evidence=directory/'source_evidence.json'
    observation={'status':'unknown','reason':'no_unique_actual_source_exit','observed_outcomes':sorted(sides)}
    if len(sides)==1 and validation['valid'] and receipt.get('all_node_eager_specs_verified'):
        observation={'status':'observed','target_id':decision['decision_id'],'outcome':next(iter(sides)),
                     'graph_sha256':receipt['graph_sha256'],'node_id':node_id,
                     'source_evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),'actual_source_input_regions':regions,
                     'input_region':next(iter({region(decision['predicate'],actual_facts(r,source_entry['formal_args'])) for r in regions})) if len({region(decision['predicate'],actual_facts(r,source_entry['formal_args'])) for r in regions})==1 else None}
    original_record=json.loads(Path(receipt['case']).read_text())['graph']
    if observation['status']=='observed':
        observation['context_signature']=context(restore(original_record),node_id,observed_facts=receipt['actual_eager_node_facts'])['signature']
    original=original_record['payload']['nodes']
    registry=json.loads((ROOT/'source_inventory_v2/inventory.json').read_text())['operators']
    public={r['registry_matches'][0]['key']:r['public_operator'] for r in registry}
    fx_names={r['name'] for r in receipt.get('dynamo_fx',[]) if r['op']=='call_function'}
    kinds=sorted({public[r['op_name']] for r in original if r['op_name'] in public and 'var_'+r['node_id'] in fx_names})
    result={'oracle':receipt.get('oracle'),'failure_type':receipt.get('error_type'),'error_message':receipt.get('error'),'runtime_elapsed_seconds':receipt.get('elapsed_seconds'),'runtime_stages':receipt.get('stages'),'receipt_status':receipt['status'],'verified_target_ids':sorted(hits),'successful_execution_target_ids':sorted(hits) if receipt['status']=='executed' else [],
            'operator_kinds_at_backend_entry':kinds,'observation':observation,'domain_validation':validation,
            'scope_failures':scope_failures,'protocol_valid':not scope_failures and validation['valid'] and not outside and receipt.get('source_unchanged_during_run',False),
            'excluded_hits':outside,'raw_calls':report.get('raw_call_count',0),'unattributed_reasons':report.get('unattributed_call_reasons',{})}
    (directory/'attribution.json').write_text(json.dumps(report,indent=2)+'\n');(directory/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
