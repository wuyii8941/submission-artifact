"""Conservative readback of exact compiler source and FX provenance evidence.

No coverage percentage is produced. A source event must match the actual ATen
registration and a unique originating Dynamo node from this serialized DAG.
Generic wrapper/initialization events remain separate from direct Inductor
operator implementation bodies.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent


def flatten(sources):
    for source in sources:
        yield source
        yield from flatten(source.get('from_node',[]))


def attribute(directory):
    receipt=json.loads((directory/'receipt.json').read_text())
    path=directory/'source_evidence.json'
    if not path.exists():
        return {'status':receipt['status'],'source_observation_started':False}
    evidence=json.loads(path.read_text())
    catalog={(entry['source']['source'],entry['source']['first_line'],entry['source']['code_sha256']):entry
             for entry in evidence['catalog']}
    fx={node['name']:node for node in receipt.get('dynamo_fx',[]) if node['op']=='call_function'}
    original=json.loads(Path(receipt['case']).read_text())['graph']['payload']['nodes']
    nodes={'var_'+node['node_id']:node for node in original}
    rows=[];unknown=Counter();attributed=set();direct=set();entered=set()
    for call in evidence['calls']:
        identity=tuple(call['source_identity']); entry=catalog[identity];meta=call['fx_provenance']
        aten=meta.get('original_aten_id')
        registrations=[r for r in entry['registrations'] if r['aten']==aten]
        sources=[s for s in flatten(meta.get('from_node_structured',[])) if s.get('name') in nodes]
        source_ids={s['name'] for s in sources}
        if not aten:reason='missing_actual_aten'
        elif not registrations:reason='source_registration_does_not_match_actual_aten'
        elif any(edge not in entry['source']['conditional_outcomes'] for edge in call['hit_outcomes']):reason='outcome_not_registered_in_source'
        elif len(source_ids)!=1:reason='missing_or_ambiguous_original_dag_node'
        else:
            node_name=next(iter(source_ids));dynamo=fx.get(node_name)
            matching=[s for s in sources if receipt.get('dynamo_graph_id') is not None and dynamo and s.get('target')==dynamo['target'] and s.get('graph_id')==receipt.get('dynamo_graph_id')]
            if not matching:reason='dynamo_node_target_or_graph_identity_not_confirmed'
            else:reason=None
        if reason:
            unknown[reason]+=1;continue
        is_direct=identity[0].endswith('/torch/_inductor/decomposition.py')
        row={'source_identity':list(identity),'original_aten':aten,'dag_node':nodes[node_name]['node_id'],
             'generated_registry_operator':nodes[node_name]['op_name'],'dynamo_node_target':dynamo['target'],
             'provenance_sources':matching,'hit_outcomes':call['hit_outcomes'],
             'entry_argument_facts':call.get('entry_argument_facts'),
             'scope':'direct_inductor_decomposition_body' if is_direct else 'other_registered_provider_or_wrapper'}
        rows.append(row);entered.add(identity)
        for src,dst in call['hit_outcomes']:
            attributed.add((identity,src,dst))
            if is_direct:direct.add((identity,src,dst))
    return {'status':receipt['status'],'source_observation_started':True,'attributed_calls':rows,
            'unattributed_call_reasons':dict(unknown),'raw_call_count':len(evidence['calls']),
            'attributed_call_count':len(rows),'entered_source_functions':len(entered),
            'attributed_source_outcome_count':len(attributed),'direct_decomposition_source_outcome_count':len(direct),
            'unit':'exact Python conditional jump outcomes, globally deduplicated by source code identity',
            'primary_denominator_committed':False,'coverage_percentage':None,
            'scope_not_all_compiler_stages_or_device_kernels':True}


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',default='cuda_dag_controls_v3');p.add_argument('--output',default='source_attribution_audit_v2');args=p.parse_args()
    directory=ROOT/args.input;out=ROOT/args.output;out.mkdir(exist_ok=False)
    rows=[]
    for case in sorted(directory.glob('case_*')):
        if not case.is_dir():continue
        report=attribute(case);report['case']=case.name
        report['evidence_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (case/'receipt.json',case/'source_evidence.json') if p.exists()}
        (out/(case.name+'.json')).write_text(json.dumps(report,indent=2)+'\n');rows.append(report)
    summary={'cases':len(rows),'executed_cases':sum(r['status']=='executed' for r in rows),
             'per_case':[{'case':r['case'],'status':r['status'],**{key:r[key] for key in
                          ('raw_call_count','attributed_call_count','direct_decomposition_source_outcome_count') if key in r}} for r in rows],
             'coverage_percentage':None,'primary_denominator_committed':False,'effect_experiment_started':False,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
