"""Independent synthetic positive/negative controls; never coverage samples."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from attribute_source_audit import attribute
ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'attribution_controls_v2';out.mkdir(exist_ok=False)
    identity=['/fixture/torch/_inductor/decomposition.py',10,'synthetic-code-identity']
    node={'node_id':'node_0','op_name':'torch.add'}
    source={'name':'var_node_0','target':'fixture.add','graph_id':123,'from_node':[]}
    fixture={
        'receipt':{'status':'executed','dynamo_graph_id':123,
                   'dynamo_fx':[{'name':'var_node_0','op':'call_function','target':'fixture.add'}]},
        'original':{'graph':{'payload':{'nodes':[node,{'node_id':'node_1','op_name':'torch.mul'}]}}},
        'evidence':{'catalog':[{'source':dict(zip(('source','first_line','code_sha256'),identity),conditional_outcomes=[[2,4],[2,10]]),
                                'registrations':[{'aten':'aten.add.Tensor'}]}],
                    'calls':[{'source_identity':identity,'fx_provenance':{'original_aten_id':'aten.add.Tensor','from_node_structured':[source]},'hit_outcomes':[[2,4]]}]}}
    controls=[]
    def check(label,change,expected,reason=None):
        f=copy.deepcopy(fixture);change(f)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            directory=Path(tmp);original=directory/'original.json'
            original.write_text(json.dumps(f['original']));f['receipt']['case']=str(original)
            (directory/'receipt.json').write_text(json.dumps(f['receipt']))
            (directory/'source_evidence.json').write_text(json.dumps(f['evidence']))
            report=attribute(directory)
        assert report['attributed_source_outcome_count']==expected,(label,report)
        if reason:assert report['unattributed_call_reasons'].get(reason)==1,(label,report)
        controls.append({'label':label,'pass':True,'expected_attributed_outcomes':expected,'report':report})
    def meta(f):return f['evidence']['calls'][0]['fx_provenance']
    check('exact provenance positive',lambda f:None,1)
    check('wrong graph identity',lambda f:meta(f)['from_node_structured'][0].update(graph_id=124),0,'dynamo_node_target_or_graph_identity_not_confirmed')
    check('missing receipt graph identity',lambda f:f['receipt'].pop('dynamo_graph_id'),0,'dynamo_node_target_or_graph_identity_not_confirmed')
    def missing_both(f):
        f['receipt'].pop('dynamo_graph_id');meta(f)['from_node_structured'][0].pop('graph_id')
    check('both graph identities missing',missing_both,0,'dynamo_node_target_or_graph_identity_not_confirmed')
    check('wrong source target',lambda f:meta(f)['from_node_structured'][0].update(target='fixture.mul'),0,'dynamo_node_target_or_graph_identity_not_confirmed')
    check('wrong actual ATen',lambda f:meta(f).update(original_aten_id='aten.mul.Tensor'),0,'source_registration_does_not_match_actual_aten')
    check('missing actual ATen',lambda f:meta(f).pop('original_aten_id'),0,'missing_actual_aten')
    check('missing source provenance',lambda f:meta(f).update(from_node_structured=[]),0,'missing_or_ambiguous_original_dag_node')
    check('ambiguous two generated nodes',lambda f:meta(f)['from_node_structured'].append(dict(source,name='var_node_1')),0,'missing_or_ambiguous_original_dag_node')
    check('invented branch event',lambda f:f['evidence']['calls'][0].update(hit_outcomes=[[2,999]]),0,'outcome_not_registered_in_source')
    check('duplicate calls do not inflate outcome union',lambda f:f['evidence']['calls'].append(copy.deepcopy(f['evidence']['calls'][0])),1)
    def nested(f):
        s=meta(f)['from_node_structured'][0];meta(f)['from_node_structured']=[{'name':'transformed_node','from_node':[s]}]
    check('nested provenance positive',nested,1)
    report={'controls':controls,'passed':len(controls),'control_only':True,'effect_experiment_started':False,
            'primary_denominator_committed':False,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('qualify_attribution.py','attribute_source_audit.py')}}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':len(controls),'scope':'synthetic attribution controls only'}))
if __name__=='__main__':main()
