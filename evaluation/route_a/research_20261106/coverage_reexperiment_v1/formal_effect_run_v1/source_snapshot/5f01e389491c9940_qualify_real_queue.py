"""General-DAG to real-source feedback queue control, excluded from effects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from inventory import ROOT
from graph_semantics import restore,manifest
from dag_binding import bind
from constraints_v4 import node_facts,evaluate,targets
from candidate_queue import CandidateQueue
from decision_feedback import resolve,observation
from attribute_source_audit import attribute
from common_domain import descriptor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(exist_ok=False);started=time.monotonic()
    # Historical predicates are reused solely as observation controls; formal
    # source-first target selection and denominator are not declared by this run.
    old=json.loads((ROOT.parents[1]/'research_20261103/unified_condition_v2/decisions_v4.json').read_text())
    specs=[]
    for item in old['decisions']:
        item=dict(item);item.update(source_sha256=old['source_sha256'])
        item['if_line']=int(item['source_location'].rsplit(':',1)[1]);item['original_aten']=item['observation']['original_aten']
        item['resolved']=resolve(item);specs.append(item)
    scan=[];selected=None
    # Fixed saved general prefix. No target insertion, no new template or
    # success-based graph repair; the scan and binding costs are retained.
    for path in sorted((ROOT/'dag_controls_v2').glob('case_*.json')):
        row=json.loads(path.read_text());entry={'case':path.name,'status':row['status'],'targets':[]}
        if row['status']=='qualified':
            graph=restore(row['graph'])
            for spec in specs:
                found,rejected=targets(graph,spec);entry['targets'].append({'id':spec['decision_id'],'nodes':found,'rejected':rejected})
                if selected is None and found:selected=(row,spec,found[0])
        scan.append(entry)
    (out/'target_scan.json').write_text(json.dumps(scan,indent=2)+'\n')
    if selected is None:raise ValueError('target_absent_in_fixed_general_prefix')
    original,spec,nid=selected;graph=restore(original['graph']);candidates=[];bindings=[]
    for val in (1,2,16,17):
        variable=spec['controllable_variables'][0]
        changed,record=bind(graph,nid,'shape',val,input_index=variable.get('input_index',0),axis=variable['axis']);bindings.append(record)
        if changed is None:continue
        payload=dict(original,graph=manifest(changed));path=out/f'bound_{val}.json';path.write_text(json.dumps(payload,indent=2)+'\n')
        prediction=evaluate(spec['predicate'],node_facts(changed,nid))
        candidates.append({'graph_sha256':payload['graph']['sha256'],'case':str(path.resolve()),
                           'action':{'attribute':'shape','value':val,'node_id':nid},
                           'predicted_outcome':'true' if prediction else 'false','context':original['graph']['sha256']})
    (out/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    b=CandidateQueue(out/'B.json',seed=5);c=CandidateQueue(out/'C.json',seed=5)
    assert b.append(spec['decision_id'],candidates)==c.append(spec['decision_id'],candidates)
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',TORCHINDUCTOR_CACHE_DIR='/data1/tzh/coverage_tooling/new_domain_inductor',CUDA_CACHE_PATH='/data1/tzh/coverage_tooling/new_domain_cuda',TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',OMP_NUM_THREADS='1')
    executions=[]
    for index in range(min(3,len(candidates))):
        sb=b.select(spec['decision_id'],use_feedback=False);sc=c.select(spec['decision_id'],use_feedback=True)
        # This control executes C's concrete choice and only feeds the same
        # event to B if B made the identical choice. This is not a B/C trial.
        directory=out/f'execute_{index}';tick=time.monotonic()
        proc=subprocess.run([sys.executable,'-S',str(ROOT/'runtime_audit.py'),'--case',sc['case'],'--output',str(directory)],env=env,capture_output=True,text=True,timeout=150)
        (out/f'{index}.stdout.txt').write_text(proc.stdout);(out/f'{index}.stderr.txt').write_text(proc.stderr)
        receipt=json.loads((directory/'receipt.json').read_text());report=attribute(directory)
        (directory/'attribution.json').write_text(json.dumps(report,indent=2)+'\n')
        obs=observation(spec,spec['resolved'],report,directory/'source_evidence.json',node_id=nid) if (directory/'source_evidence.json').exists() else {'status':'unknown','reason':receipt['status']}
        if receipt['status']!='executed':obs={'status':'unknown','reason':receipt['status']}
        c.complete(sc['id'],obs);b.complete(sb['id'],obs if sb['id']==sc['id'] else {'status':'unknown','reason':'B_not_executed_in_this_control'})
        item={'index':index,'selection':sc['selection'],'B_id':sb['id'],'C_id':sc['id'],'runtime_status':receipt['status'],'observation':obs,'full_wall_seconds':time.monotonic()-tick}
        executions.append(item);(out/'executions.json').write_text(json.dumps(executions,indent=2)+'\n');print(json.dumps(item),flush=True)
    summary={'control_only':True,'domain':descriptor(),'decision':spec,'original_general_graph_sha256':original['graph']['sha256'],
             'candidate_count':len(candidates),'binding_failures':sum(r['status']!='bound' for r in bindings),
             'real_observed_feedback':sum(r['observation']['status']=='observed' for r in executions),
             'post_feedback_selections':sum(r['selection']['verified_feedback_available'] for r in executions),
             'feedback_changed_next_action':any(r['selection']['verified_feedback_available'] and r['selection']['id']!=r['selection']['baseline_id'] for r in executions),
             'full_wall_seconds':time.monotonic()-started,'primary_denominator_committed':False,'effect_experiment_started':False,
             'not_effect_evidence':'one control sequence only; no full A/B/C comparison or positive-effect claim'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
