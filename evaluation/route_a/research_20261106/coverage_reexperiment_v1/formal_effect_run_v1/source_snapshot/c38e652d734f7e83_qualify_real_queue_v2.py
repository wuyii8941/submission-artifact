"""Fixed-prefix, all-target real-feedback control. Not an effect comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
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
    source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('qualify_real_queue_v2.py','candidate_queue.py','decision_feedback.py','runtime_audit.py','dag_binding.py','common_domain.py')}
    old=json.loads((ROOT.parents[1]/'research_20261103/unified_condition_v2/decisions_v4.json').read_text())
    specs=[]
    for item in old['decisions']:
        item=dict(item,source_sha256=old['source_sha256'])
        item['if_line']=int(item['source_location'].rsplit(':',1)[1]);item['original_aten']=item['observation']['original_aten']
        item['resolved']=resolve(item);specs.append(item)
    rows=[(path,json.loads(path.read_text())) for path in sorted((ROOT/'dag_controls_v2').glob('case_*.json'))]
    b=CandidateQueue(out/'B.json',seed=5);c=CandidateQueue(out/'C.json',seed=5)
    scan=[];bindings=[];pool_counts={};spec_by_id={s['decision_id']:s for s in specs}
    for spec in specs:
        candidates=[]
        for path,row in rows:
            tick=time.monotonic();entry={'case':path.name,'target_id':spec['decision_id'],'generation_status':row['status']}
            if row['status']!='qualified':entry['status']='generation_not_qualified'
            else:
                graph=restore(row['graph']);found,rejected=targets(graph,spec)
                entry.update(nodes=found,rejected=rejected,status='target_present' if found else 'target_absent')
                for nid in found:
                    for val in (1,2,16,17):
                        variable=spec['controllable_variables'][0]
                        changed,record=bind(graph,nid,'shape',val,input_index=variable.get('input_index',0),axis=variable['axis'])
                        record.update(original_case=path.name,target_id=spec['decision_id']);bindings.append(record)
                        if changed is None:continue
                        payload=dict(row,graph=manifest(changed));candidate_path=out/f'{spec["source_function"]}_{row["seed"]}_{nid}_{val}.json'
                        candidate_path.write_text(json.dumps(payload,indent=2)+'\n')
                        prediction=evaluate(spec['predicate'],node_facts(changed,nid))
                        candidates.append({'graph_sha256':payload['graph']['sha256'],'case':str(candidate_path.resolve()),
                                           'action':{'attribute':'shape','value':val,'node_id':nid},
                                           'predicted_outcome':'true' if prediction else 'false','context':row['graph']['sha256']})
            entry['scan_binding_wall_seconds']=time.monotonic()-tick;scan.append(entry)
        assert b.append(spec['decision_id'],candidates)==c.append(spec['decision_id'],candidates)
        pool_counts[spec['decision_id']]=len(candidates)
    (out/'target_scan.json').write_text(json.dumps(scan,indent=2)+'\n');(out/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    (out/'specs.json').write_text(json.dumps(specs,indent=2)+'\n')
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',TORCHINDUCTOR_CACHE_DIR='/data1/tzh/coverage_tooling/new_domain_inductor',CUDA_CACHE_PATH='/data1/tzh/coverage_tooling/new_domain_cuda',TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',OMP_NUM_THREADS='1')
    executions=[]
    # Fixed equal quotas, source target order, with all pending candidates
    # retained. No target success changes either quota or candidate generation.
    for round_id in range(3):
        for spec in specs:
            sb=b.select(spec['decision_id'],use_feedback=False);sc=c.select(spec['decision_id'],use_feedback=True)
            if sc is None:
                executions.append({'target_id':spec['decision_id'],'round':round_id,'status':'queue_empty'});continue
            index=len(executions);directory=out/f'execute_{index}';tick=time.monotonic()
            with (out/f'{index}.stdout.txt').open('w') as stdout,(out/f'{index}.stderr.txt').open('w') as stderr:
                proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'runtime_audit.py'),'--case',sc['case'],'--output',str(directory)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                try:proc.wait(timeout=150)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            receipt_path=directory/'receipt.json'
            receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'status':'missing_receipt'}
            report=attribute(directory) if receipt_path.exists() else {}
            if directory.exists():(directory/'attribution.json').write_text(json.dumps(report,indent=2)+'\n')
            obs=observation(spec,spec['resolved'],report,directory/'source_evidence.json',node_id=sc['action']['node_id']) if (directory/'source_evidence.json').exists() else {'status':'unknown','reason':receipt['status']}
            if receipt['status']!='executed' or proc.returncode:obs={'status':'unknown','reason':receipt['status']}
            c.complete(sc['id'],obs);b.complete(sb['id'],obs if sb['id']==sc['id'] else {'status':'unknown','reason':'B_not_executed_in_this_control'})
            item={'index':index,'target_id':spec['decision_id'],'round':round_id,'selection':sc['selection'],'B_id':sb['id'],'C_id':sc['id'],
                  'runtime_status':receipt['status'],'observation':obs,'full_wall_seconds':time.monotonic()-tick,'returncode':proc.returncode}
            executions.append(item);(out/'executions.json').write_text(json.dumps(executions,indent=2)+'\n');print(json.dumps(item),flush=True)
    summary={'control_only':True,'domain':descriptor(),'candidate_count_by_target':pool_counts,
             'input_prefix_cases':len(rows),'source_order_target_count':len(specs),'fixed_quota_per_target':3,
             'binding_attempts':len(bindings),'binding_failures':sum(r['status']!='bound' for r in bindings),
             'real_observed_feedback':sum(r.get('observation',{}).get('status')=='observed' for r in executions),
             'post_feedback_selections':sum(r.get('selection',{}).get('verified_feedback_available',False) for r in executions),
             'feedback_changed_next_action':any(r.get('selection',{}).get('verified_feedback_available') and r['selection']['id']!=r['selection']['baseline_id'] for r in executions),
             'full_wall_seconds':time.monotonic()-started,'source_sha256':source_hashes,
             'source_unchanged':all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha for f,sha in source_hashes.items()),
             'primary_denominator_committed':False,'effect_experiment_started':False,
             'not_effect_evidence':'C execution plus unexecuted shadow B; historical condition controls; original graphs pregenerated; no A/B/C effect claim'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
