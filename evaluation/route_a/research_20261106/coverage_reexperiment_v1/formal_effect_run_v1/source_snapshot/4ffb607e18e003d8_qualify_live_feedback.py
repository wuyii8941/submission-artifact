"""Production queue -> real compiler feedback -> next execution controls."""
import copy,json,os,signal,subprocess,sys,time
from pathlib import Path
from inventory import ROOT,sha
from graph_semantics import restore
from action_candidates import build
from experiment_queue import ExperimentQueue
from experiment_metrics import read as read_metrics


def main():
    out=ROOT/'live_feedback_controls_v2';out.mkdir(exist_ok=False)
    source=ROOT/'denominator_call_controls_v1/case_002';receipt=json.loads((source/'receipt.json').read_text());case=json.loads(Path(receipt['case']).read_text());graph=restore(case['graph'])
    attributed=json.loads((source/'attribution.json').read_text())['attributed_calls']
    conditions=json.loads((ROOT/'source_condition_inventory_v2/conditions.json').read_text())['conditions'];results=[]
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'),TRITON_CACHE_DIR=str(out/'triton_cache'),CUDA_CACHE_PATH=str(out/'cuda_cache'),TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',OMP_NUM_THREADS='1')
    for line in (444,565):
        source=ROOT/('denominator_call_controls_v1/case_002' if line==444 else 'arm_control_C_v1/execute_0003')
        receipt=json.loads((source/'receipt.json').read_text());case=json.loads(Path(receipt['case']).read_text());graph=restore(case['graph'])
        attributed=json.loads((source/'attribution.json').read_text())['attributed_calls']
        d=next(r for r in conditions if r['if_line']==line and r['guidance_status']=='input_condition_specified')
        calls=[r for r in attributed if r['source_identity']==d['source_identity'] and graph.nodes[r['dag_node']].op_name in d['target_ops']]
        assert calls,('fixed control target missing',line)
        nid=calls[0]['dag_node'];prior=read_metrics(source,d,nid);assert prior['protocol_valid'] and prior['observation']['status']=='observed'
        candidates,binding=build(graph,case,d,nid,out/f'actions_{line}')
        identity=next(r for r in candidates if r['action']['attribute']=='identity');assert identity['graph_sha256']==receipt['graph_sha256']
        queues={a:ExperimentQueue(out/f'{line}_{a}.json',seed=23) for a in ('B','C')}
        for arm,q in queues.items():
            q.append(d['decision_id'],[identity]);first=q.select(d['decision_id'],arm=arm,skeleton_sha256=identity['skeleton_sha256'],desired_outcome='false')
            wrong=copy.deepcopy(prior['observation']);wrong['graph_sha256']='wrong'
            try:q.complete(first['id'],wrong)
            except ValueError as exc:assert str(exc)=='feedback_program_or_node_identity_mismatch'
            else:raise AssertionError('foreign feedback accepted')
            q.complete(first['id'],prior['observation']);q.append(d['decision_id'],candidates)
        b=queues['B'].select(d['decision_id'],arm='B',skeleton_sha256=identity['skeleton_sha256'],desired_outcome='false')
        c=queues['C'].select(d['decision_id'],arm='C',skeleton_sha256=identity['skeleton_sha256'],desired_outcome='false')
        assert c['selection']['verified_feedback_available']
        target=out/f'execute_{line}'
        with (out/f'{line}.stdout.txt').open('w') as stdout,(out/f'{line}.stderr.txt').open('w') as stderr:
            proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'experiment_runtime.py'),'--case',c['case'],'--output',str(target)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:proc.wait(timeout=150)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        metric=read_metrics(target,d,nid);assert metric['protocol_valid'],metric
        queues['C'].complete(c['id'],metric['observation']);queues['B'].complete(b['id'],{'status':'unknown','reason':'shadow_B_not_executed'})
        item={'if_line':line,'heldout_dtype_condition':line==565,'binding':binding,'previous_real_observation':prior['observation'],'next_selection':c['selection'],
              'next_runtime_status':metric['receipt_status'],'next_actual_observation':metric['observation'],'source_domain_valid':metric['protocol_valid'],
              'new_observation_is_confirmed':metric['observation']['status']=='observed'}
        results.append(item);(out/'rows.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'if_line':line,'reason':c['selection']['reason'],'observed':item['new_observation_is_confirmed']}),flush=True)
    report={'control_only':True,'supported_and_heldout_real_feedback_followed_by_execution':all(r['next_selection']['verified_feedback_available'] and r['source_domain_valid'] and r['next_runtime_status'] in ('executed','value_domain_rejected','execution_error') for r in results),'both_next_source_exits_confirmed':all(r['new_observation_is_confirmed'] for r in results),
            'foreign_program_feedback_rejected':True,'controls_use_frozen_general_DAG':True,'not_effect_evidence':True,
            'source_sha256':{f:sha((ROOT/f).read_bytes()) for f in ('experiment_queue.py','graph_context.py','input_regions.py','action_candidates.py','experiment_metrics.py','experiment_runtime.py','qualify_live_feedback.py')}}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
