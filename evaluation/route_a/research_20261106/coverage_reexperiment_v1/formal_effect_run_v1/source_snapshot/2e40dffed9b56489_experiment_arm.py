"""Common full-wall arm runner. All generation/binding costs are charged."""
import time
PROCESS_STARTED=time.monotonic()
import argparse,copy,hashlib,json,os,random,signal,subprocess,sys
from pathlib import Path
from inventory import ROOT,sha,torch
from experiment_supervisor import process_start
from common_pool import snapshot
from kernelfuzz.route_a.model import GraphRequest
from kernelfuzz.route_a.torchfuzz_adapter import TorchFuzzAdapter
from dag_generator import generate
from graph_semantics import manifest,restore
from action_candidates import build,target_nodes
from source_conditions import facts,evaluate,Unknown
from experiment_queue import ExperimentQueue
from experiment_metrics import read as read_metrics
from input_regions import region


def seed(master,index):return int(hashlib.sha256(json.dumps([master,index]).encode()).hexdigest()[:8],16)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','B','C'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--wall',type=float,default=600);p.add_argument('--batch',type=int,default=24)
    p.add_argument('--max-executions',type=int,default=1000000);p.add_argument('--purpose',choices=['control','formal'],default='control')
    args=p.parse_args();torch.set_num_threads(1);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);deadline=PROCESS_STARTED+max(0,args.wall-2)
    protocol=json.loads((ROOT/'effect_protocol_v1_draft.json').read_text())
    catalog=json.loads((ROOT/protocol['condition_inventory']).read_text())
    decisions=[r for r in catalog['conditions'] if r['decision_id'] in protocol['selected_decision_ids']]
    queue=ExperimentQueue(out/'queue.json',seed=args.seed);pool=snapshot();adapter=TorchFuzzAdapter(supported_ops=pool['accepted'])
    source_files=['experiment_arm.py','experiment_runtime.py','experiment_queue.py','experiment_metrics.py','action_candidates.py','graph_context.py',
                  'experiment_supervisor.py','input_regions.py','dag_generator.py','dag_binding.py','dag_render.py','common_domain.py','functional_domain.py','source_conditions.py','candidate_queue.py','opcode_offsets.py']
    hashes={name:sha((ROOT/name).read_bytes()) for name in source_files}
    generation=[];materialization=[];skeletons=[];rows=[];metric_hits=set();success_hits=set();kinds=set();valid=True;stop='wall_budget'
    def checkpoint():
        summary={'purpose':args.purpose,'arm':args.arm,'master_seed':args.seed,'wall_budget_seconds':args.wall,
                 'elapsed_seconds':time.monotonic()-PROCESS_STARTED,'generation':generation,'materialization':materialization,
                 'executions':rows,'covered_target_ids':sorted(metric_hits),'successful_execution_target_ids':sorted(success_hits),
                 'operator_kinds_at_backend_entry':sorted(kinds),'protocol_valid':valid,'stop_reason':stop,
                 'primary_denominator':642,'primary_coverage':len(metric_hits)/642,
                 'control_only':args.purpose=='control','source_sha256':hashes,
                 'source_unchanged':all(sha((ROOT/n).read_bytes())==h for n,h in hashes.items())}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for ordinal in range(args.batch):
        if time.monotonic()>=deadline:break
        tick=time.monotonic();r={'ordinal':ordinal,'seed':seed(args.seed,ordinal)}
        try:
            graph,provenance=generate(adapter,GraphRequest(seed=r['seed'],supported_ops=tuple(pool['accepted']),target_id=None,condition=None,entry_strategy='bounded_filter'))
            record=manifest(graph);case={'seed':r['seed'],'graph':record,'generation_ordinal':ordinal};skeletons.append(record['sha256']);cache={}
            (out/f'skeleton_{ordinal:03d}.json').write_text(json.dumps(case,indent=2)+'\n')
            r.update(status='generated',skeleton_sha256=record['sha256'],provenance=provenance)
            for decision in decisions:
                if time.monotonic()>=deadline:break
                found,rejected=target_nodes(graph,decision)
                chosen=random.Random(seed(args.seed,ordinal+decision['if_line'])).choice(found) if found else graph.root_node_id
                if chosen not in cache:
                    candidates,receipt=build(graph,case,decision,chosen,out/f'candidates_{ordinal:03d}_{chosen}',target_applicable=bool(found),deadline=deadline);cache[chosen]=candidates
                    materialization.append(dict(receipt,generation_ordinal=ordinal,targets_found=found,rejected=rejected))
                else:
                    candidates=copy.deepcopy(cache[chosen])
                    for candidate in candidates:
                        candidate['target_applicable']=bool(found)
                        if not found:candidate['predicted_outcome']='unknown';candidate['input_region']=None;continue
                        changed=restore(json.loads(Path(candidate['case']).read_text())['graph'])
                        try:candidate['predicted_outcome']='true' if evaluate(decision['predicate'],facts(changed,chosen)) else 'false'
                        except Unknown:candidate['predicted_outcome']='unknown'
                        candidate['input_region']=region(decision['predicate'],facts(changed,chosen))
                queue.append(decision['decision_id'],candidates)
        except Exception as exc:
            # Generator exhaustion is a result; implementation faults are not
            # silently reclassified as normal absence or skipped successes.
            r.update(status='generation_failure' if hasattr(exc,'provenance') else 'preparation_error',error_type=type(exc).__name__,error=str(exc)[-1500:])
            if hasattr(exc,'provenance'):r['provenance']=exc.provenance
            else:valid=False;stop='preparation_error'
        r['full_wall_seconds']=time.monotonic()-tick;generation.append(r);checkpoint()
        if not valid:break
    schedules={}
    for decision in decisions:
        order=list(skeletons);random.Random(seed(args.seed,decision['if_line'])).shuffle(order);schedules[decision['decision_id']]=order
    if not skeletons:stop='no_generated_skeletons';checkpoint();return
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'),
             TRITON_CACHE_DIR=str(out/'triton_cache'),CUDA_CACHE_PATH=str(out/'cuda_cache'),TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',OMP_NUM_THREADS='1')
    count=0
    while valid and time.monotonic()<deadline and count<args.max_executions:
        decision=decisions[count%len(decisions)];round_id=count//len(decisions);schedule=schedules[decision['decision_id']]
        skeleton=schedule[round_id%len(schedule)];desired='true' if round_id%2==0 else 'false'
        choice=queue.select(decision['decision_id'],arm=args.arm,skeleton_sha256=skeleton,desired_outcome=desired)
        if choice is None:
            rows.append({'ordinal':count,'target_id':decision['decision_id'],'baseline_skeleton_sha256':skeleton,'status':'queue_empty_for_skeleton'});count+=1;checkpoint();continue
        directory=out/f'execute_{count:04d}';tick=time.monotonic();remaining=deadline-tick
        if remaining<=0:queue.complete(choice['id'],{'status':'unknown','reason':'budget_before_launch'});break
        with (out/f'{count:04d}.stdout.txt').open('w') as stdout,(out/f'{count:04d}.stderr.txt').open('w') as stderr:
            proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'experiment_runtime.py'),'--case',choice['case'],'--output',str(directory),'--purpose',args.purpose],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            active={'pid':proc.pid,'start_ticks':process_start(proc.pid),'state':'running','ordinal':count,'directory':str(directory),'case':choice['case']}
            (out/'active_worker.json').write_text(json.dumps(active,indent=2)+'\n')
            timed_out=False
            try:proc.wait(timeout=min(150,max(.001,remaining)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        active.update(state='terminal',returncode=proc.returncode)
        (out/'active_worker.json').write_text(json.dumps(active,indent=2)+'\n')
        row={'ordinal':count,'target_id':decision['decision_id'],'round':round_id,'selection':choice['selection'],
             'selected_graph_sha256':choice['graph_sha256'],'target_applicable':choice['target_applicable'],'action':choice['action'],
             'returncode':proc.returncode,'worker_wall_seconds':time.monotonic()-tick}
        receipt_path=directory/'receipt.json'
        if timed_out:
            row['status']='timeout';observation={'status':'unknown','reason':'timeout'}
        elif proc.returncode or not receipt_path.exists():
            row['status']='worker_error';observation={'status':'unknown','reason':'worker_error'}
        else:
            try:
                metrics=read_metrics(directory,decision,choice['action']['node_id']);row['status']=metrics['receipt_status'];row['metrics']=metrics
                metric_hits.update(metrics['verified_target_ids']);success_hits.update(metrics['successful_execution_target_ids']);kinds.update(metrics['operator_kinds_at_backend_entry'])
                valid=metrics['protocol_valid'];observation=metrics['observation']
                if not valid:stop='measurement_protocol_invalidated'
            except Exception as exc:
                valid=False;stop='metrics_reader_error';row.update(status='metrics_error',error=type(exc).__name__+':'+str(exc));observation={'status':'unknown','reason':'metrics_error'}
        queue.complete(choice['id'],observation);row['observation']=observation;row['full_wall_seconds']=time.monotonic()-tick;row['elapsed_seconds']=time.monotonic()-PROCESS_STARTED;rows.append(row);count+=1;checkpoint()
        print(json.dumps({'arm':args.arm,'ordinal':count,'status':row['status'],'hits':len(metric_hits),'elapsed':time.monotonic()-PROCESS_STARTED}),flush=True)
        if timed_out and time.monotonic()>=deadline:stop='wall_budget_timeout';break
    if count>=args.max_executions:stop='execution_limit_control'
    checkpoint()
if __name__=='__main__':main()
