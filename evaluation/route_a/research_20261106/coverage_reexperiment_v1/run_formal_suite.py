"""Run the frozen8-seed3-arm protocol, stopping on any definition/fairness drift."""
import argparse,hashlib,json,os,sys,time,shutil
from pathlib import Path
from experiment_supervisor import run_arm,ROOT
from analyze_experiment import analyze


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairing(a,b):
    ga=[(r['seed'],r.get('status'),r.get('skeleton_sha256')) for r in a['generation']]
    gb=[(r['seed'],r.get('status'),r.get('skeleton_sha256')) for r in b['generation']]
    if ga[:min(len(ga),len(gb))]!=gb[:min(len(ga),len(gb))]:return False
    def key(row):return (row['ordinal'],row['target_id'],row.get('selection',{}).get('baseline_skeleton_sha256',row.get('baseline_skeleton_sha256')))
    n=min(len(a['executions']),len(b['executions']))
    return [key(r) for r in a['executions'][:n]]==[key(r) for r in b['executions'][:n]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/'effect_protocol_v1.json').read_text());assert protocol['status']=='frozen_ready_for_formal_run'
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    watched=dict(protocol['runtime_source_sha256']);watched[str(ROOT/'effect_protocol_v1.json')]=digest(ROOT/'effect_protocol_v1.json');state={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'runs':[],'invalidations':[]}
    started=time.monotonic()
    snapshot=out/'source_snapshot';snapshot.mkdir()
    saved={}
    for path,expected in watched.items():
        if digest(path)!=expected:raise ValueError('source_changed_before_snapshot:'+path)
        name=hashlib.sha256(path.encode()).hexdigest()[:16]+'_'+Path(path).name
        shutil.copyfile(path,snapshot/name);saved[path]={'snapshot':name,'sha256':expected}
    (out/'source_snapshot.json').write_text(json.dumps(saved,indent=2)+'\n')
    def save():
        state['elapsed_seconds']=time.monotonic()-started;(out/'suite.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for index,seed in enumerate(protocol['master_seeds']):
        for arm in protocol['order_by_seed'][index]:
            drift=[path for path,expected in watched.items() if digest(path)!=expected]
            if drift:state['status']='invalidated';state['invalidations'].append({'reason':'source_or_definition_drift','paths':drift});save();analyze(out);return
            relative=f'seed_{index:02d}_{arm}';target=out/relative
            receipt=run_arm(target,arm,seed,protocol['per_arm_seed_wall_seconds'],protocol['common_generation']['initial_shared_skeleton_batch'],1000000,'formal')
            row={'master_seed':seed,'seed_index':index,'arm':arm,'relative_output':relative,'supervisor':receipt}
            state['runs'].append(row);summary_path=target/'summary.json'
            if not summary_path.exists():state['status']='invalidated';state['invalidations'].append({'reason':'missing_arm_summary','run':relative})
            else:
                summary=json.loads(summary_path.read_text())
                if not summary['protocol_valid'] or not summary['source_unchanged'] or receipt['cleanup']['status']=='refused_unverified_process_identity':
                    state['status']='invalidated';state['invalidations'].append({'reason':'arm_protocol_or_source_failure','run':relative})
                paths={r['arm']:out/r['relative_output']/'summary.json' for r in state['runs'] if r['master_seed']==seed}
                if 'A' in paths and 'B' in paths and not pairing(json.loads(paths['A'].read_text()),json.loads(paths['B'].read_text())):
                    state['status']='invalidated';state['invalidations'].append({'reason':'AB_skeleton_schedule_mismatch','seed':seed})
            drift=[path for path,expected in watched.items() if digest(path)!=expected]
            if drift:state['status']='invalidated';state['invalidations'].append({'reason':'source_or_definition_drift_during_arm','paths':drift})
            save();print(json.dumps({'completed_runs':len(state['runs']),'total_runs':len(protocol['master_seeds'])*3,'run':relative,'status':state['status']}),flush=True)
            if state['status']=='invalidated':analyze(out);return
    state['status']='complete';save();result=analyze(out)
    print(json.dumps({'status':'complete','positive_claim':result['positive_claim'],'analysis':str(out/'analysis.json')}),flush=True)
if __name__=='__main__':main()
