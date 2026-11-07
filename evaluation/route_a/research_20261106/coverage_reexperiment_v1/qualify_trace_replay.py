"""Replay fixed prior graph/input controls after correcting prefix tracing."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from inventory import ROOT
from attribute_source_audit import attribute
from decision_feedback import observation


def main():
    out=ROOT/'extended_arg_cuda_replay_v1';out.mkdir(exist_ok=False);started=time.monotonic()
    prior=ROOT/'real_queue_controls_v1';queue=json.loads((prior/'C.json').read_text());old_rows=json.loads((prior/'executions.json').read_text());spec=json.loads((prior/'summary.json').read_text())['decision']
    candidates={r['id']:r['candidate'] for r in queue['candidates']}
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',TORCHINDUCTOR_CACHE_DIR='/data1/tzh/coverage_tooling/new_domain_inductor',CUDA_CACHE_PATH='/data1/tzh/coverage_tooling/new_domain_cuda',TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',OMP_NUM_THREADS='1');rows=[]
    for old in old_rows:
        i=old['index'];candidate=candidates[old['C_id']];directory=out/f'execute_{i}'
        with (out/f'{i}.stdout.txt').open('w') as stdout,(out/f'{i}.stderr.txt').open('w') as stderr:
            proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'runtime_audit.py'),'--case',candidate['case'],'--output',str(directory)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:proc.wait(timeout=150)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        receipt=json.loads((directory/'receipt.json').read_text());previous=json.loads((prior/f'execute_{i}/receipt.json').read_text());report=attribute(directory)
        (directory/'attribution.json').write_text(json.dumps(report,indent=2)+'\n')
        obs=observation(spec,spec['resolved'],report,directory/'source_evidence.json',node_id=candidate['action']['node_id'])
        row={'index':i,'status':receipt['status'],'source_unchanged':receipt['source_unchanged_during_run'],
             'same_graph':receipt['graph_sha256']==previous['graph_sha256'],'same_inputs':receipt['actual_inputs']==previous['actual_inputs'],
             'previous_observation':old['observation'],'new_observation':obs,'attributed_bytecode_outcomes':report['attributed_source_outcome_count']}
        rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    result={'control_only':True,'fixed_prior_cases':len(rows),'identical_graphs_and_inputs':all(r['same_graph'] and r['same_inputs'] for r in rows),
            'all_logical_exits_observed':all(r['new_observation']['status']=='observed' for r in rows),
            'source_unchanged_all':all(r['source_unchanged'] for r in rows),'full_wall_seconds':time.monotonic()-started,
            'scope':'measurement correction only, not method coverage improvement','primary_denominator_committed':False,'effect_experiment_started':False,
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('runtime_audit.py','opcode_offsets.py','qualify_trace_replay.py')}}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
