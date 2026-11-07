"""Replay fixed general graphs to falsify proposed source-domain assumptions."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from inventory import ROOT,sha
from attribute_source_audit import attribute


def validate_report(report,audit):
    proof_ids={tuple(r['source_identity']) for r in audit['rows'] if r['proof'] is not None}
    excluded={(tuple(r['source_identity']),*r['edge']) for r in audit['exclusions']}
    checks=[];failures=[]
    for call in report.get('attributed_calls',[]):
        identity=tuple(call['source_identity'])
        for edge in call['hit_outcomes']:
            if (identity,*edge) in excluded:failures.append({'source_identity':list(identity),'reason':'actually_attributed_excluded_outcome','edge':edge})
        if identity not in proof_ids:continue
        facts=call.get('entry_argument_facts')
        tensors=[f for f in facts or [] if f['kind']=='tensor']
        confirmed=bool(tensors) and all(f['device_type']=='cuda' and f['is_complex'] is False for f in tensors) and not any(f.get('is_complex') for f in facts or [])
        checks.append({'source_identity':list(identity),'dag_node':call['dag_node'],'confirmed':confirmed,'argument_facts':facts})
        if not confirmed:failures.append({'source_identity':list(identity),'reason':'fixed_domain_entry_assumptions_not_confirmed','facts':facts})
    return {'checks':checks,'failures':failures,'valid':not failures,'fixed_domain_calls_checked':len(checks)}


def main():
    out=ROOT/'denominator_call_controls_v1';out.mkdir(exist_ok=False);started=time.monotonic()
    audit=json.loads((ROOT/'branch_domain_audit_v3/results.json').read_text());rows=[]
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',TORCHINDUCTOR_CACHE_DIR='/data1/tzh/coverage_tooling/new_domain_inductor',CUDA_CACHE_PATH='/data1/tzh/coverage_tooling/new_domain_cuda',TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',OMP_NUM_THREADS='1')
    # Fixed ordinals, not selected for domain check success. Includes unsafe
    # inputs, empty outputs, mixed ops and unsupported device dtype proposals.
    for ordinal in range(8):
        case=ROOT/'dag_controls_v2'/f'case_{ordinal:03d}.json';directory=out/f'case_{ordinal:03d}'
        tick=time.monotonic()
        with (out/f'{ordinal}.stdout.txt').open('w') as stdout,(out/f'{ordinal}.stderr.txt').open('w') as stderr:
            proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'runtime_audit.py'),'--case',str(case),'--output',str(directory)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:proc.wait(timeout=150)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        receipt=json.loads((directory/'receipt.json').read_text());report=attribute(directory);validation=validate_report(report,audit)
        (directory/'attribution.json').write_text(json.dumps(report,indent=2)+'\n');(directory/'domain_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        row={'ordinal':ordinal,'status':receipt['status'],'returncode':proc.returncode,'full_wall_seconds':time.monotonic()-tick,**{k:validation[k] for k in ('valid','fixed_domain_calls_checked','failures')}}
        rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    summary={'control_only':True,'fixed_prefix':8,'valid':all(r['valid'] for r in rows),
             'fixed_domain_calls_checked':sum(r['fixed_domain_calls_checked'] for r in rows),
             'executed_cases':sum(r['status']=='executed' for r in rows),'full_wall_seconds':time.monotonic()-started,
             'primary_denominator_committed':False,'effect_experiment_started':False,
             'source_sha256':{f:sha((ROOT/f).read_bytes()) for f in ('runtime_audit.py','attribute_source_audit.py','qualify_denominator_calls.py')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
