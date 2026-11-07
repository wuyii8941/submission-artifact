"""Fixed-prefix CUDA audit with every attempted case and full cost retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import Counter
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=8)
    p.add_argument('--input',default='dag_controls_v2');p.add_argument('--output',required=True)
    args=p.parse_args();out=ROOT/args.output;out.mkdir(exist_ok=False)
    rows=[];started=time.monotonic()
    env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',
             TORCHINDUCTOR_CACHE_DIR='/data1/tzh/coverage_tooling/new_domain_inductor',
             CUDA_CACHE_PATH='/data1/tzh/coverage_tooling/new_domain_cuda',
             TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',OMP_NUM_THREADS='1')
    for index in range(args.count):
        path=ROOT/args.input/f'case_{index:03d}.json';case=json.loads(path.read_text())
        row={'ordinal':index,'seed':case['seed'],'case_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'profile':case.get('provenance',{}).get('profile'),'control_only':True}
        tick=time.monotonic()
        if case['status']!='qualified':row['status']='generation_not_qualified'
        else:
            with (out/f'{index:03d}.stdout.txt').open('w') as stdout,(out/f'{index:03d}.stderr.txt').open('w') as stderr:
                proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'runtime_audit.py'),'--case',str(path),
                                      '--output',str(out/f'case_{index:03d}')],env=env,
                                      stdout=stdout,stderr=stderr,start_new_session=True)
                try:row['returncode']=proc.wait(timeout=150)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();row.update(status='audit_timeout',returncode=proc.returncode)
            receipt=out/f'case_{index:03d}/receipt.json'
            if receipt.exists():
                r=json.loads(receipt.read_text());row['receipt_status']=r['status'];row['oracle']=r.get('oracle')
                if 'status' not in row:row['status']=r['status'] if row['returncode']==0 else 'worker_error'
            elif 'status' not in row:row['status']='missing_receipt'
        row['full_wall_seconds']=time.monotonic()-tick;rows.append(row)
        (out/'attempts.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    summary={'requested':args.count,'selection':'fixed ordinal prefix, not selected for execution success or coverage',
             'statuses':dict(Counter(r['status'] for r in rows)), 'rows':rows,
             'full_wall_seconds':time.monotonic()-started,'excluded_from_discovery_and_effect_comparison':True,
             'primary_denominator_committed':False,'effect_experiment_started':False,
             'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('runtime_audit.py','value_domain.py','qualify_cuda_dags.py')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
