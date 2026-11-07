"""Process-launch wall budgeting and verified cleanup of this run's workers."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def process_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


def cleanup_worker(directory):
    path=Path(directory)/'active_worker.json'
    if not path.exists():return {'status':'no_worker_registered'}
    row=json.loads(path.read_text());pid=row['pid']
    if row.get('state')!='running':return {'status':'worker_already_terminal'}
    if process_start(pid)!=row.get('start_ticks'):return {'status':'worker_missing_or_pid_reused'}
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
        if str(ROOT/'experiment_runtime.py') not in command or str(Path(directory).resolve()) not in command:
            return {'status':'refused_unverified_process_identity'}
        os.killpg(pid,signal.SIGKILL);return {'status':'killed_verified_run_worker','pid':pid}
    except ProcessLookupError:return {'status':'worker_already_gone'}


def run_arm(output,arm,seed,wall,batch,limit,purpose):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('output_already_exists')
    argv=[sys.executable,'-S',str(ROOT/'experiment_arm.py'),'--arm',arm,'--seed',str(seed),'--output',str(output),'--wall',str(wall),'--batch',str(batch),'--max-executions',str(limit),'--purpose',purpose]
    started=time.monotonic();timed_out=False
    with (output.parent/(output.name+'.stdout.txt')).open('w') as stdout,(output.parent/(output.name+'.stderr.txt')).open('w') as stderr:
        proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True,
                              env=dict(os.environ,PYTHONPYCACHEPREFIX='/data1/tzh/coverage_tooling/audit_pycache',OMP_NUM_THREADS='1'))
        try:proc.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            timed_out=True;cleanup=cleanup_worker(output)
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    if not timed_out:cleanup=cleanup_worker(output)
    receipt={'arm':arm,'master_seed':seed,'requested_wall_seconds':wall,'full_process_wall_seconds':time.monotonic()-started,
             'hard_deadline_triggered':timed_out,'returncode':proc.returncode,'cleanup':cleanup,'purpose':purpose,
             'command_sha256':hashlib.sha256(json.dumps(argv).encode()).hexdigest()}
    (output.parent/(output.name+'.supervisor.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--arm',choices=['A','B','C'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--wall',type=float,required=True);p.add_argument('--batch',type=int,default=24);p.add_argument('--max-executions',type=int,default=1000000);p.add_argument('--purpose',choices=['control','formal'],default='control')
    a=p.parse_args();print(json.dumps(run_arm(a.output,a.arm,a.seed,a.wall,a.batch,a.max_executions,a.purpose)),flush=True)
if __name__=='__main__':main()
