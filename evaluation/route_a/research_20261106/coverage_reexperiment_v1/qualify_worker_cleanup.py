"""Terminate only a verified real experiment worker, testing PID identity gates."""
import json,os,signal,subprocess,sys
from pathlib import Path
from experiment_supervisor import cleanup_worker,process_start,ROOT


def main():
    out=ROOT/'worker_cleanup_controls_v1';out.mkdir(exist_ok=False)
    with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
        proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'experiment_runtime.py'),'--case',str(ROOT/'dag_controls_v2/case_002.json'),'--output',str(out/'execute')],stdout=stdout,stderr=stderr,start_new_session=True)
        record={'pid':proc.pid,'start_ticks':'wrong-start-tick','state':'running'}
        (out/'active_worker.json').write_text(json.dumps(record))
        rejected=cleanup_worker(out);assert rejected['status']=='worker_missing_or_pid_reused'
        record['start_ticks']=process_start(proc.pid);(out/'active_worker.json').write_text(json.dumps(record))
        killed=cleanup_worker(out);proc.wait(timeout=5)
        assert killed['status']=='killed_verified_run_worker' and proc.returncode==-signal.SIGKILL,(killed,proc.returncode)
    result={'control_only':True,'wrong_process_start_refused':True,'verified_real_worker_group_terminated':True,'returncode':proc.returncode}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
