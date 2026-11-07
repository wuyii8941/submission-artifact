"""Independent concrete boundary controls for proposed source-domain exclusions."""
import itertools,json,time
from pathlib import Path
from inventory import ROOT,sha
from common_domain import DTYPES
from runtime_audit import SourceAudit,source_catalog
import torch
import torch._inductor.decomposition as decomp


def main():
    out=ROOT/'branch_domain_controls_v1';out.mkdir(exist_ok=False);started=time.monotonic()
    audit=json.loads((ROOT/'branch_domain_audit_v3/results.json').read_text());catalog=source_catalog()
    excluded={(tuple(r['source_identity']),*r['edge']) for r in audit['exclusions']}
    functions={}
    for row in audit['rows']:
        if row['proof'] is None:continue
        fn=getattr(decomp,row['qualname'])
        while hasattr(fn,'__wrapped__'):fn=fn.__wrapped__
        functions[row['qualname']]=fn
    observer=SourceAudit(catalog);rows=[];errors=[]
    def run(name,args,label):
        try:
            with observer:functions[name](*args)
            rows.append({'function':name,'label':label,'status':'executed'})
        except Exception as exc:errors.append({'function':name,'label':label,'error':type(exc).__name__+':'+str(exc)})
    for dtype in DTYPES:
        for m,k,n in itertools.product((0,1,2),repeat=3):
            a=torch.ones((m,k),device='cuda',dtype=dtype);b=torch.ones((k,n),device='cuda',dtype=dtype)
            run('mm',(a,b),[str(dtype),m,k,n]);run('addmm',(torch.ones((m,n),device='cuda',dtype=dtype),a,b),[str(dtype),m,k,n])
        for batch,m,k,n in itertools.product((0,1,2),repeat=4):
            run('bmm',(torch.ones((batch,m,k),device='cuda',dtype=dtype),torch.ones((batch,k,n),device='cuda',dtype=dtype)),[str(dtype),batch,m,k,n])
        for shape in ((),(0,),(1,),(2,3)):
            t=torch.ones(shape,device='cuda',dtype=dtype)
            run('add',(t,t),[str(dtype),list(shape)])
            run('amax',(t,),[str(dtype),list(shape)]);run('amin',(t,),[str(dtype),list(shape)])
    actual={(tuple(c['source_identity']),*e) for c in observer.calls for e in c['hit_outcomes']}
    violations=sorted(actual&excluded);assert not violations,violations
    negative=SourceAudit(catalog)
    # Deliberately violate fixed CUDA or noncomplex assumptions. Their events
    # must reach exclusions, confirming that these really are domain bounds.
    with negative:
        functions['bmm'](torch.ones((2,1,2)),torch.ones((2,2,1)))
        functions['mm'](torch.ones((2,1)),torch.ones((1,2)))
        functions['add'](torch.ones((2,3),device='cuda',dtype=torch.complex64),torch.ones((2,3),device='cuda',dtype=torch.complex64))
    negative_hits={(tuple(c['source_identity']),*e) for c in negative.calls for e in c['hit_outcomes']}
    crossed=negative_hits&excluded;assert crossed,'outside-domain controls did not cross exclusions'
    assert not errors,errors[:10]
    torch.cuda.synchronize()
    result={'control_only':True,'boundary_cases':len(rows),'dtype_values':[str(d) for d in DTYPES],
            'matrix_dimension_representatives':[0,1,2],'scope':'all CUDA equality-to-one decision regions; concrete control does not replace abstract proof',
            'in_domain_observed_excluded_outcomes':violations,'out_of_domain_observed_excluded_outcomes':[list(k) for k in sorted(crossed)],
            'all_in_domain_calls_executed':True,'primary_denominator_committed':False,'effect_experiment_started':False,
            'full_wall_seconds':time.monotonic()-started,'source_sha256':{f:sha((ROOT/f).read_bytes()) for f in ('branch_domain_audit.py','qualify_branch_domain.py','runtime_audit.py')}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('boundary_cases','in_domain_observed_excluded_outcomes','full_wall_seconds')}))
if __name__=='__main__':main()
