"""All-pool semantic controls: private state, serialization and RNG isolation."""
import copy
import hashlib
import json
import random
from inventory import ROOT
from frozen_codegen import freeze,render
from probe_operator_semantics import check
import torch
from torchfuzz.operators import list_operators
from torchfuzz.tensor_fuzzer import TensorSpec

def main():
    out=ROOT/'frozen_codegen_controls_v1';out.mkdir(exist_ok=False)
    inv=json.loads((ROOT/'source_inventory_v2/inventory.json').read_text());registry=list_operators();rows=[]
    for i,entry in enumerate(inv['operators']):
        row={'operator':entry['public_operator'],'control_only':True}
        try:
            key=entry['registry_matches'][0]['key'];op=copy.deepcopy(registry[key]);random.seed(2026110600+i)
            output=None
            for dtype in (torch.float32,torch.int64):
                for shape in ((),(3,),(2,3),(2,3,4)):
                    stride=[];s=1
                    for n in reversed(shape):stride.insert(0,s);s*=max(1,n)
                    candidate=TensorSpec(shape,tuple(stride),dtype)
                    if op.can_produce(candidate):output=candidate;break
                if output is not None:break
            if output is None:raise ValueError('no_control_spec')
            inputs=op.fuzz_inputs_specs(output);names=[f'input_{j}' for j in range(len(inputs))];seed=202611066000+i
            snapshot=freeze(op,key,seed);before=json.dumps(snapshot,sort_keys=True)
            random.seed(seed);baseline=op.codegen('probe_output',names,output)
            a=render(snapshot,'probe_output',names,output)
            # Interleave different state from the same operator before replay.
            op.fuzz_inputs_specs(output);interference=freeze(op,key,seed+1)
            render(interference,'other_output',names,output)
            b=render(json.loads(json.dumps(snapshot)),'probe_output',names,output)
            meta=check(a,inputs,names,output)
            row.update(status='probed',snapshot=snapshot,code=a,meta=meta,
                checks={'first_emission_preserved':a==baseline,'interleaved_repeat_identical':a==b,
                        'snapshot_unchanged':before==json.dumps(snapshot,sort_keys=True)})
            # Measure RNG preservation independently of deliberate interference.
            state=random.getstate();render(snapshot,'probe_output',names,output)
            row['checks']['rng_preserved_during_render']=state==random.getstate()
            bad=copy.deepcopy(snapshot);bad['payload']['render_seed']+=1
            try:render(bad,'probe_output',names,output);tamper=False
            except ValueError:tamper=True
            row['checks']['tampered_snapshot_rejected']=tamper
        except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[-1200:])
        rows.append(row)
    qualified=len(rows)==55 and all(r['status']=='probed' and all(r['checks'].values()) and r['meta']['status']=='meta_executable' for r in rows)
    result={'qualified_for_stable_reemission':qualified,'not_qualified_for_graph_binding_or_full_domain':True,
            'excluded_from_discovery':True,'rows':rows,
            'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('frozen_codegen.py','qualify_frozen_codegen.py')}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    summary={'operators':len(rows),'stable_reemission_qualified':qualified,'errors':[r['operator'] for r in rows if r['status']=='error'],
        'remaining_output_spec_mismatches':[r['operator'] for r in rows if r.get('meta',{}).get('matches_output') is False],
        'domain_fully_qualified':False,'effect_experiment_started':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    if not qualified:raise SystemExit(1)
if __name__=='__main__':main()
