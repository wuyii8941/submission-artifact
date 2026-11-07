"""Isolated semantic controls; never discovery tests or a DAG generator."""
import copy
import hashlib
import json
from pathlib import Path
import random
from inventory import ROOT
import torch
from torchfuzz.operators import list_operators
from torchfuzz.tensor_fuzzer import TensorSpec

def check(code,inputs,names,output):
    try:
        ns={'torch':torch}
        for name,spec in zip(names,inputs):
            if isinstance(spec,TensorSpec):ns[name]=torch.empty_strided(spec.size,spec.stride,dtype=spec.dtype,device='meta')
            elif spec.constant is not None:ns[name]=spec.constant
            elif spec.dtype.is_floating_point:ns[name]=1.0
            elif spec.dtype.is_complex:ns[name]=complex(1,0)
            elif spec.dtype==torch.bool:ns[name]=True
            else:ns[name]=1
        with torch.device('meta'):exec(compile(code,'<operator-semantics-control>','exec'),ns)
        result=ns['probe_output']
        return {'status':'meta_executable','shape':list(result.shape) if isinstance(result,torch.Tensor) else None,
                'dtype':str(result.dtype) if isinstance(result,torch.Tensor) else type(result).__name__,
                'matches_output':isinstance(result,torch.Tensor) and tuple(result.shape)==output.size and result.dtype==output.dtype}
    except Exception as exc:return {'status':'error','type':type(exc).__name__,'error':str(exc)[-1200:]}

def main():
    inventory=json.loads((ROOT/'source_inventory_v2/inventory.json').read_text());registry=list_operators()
    out=ROOT/'semantic_controls_v3';out.mkdir(exist_ok=False);rows=[]
    shapes=[(),(3,),(2,3),(2,3,4)]
    for index,entry in enumerate(inventory['operators']):
        name=entry['public_operator'];row={'operator':name,'excluded_from_discovery':True,'control_only':True}
        random.seed(2026110600+index)
        try:
            assert len(entry['registry_matches'])==1
            op=copy.deepcopy(registry[entry['registry_matches'][0]['key']]);output=None
            for dtype in (torch.float32,torch.int64):
                for shape in shapes:
                    stride=[];s=1
                    for n in reversed(shape):stride.insert(0,s);s*=max(1,n)
                    candidate=TensorSpec(size=shape,stride=tuple(stride),dtype=dtype)
                    if op.can_produce(candidate):output=candidate;break
                if output is not None:break
            if output is None:raise ValueError('no_supported_control_output_spec')
            inputs=op.fuzz_inputs_specs(output);names=[f'probe_input_{i}' for i in range(len(inputs))]
            render_seed=202611066000+index;random.seed(render_seed)
            first=op.codegen('probe_output',names,output)
            random.seed(render_seed)
            second=op.codegen('probe_output',names,output)
            row.update(first_code=first,second_code=second,render_identical=first==second,
                render_seed=render_seed,
                first_meta=check(first,inputs,names,output),second_meta=check(second,inputs,names,output),
                status='probed',output={'shape':list(output.size),'dtype':str(output.dtype)})
        except Exception as exc:row.update(status='probe_error',error_type=type(exc).__name__,error=str(exc)[-1200:])
        rows.append(row)
    report={'scope':'one isolated semantic control per common operator; no claim of complete legal domain or compiler coverage',
            'all55retained':len(rows)==55,'denominator_selection_uses_these_results':False,'same_random_seed_before_each_codegen':True,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={'operators_probed':len(rows),'render_changes':[r['operator'] for r in rows if r.get('render_identical') is False],
        'first_meta_failures':[r['operator'] for r in rows if r.get('first_meta',{}).get('status')=='error'],
        'second_meta_failures':[r['operator'] for r in rows if r.get('second_meta',{}).get('status')=='error'],
        'first_output_spec_mismatches':[r['operator'] for r in rows if r.get('first_meta',{}).get('matches_output') is False],
        'second_output_spec_mismatches':[r['operator'] for r in rows if r.get('second_meta',{}).get('matches_output') is False],
        'probe_errors':[r['operator'] for r in rows if r['status']=='probe_error'],'effect_experiment_started':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
