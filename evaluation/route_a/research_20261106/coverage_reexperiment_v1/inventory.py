"""Pre-experiment source inventory, never an observed-hit denominator.

Nominal public-name -> ATen overload associations are hypotheses here. Actual
schema/FX attribution and domain reachability must qualify before committing
any denominator. All operators, missing mappings and source failures remain.
"""
import dis
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
OLD=REPO/'evaluation/route_a/research_20261103/unified_condition_v2'
SITE='/data1/tzh/pt_nightly_env/lib/python3.11/site-packages'
sys.path[:0]=[str(OLD),str(REPO),str(REPO/'third_party'),SITE]
import torch
import torch._inductor.decomposition as decomp
import torch._inductor.lowering as lowering
from common_pool import snapshot
from torchfuzz.operators import list_operators

def sha(data):return hashlib.sha256(data).hexdigest()
def describe(fn):
    chain=[];seen=set()
    while callable(fn) and id(fn) not in seen:
        seen.add(id(fn))
        if isinstance(fn,functools.partial):fn=fn.func;continue
        code=getattr(fn,'__code__',None)
        row={'module':getattr(fn,'__module__',None),'qualname':getattr(fn,'__qualname__',str(fn))}
        if isinstance(code,types.CodeType):
            path=Path(code.co_filename)
            instructions=list(dis.get_instructions(code));edges=[]
            for i,op in enumerate(instructions[:-1]):
                if ('JUMP' in op.opname and 'IF' in op.opname) or op.opname=='FOR_ITER':
                    edges.extend([op.offset,dst] for dst in sorted({op.argval,instructions[i+1].offset}))
            row.update(source=str(path),first_line=code.co_firstlineno,code_sha256=sha(code.co_code),
                       source_sha256=sha(path.read_bytes()) if path.is_file() else None,
                       conditional_outcomes=edges,nested_code_objects_excluded=True)
        else:row['source_status']='no_python_code_object'
        chain.append(row)
        if not hasattr(fn,'__wrapped__'):break
        fn=fn.__wrapped__
    return chain

def main():
    out=ROOT/'source_inventory_v2';out.mkdir(exist_ok=False)
    pool=snapshot();operators=[];registry=list_operators()
    # Semantic alias hypothesis, not a target-specific graph or an assertion
    # that every overload can occur in this generator's actual forward domain.
    aliases={'softmax':['_softmax','softmax']}
    for name in pool['accepted']:
        leaf=name.rsplit('.',1)[-1];names=aliases.get(leaf,[leaf]);entries=[]
        for family in names:
            try:packet=getattr(torch.ops.aten,family);overloads=packet.overloads()
            except (AttributeError,RuntimeError):continue
            for overload_name in overloads:
                op=getattr(packet,overload_name)
                providers=[]
                for stage,table in [('inductor_decomposition',decomp.decompositions),('inductor_lowering',lowering.lowerings)]:
                    if op in table:providers.append({'stage':stage,'registration_key':str(op),'callable_chain':describe(table[op])})
                    elif packet in table:providers.append({'stage':stage,'registration_key':str(packet),'callable_chain':describe(table[packet])})
                entries.append({'aten_overload':str(op),'schema':str(op._schema),'providers':providers,
                                'membership_in_generator_domain':'unverified'})
        matches=[(key,op) for key,op in registry.items() if key==name or op.torch_op_name==name]
        operator=matches[0][1] if len(matches)==1 else None
        operators.append({'public_operator':name,'torchfuzz_registry_present':operator is not None,
            'registry_matches':[{'key':key,'class':type(op).__name__} for key,op in matches],
            'torchfuzz_class':type(operator).__name__ if operator is not None else None,
            'torchfuzz_source':inspect.getsourcefile(type(operator)) if operator is not None else None,
            'nominal_aten_family_candidates':names,'aten_candidates':entries,
            'mapping_status':'requires_runtime_schema_confirmation' if entries else 'unresolved_public_to_aten_mapping',
            'reachability_status':'not_audited','primary_denominator_eligible':False})
    report={'schema':'operator-implementation-source-inventory-v1','torch':torch.__version__,'commit':torch.version.git_version,
        'pool':pool,'operators':operators,'purpose':'enumerate sources before choosing targets; not count coverage or claim reachability',
        'denominator_committed':False,'effect_experiment_authorized_by_gates':False,
        'rule':'all55common public operators retained; enumerate all nominal ATen overloads and registered providers, including non-forward/out overloads pending explicit domain audit',
        'source_sha256':sha(Path(__file__).read_bytes())}
    assert len(operators)==55 and report['commit']=='11ff36ef30d23635f4e9bfe414ae027471901134'
    (out/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={'operators':len(operators),'registered_torchfuzz_operators':sum(r['torchfuzz_registry_present'] for r in operators),
        'nominal_overloads':sum(len(r['aten_candidates']) for r in operators),
        'operators_without_nominal_mapping':[r['public_operator'] for r in operators if not r['aten_candidates']],
        'operators_without_inductor_provider':[r['public_operator'] for r in operators if not any(e['providers'] for e in r['aten_candidates'])],
        'denominator_committed':False,'effect_experiment_started':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
