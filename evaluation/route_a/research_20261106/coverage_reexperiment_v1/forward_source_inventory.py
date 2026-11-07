"""Source-first registry/schema audit, independent of trial coverage hits.

All common operators and uncertain functional overloads remain. Only mutating
schemas are excluded from the original-ATen membership domain, using the
common emission guard. Compiler-generated extra ATen aliases remain an open
world limitation; out-of-catalog events must be reported as such.
"""
import ast,dis,functools,hashlib,inspect,json,types
from pathlib import Path
from collections import Counter
from inventory import ROOT,describe,sha
from common_pool import snapshot
from torchfuzz.operators import list_operators
import torch
import torch._inductor.decomposition as decomp
import torch._inductor.lowering as lowering


def identity(code):return (code.co_filename,code.co_firstlineno,sha(code.co_code))


def main():
    out=ROOT/'forward_source_inventory_v1';out.mkdir(exist_ok=False)
    aliases={'softmax':['_softmax','softmax'],'matmul':['matmul','dot','mv','mm','bmm'],'square':['square','pow']}
    operators=[];sources={};functions={};excluded=[]
    def register(fn,registration):
        seen=set()
        while callable(fn) and id(fn) not in seen:
            seen.add(id(fn))
            if isinstance(fn,functools.partial):fn=fn.func;continue
            code=getattr(fn,'__code__',None)
            if isinstance(code,types.CodeType):
                key=identity(code)
                if key not in sources:sources[key]={'source':describe(fn)[0],'registrations':[],'kind':'registered_entry'};functions[key]=code
                if registration not in sources[key]['registrations']:sources[key]['registrations'].append(registration)
            fn=getattr(fn,'__wrapped__',None)
    for public in snapshot()['accepted']:
        names=aliases.get(public.rsplit('.',1)[-1],[public.rsplit('.',1)[-1]])
        overloads=[]
        for family in names:
            packet=getattr(torch.ops.aten,family,None)
            if packet is None:continue
            for name in packet.overloads():
                op=getattr(packet,name);schema=op._schema
                mutable=bool(schema.is_mutable) or any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments)
                row={'aten':str(op),'schema':str(schema),'family_rule':family,'membership':'excluded_mutating_schema' if mutable else 'functional_candidate_unknown_retained',
                     'proof':'functional_domain.py rejects out=,inplace=True and mutation in emitted code; original public call is forward functional' if mutable else None}
                overloads.append(row)
                if mutable:excluded.append({'public_operator':public,**row});continue
                for stage,table in [('inductor_decomposition',decomp.decompositions),('inductor_lowering',lowering.lowerings)]:
                    key=op if op in table else packet if packet in table else None
                    if key is not None:register(table[key],{'public_operator':public,'aten':str(op),'stage':stage,'mapping_evidence':'source_schema_candidate_not_runtime_claim'})
        operators.append({'public_operator':public,'family_rules':names,'overloads':overloads,'unresolved_nonmutating_candidates_retained':True})
    roots=list(functions)
    def descend(code,parent):
        for child in code.co_consts:
            if not isinstance(child,types.CodeType):continue
            key=identity(child)
            if key not in sources:
                instructions=list(dis.get_instructions(child));edges=[]
                for i,op in enumerate(instructions[:-1]):
                    if ('JUMP' in op.opname and 'IF' in op.opname) or op.opname=='FOR_ITER':edges.extend([op.offset,dst] for dst in sorted({op.argval,instructions[i+1].offset}))
                sources[key]={'source':{'source':key[0],'first_line':key[1],'code_sha256':key[2],'source_sha256':sha(Path(key[0]).read_bytes()),'qualname':child.co_qualname,'conditional_outcomes':edges},
                              'registrations':list(sources[parent]['registrations']),'kind':'nested_registered_provider_code','parent_identity':list(parent)}
                functions[key]=child
            descend(child,key)
    for key in roots:descend(functions[key],key)
    for key,row in sources.items():
        code=functions[key];row['source']['qualname']=code.co_qualname
        row['formal_args']=list(code.co_varnames[:code.co_argcount]);row['varargs']=bool(code.co_flags&inspect.CO_VARARGS)
        row['outcome_membership']='unknown_retained_pending_branch_domain_audit'
    report={'schema':1,'torch':torch.__version__,'commit':torch.version.git_version,'operators':operators,'sources':list(sources.values()),
            'mutating_schema_exclusions':excluded,'aliases':aliases,'root_entry_code_count':len(roots),
            'selection_rule':'all nonmutating nominal forward schemas for common55 with source-known frontend aliases; all provider chains and nested code recursively included; no trial outcomes read',
            'scope':'registered decomposition/lowering provider code and lexical nested code; not every compiler helper or GPU kernel',
            'unknown_membership_policy':'retain unknown nonmutating overloads and branch reachability; do not exclude from poor hit rate',
            'mapping_completeness':'source-declared alias families; not all compiler-created ATen intermediates',
            'primary_denominator_committed':False,'effect_experiment_started':False,
            'source_sha256':{f:sha((ROOT/f).read_bytes()) for f in ('forward_source_inventory.py','functional_domain.py','dag_render.py')}}
    (out/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={'operators':len(operators),'nominal_and_alias_overloads':sum(len(r['overloads']) for r in operators),'mutating_schema_exclusions':len(excluded),
             'root_code_objects':len(roots),'total_code_objects':len(sources),'source_outcomes':sum(len(r['source']['conditional_outcomes']) for r in sources.values()),
             'primary_denominator_committed':False,'effect_experiment_started':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
