"""Conservative bytecode domain proof; unsupported paths exclude nothing.

No compiler code is executed. The only fixed tensor facts are CUDA placement
and non-complex default-template inputs. Unknown dimensions/dtypes fork both
paths. Results are source-domain exclusions, never trial-hit exclusions.
"""
from dataclasses import dataclass
import dis,hashlib,json,operator,types
from pathlib import Path
from inventory import ROOT,sha
from common_domain import descriptor
import torch._inductor.decomposition as decomp
import torch._inductor.config as config

@dataclass(frozen=True)
class Symbol:
    kind:str
    value:object=None
UNKNOWN=Symbol('unknown')
BOOL_UNKNOWN=Symbol('unknown_bool')


def known(value):
    if isinstance(value,Symbol):return False
    if isinstance(value,(tuple,list,set)):return all(known(v) for v in value)
    return value is None or isinstance(value,(bool,int,float,str))


def truth(value):return bool(value) if known(value) else None


def attr(value,name):
    if value==Symbol('global','config') and name=='coordinate_descent_tuning':return False
    if name=='type' and isinstance(value,Symbol) and value.kind=='attr' and value.value[1]=='device' and isinstance(value.value[0],Symbol) and value.value[0].kind=='arg':return 'cuda'
    return Symbol('attr',(value,name))


def invoke(fn,args):
    if isinstance(fn,Symbol) and fn.kind=='global' and fn.value in ('statically_known_true','guard_or_false'):
        return args[0] if args and known(args[0]) else BOOL_UNKNOWN
    if isinstance(fn,Symbol) and fn.kind=='attr':
        owner,name=fn.value
        if name=='is_complex' and isinstance(owner,Symbol) and owner.kind=='arg':return False
        if owner==Symbol('global','torch') and name=='is_tensor':return BOOL_UNKNOWN
    return UNKNOWN


def prove(code):
    ins=list(dis.get_instructions(code));lookup={i.offset:n for n,i in enumerate(ins)}
    initial={name:Symbol('arg',name) for name in code.co_varnames[:code.co_argcount]}
    todo=[(0,[],initial)];seen=set();hit=set();created=set();errors=[];states=0
    def enqueue(index,stack,env):todo.append((index,list(stack),dict(env)))
    try:
        while todo:
            index,stack,env=todo.pop();key=(index,repr(stack),repr(sorted(env.items())))
            if key in seen:continue
            seen.add(key);states+=1
            if states>20000:raise ValueError('abstract_state_limit')
            i=ins[index];op=i.opname;next_index=index+1
            if op in ('RESUME','NOP','EXTENDED_ARG','PRECALL','PUSH_NULL','KW_NAMES','MAKE_CELL','COPY_FREE_VARS'):pass
            elif op=='LOAD_CONST':stack.append(i.argval)
            elif op=='LOAD_GLOBAL':stack.append(Symbol('global',i.argval))
            elif op in ('LOAD_FAST','LOAD_DEREF','LOAD_CLOSURE'):stack.append(env.get(i.argval,UNKNOWN))
            elif op in ('STORE_FAST','STORE_DEREF'):env[i.argval]=stack.pop()
            elif op in ('LOAD_ATTR','LOAD_METHOD'):stack.append(attr(stack.pop(),i.argval))
            elif op=='CALL':
                args=stack[-i.arg:] if i.arg else []
                if i.arg:del stack[-i.arg:]
                fn=stack.pop();stack.append(invoke(fn,args))
            elif op in ('BUILD_LIST','BUILD_TUPLE','BUILD_SET'):
                args=stack[-i.arg:] if i.arg else []
                if i.arg:del stack[-i.arg:]
                stack.append(tuple(args))
            elif op=='BUILD_SLICE':
                for _ in range(i.arg):stack.pop()
                stack.append(UNKNOWN)
            elif op=='BINARY_SUBSCR':
                sub=stack.pop();value=stack.pop()
                stack.append(value[sub] if known(value) and known(sub) else UNKNOWN)
            elif op=='BINARY_OP':
                right=stack.pop();left=stack.pop();fn={'+':operator.add,'-':operator.sub,'*':operator.mul,'//':operator.floordiv,'%':operator.mod,'/':operator.truediv}.get(i.argrepr)
                stack.append(fn(left,right) if fn and known(left) and known(right) else UNKNOWN)
            elif op=='COMPARE_OP':
                right=stack.pop();left=stack.pop();fn={'==':operator.eq,'!=':operator.ne,'<':operator.lt,'<=':operator.le,'>':operator.gt,'>=':operator.ge}.get(i.argval)
                stack.append(fn(left,right) if fn and known(left) and known(right) else BOOL_UNKNOWN)
            elif op=='CONTAINS_OP':
                right=stack.pop();left=stack.pop();stack.append(((left in right)^bool(i.arg)) if known(left) and known(right) else BOOL_UNKNOWN)
            elif op=='IS_OP':
                right=stack.pop();left=stack.pop();stack.append(((left is right)^bool(i.arg)) if known(left) and known(right) else BOOL_UNKNOWN)
            elif op=='UNARY_NOT':
                value=truth(stack.pop());stack.append(not value if value is not None else BOOL_UNKNOWN)
            elif op=='UNARY_NEGATIVE':
                value=stack.pop();stack.append(-value if known(value) else UNKNOWN)
            elif op=='STORE_SUBSCR':
                sub=stack.pop();container=stack.pop();value=stack.pop()
                if known(container):raise ValueError('unsupported_write_to_known_container')
            elif op=='POP_TOP':stack.pop()
            elif op=='COPY':stack.append(stack[-i.arg])
            elif op=='SWAP':stack[-1],stack[-i.arg]=stack[-i.arg],stack[-1]
            elif op=='MAKE_FUNCTION':
                nested=stack.pop()
                if isinstance(nested,types.CodeType):created.add((nested.co_filename,nested.co_firstlineno,sha(nested.co_code)))
                for flag in (1,2,4,8):
                    if i.arg&flag:stack.pop()
                stack.append(UNKNOWN)
            elif op in ('RETURN_VALUE','RAISE_VARARGS'):continue
            elif op.startswith('JUMP') and 'IF' not in op:
                enqueue(lookup[i.argval],stack,env);continue
            elif 'JUMP' in op and 'IF' in op:
                conditional=stack[-1]
                if op.endswith('IF_NONE'):result=(conditional is None) if known(conditional) else None
                elif op.endswith('IF_NOT_NONE'):result=(conditional is not None) if known(conditional) else None
                else:
                    result=truth(conditional)
                    if 'IF_FALSE' in op and result is not None:result=not result
                for take in ([False,True] if result is None else [result]):
                    target=i.argval if take else ins[index+1].offset;branch_stack=list(stack)
                    if op.startswith('POP_JUMP') or not take:branch_stack.pop()
                    elif conditional==BOOL_UNKNOWN:
                        branch_stack[-1]=('IF_TRUE' in op)
                    hit.add((i.offset,target));enqueue(lookup[target],branch_stack,env)
                continue
            else:raise ValueError('unsupported_opcode:'+op)
            if next_index<len(ins):enqueue(next_index,stack,env)
    except Exception as exc:errors.append(type(exc).__name__+':'+str(exc))
    return {'complete_for_supported_abstract_model':not errors,'possible_outcomes':sorted(hit),'created_nested_identities':[list(k) for k in sorted(created)],'abstract_states':states,'errors':errors}


def main():
    out=ROOT/'branch_domain_audit_v3';out.mkdir(exist_ok=False)
    inventory=json.loads((ROOT/'forward_source_inventory_v1/inventory.json').read_text());domain=descriptor()
    assert config.coordinate_descent_tuning is False
    rows=[];results={};excluded=[]
    for row in inventory['sources']:
        source=row['source'];identity=(source['source'],source['first_line'],source['code_sha256'])
        audit={'source_identity':list(identity),'qualname':source['qualname'],'outcomes':[], 'proof':None}
        # Source selection is by compiler module identity, not trial success.
        direct=identity[0].endswith('/torch/_inductor/decomposition.py') and row['kind']=='registered_entry'
        if direct:
            fn=getattr(decomp,source['qualname'],None)
            if fn is not None:
                while hasattr(fn,'__wrapped__'):fn=fn.__wrapped__
                if sha(fn.__code__.co_code)!=identity[2]:raise ValueError('audit_code_identity_mismatch')
                result=prove(fn.__code__);results[identity]=result;audit['proof']=result
                if result['complete_for_supported_abstract_model']:
                    possible={tuple(e) for e in result['possible_outcomes']}
                    for edge in source['conditional_outcomes']:
                        outside=tuple(edge) not in possible
                        item={'edge':edge,'status':'proven_outside_fixed_domain' if outside else 'unknown_or_possible_retained'}
                        audit['outcomes'].append(item)
                        if outside:excluded.append({'source_identity':list(identity),**item,'proof':'sound abstract CUDA/noncomplex/configFalse execution overapproximates all unknown shapes/dtypes'})
        if not audit['outcomes']:
            audit['outcomes']=[{'edge':e,'status':'unknown_retained'} for e in source['conditional_outcomes']]
        rows.append(audit)
    # A lexical nested function cannot execute if its MAKE_FUNCTION is never
    # reached on any parent path. Apply only to fully supported direct proofs.
    for row,audit in zip(inventory['sources'],rows):
        parent=tuple(row.get('parent_identity',[]));result=results.get(parent)
        identity=tuple(audit['source_identity'])
        if result and result['complete_for_supported_abstract_model'] and identity not in {tuple(k) for k in result['created_nested_identities']}:
            for item in audit['outcomes']:
                item['status']='proven_outside_fixed_domain';excluded.append({'source_identity':list(identity),**item,'proof':'nested MAKE_FUNCTION unreachable on all supported parent paths'})
    report={'control_only':True,'domain':domain,'fixed_config':{'coordinate_descent_tuning':False},'rows':rows,'exclusions':excluded,
            'abstract_assumptions':['all compiler tensor arguments attributable to generated DAG stay on CUDA','default-template tensor/scalar dtypes contain no complex','unknown shape,dtype,param branches are forked; unsupported functions exclude nothing'],
            'primary_denominator_committed':False,'effect_experiment_started':False,'proof_not_yet_independently_qualified':True,
            'source_sha256':sha(Path(__file__).read_bytes())}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={'source_outcome_candidates':sum(len(r['outcomes']) for r in rows),'proposed_domain_exclusions':len(excluded),
             'fully_supported_direct_proofs':sum(r['complete_for_supported_abstract_model'] for r in results.values()),
             'unsupported_proofs':{r['qualname']:r['proof']['errors'] for r in rows if r['proof'] and not r['proof']['complete_for_supported_abstract_model']},
             'primary_denominator_committed':False,'independent_qualification_required':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
