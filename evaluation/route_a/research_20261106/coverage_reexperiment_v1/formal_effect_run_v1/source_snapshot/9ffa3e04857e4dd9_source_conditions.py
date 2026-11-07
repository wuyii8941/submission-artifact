"""Source-order declarative condition inventory, no generated graph templates."""
import ast,dis,hashlib,json,types
from pathlib import Path
from inventory import ROOT,sha
from constraints import evaluate,Unknown
from constraints_v4 import node_facts as base_facts


def expr(op,*args):return {'op':op,'args':list(args)}
class Unsupported(ValueError):pass


def integer(node):
    if isinstance(node,ast.Constant) and type(node.value) is int:return node.value
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -integer(node.operand)
    raise Unsupported('nonliteral_axis')


def translate(node,parameters,definitions,seen=()):
    t=lambda n:translate(n,parameters,definitions,seen)
    def tensor_fact(base,prop):
        if not isinstance(base,ast.Name) or base.id not in parameters:raise Unsupported('unmapped_tensor_expression:'+ast.unparse(base))
        return expr('fact',f'input.{parameters.index(base.id)}.{prop}')
    if isinstance(node,ast.Constant) and isinstance(node.value,(bool,int,float,str,type(None))):return node.value
    if isinstance(node,ast.Name) and node.id in definitions and node.id not in seen:
        return translate(definitions[node.id],parameters,definitions,seen+(node.id,))
    if isinstance(node,ast.BoolOp):return expr('and' if isinstance(node.op,ast.And) else 'or',*(t(n) for n in node.values))
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):return expr('not',t(node.operand))
    if isinstance(node,ast.Compare) and len(node.ops)==1:
        relation={ast.Eq:'eq',ast.NotEq:'ne',ast.Lt:'lt',ast.LtE:'le',ast.Gt:'gt',ast.GtE:'ge',ast.Is:'eq',ast.IsNot:'ne'}.get(type(node.ops[0]))
        if relation:return expr(relation,t(node.left),t(node.comparators[0]))
        if isinstance(node.ops[0],(ast.In,ast.NotIn)) and isinstance(node.comparators[0],(ast.List,ast.Tuple,ast.Set)):
            value=expr('in',t(node.left),*(t(n) for n in node.comparators[0].elts))
            return expr('not',value) if isinstance(node.ops[0],ast.NotIn) else value
    if isinstance(node,ast.Attribute):
        text=ast.unparse(node)
        if text=='config.coordinate_descent_tuning':return False
        if text.startswith('torch.') and node.attr in ('bool','float16','bfloat16','float32','float64','int8','int16','int32','int64'):return text
        if node.attr=='ndim':return tensor_fact(node.value,'rank')
        if node.attr=='dtype':return tensor_fact(node.value,'dtype')
        if node.attr=='type' and isinstance(node.value,ast.Attribute) and node.value.attr=='device':return 'cuda'
    if isinstance(node,ast.Subscript):
        if isinstance(node.value,ast.Attribute) and node.value.attr=='shape':return tensor_fact(node.value.value,'size.'+str(integer(node.slice)))
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='stride' and not node.value.args:
            return tensor_fact(node.value.func.value,'stride.'+str(integer(node.slice)))
    if isinstance(node,ast.Call):
        func=ast.unparse(node.func)
        if func in ('statically_known_true','guard_or_false') and len(node.args)==1:return t(node.args[0])
        if func=='torch.is_tensor' and len(node.args)==1:return tensor_fact(node.args[0],'is_tensor')
        if isinstance(node.func,ast.Attribute):
            if node.func.attr in ('dim','is_complex','is_contiguous') and not node.args:
                return tensor_fact(node.func.value,{'dim':'rank','is_complex':'is_complex','is_contiguous':'is_contiguous'}[node.func.attr])
            if node.func.attr in ('size','stride') and len(node.args)==1:
                return tensor_fact(node.func.value,node.func.attr+'.'+str(integer(node.args[0])))
    raise Unsupported('unsupported_source_expression:'+ast.unparse(node))


def local_ifs(fn):
    result=[];assignments={}
    def visit(node):
        if node is not fn and isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):return
        if isinstance(node,ast.If):result.append(node)
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):assignments.setdefault(target.id,[]).append(node.value)
        for child in ast.iter_child_nodes(node):visit(child)
    visit(fn)
    return result,{name:values[0] for name,values in assignments.items() if len(values)==1}


def exits(code,node):
    instructions=list(dis.get_instructions(code));lookup={i.offset:i for i in instructions}
    start=min(i.offset for i in instructions if i.positions.lineno and node.test.lineno<=i.positions.lineno<=node.test.end_lineno)
    body=min(i.offset for i in instructions if i.positions.lineno==node.body[0].lineno)
    result=[]
    for index,i in enumerate(instructions[:-1]):
        if not start<=i.offset<body or not ('JUMP' in i.opname and 'IF' in i.opname):continue
        for dst in (i.argval,instructions[index+1].offset):
            if dst==body:result.append({'edge':[i.offset,dst],'outcome':'true'})
            elif dst>body and lookup[dst].positions.lineno is not None and lookup[dst].positions.lineno>node.end_lineno:result.append({'edge':[i.offset,dst],'outcome':'false'})
    if set(r['outcome'] for r in result)!={'true','false'}:raise Unsupported('logical_exit_mapping_not_supported')
    return result


def facts(graph,node_id):
    result=base_facts(graph,node_id)
    for index,nid in enumerate(graph.nodes[node_id].input_nodes):
        spec=graph.nodes[nid].output_spec;prefix=f'input.{index}';result[prefix+'.is_tensor']=hasattr(spec,'size');result[prefix+'.is_complex']=spec.dtype.is_complex
        if hasattr(spec,'size'):
            for axis in range(len(spec.size)):
                result[f'{prefix}.size.{axis-len(spec.size)}']=spec.size[axis]
                result[f'{prefix}.stride.{axis-len(spec.size)}']=spec.stride[axis]
    return result


def main():
    out=ROOT/'source_condition_inventory_v2';out.mkdir(exist_ok=False)
    inventory=json.loads((ROOT/'forward_source_inventory_v1/inventory.json').read_text())
    original=json.loads((ROOT/'source_inventory_v2/inventory.json').read_text());registry={r['public_operator']:r['registry_matches'][0]['key'] for r in original['operators']}
    audit=json.loads((ROOT/'branch_domain_audit_v3/results.json').read_text());excluded={(tuple(r['source_identity']),*r['edge']) for r in audit['exclusions']}
    parsed={};compiled={};conditions=[];unparsed=[]
    for entry in sorted(inventory['sources'],key=lambda r:(r['source']['source'],r['source']['first_line'],r['source']['code_sha256'])):
        source=entry['source'];path=Path(source['source']);identity=(str(path),source['first_line'],source['code_sha256'])
        if path not in parsed:
            text=path.read_text();parsed[path]=ast.parse(text);objects={}
            def collect(code):
                objects[(str(path),code.co_firstlineno,sha(code.co_code))]=code
                for nested in code.co_consts:
                    if isinstance(nested,types.CodeType):collect(nested)
            collect(compile(text,str(path),'exec',dont_inherit=True));compiled[path]=objects
        candidates=[node for node in ast.walk(parsed[path]) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and
                    min([node.lineno]+[d.lineno for d in node.decorator_list])==source['first_line'] and node.name==source['qualname'].rsplit('.',1)[-1]]
        code=compiled[path].get(identity)
        if len(candidates)!=1 or code is None:
            unparsed.append({'source_identity':list(identity),'reason':'code_without_unique_statement_function_ast','outcomes_retained':len(source['conditional_outcomes'])});continue
        fn=candidates[0];ifs,definitions=local_ifs(fn)
        for node in ifs:
            row={'decision_id':sha(json.dumps([list(identity),node.lineno]).encode())[:24],
                 'source_identity':list(identity),'source_sha256':source['source_sha256'],'if_line':node.lineno,
                 'source_expression':ast.unparse(node.test),'target_ops':sorted({registry[r['public_operator']] for r in entry['registrations']}),
                 'actual_aten_candidates':sorted({r['aten'] for r in entry['registrations']}),'predicate':None,'observation':None}
            try:
                row['observation']={'kind':'actual_attributed_logical_jump_exit','edges':exits(code,node)}
                if all((identity,*edge['edge']) in excluded for edge in row['observation']['edges']):row['domain_status']='outside_fixed_domain'
                else:row['domain_status']='possible_or_unknown_retained'
                if entry['varargs']:raise Unsupported('variadic_provider_arguments_not_mapped_to_graph_inputs')
                row['predicate']=translate(node.test,entry['formal_args'],definitions)
                text=json.dumps(row['predicate']);row['controllable_attributes']=[name for name,needle in [('shape','.size.'),('dtype','.dtype'),('dtype','.is_complex'),('layout','.stride.'),('layout','.is_contiguous')] if needle in text]
                row['controllable_attributes']=sorted(set(row['controllable_attributes']))
                row['guidance_status']='input_condition_specified' if row['controllable_attributes'] else 'fixed_or_structural_no_binding_action'
            except (Unsupported,ValueError) as exc:row.update(guidance_status='unsupported_retained',unsupported_reason=str(exc))
            conditions.append(row)
    report={'selection_rule':'every statement-if in every registered/lexically-nested provider from the fixed forward-source inventory; no trial hit data read',
            'conditions':conditions,'unparsed_source_records':unparsed,'primary_coverage_unit_unchanged':'conditional jump outcome; logical ifs here only define guidance and feedback',
            'limitations':['source formal tensor argument position is a prediction; runtime actual ATen and provenance remain authoritative','unsupported expressions and non-if control flow retained in primary source universe'],
            'primary_denominator_committed':False,'effect_experiment_started':False,'source_sha256':sha(Path(__file__).read_bytes())}
    (out/'conditions.json').write_text(json.dumps(report,indent=2)+'\n')
    from collections import Counter
    summary={'statement_conditions':len(conditions),'guidance_statuses':dict(Counter(r['guidance_status'] for r in conditions)),
             'domain_statuses':dict(Counter(r.get('domain_status','unresolved') for r in conditions)),
             'source_records_without_statement_if_mapping':len(unparsed),'primary_denominator_committed':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
