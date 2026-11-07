"""Map source-verified jump events to a declared AST decision for scheduling.

Logical decision outcomes here are feedback units, not a replacement for the
primary conditional-bytecode coverage unit. No graph generation occurs here.
"""
import ast
import dis
import hashlib
import inspect
from pathlib import Path
from inventory import describe
import torch._inductor.decomposition as decomp


def resolve(spec):
    fn=getattr(decomp,spec['source_function'])
    while hasattr(fn,'__wrapped__'):fn=fn.__wrapped__
    code=fn.__code__;path=Path(code.co_filename);source=path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['source_sha256']:raise ValueError('decision_source_changed')
    nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.If) and n.lineno==spec['if_line']]
    if len(nodes)!=1:raise ValueError('decision_if_not_unique')
    node=nodes[0];ins=list(dis.get_instructions(code));by_offset={i.offset:i for i in ins}
    start=min(i.offset for i in ins if i.positions.lineno and node.test.lineno<=i.positions.lineno<=node.test.end_lineno)
    body=min(i.offset for i in ins if i.positions.lineno==node.body[0].lineno)
    edges={}
    for index,i in enumerate(ins[:-1]):
        if not start<=i.offset<body or not ('JUMP' in i.opname and 'IF' in i.opname):continue
        for dst in (i.argval,ins[index+1].offset):
            if dst==body:edges[(i.offset,dst)]='true'
            elif dst>body and by_offset[dst].positions.lineno>node.end_lineno:edges[(i.offset,dst)]='false'
    if set(edges.values())!={'true','false'}:raise ValueError('logical_decision_exit_mapping_unsupported')
    d=describe(fn)[0]
    return {'source_identity':[d['source'],d['first_line'],d['code_sha256']],
            'if_expression':ast.unparse(node.test),'edges':[{'edge':list(k),'outcome':v} for k,v in edges.items()]}


def observation(spec,resolved,report,source_path,*,node_id):
    outcomes=set();events=[]
    exits={tuple(r['edge']):r['outcome'] for r in resolved['edges']}
    for call in report.get('attributed_calls',[]):
        if call['source_identity']!=resolved['source_identity'] or call['dag_node']!=node_id:continue
        if call['original_aten']!=spec['original_aten']:continue
        for edge in call['hit_outcomes']:
            if tuple(edge) in exits:outcomes.add(exits[tuple(edge)]);events.append(edge)
    if len(outcomes)!=1:return {'status':'unknown','reason':'no_unique_observed_logical_exit','observed_outcomes':sorted(outcomes)}
    return {'status':'observed','target_id':spec['decision_id'],'outcome':next(iter(outcomes)),
            'source_evidence_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
            'source_identity':resolved['source_identity'],'if_line':spec['if_line'],'events':events,
            'logical_feedback_only_not_primary_bytecode_coverage':True}
