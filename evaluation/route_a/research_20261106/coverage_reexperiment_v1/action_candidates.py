"""One legal-action enumerator for A/B/C; target predicates only label choices."""
import ast,json,time
from pathlib import Path
from inventory import ROOT
from common_domain import DTYPES,LAYOUTS,SHAPE_VALUES
from graph_semantics import manifest
from dag_binding import bind
from dag_render import validate
from graph_context import context
from source_conditions import facts,evaluate,Unknown
from input_regions import region


def target_nodes(graph,decision):
    # Public semantic arity/rank checks belong to operator compatibility,
    # never to the desired condition side. No topology changes are made.
    matrix={'bmm':(3,3),'mm':(2,2),'addmm':(None,2,2)}
    paths=decision['source_identity'][0];line=decision['source_identity'][1]
    inventory=json.loads((ROOT/'forward_source_inventory_v1/inventory.json').read_text())
    entry=next(r for r in inventory['sources'] if [r['source']['source'],r['source']['first_line'],r['source']['code_sha256']]==decision['source_identity'])
    expected=matrix.get(entry['source']['qualname']);found=[];rejected=[]
    for nid in graph.get_topological_order():
        node=graph.nodes[nid]
        if node.op_name not in decision['target_ops']:continue
        specs=[graph.nodes[p].output_spec for p in node.input_nodes]
        if expected and (len(specs)!=len(expected) or any(rank is not None and (not hasattr(spec,'size') or len(spec.size)!=rank) for spec,rank in zip(specs,expected))):
            rejected.append({'node_id':nid,'reason':'public_to_ATen_rank_not_applicable'});continue
        found.append(nid)
    return found,rejected


def actions(graph,nid):
    result=[{'attribute':'identity'}]+[{'attribute':'dtype','value':str(dtype)} for dtype in DTYPES]
    for index,parent in enumerate(graph.nodes[nid].input_nodes):
        spec=graph.nodes[parent].output_spec
        if not hasattr(spec,'size'):continue
        for axis in range(len(spec.size)):
            result.extend({'attribute':'shape','value':value,'input_index':index,'axis':axis} for value in SHAPE_VALUES)
        result.extend({'attribute':'layout','value':layout,'input_index':index} for layout in LAYOUTS)
    return result


def build(graph,case,decision,nid,directory,*,target_applicable=True,deadline=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    original=manifest(graph);candidates=[];attempts=[];graphs={}
    stopped=False
    for index,action in enumerate(actions(graph,nid)):
        if deadline is not None and time.monotonic()>=deadline:stopped=True;break
        tick=time.monotonic()
        if action['attribute']=='identity':changed=graph;receipt={'status':'bound','original_graph_sha256':original['sha256'],'meta':validate(graph)}
        else:
            kwargs={k:v for k,v in action.items() if k not in ('attribute','value')}
            changed,receipt=bind(graph,nid,action['attribute'],action['value'],**kwargs)
        attempts.append({'action':action,'receipt':receipt,'full_wall_seconds':time.monotonic()-tick})
        if changed is None:continue
        record=manifest(changed);path=directory/(record['sha256']+'.json')
        if not path.exists():path.write_text(json.dumps({**case,'graph':record},indent=2)+'\n')
        try:prediction=('true' if evaluate(decision['predicate'],facts(changed,nid)) else 'false') if target_applicable else 'unknown'
        except Unknown:prediction='unknown'
        neighborhood=context(changed,nid,observed_facts=receipt['meta']['node_facts'])
        candidates.append({'graph_sha256':record['sha256'],'skeleton_sha256':original['sha256'],'case':str(path.resolve()),
                           'action':dict(action,node_id=nid),'target_applicable':target_applicable,'predicted_outcome':prediction,'context':neighborhood['signature'],'input_region':region(decision['predicate'],facts(changed,nid)) if target_applicable else None})
    (directory/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    return candidates,{'skeleton_sha256':original['sha256'],'target_id':decision['decision_id'],'node_id':nid,
                       'attempts':len(attempts),'legal_candidates':len(candidates),'failures':sum(r['receipt']['status']!='bound' for r in attempts),
                       'full_wall_seconds':time.monotonic()-started,'stopped_at_deadline':stopped}
