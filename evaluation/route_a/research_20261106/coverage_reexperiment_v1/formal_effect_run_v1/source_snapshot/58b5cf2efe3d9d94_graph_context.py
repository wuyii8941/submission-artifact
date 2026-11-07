"""Canonical original-DAG neighborhoods, preserving alias edges and parameters."""
import ast,hashlib,json
from graph_semantics import render_node


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def descriptor(node,observed=None):
    code=render_node(node.semantics,'context_output',[f'context_input_{i}' for i in range(len(node.input_nodes))],node.output_spec)
    # Only canonical placeholder identifiers are rendered. Source snapshots,
    # render seeds, original node names and artifact paths are never hashed.
    spec=node.output_spec
    layout=None
    if observed and observed.get('kind')=='tensor':
        strides=observed['stride'];layout={'contiguous':observed['contiguous'],'axis_order':sorted(range(len(strides)),key=lambda i:(strides[i],i)),'zero_stride_axes':[i for i,v in enumerate(strides) if v==0]}
    elif node.semantics['kind']=='argument' and hasattr(spec,'size'):
        strides=spec.stride;layout={'axis_order':sorted(range(len(strides)),key=lambda i:(strides[i],i)),'zero_stride_axes':[i for i,v in enumerate(strides) if v==0]}
    return {'layout':layout,'op':str(node.op_name),'kind':node.semantics['kind'],
            'code':ast.dump(ast.parse(code),include_attributes=False),
            'rank':len(spec.size) if hasattr(spec,'size') else None,'dtype':str(spec.dtype) if hasattr(spec,'size') else ('python_complex' if spec.dtype.is_complex else 'python_float' if spec.dtype.is_floating_point else 'python_bool' if str(spec.dtype)=='torch.bool' else 'python_int')}


def context(graph,target,radius=2,observed_facts=None):
    observed_facts=observed_facts or {}
    if target not in graph.nodes:raise ValueError('context_target_absent')
    users={nid:[] for nid in graph.nodes}
    for nid,node in graph.nodes.items():
        for index,parent in enumerate(node.input_nodes):users[parent].append((nid,index))
    order=[];seen=set()
    def visit(nid):
        if nid in seen:return
        seen.add(nid)
        for parent in graph.nodes[nid].input_nodes:visit(parent)
        order.append(nid)
    visit(graph.root_node_id)
    # Generated DAGs contain no disconnected nodes. Reject rather than using
    # arbitrary dictionary/name ordering as a novelty signal.
    if len(order)!=len(graph.nodes):raise ValueError('context_disconnected_graph_unsupported')
    region={target};frontier={target}
    for _ in range(radius):
        other={p for nid in frontier for p in graph.nodes[nid].input_nodes}|{u for nid in frontier for u,_ in users[nid]}
        frontier=other-region;region|=other
    local=[nid for nid in order if nid in region];labels={nid:i for i,nid in enumerate(local)};external={}
    rows=[]
    for nid in local:
        edges=[]
        for parent in graph.nodes[nid].input_nodes:
            if parent in labels:edges.append({'local':labels[parent]})
            else:
                if parent not in external:external[parent]=len(external)
                edges.append({'external':external[parent],'descriptor':descriptor(graph.nodes[parent],observed_facts.get(parent))})
        rows.append({'descriptor':descriptor(graph.nodes[nid],observed_facts.get(nid)),'inputs':edges,'num_users':len({u for u,_ in users[nid]}),'is_target':nid==target})
    payload={'radius':radius,'nodes':rows,'target':labels[target],'unit':'original general DAG neighborhood, not optimized compiler IR'}
    return {'signature':digest(payload),'payload':payload}
