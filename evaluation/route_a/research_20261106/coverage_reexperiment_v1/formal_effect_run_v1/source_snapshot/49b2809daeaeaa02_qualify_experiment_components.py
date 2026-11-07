"""Shared-domain, context and queue controls; no effect claims."""
import copy,json
from pathlib import Path
from inventory import ROOT
from graph_semantics import restore
from graph_context import context
from experiment_queue import ExperimentQueue
from action_candidates import build,target_nodes


def main():
    out=ROOT/'experiment_component_controls_v5';out.mkdir(exist_ok=False)
    row=json.loads((ROOT/'dag_controls_v2/case_002.json').read_text());graph=restore(row['graph']);nid=graph.root_node_id
    renamed=copy.deepcopy(graph);names={old:'renamed_'+str(i) for i,old in enumerate(reversed(list(graph.nodes)))};nodes={}
    for old in reversed(list(graph.nodes)):
        node=renamed.nodes[old];node.node_id=names[old];node.input_nodes=[names[p] for p in node.input_nodes];nodes[names[old]]=node
    renamed.nodes=nodes;renamed.root_node_id=names[graph.root_node_id]
    assert context(graph,nid)['signature']==context(renamed,names[nid])['signature']
    decisions=json.loads((ROOT/'source_condition_inventory_v2/conditions.json').read_text())['conditions']
    decision=next(d for d in decisions if d['if_line']==444 and d['guidance_status']=='input_condition_specified')
    found,_=target_nodes(graph,decision);assert found
    candidates,receipt=build(graph,row,decision,found[0],out/'actions')
    assert candidates and receipt['attempts']>receipt['legal_candidates']
    queues={arm:ExperimentQueue(out/(arm+'.json'),seed=23) for arm in ('A','B','C')}
    identities=[q.append(decision['decision_id'],candidates) for q in queues.values()];assert identities[0]==identities[1]==identities[2]
    skeleton=candidates[0]['skeleton_sha256'];choices={a:q.select(decision['decision_id'],arm=a,skeleton_sha256=skeleton,desired_outcome='true') for a,q in queues.items()}
    assert choices['B']['id']==choices['C']['id']
    assert choices['A']['skeleton_sha256']==choices['B']['skeleton_sha256']
    for arm,choice in choices.items():queues[arm].complete(choice['id'],{'status':'unknown'})
    # A's selection must not change with the condition goal.
    q1=ExperimentQueue(out/'A_goal_true.json',seed=55);q2=ExperimentQueue(out/'A_goal_false.json',seed=55)
    for q in (q1,q2):q.append(decision['decision_id'],candidates)
    assert q1.select(decision['decision_id'],arm='A',skeleton_sha256=skeleton,desired_outcome='true')['id']==q2.select(decision['decision_id'],arm='A',skeleton_sha256=skeleton,desired_outcome='false')['id']
    result={'control_only':True,'canonical_context_renaming_and_dictionary_reordering':True,'common_action_ids_all_arms':True,
            'A_ignores_condition_goal':True,'AB_same_skeleton':True,'no_feedback_B_equals_C':True,'candidate_receipt':receipt,
            'source_sha256':{f:__import__('hashlib').sha256((ROOT/f).read_bytes()).hexdigest() for f in ('graph_context.py','experiment_queue.py','action_candidates.py','input_regions.py')},'primary_denominator':642,'effect_experiment_started':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
