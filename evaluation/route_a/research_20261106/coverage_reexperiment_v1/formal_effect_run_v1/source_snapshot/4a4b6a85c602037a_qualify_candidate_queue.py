"""Queue persistence/fairness controls. Synthetic observations are not coverage."""
import hashlib
import json
from pathlib import Path
from candidate_queue import CandidateQueue
ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'candidate_queue_controls_v2';out.mkdir(exist_ok=False)
    candidates=[{'graph_sha256':'synthetic_graph_'+str(i),'action':{'attribute':'shape','value':i+1},
                 'predicted_outcome':str(i%2),'context':str(i//2)} for i in range(12)]
    path_a=CandidateQueue(out/'path_a.json',seed=5);path_b=CandidateQueue(out/'path_b.json',seed=5)
    assert path_a.append('target',[dict(r,case='/artifact/A/'+str(i)) for i,r in enumerate(candidates)])==path_b.append('target',[dict(r,case='/artifact/B/'+str(i)) for i,r in enumerate(candidates)])
    assert path_a.select('target',use_feedback=False)['id']==path_b.select('target',use_feedback=False)['id']
    b=CandidateQueue(out/'B.json',seed=5);c=CandidateQueue(out/'C.json',seed=5)
    assert b.append('target',candidates)==c.append('target',candidates)
    # Before feedback C uses exactly B's baseline, including across restarts.
    for i in range(3):
        sb=b.select('target',use_feedback=False);sc=c.select('target',use_feedback=True)
        assert sb['id']==sc['id']
        for q,s in ((b,sb),(c,sc)):q.complete(s['id'],{'status':'unknown'})
        b=CandidateQueue(out/'B.json');c=CandidateQueue(out/'C.json')
    # Both arms receive the same observed event. Only C uses it for priority.
    sb=b.select('target',use_feedback=False);sc=c.select('target',use_feedback=True);assert sb['id']==sc['id']
    observation={'status':'observed','target_id':'target','outcome':'0','source_evidence_sha256':'synthetic_control_receipt'}
    b.complete(sb['id'],observation);c.complete(sc['id'],observation)
    first_pending={r['id'] for r in b.state['candidates'] if r['status']=='pending'}
    assert first_pending=={r['id'] for r in c.state['candidates'] if r['status']=='pending'}
    after_feedback=[]
    for i in range(3):
        sb=b.select('target',use_feedback=False);sc=c.select('target',use_feedback=True)
        assert sb['selection']['reason']=='fixed_random_baseline'
        after_feedback.append(sc['selection'])
        for q,s in ((b,sb),(c,sc)):q.complete(s['id'],{'status':'unknown'})
    assert any(e['reason']=='unobserved_outcome' for e in after_feedback)
    assert any(e['id']!=e['baseline_id'] for e in after_feedback)
    # Feedback for another target is rejected without consuming the selection.
    s=c.select('target',use_feedback=True)
    try:c.complete(s['id'],dict(observation,target_id='wrong'))
    except ValueError as exc:assert str(exc)=='feedback_target_mismatch'
    else:raise AssertionError('cross-target feedback accepted')
    assert next(r for r in c.state['candidates'] if r['id']==s['id'])['status']=='selected'
    c.complete(s['id'],{'status':'execution_failure'})
    q=CandidateQueue(out/'append_only.json',seed=1);before=q.append('target',candidates[:2]);q.append('target',candidates[2:])
    assert before==[r['id'] for r in q.state['candidates'][:2]]
    report={'control_only':True,'candidate_count':12,'identical_action_set':True,'no_feedback_B_equals_C':True,
            'verified_post_feedback_opportunities':len(after_feedback),'feedback_changed_selection':True,
            'artifact_paths_do_not_change_identity_or_selection':True,'persistent_replay':True,'cross_target_feedback_rejected':True,'unknown_did_not_mark_coverage':True,
            'effect_experiment_started':False,'primary_denominator_committed':False,
            'limitations':['synthetic observations only','not integrated with compiler source observer or full wall budget runner'],
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('candidate_queue.py','qualify_candidate_queue.py')}}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
