"""Aggregate concrete runner gates and verify frozen analysis arithmetic."""
import json,math
from pathlib import Path
from analyze_experiment import sign_p,interval
from run_formal_suite import pairing
from experiment_queue import ExperimentQueue
ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'final_pipeline_controls_v1';out.mkdir(exist_ok=False)
    load=lambda p:json.loads((ROOT/p).read_text())
    a=load('arm_control_A_v2/summary.json');b=load('arm_control_B_v2/summary.json')
    assert pairing(a,b) and a['source_sha256']==b['source_sha256']
    live=load('live_feedback_controls_v2/summary.json');assert live['supported_and_heldout_real_feedback_followed_by_execution'] and live['both_next_source_exits_confirmed']
    components=load('experiment_component_controls_v5/summary.json')
    assert all(components[k] for k in ('canonical_context_renaming_and_dictionary_reordering','common_action_ids_all_arms','A_ignores_condition_goal','AB_same_skeleton','no_feedback_B_equals_C'))
    scope=load('scope_controls_v1/summary.json')['scope_checks'];assert [r['protocol_valid'] for r in scope]==[True,False,False,False]
    deadline=load('hard_deadline_control_v1.supervisor.json');assert deadline['hard_deadline_triggered'] and deadline['full_process_wall_seconds']<2
    cleanup=load('worker_cleanup_controls_v1/summary.json');assert cleanup['verified_real_worker_group_terminated'] and cleanup['wrong_process_start_refused']
    assert sign_p([1]*8)==.0078125 and sign_p([0]*8)==1 and sign_p([1,-1]*4)==1
    assert interval([2]*8)==[2,2]
    q=ExperimentQueue(out/'fallback_queue.json',seed=7)
    candidates=[{'graph_sha256':'g'+str(i),'skeleton_sha256':'s'+str(i),'action':{'node_id':'n'},'context':'c'+str(i),'predicted_outcome':str(bool(i)).lower()} for i in range(2)]
    q.append('target',candidates);first=q.select('target',arm='C',skeleton_sha256='s0',desired_outcome='false')
    q.complete(first['id'],{'status':'observed','target_id':'target','outcome':'false','graph_sha256':'g0','node_id':'n','source_evidence_sha256':'synthetic-control-only'})
    second=q.select('target',arm='C',skeleton_sha256='s0',desired_outcome='false');assert second and second['skeleton_sha256']=='s1'
    report={'control_only':True,'AB_end_to_end_shared_skeletons':True,'A_ignores_predicate_and_no_feedback_B_equals_C':True,'name_independent_contexts':True,
            'supported_and_heldout_actual_post_feedback_execution':True,'identity_safe_worker_cleanup':True,'imports_in_wall_deadline':True,
            'scope_drift_rejected':True,'feedback_does_not_stall_on_consumed_baseline_skeleton':True,'analysis_ties_sign_test_and_bootstrap_controls':True,
            'primary_denominator':642,'formal_effects_started':False,
            'evidence':['experiment_component_controls_v5/summary.json','live_feedback_controls_v2/summary.json','scope_controls_v1/summary.json','arm_control_A_v2/summary.json','arm_control_B_v2/summary.json','worker_cleanup_controls_v1/summary.json','hard_deadline_control_v1.supervisor.json']}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
