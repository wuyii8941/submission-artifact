"""Freeze a source-audited target set before any effect comparison.

Unknown feasibility remains in the denominator. The resulting fraction is
observed coverage of this explicit source target set, never a claim about all
reachable compiler branches, all operators, or device-kernel coverage.
"""
import json
from pathlib import Path
from inventory import ROOT,sha


def main():
    out=ROOT/'coverage_universe_v1';out.mkdir(exist_ok=False)
    inventory=json.loads((ROOT/'forward_source_inventory_v1/inventory.json').read_text())
    audit=json.loads((ROOT/'branch_domain_audit_v3/results.json').read_text())
    controls=json.loads((ROOT/'branch_domain_controls_v1/results.json').read_text())
    runtime=json.loads((ROOT/'denominator_call_controls_v1/summary.json').read_text())
    assert not controls['in_domain_observed_excluded_outcomes'] and controls['out_of_domain_observed_excluded_outcomes']
    assert runtime['valid'] and runtime['fixed_domain_calls_checked']>0
    assert inventory['commit']=='11ff36ef30d23635f4e9bfe414ae027471901134'
    excluded={(tuple(r['source_identity']),*r['edge']):r for r in audit['exclusions']}
    all_outcomes=set();targets=[];proofs=[]
    for row in inventory['sources']:
        source=row['source'];identity=(source['source'],source['first_line'],source['code_sha256'])
        assert sha(Path(identity[0]).read_bytes())==source['source_sha256']
        for edge in source['conditional_outcomes']:
            key=(identity,*edge)
            if key in all_outcomes:continue
            all_outcomes.add(key)
            item={'target_id':sha(json.dumps([list(identity),edge],sort_keys=True).encode()),'source_identity':list(identity),
                  'source_sha256':source['source_sha256'],'qualname':source['qualname'],'edge':edge,'registrations':row['registrations']}
            if key in excluded:proofs.append({**item,'exclusion_proof':excluded[key]['proof']})
            else:targets.append({**item,'feasibility':'unknown_or_possible_retained','no_proof_of_unreachability':True})
    assert set(excluded).issubset(all_outcomes)
    assert len(all_outcomes)==len(targets)+len(proofs)
    inputs=['forward_source_inventory_v1/inventory.json','branch_domain_audit_v3/results.json','branch_domain_controls_v1/results.json','denominator_call_controls_v1/summary.json','device_domain_audit_v1/results.json','source_condition_inventory_v2/conditions.json']
    sources=['common_domain.py','functional_domain.py','dag_render.py','dag_generator.py','dag_binding.py','graph_semantics.py','frozen_codegen.py','runtime_audit.py','opcode_offsets.py','attribute_source_audit.py','source_conditions.py','candidate_queue.py','qualify_denominator_calls.py','freeze_coverage_universe.py']
    result={'schema':1,'status':'frozen_before_effect_experiment','torch':inventory['torch'],'commit':inventory['commit'],
            'metric_name':'预登记算子实现分支结果的观测覆盖率',
            'scope':inventory['scope'],'unit':'directed Python conditional-jump outcome; global source-identity deduplication; includes FOR_ITER outcomes; excludes exceptions and unconditional jumps',
            'formula':'verified attributed target outcome union / fixed target outcome count',
            'operator_public_names':[r['public_operator'] for r in inventory['operators']],
            'operator_kind_metric':'separate: public operator kinds actually present at compiler backend entry / common55; generated-node appearances are generation statistics only',
            'branch_candidate_count':len(all_outcomes),'excluded_outcome_count':len(proofs),'denominator':len(targets),
            'targets':targets,'excluded':proofs,'domain':audit['domain'],'fixed_config':audit['fixed_config'],
            'unknowns':'Unknown reachability/membership remains in the target set. This is measured coverage of a declared source universe, not the fraction of all feasible compiler/kernel paths. Unattributed events are unknown and never hits.',
            'numerator_population':'all legal-input compilation attempts, including verified decisions reached before later compilation failure; also report successful-execution-only hits separately',
            'invalidations':['any attributed hit of a proved-excluded outcome','actual proof-function entry facts fail fixed CUDA/noncomplex assumptions','source/config/device/generator/action/oracle drift','A/B/C action-set or total-cost mismatch'],
            'invalidations_action':'stop inference under this protocol, retain failures and costs, audit and create a new version; never silently enlarge denominator or remove hard targets',
            'controls_excluded_from_effect_results':True,'primary_denominator_committed':True,'effect_experiment_started':False,
            'input_sha256':{p:sha((ROOT/p).read_bytes()) for p in inputs},'implementation_sha256':{p:sha((ROOT/p).read_bytes()) for p in sources}}
    (out/'universe.json').write_text(json.dumps(result,indent=2)+'\n')
    summary={k:result[k] for k in ('status','metric_name','branch_candidate_count','excluded_outcome_count','denominator','primary_denominator_committed','effect_experiment_started')}
    summary['universe_sha256']=sha((out/'universe.json').read_bytes());(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
