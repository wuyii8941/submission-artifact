"""Predeclared master-seed-level analysis; no request pseudoreplication."""
import csv,itertools,json,math,random,statistics
from collections import Counter
from pathlib import Path


def sign_p(differences):
    nonzero=[d for d in differences if d!=0]
    if not nonzero:return 1.0
    n=len(nonzero);small=min(sum(d>0 for d in nonzero),sum(d<0 for d in nonzero))
    return min(1.0,2*sum(math.comb(n,k) for k in range(small+1))/2**n)


def interval(differences):
    rng=random.Random(2026110607);n=len(differences)
    boot=sorted(sum(rng.choice(differences) for _ in range(n))/n for _ in range(10000))
    return [boot[249],boot[9749]]


def analyze(directory):
    directory=Path(directory);protocol=json.loads((directory/'protocol.json').read_text());state=json.loads((directory/'suite.json').read_text())
    rows=[];data={};resource=[]
    for run in state['runs']:
        path=directory/run['relative_output']/'summary.json'
        if not path.exists():continue
        summary=json.loads(path.read_text());key=(run['master_seed'],run['arm']);data[key]=summary
        rows.append({'master_seed':run['master_seed'],'arm':run['arm'],'branch_outcomes':len(summary['covered_target_ids']),
                     'branch_fraction':len(summary['covered_target_ids'])/642,'successful_execution_branch_outcomes':len(summary['successful_execution_target_ids']),
                     'operator_kinds':len(summary['operator_kinds_at_backend_entry']),'operator_kind_fraction':len(summary['operator_kinds_at_backend_entry'])/55,
                     'generation_attempts':len(summary['generation']),'generation_failures':sum(r['status']!='generated' for r in summary['generation']),
                     'binding_attempts':sum(r['attempts'] for r in summary['materialization']),'binding_failures':sum(r['failures'] for r in summary['materialization']),
                     'status_counts':dict(Counter(r['status'] for r in summary['executions'])),
                     'bf16_support_rejections':sum('BF16 is not supported' in (r.get('metrics',{}).get('error_message') or '') for r in summary['executions']),
                     'oracle_statuses':dict(Counter((r.get('metrics',{}).get('oracle') or {}).get('status','not_available') for r in summary['executions'])),
                     'actual_outcome_contexts':len({(r['target_id'],r['observation']['outcome'],r['observation'].get('context_signature')) for r in summary['executions'] if r.get('observation',{}).get('status')=='observed'}),
                     'requests':len(summary['executions']),'observed_feedback':sum(r.get('observation',{}).get('status')=='observed' for r in summary['executions']),
                     'post_feedback_selections':sum(bool(r.get('selection',{}).get('verified_feedback_available')) for r in summary['executions']),
                     'supervisor_wall_seconds':run['supervisor']['full_process_wall_seconds'],'protocol_valid':summary['protocol_valid'] and summary['source_unchanged']})
    complete=state['status']=='complete' and len(rows)==len(protocol['master_seeds'])*3 and all(r['protocol_valid'] for r in rows)
    result={'status':'complete' if complete else 'incomplete_no_effect_inference','rows':rows,'denominator':642,'operator_kind_denominator':55,
            'scope':'registered source-implementation branch observations, not all compiler/kernel paths','comparisons':[],'positive_claim':False}
    if rows:
        with (directory/'per_seed.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if complete:
        for a,b in [('B','A'),('C','B')]:
            diffs=[len(data[(seed,a)]['covered_target_ids'])-len(data[(seed,b)]['covered_target_ids']) for seed in protocol['master_seeds']]
            result['comparisons'].append({'name':a+'-'+b,'paired_count_differences':diffs,'mean_difference':statistics.mean(diffs),
                                          'mean_fraction_difference':statistics.mean(diffs)/642,'bootstrap95_count_difference':interval(diffs),'sign_test_p':sign_p(diffs)})
        previous=0
        for rank,item in enumerate(sorted(result['comparisons'],key=lambda r:r['sign_test_p'])):
            item['holm_p']=min(1,max(previous,item['sign_test_p']*(2-rank)));previous=item['holm_p']
            item['stable_positive_evidence']=item['mean_difference']>0 and item['bootstrap95_count_difference'][0]>0 and item['holm_p']<.05
        result['positive_claim']=all(r['stable_positive_evidence'] for r in result['comparisons'])
        result['AB_matched_prefixes']=[]
        for seed in protocol['master_seeds']:
            a=data[(seed,'A')]['executions'];b=data[(seed,'B')]['executions'];n=min(len(a),len(b))
            sets=[{t for row in stream[:n] for t in row.get('metrics',{}).get('verified_target_ids',[])} for stream in (a,b)]
            result['AB_matched_prefixes'].append({'master_seed':seed,'paired_skeleton_requests':n,'A_outcomes':len(sets[0]),'B_outcomes':len(sets[1]),'B_minus_A':len(sets[1])-len(sets[0])})
        result['operator_kind_claim']='descriptive separate metric; branch gains never imply operator-kind gains'
    (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();print(json.dumps(analyze(a.directory)))
