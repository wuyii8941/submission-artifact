"""Common action queue with an explicit, condition-independent skeleton schedule."""
import copy,random,time
from candidate_queue import CandidateQueue,digest


class ExperimentQueue(CandidateQueue):
    def select(self,target_id,*,arm,skeleton_sha256,desired_outcome):
        if arm not in ('A','B','C'):raise ValueError('unknown_arm')
        tick=time.monotonic();pending=[r for r in self.state['candidates'] if r['target_id']==target_id and r['status']=='pending']
        if not pending:return None
        eligible=[r for r in pending if r['candidate']['skeleton_sha256']==skeleton_sha256]
        if not eligible:
            if arm=='C' and self.state['seen'].get(target_id):eligible=pending
            else:return None
        selections=sum(e['kind']=='select' and e['target_id']==target_id for e in self.state['events'])
        rng=random.Random(digest([self.state['seed'],target_id,selections]));baseline_pool=eligible
        if arm in ('B','C'):
            guided=[r for r in eligible if r['candidate']['predicted_outcome']==desired_outcome]
            if guided:baseline_pool=guided
        baseline=rng.choice(baseline_pool);chosen=baseline;reason='random_legal_input' if arm=='A' else 'fixed_condition_baseline'
        seen=self.state['seen'].get(target_id,[])
        if arm=='C' and seen and selections%5!=4:
            outcomes={r['outcome'] for r in seen};contexts={(r['outcome'],r['context']) for r in seen}
            fresh=[r for r in pending if r['candidate']['predicted_outcome'] in ('true','false') and r['candidate']['predicted_outcome'] not in outcomes]
            if fresh:chosen=rng.choice(fresh);reason='unobserved_outcome'
            else:
                seen_regions={(r['outcome'],r.get('input_region')) for r in seen if r.get('input_region') is not None}
                fresh=[r for r in pending if r['candidate'].get('input_region') is not None and (r['candidate']['predicted_outcome'],r['candidate']['input_region']) not in seen_regions]
                if fresh:chosen=rng.choice(fresh);reason='new_input_region'
                else:
                    fresh=[r for r in pending if r['candidate']['predicted_outcome'] in ('true','false') and (r['candidate']['predicted_outcome'],r['candidate']['context']) not in contexts]
                    if fresh:chosen=rng.choice(fresh);reason='new_outcome_context'
        chosen['status']='selected'
        event={'kind':'select','target_id':target_id,'id':chosen['id'],'baseline_id':baseline['id'],'arm':arm,
               'baseline_skeleton_sha256':skeleton_sha256,'selected_skeleton_sha256':chosen['candidate']['skeleton_sha256'],
               'desired_outcome':desired_outcome,'desired_outcome_available':any(r['candidate']['predicted_outcome']==desired_outcome for r in eligible),
               'reason':reason,'verified_feedback_available':bool(seen),'pending_ids':sorted(r['id'] for r in pending),
               'elapsed_seconds':time.monotonic()-tick}
        self.state['events'].append(event);self._save()
        return copy.deepcopy({'id':chosen['id'],**chosen['candidate'],'selection':event})

    def complete(self,identity,observation):
        if observation.get('status')=='observed':
            row=next(r for r in self.state['candidates'] if r['id']==identity)
            if observation.get('graph_sha256')!=row['candidate']['graph_sha256'] or observation.get('node_id')!=row['candidate']['action']['node_id']:raise ValueError('feedback_program_or_node_identity_mismatch')
        super().complete(identity,observation)
        if observation.get('status')=='observed':
            self.state['seen'][observation['target_id']][-1]['input_region']=observation.get('input_region')
            if observation.get('context_signature'):self.state['seen'][observation['target_id']][-1]['context']=observation['context_signature']
            self._save()
