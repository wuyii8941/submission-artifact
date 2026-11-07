"""Persistent within-target queue; feedback changes priority, never actions.

Callers supply the same already-bound candidates to B/C. Predictions only rank
candidates; observed coverage must come from an independently verified source
receipt. This component does not establish a coverage denominator or effect.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class CandidateQueue:
    def __init__(self,path,seed=None):
        self.path=Path(path)
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if seed is not None and seed!=self.state['seed']:raise ValueError('queue_seed_mismatch')
        else:
            if seed is None:raise ValueError('new_queue_requires_seed')
            self.state={'schema':1,'seed':seed,'candidates':[],'events':[],'seen':{}}
            self._save()

    def _save(self):
        tmp=self.path.with_suffix(self.path.suffix+'.tmp')
        tmp.write_text(json.dumps(self.state,indent=2)+'\n');os.replace(tmp,self.path)

    def append(self,target_id,candidates):
        tick=time.monotonic();known={row['id'] for row in self.state['candidates']};added=[]
        for candidate in candidates:
            # A candidate includes a frozen graph reference and a concrete
            # binding action, not instructions to create a new graph template.
            item=copy.deepcopy(candidate)
            for required in ('graph_sha256','action','predicted_outcome','context'):
                if required not in item:raise ValueError('missing_candidate_field:'+required)
            identity=digest({'target':target_id,'candidate':{key:item[key] for key in ('graph_sha256','action','predicted_outcome','context')}})
            if identity in known:continue
            known.add(identity);added.append({'id':identity,'target_id':target_id,'candidate':item,'status':'pending'})
        self.state['candidates'].extend(added)
        self.state['events'].append({'kind':'append','target_id':target_id,'added':[r['id'] for r in added],
                                     'elapsed_seconds':time.monotonic()-tick})
        self._save();return [r['id'] for r in added]

    def select(self,target_id,*,use_feedback):
        tick=time.monotonic()
        candidates=[r for r in self.state['candidates'] if r['target_id']==target_id and r['status']=='pending']
        if not candidates:return None
        selections=sum(e['kind']=='select' and e['target_id']==target_id for e in self.state['events'])
        randomizer=random.Random(digest([self.state['seed'],target_id,selections]))
        baseline=randomizer.choice(candidates);chosen=baseline;reason='fixed_random_baseline'
        seen=self.state['seen'].get(target_id,[])
        # Same baseline before real feedback; one in five later choices keeps
        # random exploration. Per-target counters preserve fixed target quotas.
        if use_feedback and seen and selections%5!=4:
            branches={row['outcome'] for row in seen}
            contexts={(row['outcome'],row['context']) for row in seen}
            fresh=[r for r in candidates if r['candidate']['predicted_outcome'] not in branches]
            if fresh:
                chosen=randomizer.choice(fresh);reason='unobserved_outcome'
            else:
                fresh=[r for r in candidates if (r['candidate']['predicted_outcome'],r['candidate']['context']) not in contexts]
                if fresh:chosen=randomizer.choice(fresh);reason='new_outcome_context'
        chosen['status']='selected'
        event={'kind':'select','target_id':target_id,'id':chosen['id'],'baseline_id':baseline['id'],
               'reason':reason,'feedback_enabled':use_feedback,'verified_feedback_available':bool(seen),
               'candidate_ids_before_selection':sorted(r['id'] for r in candidates),
               'elapsed_seconds':time.monotonic()-tick}
        self.state['events'].append(event);self._save()
        return copy.deepcopy({'id':chosen['id'],**chosen['candidate'],'selection':event})

    def complete(self,identity,observation):
        tick=time.monotonic();rows=[r for r in self.state['candidates'] if r['id']==identity]
        if len(rows)!=1 or rows[0]['status']!='selected':raise ValueError('candidate_not_selected')
        row=rows[0];observation=copy.deepcopy(observation)
        verified=observation.get('status')=='observed'
        if verified:
            if observation.get('target_id')!=row['target_id']:raise ValueError('feedback_target_mismatch')
            if not observation.get('source_evidence_sha256') or not isinstance(observation.get('outcome'),str):
                raise ValueError('feedback_requires_source_receipt_and_outcome')
            self.state['seen'].setdefault(row['target_id'],[]).append({
                'outcome':observation['outcome'],'context':row['candidate']['context'],
                'source_evidence_sha256':observation['source_evidence_sha256']})
        row['status']='completed';row['observation']=observation
        self.state['events'].append({'kind':'complete','target_id':row['target_id'],'id':identity,
                                     'verified_feedback':verified,'elapsed_seconds':time.monotonic()-tick})
        self._save()
