"""Seal the executable protocol before any formal source-coverage observations."""
import ast,hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target=ROOT/'effect_protocol_v1.json'
    if target.exists():raise ValueError('formal_protocol_already_frozen')
    gates=json.loads((ROOT/'final_pipeline_controls_v1/summary.json').read_text())
    assert gates['formal_effects_started'] is False
    assert all(v for k,v in gates.items() if isinstance(v,bool) and k!='formal_effects_started')
    protocol=json.loads((ROOT/'effect_protocol_v1_draft.json').read_text())
    assert sha(ROOT/protocol['universe'])==protocol['universe_sha256']
    protocol['status']='frozen_ready_for_formal_run';protocol['effect_experiment_started']=False
    # Balance run positions before outcome data; every position differs by at
    # most one assignment across arms. No effect values inform this ordering.
    permutations=list(itertools.permutations(('A','B','C')))
    protocol['order_by_seed']=[list(p) for p in permutations]+[list(permutations[0]),list(permutations[3])]
    protocol['wall_deadline']='600s from process launch, including imports and all preparation;2s reserved inside budget for cleanup; supervisor enforces outer deadline and reports observed overshoot'
    protocol['context_unit']='canonical original-DAG radius2 neighborhood; aliases,semantic parameters,rank,tensor dtype and observed layout retained; Python scalar dtype labels collapsed to real Python types'
    protocol['feedback']='actual source exit with matching source,ATen,original graph/node; observed input regions before context priority; candidate context predictions replaced by actual eager layout context on observation'
    protocol['evidence_gate_report']='final_pipeline_controls_v1/summary.json'
    files=set(ROOT.glob('*.py'))
    for directory in (REPO/'third_party/torchfuzz',REPO/'kernelfuzz/route_a'):
        files.update(directory.rglob('*.py'))
    old=ROOT.parents[1]/'research_20261103/unified_condition_v2'
    files.update(old/name for name in ('generator_v4.py','relations.py','constraints.py','constraints_v4.py','common_pool.py','runtime_oracle_v4.py'))
    inputs=['effect_protocol_v1_draft.json','coverage_universe_v1/universe.json','forward_source_inventory_v1/inventory.json','branch_domain_audit_v3/results.json','source_inventory_v2/inventory.json','source_condition_inventory_v2/conditions.json','device_domain_audit_v1/results.json','installed_source_audit_v1.json','final_pipeline_controls_v1/summary.json']
    files.update(ROOT/name for name in inputs)
    universe=json.loads((ROOT/'coverage_universe_v1/universe.json').read_text())
    files.update(Path(r['source_identity'][0]) for r in universe['targets']+universe['excluded'])
    site=Path('/data1/tzh/pt_nightly_env/lib/python3.11/site-packages/torch')
    files.update(site/name for name in ('_inductor/compile_fx.py','_inductor/config.py','_dynamo/device_interface.py','fx/traceback.py','_subclasses/fake_tensor.py'))
    for path in files:
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
    protocol['runtime_source_sha256']={str(path.resolve()):sha(path) for path in sorted(files)}
    protocol['gate_evidence_sha256']={name:sha(ROOT/name) for name in gates['evidence']}
    protocol['source_audit_policy']='all10registered implementation source files match this build wheel RECORD; runtime file hashes and graph/source-domain assumptions remain enforced; no all-kernel coverage claim'
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'effect_protocol_v1.sha256').write_text(sha(target)+'\n')
    print(json.dumps({'status':protocol['status'],'seed_count':len(protocol['master_seeds']),'arms':3,'seconds_per_arm_seed':600,'planned_arm_wall_hours':4,'watched_files':len(files),'protocol_sha256':sha(target)}))
if __name__=='__main__':main()
