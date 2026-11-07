"""Build/device/input-domain mismatches must invalidate coverage inference."""
import json,shutil
from pathlib import Path
from inventory import ROOT
from experiment_metrics import read


def main():
    out=ROOT/'scope_controls_v1';out.mkdir(exist_ok=False)
    decision=next(r for r in json.loads((ROOT/'source_condition_inventory_v2/conditions.json').read_text())['conditions'] if r['if_line']==444 and r['guidance_status']=='input_condition_specified')
    source=ROOT/'live_feedback_controls_v2/execute_444';original=json.loads((source/'receipt.json').read_text());case=json.loads(Path(original['case']).read_text())
    node=next(r for r in json.loads((source/'metrics.json').read_text())['observation'].items() if r[0]=='node_id')[1]
    results=[]
    for label,field,value in [('same_scope',None,None),('wrong_compiler','commit','wrong'),('wrong_hardware','capability',[9,0]),('wrong_domain','common_domain',{})]:
        directory=out/label;directory.mkdir();receipt=dict(original)
        if field:receipt[field]=value
        (directory/'receipt.json').write_text(json.dumps(receipt));shutil.copyfile(source/'source_evidence.json',directory/'source_evidence.json')
        metric=read(directory,decision,node);assert metric['protocol_valid']==(field is None),(label,metric['scope_failures'])
        results.append({'label':label,'protocol_valid':metric['protocol_valid'],'scope_failures':metric['scope_failures']})
    (out/'summary.json').write_text(json.dumps({'control_only':True,'scope_checks':results},indent=2)+'\n');print(json.dumps(results))
if __name__=='__main__':main()
