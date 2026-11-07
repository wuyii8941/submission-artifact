"""Positive/negative long-jump controls with independent raw trace evidence."""
import dis
import hashlib
import importlib.util
import json
from pathlib import Path
from inventory import ROOT,describe
from runtime_audit import SourceAudit
import torch
import torch._inductor.decomposition as decomp


def observe(fn,call):
    s=describe(fn)[0];key=(s['source'],s['first_line'],s['code_sha256']);o=SourceAudit({key:{'source':s,'registrations':[]}})
    with o:call()
    return sorted({tuple(e) for c in o.calls for e in c['hit_outcomes']})


def main():
    out=ROOT/'extended_arg_controls_v1';out.mkdir(exist_ok=False)
    fn=decomp.mm
    while hasattr(fn,'__wrapped__'):fn=fn.__wrapped__
    a=torch.empty(2,2,device='meta');b=torch.empty(2,3,device='meta')
    fixed=observe(fn,lambda:fn(a,b))
    diagnostic=json.loads((ROOT/'extended_arg_diagnostic_v1/results.json').read_text())
    # Independent raw trace identifies the prefix actually executed and the
    # disassembled jump target. Assert that precise edge is now observed.
    missed=diagnostic['missed_by_jump_offset_only_collector'];expected={(r['jump_offset'],r['jump_target']) for r in missed}
    assert expected.issubset(set(fixed)),(expected,fixed)
    legacy=ROOT.parents[1]/'research_20261103/coverage_measurement_v1/bytecode_edges_v2.py'
    spec=importlib.util.spec_from_file_location('legacy_bytecode_collector',legacy);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    description=describe(fn)[0]
    catalog={'source':fn.__code__.co_filename,'source_sha256':description['source_sha256'],'catalog_sha256':'control-only',
             'functions':{'mm':{'key':list(mod.code_key(fn.__code__)),'edges':description['conditional_outcomes']}}}
    collector=mod.Collector(catalog)
    with collector:fn(a,b)
    old={(src,dst) for name,src,dst in collector.hits};assert not expected.intersection(old)
    controls=[]
    for statements in (100,35000):
        source='def synthetic(flag):\n    if flag:\n'+''.join('        x=0\n' for _ in range(statements))+'        return 1\n    return 0\n'
        namespace={};exec(compile(source,'<independent-extended-jump-control>','exec'),namespace);g=namespace['synthetic']
        ins=list(dis.get_instructions(g));jump=next(i for i in ins if 'JUMP' in i.opname and 'IF' in i.opname)
        prefix_count=sum(i.opname=='EXTENDED_ARG' and i.offset<jump.offset for i in ins)
        hits=observe(g,lambda:(g(False),g(True)))
        expected_edges={tuple(e) for e in describe(g)[0]['conditional_outcomes']}
        assert set(hits)==expected_edges,(statements,hits,expected_edges)
        controls.append({'body_statements':statements,'prefix_count':prefix_count,'expected_edges':sorted(expected_edges),'observed_edges':hits})
    assert controls[1]['prefix_count']>=2
    result={'control_only':True,'mm_expected_previously_missing_edges':sorted(expected),'legacy_mm_hits':sorted(old),'fixed_mm_hits':fixed,
            'legacy_collector_undercount_confirmed':True,'synthetic_both_sides':controls,
            'primary_denominator_committed':False,'effect_experiment_started':False,
            'legacy_source_sha256':hashlib.sha256(legacy.read_bytes()).hexdigest(),
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('runtime_audit.py','opcode_offsets.py','qualify_extended_arg.py')}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
