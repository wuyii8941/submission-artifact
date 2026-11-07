"""Source-first audit ledger. Does not read trial outcomes or freeze a denominator."""
import ast
import dis
import functools
import hashlib
import json
from pathlib import Path
import types
from collections import Counter
from inventory import ROOT, describe
from runtime_audit import source_catalog
import torch
import torch._inductor.config as config
import torch._inductor.decomposition as decomp
import torch._inductor.lowering as lowering


def code_key(code):
    return (code.co_filename,code.co_firstlineno,hashlib.sha256(code.co_code).hexdigest())


def main():
    out=ROOT/'source_domain_ledger_v2';out.mkdir(exist_ok=False)
    catalog=source_catalog();objects={}
    for table in (decomp.decompositions,lowering.lowerings):
        for fn in table.values():
            seen=set()
            while callable(fn) and id(fn) not in seen:
                seen.add(id(fn))
                if isinstance(fn,functools.partial):fn=fn.func;continue
                code=getattr(fn,'__code__',None)
                if isinstance(code,types.CodeType):objects[code_key(code)]=code
                fn=getattr(fn,'__wrapped__',None)
    rows=[];trees={};missing=[];nested=[]
    for identity,entry in sorted(catalog.items()):
        code=objects.get(identity)
        if code is None:missing.append(identity);continue
        path=Path(identity[0]);text=path.read_text()
        if path not in trees:
            parsed=ast.parse(text);tests=[]
            for node in ast.walk(parsed):
                if isinstance(node,(ast.If,ast.While,ast.IfExp,ast.Assert)):
                    tests.append((node.test,ast.unparse(node.test)))
            trees[path]=tests
        tests=trees[path]
        instructions=list(dis.get_instructions(code));by_offset={ins.offset:ins for ins in instructions}
        jumps={edge[0] for edge in entry['source']['conditional_outcomes']}
        branches=[]
        for offset in sorted(jumps):
            ins=by_offset[offset];pos=ins.positions
            relevant=[{'line':test.lineno,'end_line':test.end_lineno,'expression':expr}
                      for test,expr in tests if pos.lineno and test.lineno<=pos.lineno<=test.end_lineno]
            branches.append({'offset':offset,'opcode':ins.opname,'positions':list(pos),
                             'outcomes':[edge for edge in entry['source']['conditional_outcomes'] if edge[0]==offset],
                             'source_test_candidates':relevant,'reachability':'unknown_retained',
                             'exclusion_proof':None,'binding_support':'not_audited'})
        nested_codes=[]
        def collect_nested(parent):
            for item in parent.co_consts:
                if not isinstance(item,types.CodeType):continue
                ins=list(dis.get_instructions(item));edges=[]
                for i,op in enumerate(ins[:-1]):
                    if ('JUMP' in op.opname and 'IF' in op.opname) or op.opname=='FOR_ITER':
                        edges.extend([op.offset,dst] for dst in sorted({op.argval,ins[i+1].offset}))
                child={'source_identity':list(code_key(item)),'qualname':item.co_qualname,
                       'conditional_outcomes':edges,'current_observer_tracks':False,
                       'reachability':'unknown_retained','parent_identity':list(code_key(parent))}
                nested_codes.append(child);nested.append(child);collect_nested(item)
        collect_nested(code)
        rows.append({'source_identity':list(identity),'qualname':code.co_qualname,
                     'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'registrations':entry['registrations'],'branches':branches,'nested_code_objects':nested_codes,
                     'provider_scope':'direct_inductor_decomposition_body' if str(path).endswith('/torch/_inductor/decomposition.py') else 'other_registered_provider_or_wrapper',
                     'current_observer_tracks':True,'membership':'pending_public_to_actual_ATen_domain_audit'})
    # This is an inventory of uncertainties, never an empirical reachable subset.
    inventory=json.loads((ROOT/'source_inventory_v2/inventory.json').read_text())
    ledger={'torch':torch.__version__,'commit':torch.version.git_version,
            'selection':'all code identities in the common55 nominal and independent CPU actual-ATen provider inventory; no trial hit data read',
            'operator_count':len(inventory['operators']),'nominal_overloads':sum(len(r['aten_candidates']) for r in inventory['operators']),
            'functions':rows,'missing_code_objects':missing,'nested_code_objects':nested,
            'unit':'directed conditional bytecode jump outcome, implementation identity globally deduplicated',
            'membership_policy':'unknown and hard outcomes retained, no exclusion without source/domain proof',
            'compiler_config_observed':{'coordinate_descent_tuning':config.coordinate_descent_tuning},
            'unresolved_domain_items':['nominal overload membership incl out/backward','actual alias completeness',
                                       'binding dtype domain currently broader than default template',
                                       'nested registered-provider code not yet tracked',
                                       'lowering-created helper control flow outside registered entry code'],
            'primary_denominator_committed':False,'effect_experiment_started':False,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    summary={'registered_entry_code_objects':len(rows),'missing_code_objects':len(missing),
             'registered_entry_outcomes':sum(len(b['outcomes']) for r in rows for b in r['branches']),
             'scope_counts':dict(Counter(r['provider_scope'] for r in rows)),
             'nested_code_objects':len(nested),'nested_outcomes_not_tracked':sum(len(r['conditional_outcomes']) for r in nested),
             'all_reachability_unknown':True,'coverage_percentage':None,'primary_denominator_committed':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
