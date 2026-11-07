"""Independent raw CPython opcode trace evidence for long conditional jumps."""
import dis
import hashlib
import json
import sys
from pathlib import Path
from inventory import ROOT,describe
import torch
import torch._inductor.decomposition as decomp


def main():
    out=ROOT/'extended_arg_diagnostic_v1';out.mkdir(exist_ok=False)
    fn=decomp.mm
    while hasattr(fn,'__wrapped__'):fn=fn.__wrapped__
    steps=[]
    def trace(frame,event,arg):
        if frame.f_code is not fn.__code__:return None
        if event=='call':frame.f_trace_opcodes=True;frame.f_trace_lines=False
        elif event=='opcode':steps.append(frame.f_lasti)
        return trace
    prior=sys.gettrace();sys.settrace(trace)
    try:result=fn(torch.empty(2,2,device='meta'),torch.empty(2,3,device='meta'))
    finally:sys.settrace(prior)
    assert result is NotImplemented
    ins=list(dis.get_instructions(fn));rows=[]
    for i,op in enumerate(ins):
        if op.opname=='EXTENDED_ARG' and i+1<len(ins) and 'JUMP' in ins[i+1].opname and 'IF' in ins[i+1].opname:
            nxt=ins[i+1]
            rows.append({'prefix_offset':op.offset,'jump_offset':nxt.offset,'jump_target':nxt.argval,
                         'prefix_executed':op.offset in steps,'jump_opcode_trace_event_seen':nxt.offset in steps})
    missed=[r for r in rows if r['prefix_executed'] and not r['jump_opcode_trace_event_seen']]
    assert missed,'no EXTENDED_ARG trace discrepancy reproduced'
    report={'control_only':True,'source':describe(fn)[0],'raw_opcode_offsets':steps,'extended_conditional_jumps':rows,
            'missed_by_jump_offset_only_collector':missed,'numerator_measurement_incomplete':True,
            'primary_denominator_committed':False,'effect_experiment_started':False,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(missed))
if __name__=='__main__':main()
