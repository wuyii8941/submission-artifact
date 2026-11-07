"""Independent arithmetic value controls; excluded from discovery/coverage."""
import hashlib
import json
from pathlib import Path
from inventory import ROOT
from value_domain import ValueDomain, ValueDomainViolation
import torch


def main():
    out = ROOT/'value_domain_controls_v1'; out.mkdir(exist_ok=False)
    controls = [
        ('fmod_integer_zero', lambda: torch.fmod(torch.tensor([0,1]),torch.tensor([0,1])), False),
        ('remainder_integer_zero', lambda: torch.remainder(torch.tensor([1,2]),torch.tensor([0,1])), False),
        ('rounded_integer_div_zero', lambda: torch.div(torch.tensor([1,2]),torch.tensor([0,1]),rounding_mode='trunc'), False),
        ('integer_signed_overflow', lambda: torch.fmod(torch.tensor([-2**63]),torch.tensor([-1])), False),
        ('integer_negative_exponent', lambda: torch.pow(torch.tensor([2,3]),torch.tensor([-1,1])), False),
        ('integer_fmod_legal', lambda: torch.fmod(torch.tensor([0,3]),torch.tensor([1,2])), True),
        ('integer_remainder_legal', lambda: torch.remainder(torch.tensor([2,3]),torch.tensor([1,2])), True),
        ('floating_zero_divisor', lambda: torch.fmod(torch.tensor([0.,1.]),torch.tensor([0.,1.])), True),
        ('integer_true_div_zero', lambda: torch.div(torch.tensor([0,1]),torch.tensor([0,1])), True),
        ('positive_integer_power', lambda: torch.pow(torch.tensor([2,3]),torch.tensor([0,1])), True),
        ('float_negative_power', lambda: torch.pow(torch.tensor([2.,3.]),torch.tensor([-1.,1.])), True),
    ]
    rows=[]
    for name, fn, expected in controls:
        screen=ValueDomain()
        try:
            with screen: fn()
            accepted=True
        except ValueDomainViolation as exc:
            accepted=False
        rows.append({'control':name,'expected_accepted':expected,'accepted':accepted,'passed':accepted==expected,
                     'screen':screen.receipt()})
    try:
        torch.fmod(torch.tensor([1]),torch.tensor([0]))
        cpu_raises=False
    except RuntimeError as exc:
        cpu_raises=True; cpu_error=str(exc)
    report={'controls':rows,'all_passed':all(r['passed'] for r in rows),
            'cpu_integer_fmod_zero_raises':cpu_raises,'cpu_error':cpu_error if cpu_raises else None,
            'excluded_from_discovery_and_coverage':True,
            'scope':'the checked rules only; no claim of exhaustive semantic domain validation',
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('value_domain.py','qualify_value_domain.py')}}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'controls':len(rows),'all_passed':report['all_passed'],'cpu_integer_fmod_zero_raises':cpu_raises}))
    assert report['all_passed'] and cpu_raises
if __name__=='__main__':main()
