"""Independent trace control for caught exceptions and nested/generator calls."""
import hashlib
import json
from pathlib import Path
from inventory import describe
from runtime_audit import SourceAudit
ROOT=Path(__file__).resolve().parent


def ordinary(value):
    if value:return 1
    return 0


def catches(value):
    try:
        raise ValueError('control')
    except ValueError:
        pass
    if value:return 1
    return 0


def generator(value):
    if value:yield 1
    else:yield 0


def main():
    out=ROOT/'source_trace_controls_v1';out.mkdir(exist_ok=False)
    functions=(ordinary,catches,generator);catalog={}
    for fn in functions:
        source=describe(fn)[0];key=(source['source'],source['first_line'],source['code_sha256'])
        catalog[key]={'source':source,'registrations':[]}
    observer=SourceAudit(catalog)
    with observer:
        assert ordinary(True)==1 and ordinary(False)==0
        assert catches(True)==1 and catches(False)==0
        assert list(generator(True))==[1] and list(generator(False))==[0]
    report=observer.receipt();counts={}
    for fn in functions:
        source=describe(fn)[0];calls=[c for c in report['calls'] if c['source_identity'][1]==source['first_line']]
        hits={tuple(edge) for call in calls for edge in call['hit_outcomes']}
        assert len(hits)>=2,(fn.__name__,calls)
        assert all(list(edge) in source['conditional_outcomes'] for edge in hits)
        counts[fn.__name__]={'calls':len(calls),'distinct_outcomes':len(hits)}
    result={'control_only':True,'counts':counts,'caught_exception_trace_survives':True,
            'generator_resumption_trace_survives':True,'coverage_percentage':None,
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('runtime_audit.py','qualify_source_trace.py')}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
