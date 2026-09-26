import signal

import tvm
from tvm.ir.instrument import pass_instrument

import recipes

TIMEOUT = 60


class Timeout(BaseException):
    pass


def _alarm(_signum, _frame):
    raise Timeout()


def install_alarm():
    signal.signal(signal.SIGALRM, _alarm)


def timed(fn, *args, heartbeat=None):
    if heartbeat is not None:
        heartbeat()
    signal.alarm(TIMEOUT)
    try:
        return fn(*args)
    finally:
        signal.alarm(0)


@pass_instrument
class Recorder:

    def __init__(self):
        self.stack, self.events = [], []

    def run_before_pass(self, mod, info):
        self.stack.append(mod)

    def run_after_pass(self, mod, info):
        before = self.stack.pop()
        self.events.append((info.name, len(self.stack),
                            not tvm.ir.structural_equal(before, mod)))


def synthesis_hit(target, synth_events):
    return any(depth == 0 and changed and target in recipes.EVENT_TO_TARGETS.get(name, ())
               for name, depth, changed in synth_events)


def _changed_by(target, mod, heartbeat):
    rec = Recorder()
    try:
        with tvm.target.Target("llvm"), recipes.context(target, [rec]):
            after = timed(recipes.make_pass(target), mod, heartbeat=heartbeat)
    except Timeout:
        return "timeout", rec.events
    except Exception:
        return "pass_error", rec.events
    return ("transformed" if not tvm.ir.structural_equal(mod, after)
            else "unchanged"), rec.events


def evaluate(target, mod, heartbeat=None):
    result = {"direct": None, "pipeline": None}
    events = []
    result["direct"], ev = _changed_by(target, mod, heartbeat)
    events += [("eval_direct",) + e for e in ev]
    prefix = recipes.prefix_of(target)
    if prefix is not None:
        rec = Recorder()
        try:
            with tvm.target.Target("llvm"), recipes.context(target, [rec]):
                x = mod
                for _, factory in prefix:
                    x = timed(factory(), x, heartbeat=heartbeat)
        except Timeout:
            result["pipeline"] = "prefix_timeout"
        except Exception:
            result["pipeline"] = "prefix_error"
        else:
            result["pipeline"], ev = _changed_by(target, x, heartbeat)
            events += [("eval_pipeline",) + e for e in ev]
        events += [("eval_prefix",) + e for e in rec.events]
    return result, events


def selected_stages(target, synth_events, outcome):
    stages = []
    if synthesis_hit(target, synth_events):
        stages.append("synthesis")
    if outcome and outcome["direct"] == "transformed":
        stages.append("direct")
    if outcome and outcome["pipeline"] == "transformed":
        stages.append("pipeline")
    return stages
