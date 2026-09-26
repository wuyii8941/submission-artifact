#!/usr/bin/env python3
import argparse
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import pickle
import random
import select
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("OATEST_SRC", HERE / "oatest")) / "res" / "tvm_ut"
OATEST = Path(os.environ.get("OATEST_PATCHED", HERE / "oatest_patched"))
ARMS = ("pattern_uniform", "pass_uniform", "hit_feedback")
HANG = 600


def supervise(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"arm": args.arm, "seed": args.seed, "targets_file": str(args.targets),
                "budget_seconds": args.seconds, "start": time.time(),
                "deadline": time.time() + args.seconds, "restarts": []}
        meta_path.write_text(json.dumps(meta, indent=1))
    skip = [r["attempt"] for r in meta["restarts"] if r.get("attempt") is not None]
    idle_deaths = 0
    while time.time() < meta["deadline"] and idle_deaths < 5:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
               "--arm", meta["arm"], "--seed", str(meta["seed"]),
               "--targets", meta["targets_file"], "--output", str(out),
               "--deadline", str(meta["deadline"]), "--skip", json.dumps(skip)]
        with open(out / "worker.log", "a") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            heartbeat = out / "heartbeat"
            reason = None
            while proc.poll() is None:
                time.sleep(5)
                last = heartbeat.stat().st_mtime if heartbeat.exists() else meta["start"]
                if time.time() - last > HANG:
                    proc.kill()
                    proc.wait()
                    reason = "hang"
            code = proc.returncode
        if code == 0 and reason is None:
            break
        state = _load(out / "checkpoint.json") or {"next_attempt": 0}
        attempt = state["next_attempt"]
        begun = _load(out / "inflight.json")
        if not begun or begun.get("attempt") != attempt:
            attempt = None
        meta["restarts"].append({"time": time.time(), "exit_code": code,
                                 "reason": reason or "crash", "attempt": attempt})
        meta_path.write_text(json.dumps(meta, indent=1))
        if attempt is not None:
            skip.append(attempt)
            idle_deaths = 0
        else:
            idle_deaths += 1
    meta["finished"] = time.time()
    meta_path.write_text(json.dumps(meta, indent=1))
    attempts = out / "attempts.jsonl"
    if not attempts.exists() or attempts.stat().st_size == 0:
        sys.exit(f"no attempts recorded in {out}; see {out / 'worker.log'}")


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def worker(args):
    import numpy as np
    import tvm
    from tvm import relax
    sys.path.insert(0, str(OATEST))
    sys.path.insert(0, str(HERE))
    import synthesis_mode
    import get_pass_level
    import parse_utils
    import transform_decompile
    import matcher

    matcher.install_alarm()
    out = args.output
    targets = tuple(json.loads(Path(args.targets).read_text())["targets"])
    beat = lambda: (out / "heartbeat").touch()

    donors_by_pass = {t: sorted(str(p.relative_to(CORPUS)) for p in (CORPUS / t).rglob("*.py"))
                      for t in targets}
    donor_list = [d for t in targets for d in donors_by_pass[t]]
    base_list = sorted(str(p.relative_to(CORPUS)) for p in CORPUS.rglob("*.py"))
    pool_dir = out / "pool"
    pool_dir.mkdir(exist_ok=True)

    state = _load(out / "checkpoint.json")
    if state is None:
        rng = random.Random(args.seed)
        state = {"next_attempt": 0, "hits": {t: 0 for t in targets},
                 "attempts": {t: 0 for t in targets}, "grown": [], "rng": None}
        import recipes
        (out / "config.json").write_text(json.dumps({
            "arm": args.arm, "seed": args.seed, "tvm": tvm.__version__,
            "targets_file": str(args.targets), "targets": list(targets),
            "pipeline": list(recipes.PIPELINE_NAMES), "pass_config": recipes.PASS_CONFIG,
            "oatest": OATEST.name, "n_donors": len(donor_list),
            "donors_per_target": {t: len(v) for t, v in donors_by_pass.items()},
            "n_bases_initial": len(base_list), "timeout": matcher.TIMEOUT,
        }, indent=1))
    else:
        rng = random.Random()
        rng.setstate((state["rng"][0], tuple(state["rng"][1]), state["rng"][2]))
    skip = set(json.loads(args.skip))
    records = open(out / "attempts.jsonl", "a")

    def choose():
        if args.arm == "pattern_uniform":
            donor = rng.choice(donor_list)
            target = donor.split("/")[0]
        else:
            if args.arm == "pass_uniform":
                target = rng.choice(targets)
            else:
                weights = [1.0 / (1 + state["hits"][t]) for t in targets]
                target = rng.choices(targets, weights=weights)[0]
            donor = rng.choice(donors_by_pass[target])
        return target, donor, rng.choice(base_list + state["grown"])

    def read_ir(name):
        path = pool_dir / name if name.startswith("grown_") else CORPUS / name
        return tvm.script.from_source(path.read_text(encoding="utf-8"))

    def forked(fn):
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            with os.fdopen(write_fd, "wb") as pipe:
                def stage(name):
                    pickle.dump(("stage", name), pipe)
                    pipe.flush()
                try:
                    result = fn(stage)
                except BaseException as exc:
                    result = {"status": "exception", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                pickle.dump(("result", result), pipe)
                pipe.flush()
            os._exit(0)
        os.close(write_fd)
        last_stage, result = None, None
        with os.fdopen(read_fd, "rb", buffering=0) as pipe:
            deadline = time.time() + HANG - 60
            while True:
                ready, _, _ = select.select([pipe], [], [], max(0.0, deadline - time.time()))
                if not ready:
                    os.kill(pid, signal.SIGKILL)
                    break
                try:
                    kind, value = pickle.load(pipe)
                except (EOFError, pickle.UnpicklingError, ValueError):
                    break
                if kind == "stage":
                    last_stage = value
                    beat()
                else:
                    result = value
                    break
        _, status = os.waitpid(pid, 0)
        if result is not None:
            return result
        signaled = os.WIFSIGNALED(status)
        return {"status": "hang" if signaled and os.WTERMSIG(status) == signal.SIGKILL else "crash",
                "crash_stage": last_stage,
                "exit_signal": os.WTERMSIG(status) if signaled else None}

    def attempt_once(target, donor, base, stage):
        rec = {}
        t0 = time.time()
        stage("synthesis")
        synth = matcher.Recorder()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                donor_ir, base_ir = read_ir(donor), read_ir(base)
                level = get_pass_level.get_level(donor.split("/")[0].split("__"))
                if level == "block":
                    try:
                        donor_ir = transform_decompile.decompile_ir(
                            parse_utils.update_irs_inner_fun_name(donor_ir))
                    except Exception:
                        level = "dataflow"
                rec["level"] = level
                with tvm.transform.PassContext(opt_level=3, instruments=[synth]):
                    value = matcher.timed(synthesis_mode.synthesize, base_ir, donor_ir, level,
                                          heartbeat=beat)
            except (matcher.Timeout, Exception) as exc:
                value = "exception"
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["synth_seconds"] = round(time.time() - t0, 3)
        if value == "exception":
            rec["status"] = "timeout" if rec["error"].startswith("Timeout") else "exception"
            return rec
        rec["status"] = ("generated" if value else
                         "invalid" if value is False else "no_connection")
        events = [("synthesis",) + e for e in synth.events]
        outcome = None
        if value:
            rec["source"], rec["function"] = value[0], value[2]
            t1 = time.time()
            try:
                mod = tvm.script.from_source(value[0])
            except Exception as exc:
                rec["status"] = "emitted_parse_error"
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                mod = None
            if mod is not None:
                rec["well_formed"] = bool(relax.analysis.well_formed(mod))
                rec["grow"] = rec["well_formed"]
                stage("evaluate")
                outcome, eval_events = matcher.evaluate(target, mod, heartbeat=beat)
                rec["eval"] = outcome
                events += eval_events
            rec["eval_seconds"] = round(time.time() - t1, 3)
        stages = matcher.selected_stages(target, synth.events, outcome)
        rec["selected_hit"] = bool(stages)
        rec["selected_stages"] = stages
        observed = {}
        for stage, name, depth, changed in events:
            if changed:
                for t in matcher.recipes.EVENT_TO_TARGETS.get(name, ()):
                    observed.setdefault(t, set()).add(stage if depth == 0 else stage + ":nested")
        rec["observed"] = {t: sorted(s) for t, s in observed.items()}
        return rec

    while time.time() < args.deadline:
        i = state["next_attempt"]
        beat()
        target, donor, base = choose()
        state["attempts"][target] += 1
        record = {"attempt": i, "t": round(time.time(), 2), "target": target,
                  "donor": donor, "base": base}
        if i in skip:
            record["status"] = "worker_died"
        else:
            _atomic(out / "inflight.json", {"attempt": i})
            random.seed(args.seed * 1000003 + i)
            np.random.seed((args.seed * 1000003 + i) % 2**32)
            record.update(forked(lambda stage: attempt_once(target, donor, base, stage)))
            if record.get("selected_hit"):
                state["hits"][target] += 1
            source = record.pop("source", None)
            if source is not None:
                with gzip.open(out / "programs.jsonl.gz", "at") as programs:
                    programs.write(json.dumps({"attempt": i, "target": target, "donor": donor,
                                               "base": base, "function": record.get("function"),
                                               "source": source}) + "\n")
                if record.get("grow"):
                    name = f"grown_{i:07d}.py"
                    (pool_dir / name).write_text(source, encoding="utf-8")
                    state["grown"].append(name)
        record["seconds"] = round(time.time() - record["t"], 3)
        records.write(json.dumps(record) + "\n")
        records.flush()
        state["next_attempt"] = i + 1
        rs = rng.getstate()
        state["rng"] = [rs[0], list(rs[1]), rs[2]]
        _atomic(out / "checkpoint.json", state)
    records.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--skip", default="[]")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        args.targets = args.targets.resolve()
        supervise(args)
