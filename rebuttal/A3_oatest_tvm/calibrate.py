#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("OATEST_SRC", HERE / "oatest")) / "res" / "tvm_ut"
OATEST = Path(os.environ.get("OATEST_PATCHED", HERE / "oatest_patched"))
STALL = 300


def shard_worker(jobs_file, results_file, start):
    import numpy as np
    import tvm
    sys.path.insert(0, str(OATEST))
    sys.path.insert(0, str(HERE))
    import synthesis_mode, get_pass_level, parse_utils, transform_decompile, matcher
    matcher.install_alarm()
    jobs = json.loads(Path(jobs_file).read_text())
    with open(results_file, "a") as out:
        for index in range(start, len(jobs)):
            donor, base, seed = jobs[index]
            random.seed(seed)
            np.random.seed(seed % 2**32)
            target = donor.split("/")[0]
            synth = matcher.Recorder()
            outcome = None
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    donor_ir = tvm.script.from_source((CORPUS / donor).read_text())
                    base_ir = tvm.script.from_source((CORPUS / base).read_text())
                    level = get_pass_level.get_level(target.split("__"))
                    if level == "block":
                        try:
                            donor_ir = transform_decompile.decompile_ir(
                                parse_utils.update_irs_inner_fun_name(donor_ir))
                        except Exception:
                            level = "dataflow"
                    with tvm.transform.PassContext(opt_level=3, instruments=[synth]):
                        value = matcher.timed(synthesis_mode.synthesize, base_ir, donor_ir, level)
                    if value:
                        mod = tvm.script.from_source(value[0])
                        status = "generated"
                        outcome, _ = matcher.evaluate(target, mod)
                    else:
                        status = "invalid" if value is False else "no_connection"
                except matcher.Timeout:
                    status = "timeout"
                except Exception:
                    status = "exception"
            hit = status == "generated" and bool(
                matcher.selected_stages(target, synth.events, outcome))
            out.write(json.dumps([index, status, hit]) + "\n")
            out.flush()


def run_shards(jobs, workdir, n_shards):
    workdir.mkdir(parents=True, exist_ok=True)
    shards = []
    for k in range(n_shards):
        part = jobs[k::n_shards]
        jobs_file, results_file = workdir / f"jobs_{k}.json", workdir / f"results_{k}.jsonl"
        jobs_file.write_text(json.dumps(part))
        shards.append({"k": k, "n": len(part), "jobs": jobs_file, "results": results_file,
                       "proc": None, "deaths": []})

    def done_count(shard):
        if not shard["results"].exists():
            return 0
        return sum(1 for _ in open(shard["results"]))

    def launch(shard):
        start = done_count(shard)
        if start >= shard["n"]:
            shard["proc"] = None
            return
        shard["proc"] = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--shard",
             str(shard["jobs"]), str(shard["results"]), str(start)],
            stdout=open(workdir / f"log_{shard['k']}.txt", "a"), stderr=subprocess.STDOUT)
        shard["last_progress"], shard["last_count"] = time.time(), start

    for shard in shards:
        launch(shard)
    while any(s["proc"] is not None for s in shards):
        time.sleep(5)
        for shard in shards:
            proc = shard["proc"]
            if proc is None:
                continue
            count = done_count(shard)
            if count != shard["last_count"]:
                shard["last_count"], shard["last_progress"] = count, time.time()
            stalled = time.time() - shard["last_progress"] > STALL
            if proc.poll() is None and not stalled:
                continue
            if stalled and proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.returncode == 0 and count >= shard["n"]:
                shard["proc"] = None
                continue
            with open(shard["results"], "a") as out:
                out.write(json.dumps([count, "worker_died", False]) + "\n")
            shard["deaths"].append({"index": count, "code": proc.returncode, "stalled": stalled})
            launch(shard)
    results = [None] * len(jobs)
    for shard in shards:
        for line in open(shard["results"]):
            index, status, hit = json.loads(line)
            results[shard["k"] + index * n_shards] = (status, hit)
    return results, [dict(k=s["k"], deaths=s["deaths"]) for s in shards]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases-per-donor", type=int, default=10)
    parser.add_argument("--min-attempts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard", nargs=3)
    args = parser.parse_args()
    if args.shard:
        shard_worker(args.shard[0], args.shard[1], int(args.shard[2]))
        return
    sys.path.insert(0, str(HERE))
    import recipes
    rng = random.Random(args.seed)
    bases = sorted(str(p.relative_to(CORPUS)) for p in CORPUS.rglob("*.py"))
    jobs = []
    for target in recipes.TARGETS:
        donors = sorted(str(p.relative_to(CORPUS)) for p in (CORPUS / target).rglob("*.py"))
        per = max(args.bases_per_donor, math.ceil(args.min_attempts / len(donors)))
        for donor in donors:
            for _ in range(per):
                jobs.append([donor, rng.choice(bases), rng.randrange(2**31)])
    results, deaths = run_shards(jobs, args.output.with_suffix(".work"), args.workers)
    per_target = {}
    for target in recipes.TARGETS:
        rows = [(job[0], res) for job, res in zip(jobs, results) if job[0].split("/")[0] == target]
        statuses = [res[0] for _, res in rows]
        per_target[target] = {
            "patterns": len({d for d, _ in rows}),
            "usable_patterns": len({d for d, res in rows if res[0] == "generated"}),
            "attempts": len(rows),
            "generated": statuses.count("generated"),
            "confirmed": sum(res[1] for _, res in rows),
            "status": {s: statuses.count(s) for s in sorted(set(statuses))},
        }
        row = per_target[target]
        row["generatable"] = row["generated"] > 0
        row["confirmable"] = row["confirmed"] > 0
    args.output.write_text(json.dumps({
        "seed": args.seed, "bases_per_donor": args.bases_per_donor,
        "min_attempts": args.min_attempts, "oatest": OATEST.name, "shard_deaths": deaths,
        "per_target": per_target,
        "jobs": [job + list(res) for job, res in zip(jobs, results)]}, indent=1))
    for target, row in per_target.items():
        print(f"{target:30s} {row}")
    print("generatable:", sum(r["generatable"] for r in per_target.values()),
          "confirmable:", sum(r["confirmable"] for r in per_target.values()), "/", len(per_target))


if __name__ == "__main__":
    main()
