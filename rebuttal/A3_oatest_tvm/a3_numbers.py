#!/usr/bin/env python3
import gzip, json, os, re, statistics, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a3_summarize import load, per_target

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("A3_RUNS", HERE / "results" / "runs")) / "formal_primary"
ARMS, SEEDS = ("pattern_uniform", "pass_uniform", "hit_feedback"), (42, 123, 456)
HARD = ("ComputePrimValue", "RealizeVDevice", "MergeCompositeFunctions")

runs = {(a, s): load(RUNS / f"{a}_seed{s}") for a in ARMS for s in SEEDS}
targets = runs[ARMS[0], SEEDS[0]][0]["targets"]


def hits(arm, seed, max_attempts=None):
    config, _, records = runs[arm, seed]
    if max_attempts is not None:
        records = [r for r in records if r["attempt"] < max_attempts]
    rows = per_target(config, records)
    return {t: rows[t]["selected_hits"] for t in targets}


def msd(v):
    return {"mean": round(statistics.mean(v), 1), "sd": round(statistics.stdev(v), 1), "per_seed": v}


out = {"passes": len(targets), "tvm_commit": "292ecfd", "seeds": list(SEEDS), "budget_seconds": 7200, "targets": targets}
out["passes_hit"] = {}
for a in ARMS:
    h = [hits(a, s) for s in SEEDS]
    out["passes_hit"][a] = {"ge1": msd([sum(v >= 1 for v in x.values()) for x in h]),
                            "ge10": msd([sum(v >= 10 for v in x.values()) for x in h])}
out["passes_hit_equal_attempts"] = []
for s in SEEDS:
    n = min(len(runs[a, s][2]) for a in ARMS)
    row = {"seed": s, "attempts": n}
    for a in ARMS:
        h = hits(a, s, n)
        row[a] = {"ge1": sum(v >= 1 for v in h.values()), "ge10": sum(v >= 10 for v in h.values())}
    out["passes_hit_equal_attempts"].append(row)

out["hit_feedback_vs_pattern_uniform"] = []
for s in SEEDS:
    o, f = hits("pattern_uniform", s), hits("hit_feedback", s)
    row = {"seed": s}
    for k in (1, 10):
        so, sf = {t for t in targets if o[t] >= k}, {t for t in targets if f[t] >= k}
        row[f"covers_all_ge{k}"] = so <= sf
        row[f"extra_ge{k}"] = len(sf - so)
    out["hit_feedback_vs_pattern_uniform"].append(row)

out["hard_passes_hits_sum_of_seeds"] = {t: {a: sum(hits(a, s)[t] for s in SEEDS) for a in ARMS} for t in HARD}

out["RealizeVDevice"] = {}
for a in ARMS:
    c = Counter()
    for s in SEEDS:
        progs = {}
        run_dir = RUNS / f"{a}_seed{s}"
        prog_file = run_dir / "programs_RealizeVDevice.jsonl.gz"
        if not prog_file.exists():
            prog_file = run_dir / "programs.jsonl.gz"
        with gzip.open(prog_file, "rt") as fh:
            for line in fh:
                p = json.loads(line)
                progs[p["attempt"]] = p["source"]
        for r in runs[a, s][2]:
            if r["target"] != "RealizeVDevice":
                continue
            c["selected"] += 1
            if r["status"] != "generated":
                continue
            c["generated"] += 1
            c[bool(r.get("selected_hit")), bool(re.search(r"I\.vdevice", progs[r["attempt"]]))] += 1
    out["RealizeVDevice"][a] = {"selected": c["selected"], "programs_generated": c["generated"],
                                "rewritten_declares_vdevice": c[True, True], "rewritten_no_vdevice": c[True, False],
                                "not_rewritten_declares_vdevice": c[False, True], "not_rewritten_no_vdevice": c[False, False]}

att = sum(len(runs["pattern_uniform", s][2]) for s in SEEDS)
leg = sum(r["target"] == "LegalizeOps" for s in SEEDS for r in runs["pattern_uniform", s][2])
out["pattern_uniform_budget"] = {"attempts": att, "LegalizeOps_attempts": leg, "LegalizeOps_share": round(leg / att, 3),
                                 "passes_never_rewritten": [t for t in targets if sum(hits("pattern_uniform", s)[t] for s in SEEDS) == 0]}

(HERE / "results").mkdir(exist_ok=True)
json.dump(out, open(HERE / "results" / "a3_numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
