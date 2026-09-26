#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def load(run, max_attempts=None):
    by_index = {}
    for line in open(run / "attempts.jsonl"):
        record = json.loads(line)
        by_index[record["attempt"]] = record
    records = [by_index[i] for i in sorted(by_index)]
    if max_attempts is not None:
        records = [r for r in records if r["attempt"] < max_attempts]
    config = json.loads((run / "config.json").read_text())
    meta = json.loads((run / "run_meta.json").read_text())
    return config, meta, records


def per_target(config, records):
    rows = {t: Counter() for t in config["targets"]}
    for r in records:
        row = rows[r["target"]]
        row["attempts"] += 1
        row[r["status"]] += 1
        if r.get("selected_hit"):
            row["selected_hits"] += 1
        for stage in r.get("selected_stages", []):
            row["sel_" + stage] += 1
        ev = r.get("eval")
        if ev:
            row["direct_" + ev["direct"]] += 1
            if ev["pipeline"]:
                row["pipe_" + ev["pipeline"]] += 1
        for target, stages in r.get("observed", {}).items():
            if target not in rows:
                continue
            top = [s for s in stages if not s.endswith(":nested")]
            if top:
                rows[target]["observed"] += 1
            rows[target]["observed_incl_nested"] += 1
    return rows


def run_summary(run, max_attempts=None):
    config, meta, records = load(run, max_attempts)
    rows = per_target(config, records)
    sel = [rows[t]["selected_hits"] for t in config["targets"]]
    obs = [rows[t]["observed"] for t in config["targets"]]
    return {
        "run": str(run), "arm": config["arm"], "seed": config["seed"],
        "attempts": len(records),
        "status": dict(Counter(r["status"] for r in records)),
        "restarts": len(meta.get("restarts", [])),
        "selected_ge1": sum(n >= 1 for n in sel), "selected_ge10": sum(n >= 10 for n in sel),
        "observed_ge1": sum(n >= 1 for n in obs), "observed_ge10": sum(n >= 10 for n in obs),
        "selected_hits_total": sum(sel),
        "targets": len(config["targets"]),
        "per_target": {t: dict(rows[t]) for t in config["targets"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--per-target", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summaries = [run_summary(r, args.max_attempts) for r in args.runs]
    for s in summaries:
        print(f"{s['arm']:16s} seed={s['seed']:<4d} attempts={s['attempts']:<6d} "
              f"sel>=1={s['selected_ge1']:<3d} sel>=10={s['selected_ge10']:<3d} "
              f"obs>=1={s['observed_ge1']:<3d} obs>=10={s['observed_ge10']:<3d} "
              f"sel_total={s['selected_hits_total']:<6d} restarts={s['restarts']} {s['status']}")
    by_arm = defaultdict(list)
    for s in summaries:
        by_arm[s["arm"]].append(s)
    if any(len(v) > 1 for v in by_arm.values()):
        for arm, group in by_arm.items():
            def ms(key):
                values = [g[key] for g in group]
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                return f"{statistics.mean(values):.1f}±{sd:.1f}"
            union1 = {t for g in group for t, row in g["per_target"].items()
                      if row.get("selected_hits", 0) >= 1}
            print(f"{arm:16s} n={len(group)} sel>=1 {ms('selected_ge1')} sel>=10 {ms('selected_ge10')} "
                  f"obs>=1 {ms('observed_ge1')} obs>=10 {ms('observed_ge10')} union sel>=1 {len(union1)}")
    if args.per_target:
        keys = ["attempts", "generated", "selected_hits", "sel_synthesis", "sel_direct",
                "sel_pipeline", "direct_pass_error", "pipe_prefix_error", "pipe_pass_error",
                "observed"]
        for s in summaries:
            print(f"\n## {s['arm']} seed={s['seed']}")
            print("target".ljust(30) + "".join(k[:11].rjust(12) for k in keys))
            for t, row in s["per_target"].items():
                print(t.ljust(30) + "".join(str(row.get(k, 0)).rjust(12) for k in keys))
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
