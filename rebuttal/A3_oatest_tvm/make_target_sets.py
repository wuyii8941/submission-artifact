#!/usr/bin/env python3
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
calibration = json.loads((HERE / "results" / "calibration_seed2024.json").read_text())
rows = calibration["per_target"]
source = {"calibration": "results/calibration_seed2024.json", "seed": calibration["seed"],
          "min_attempts": calibration["min_attempts"],
          "bases_per_donor": calibration["bases_per_donor"]}
sets = {
    "primary": {"rule": "generatable and confirmable in calibration",
                "targets": [t for t, r in rows.items() if r["generatable"] and r["confirmable"]]},
    "generatable": {"rule": "generatable in calibration",
                    "targets": [t for t, r in rows.items() if r["generatable"]]},
}
for name, spec in sets.items():
    spec.update(name=name, **source)
    spec["excluded"] = {t: {k: rows[t][k] for k in ("patterns", "attempts", "generated",
                                                  "confirmed", "status")}
                        for t in rows if t not in spec["targets"]}
    (HERE / f"targets_{name}.json").write_text(json.dumps(spec, indent=1))
    print(name, len(spec["targets"]), spec["targets"])
