"""Pre-registered v4 feedback scheduler for the compiler-consistency pilot.

The branch schedule is a deterministic ten-request cycle: 2 uniform-admission
draws (20%), 5 semantic/structure coverage draws (50%), and 3 behavior
exploration draws (30%).  Quotas are applied before exploitation scores, and
positive novelty is gated by domain admission, eager/reference validity, and
oracle applicability.
"""

from collections import Counter, defaultdict
import math
import random
import re

from .conditions_v3 import AXES, OBJECTIVES, compatible_cells
from .construction_v3 import digest


SCHEMA = "route-a-feedback-v4.2"
BRANCH_CYCLE = (
    "uniform_random_20pct", "uniform_random_20pct",
    "semantic_structure_50pct", "semantic_structure_50pct", "semantic_structure_50pct",
    "semantic_structure_50pct", "semantic_structure_50pct",
    "behavior_exploration_30pct", "behavior_exploration_30pct", "behavior_exploration_30pct",
)
RANDOM_FRACTION = 0.20
SEMANTIC_FRACTION = 0.50
BEHAVIOR_FRACTION = 0.30
MINIMUM_AXIS_QUOTA = 5


def normalize_text(value):
    """Remove runtime identity while preserving semantic/compiler text."""
    text = str(value)
    text = re.sub(r"0x[0-9a-fA-F]+", "<address>", text)
    text = re.sub(r"\b(?:pid|process)[=: ]+\d+\b", "pid=<pid>", text, flags=re.I)
    text = re.sub(r"/data1/tzh/[^\s\"']+", "<data1-path>", text)
    text = re.sub(r"(?:tmp|buf)\d+", lambda m: re.sub(r"\d+", "<N>", m.group(0)), text)
    text = re.sub(r"line \d+", "line <N>", text)
    return text


def stable_normalized(value):
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, dict):
        return {str(k): stable_normalized(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [stable_normalized(v) for v in value]
    if isinstance(value, set):
        return sorted(stable_normalized(v) for v in value)
    return value


def stable_signature(value):
    return digest(stable_normalized(value))


class Scheduler:
    def __init__(self, capability, seed=20261102, historical_signatures=None):
        pool = list(compatible_cells(capability))
        rng = random.Random(seed)
        rng.shuffle(pool)
        self.menu = pool[:4096]
        self.menu_hash = digest([cell.to_dict() for cell in self.menu])
        self._cell_keys = {cell: digest(cell.to_dict()) for cell in self.menu}
        self._structure_proxies = {
            cell: stable_signature({"family": cell.family, "variant": cell.variant,
                                    "layout": cell.layout, "mode": cell.mode,
                                    "context": cell.context}) for cell in self.menu
        }
        self.seed = int(seed)
        historical = historical_signatures or {}
        self.historical = {key: set(map(str, historical.get(key, [])))
                           for key in ("graphs", "decisions", "behaviors", "failures")}
        self.rows = []
        self.visits = Counter()
        self.rewards = Counter()
        self.selected = {axis: Counter() for axis in AXES}
        self.legal = {axis: Counter() for axis in AXES}
        self.graphs = set()
        self.decisions = set()
        self.behaviors = set()
        self.cells = set()
        self.failures = set()
        self.cell_behaviors = defaultdict(set)

    def _counts(self):
        return {axis: self.selected[axis] for axis in AXES}

    def _quota_deficits(self, cell):
        return {axis: max(0, MINIMUM_AXIS_QUOTA - self.legal[axis][value])
                for axis, value in cell.tags().items()}

    def _cell_structure_signature(self, cell):
        # Pre-execution structure proxy used only to spread menu selection;
        # observed graph signatures remain the authoritative novelty signal.
        return self._structure_proxies[cell]

    def _score(self, cell, branch, request_id):
        tags = cell.tags()
        deficits = self._quota_deficits(cell)
        key = self._cell_keys[cell]
        visits = self.visits[key]
        mean = self.rewards[key] / max(1, visits)
        ucb = mean + math.sqrt(2 * math.log(len(self.rows) + 2) / (1 + visits))
        return {
            "semantic_deficit": float(sum(deficits.values())),
            "per_axis_deficit": deficits,
            "selected_counts": {axis: self.selected[axis][value] for axis, value in tags.items()},
            "legal_counts": {axis: self.legal[axis][value] for axis, value in tags.items()},
            "structure_proxy_seen": self._cell_structure_signature(cell) in self.graphs,
            "behavior_cells_seen": len(self.cell_behaviors.get(key, ())),
            "cell_visits": visits,
            "mean_reward": mean,
            "ucb": ucb,
            "tie_break": digest([len(self.rows), request_id, key]),
        }

    @staticmethod
    def _semantic_key(item):
        score, cell = item
        return (-score["semantic_deficit"], score["cell_visits"],
                int(score["structure_proxy_seen"]), score["tie_break"], cell.to_dict())

    @staticmethod
    def _behavior_key(item):
        score, cell = item
        return (score["behavior_cells_seen"], -score["ucb"],
                score["cell_visits"], score["tie_break"], cell.to_dict())

    def select(self, request_id):
        if any(row["request_id"] == request_id for row in self.rows):
            raise ValueError("duplicate_request")
        n = len(self.rows)
        branch = BRANCH_CYCLE[n % len(BRANCH_CYCLE)]
        if branch == "uniform_random_20pct":
            rng = random.Random(int(digest([self.seed, n, request_id, self.menu_hash]), 16))
            chosen = self.menu[rng.randrange(len(self.menu))]
            reason = "uniform_random_from_admitted_menu"
        else:
            scored = [(self._score(cell, branch, request_id), cell) for cell in self.menu]
            if branch == "semantic_structure_50pct":
                _, chosen = min(scored, key=self._semantic_key)
                reason = "quota_first_semantic_and_structure_coverage"
            else:
                _, chosen = min(scored, key=self._behavior_key)
                reason = "quota_first_behavior_ucb_and_underexplored_cells"
        score = self._score(chosen, branch, request_id)
        selection = {
            "schema": SCHEMA,
            "objective": OBJECTIVES[n % len(OBJECTIVES)],
            "branch": branch,
            "reason": reason,
            "prior_rows": n,
            "menu_sha256": self.menu_hash,
            "fractions": {"uniform_random": RANDOM_FRACTION,
                          "semantic_structure": SEMANTIC_FRACTION,
                          "behavior_exploration": BEHAVIOR_FRACTION},
            "score": score,
        }
        return chosen, selection

    @staticmethod
    def _is_domain_admitted(outcome, evidence):
        return bool(evidence.get("domain_admitted", outcome.get("domain_admitted", True)))

    @staticmethod
    def _is_reference_valid(outcome, evidence):
        if "reference_valid" in evidence or "reference_valid" in outcome:
            return bool(evidence.get("reference_valid", outcome.get("reference_valid")))
        return bool(outcome.get("applicable_valid"))

    def observe(self, request_id, cell, selection, outcome, evidence=None):
        evidence = evidence or {}
        rejection = outcome.get("rejection")
        domain_admitted = self._is_domain_admitted(outcome, evidence)
        reference_valid = self._is_reference_valid(outcome, evidence)
        oracle_applicable = bool(evidence.get("oracle_applicable",
                                           outcome.get("oracle_applicability") == "applicable"
                                           or outcome.get("oracle_status") == "pass"))
        eligible = bool(domain_admitted and reference_valid and oracle_applicable)
        graph = evidence.get("graph_structure_signature")
        decisions = set(map(str, evidence.get("compiler_decision_signatures",
                                               outcome.get("decision_signatures", []))))
        kernels = set(map(str, evidence.get("kernel_signatures",
                                            outcome.get("kernel_signatures", []))))
        guards = set(map(str, evidence.get("guard_signatures", outcome.get("guards", []))))
        origins = set(map(str, evidence.get("lowering_origins", [])))
        recompile = stable_signature({"backend_invocations": outcome.get("backend_invocations", 0),
                                      "recompiles": outcome.get("recompiles", 0)})
        behaviors = kernels | guards | origins
        if kernels or guards or origins or outcome.get("backend_invocations", 0):
            behaviors.add(recompile)
        graph_new = bool(graph and graph not in self.graphs and graph not in self.historical["graphs"])
        decision_new = bool(decisions - self.decisions - self.historical["decisions"])
        behavior_new = bool(behaviors - self.behaviors - self.historical["behaviors"])
        cell_key = digest(cell.to_dict())
        failure_signature = evidence.get("normalized_failure_signature")
        failure_signature = str(failure_signature) if failure_signature else None
        duplicate = bool(evidence.get("known_duplicate")
                         or (graph and graph in self.graphs and not decision_new and not behavior_new)
                         or (graph and graph in self.historical["graphs"])
                         or (failure_signature and failure_signature in self.historical["failures"])
                         or (failure_signature and failure_signature in self.failures))
        candidate_validated = bool(evidence.get("candidate_validated"))
        positive_allowed = eligible
        cost = max(0.0, float(outcome.get("cost_seconds", 0.0)))
        components = {
            "semantic_deficit": 0.50 * int(positive_allowed and cell_key not in self.cells),
            "graph_novelty": 1.00 * int(positive_allowed and graph_new),
            "decision_novelty": 1.00 * int(positive_allowed and decision_new),
            "runtime_novelty": 1.00 * int(positive_allowed and behavior_new),
            "candidate_frontier": 1.50 * int(positive_allowed and candidate_validated and not duplicate),
            "duplicate_penalty": -1.00 * int(duplicate),
            "cost_penalty": -min(0.75, cost / 120.0 * 0.75),
            "ineligible_penalty": -0.25 * int(not eligible),
        }
        row = {
            "schema": SCHEMA, "request_id": request_id, "cell": cell.to_dict(),
            "selection": selection,
            "eligibility": {"eligible": eligible, "domain_admitted": domain_admitted,
                            "reference_valid": reference_valid,
                            "oracle_applicable": oracle_applicable,
                            "compiled_agrees": bool(evidence.get("compiled_agrees",
                                                               outcome.get("compiled_agrees", False))),
                            "rejection_category": rejection.get("category") if rejection else None,
                            "candidate_validated": candidate_validated},
            "signatures": {"graph": graph, "decisions": sorted(decisions),
                           "behaviors": sorted(behaviors), "failure": failure_signature},
            "novelty": {"graph_new": graph_new, "decision_new": decision_new,
                        "behavior_new": behavior_new, "duplicate": duplicate},
            "components": components,
            "reward": sum(components.values()),
        }
        tags = cell.tags()
        for axis, value in tags.items():
            self.selected[axis][value] += 1
            if domain_admitted and reference_valid:
                self.legal[axis][value] += 1
        if eligible:
            if graph:
                self.graphs.add(str(graph))
            self.decisions.update(decisions)
            self.behaviors.update(behaviors)
            self.cells.add(cell_key)
            self.cell_behaviors[cell_key].update(behaviors)
        if failure_signature:
            self.failures.add(failure_signature)
        self.rows.append(row)
        self.visits[cell_key] += 1
        self.rewards[cell_key] += row["reward"]
        return row

    def state(self):
        return {
            "schema": SCHEMA, "seed": self.seed, "menu_sha256": self.menu_hash,
            "rows": self.rows,
            "graphs": sorted(self.graphs), "decisions": sorted(self.decisions),
            "behaviors": sorted(self.behaviors), "cells": sorted(self.cells),
            "failures": sorted(self.failures),
            "selected": {axis: dict(sorted(values.items())) for axis, values in self.selected.items()},
            "legal": {axis: dict(sorted(values.items())) for axis, values in self.legal.items()},
            "visits": dict(sorted(self.visits.items())),
            "rewards": dict(sorted(self.rewards.items())),
            "objectives": dict(Counter(row["selection"]["objective"] for row in self.rows)),
            "branches": dict(Counter(row["selection"]["branch"] for row in self.rows)),
            "historical": {key: sorted(value) for key, value in self.historical.items()},
            "policy": {"branch_cycle": list(BRANCH_CYCLE),
                       "fractions": {"uniform_random": RANDOM_FRACTION,
                                     "semantic_structure": SEMANTIC_FRACTION,
                                     "behavior_exploration": BEHAVIOR_FRACTION},
                       "minimum_axis_quota": MINIMUM_AXIS_QUOTA,
                       "positive_novelty_requires_domain_reference_oracle": True,
                       "zero_kernel_has_no_intrinsic_reward": True,
                       "bounded_component_max_abs": 1.5},
        }
