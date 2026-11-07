"""Target and condition selection policies with explicit accounting."""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class CellStats:
    attempts: int = 0
    valid: int = 0
    observable: int = 0
    selected_hits: int = 0
    novel_behaviors: int = 0
    cost_seconds: float = 0.0

    def hit_rate(self) -> float:
        return self.selected_hits / max(1, self.observable)


class ExplorationHistory:
    def __init__(self, target_ids: Iterable[str]):
        self.target_stats: Dict[str, CellStats] = {tid: CellStats() for tid in target_ids}
        self.cell_stats: Dict[Tuple[str, str], CellStats] = defaultdict(CellStats)
        self.total_attempts = 0

    def cell_stat(self, target_id: str, cell: str) -> CellStats:
        """Read a cell without materializing an unvisited entry."""
        return self.cell_stats.get((target_id, cell), CellStats())

    def record(self, target_id: Optional[str], cell: Optional[str], *, valid: bool,
               observable: bool, selected_hit: bool, novel: bool, cost: float) -> None:
        self.total_attempts += 1
        if target_id is None:
            return
        target = self.target_stats.setdefault(target_id, CellStats())
        target.attempts += 1
        target.valid += int(valid)
        target.observable += int(observable)
        target.selected_hits += int(selected_hit)
        target.novel_behaviors += int(novel)
        target.cost_seconds += float(cost)
        if cell is not None:
            stats = self.cell_stats[(target_id, cell)]
            stats.attempts += 1
            stats.valid += int(valid)
            stats.observable += int(observable)
            stats.selected_hits += int(selected_hit)
            stats.novel_behaviors += int(novel)
            stats.cost_seconds += float(cost)


class TargetPolicy:
    def __init__(self, target_ids: Iterable[str], seed: int):
        self.target_ids = tuple(target_ids)
        self.rng = random.Random(seed)

    def select(self, history: ExplorationHistory) -> str:
        raise NotImplementedError


class UniformTargetPolicy(TargetPolicy):
    def select(self, history: ExplorationHistory) -> str:
        del history
        return self.rng.choice(self.target_ids)


class AttemptBalancedTargetPolicy(TargetPolicy):
    def select(self, history: ExplorationHistory) -> str:
        minimum = min(history.target_stats[tid].attempts for tid in self.target_ids)
        choices = [tid for tid in self.target_ids if history.target_stats[tid].attempts == minimum]
        return self.rng.choice(choices)


class ArtifactFeedbackTargetPolicy(TargetPolicy):
    def select(self, history: ExplorationHistory) -> str:
        if self.rng.random() < 0.10:
            return self.rng.choice(self.target_ids)
        if history.total_attempts < len(self.target_ids):
            unseen = [tid for tid in self.target_ids if history.target_stats[tid].attempts == 0]
            if unseen:
                return self.rng.choice(unseen)
        total = max(1, history.total_attempts)
        scores = []
        for tid in self.target_ids:
            stats = history.target_stats[tid]
            uncertainty = math.sqrt(math.log(total + 1.0) / max(1, stats.attempts))
            underexplored = 1.0 / math.sqrt(1.0 + stats.selected_hits)
            if stats.attempts >= 8:
                underexplored *= (stats.selected_hits + 1.0) / (stats.attempts + 2.0)
            failure_cost = 0.10 * (stats.attempts - stats.valid) / max(1, stats.attempts)
            scores.append((underexplored + uncertainty - failure_cost, tid))
        best = max(score for score, _ in scores)
        choices = [tid for score, tid in scores if abs(score - best) < 1e-12]
        return self.rng.choice(choices)


class ConditionPolicy:
    def __init__(self, seed: int, exploration_probability: float = 0.20):
        self.rng = random.Random(seed)
        self.exploration_probability = exploration_probability
        self.last_branch = "not_selected"

    def select(self, target_id: str, cells: List[Tuple[str, str, object, object]],
               history: ExplorationHistory, adaptive: bool):
        if not cells:
            self.last_branch = "no_cells"
            return None
        if not adaptive or self.rng.random() < self.exploration_probability:
            self.last_branch = "exploration" if adaptive else "fixed_prior"
            return self.rng.choice(cells)
        stats = [(history.cell_stat(target_id, cell[1]), cell) for cell in cells]
        unseen = [cell for stat, cell in stats if stat.attempts == 0]
        if unseen:
            self.last_branch = "warmup"
            return self.rng.choice(unseen)
        # UCB reward combines selected-node hits and newly observed behaviors
        # attributable to that node. Failed or unobservable cells get no
        # reward; uncertainty and epsilon preserve exploration.
        def score(item):
            stat, _cell = item
            reward = (stat.selected_hits + stat.novel_behaviors) / (2.0 * stat.attempts)
            uncertainty = math.sqrt(2.0 * math.log(history.total_attempts + 1) / stat.attempts)
            average_cost = stat.cost_seconds / stat.attempts
            return (reward + uncertainty) / math.sqrt(1.0 + average_cost)
        self.last_branch = "adaptive_score"
        best = max(score(item) for item in stats)
        return self.rng.choice([cell for stat, cell in stats if abs(score((stat, cell)) - best) < 1e-12])
