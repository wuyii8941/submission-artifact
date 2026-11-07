"""Serial receipt-replayable v3 scheduler, six explicit rotating objectives.

Family quotas and coverage deficits precede bounded UCB/exploration. Positive
feedback requires an applicable oracle pass; anomalies remain unreviewed and
receive no novelty reward. No compiler-defect discovery claim is automated.
"""
from collections import Counter
import math
import random

from .conditions_v3 import AXES, OBJECTIVES, SemanticCell, compatible_cells
from .construction_v3 import digest


class Scheduler:
    def __init__(self, capability, seed=20261030):
        pool = list(compatible_cells(capability))
        rng = random.Random(seed)
        rng.shuffle(pool)
        self.menu = pool[:4096]
        self.menu_hash = digest([c.to_dict() for c in self.menu])
        self.rows = []
        self.visits, self.rewards = Counter(), Counter()
        self.decisions, self.kernels, self.cells = set(), set(), set()

    def select(self, request_id):
        if any(r['request_id'] == request_id for r in self.rows):
            raise ValueError('duplicate_request')
        n = len(self.rows)
        objective = OBJECTIVES[n % len(OBJECTIVES)]
        counts = {a: Counter(r['cell'][k] for r in self.rows) for a, k in zip(OBJECTIVES,
                  ('family', 'dtype', 'shape', 'layout', 'mode', 'context'))}
        low = min((counts['operator'][x] for x in AXES['operator']), default=0)
        pool = [c for c in self.menu if counts['operator'][c.family] == low]
        def score(c):
            tags = c.tags()
            # Fill each declared value at least five times before exploitation.
            deficits = sum(max(0, 5 - counts[a][v]) for a, v in tags.items())
            focus = counts[objective][tags[objective]] * len(AXES[objective])
            visits = self.visits[c]
            mean = self.rewards[c] / max(1, visits)
            ucb = mean + math.sqrt(2 * math.log(n + 2) / (1 + visits))
            tie = digest([n, request_id, c.to_dict()])
            return (-deficits, focus, -ucb, tie)
        chosen = min(pool, key=score)
        return chosen, {'objective': objective, 'branch': 'family_quota_coverage_ucb',
                        'prior_rows': n, 'menu_sha256': self.menu_hash}

    def observe(self, request_id, cell, selection, outcome):
        eligible = outcome['applicable_valid']
        decisions = set(outcome['decision_signatures'])
        kernels = set(outcome['kernel_signatures'])
        key = digest(cell.to_dict())
        components = {'condition_coverage': int(eligible and key not in self.cells),
                      'decision_novelty': int(eligible and bool(decisions - self.decisions)),
                      'kernel_novelty': int(eligible and bool(kernels - self.kernels)),
                      'semantic_anomaly': 0, 'cost': -outcome['cost_seconds'] / 100,
                      'rejection': -int(not eligible)}
        if eligible:
            self.decisions.update(decisions)
            self.kernels.update(kernels)
            self.cells.add(key)
        row = {'request_id': request_id, 'cell': cell.to_dict(), 'selection': selection,
               'components': components, 'reward': sum(components.values())}
        self.rows.append(row)
        self.visits[cell] += 1
        self.rewards[cell] += row['reward']
        return row

    def state(self):
        return {'rows': self.rows, 'decisions': sorted(self.decisions), 'kernels': sorted(self.kernels),
                'cells': sorted(self.cells), 'objectives': dict(Counter(r['selection']['objective'] for r in self.rows)),
                'menu_sha256': self.menu_hash}
