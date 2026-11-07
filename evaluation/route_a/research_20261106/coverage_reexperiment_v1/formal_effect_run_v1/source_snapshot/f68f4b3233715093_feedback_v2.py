"""Deterministic, quota-constrained multi-objective UCB with reviewed rewards.

Selection reserves an attempt immediately. Observation and adjudication are
separate, checkpointed operations. Inapplicable outcomes never enter positive
coverage or novelty sets. Costs and rejection penalties remain separate.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Tuple
from .conditions_v2 import ConditionSpec, Dimension
from .storage import EventStore

SCHEMA = 'route-a-feedback-v2.1'
COMPONENTS = ('decision_novelty', 'condition_coverage', 'kernel_novelty', 'semantic_anomaly',
              'target_confirmation', 'illegal_generation', 'capability_rejection',
              'semantic_rejection', 'timeout', 'cost')


@dataclass(frozen=True, order=True)
class Candidate:
    target: str
    dimension: str
    value: str
    family: Tuple[int, ...] = ()
    axis: int = -1

    @property
    def key(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))

    def condition(self):
        return ConditionSpec(Dimension(self.dimension), self.value, self.axis, self.family)


@dataclass(frozen=True)
class SchedulerConfig:
    exploration_floor: float = 0.20
    ucb_scale: float = 1.0
    dimension_quotas: dict = field(default_factory=lambda: {'layout': .20, 'dynamic_shape': .20, 'autograd': .20, 'alias_mutation': .20})
    weights: dict = field(default_factory=lambda: dict(decision_novelty=1.0, condition_coverage=1.0,
        kernel_novelty=.75, semantic_anomaly=3.0, target_confirmation=.25,
        illegal_generation=-1., capability_rejection=-.25, semantic_rejection=-.1, timeout=-1., cost=-.01))
    cost_unit_seconds: float = 10.0
    reproducibility_repeats: int = 2

    def __post_init__(self):
        if not 0 < self.exploration_floor <= 1 or not math.isfinite(self.ucb_scale) or self.ucb_scale < 0:
            raise ValueError('invalid_exploration_or_ucb')
        if set(self.weights) != set(COMPONENTS) or not all(math.isfinite(x) for x in self.weights.values()):
            raise ValueError('invalid_weights')
        if any(self.weights[x] < 0 for x in COMPONENTS[:5]) or any(self.weights[x] > 0 for x in COMPONENTS[5:]):
            raise ValueError('reward_penalty_sign')
        if any(not math.isfinite(q) or q < 0 for q in self.dimension_quotas.values()) or sum(self.dimension_quotas.values()) > 1 + 1e-12:
            raise ValueError('invalid_dimension_quotas')
        if not math.isfinite(self.cost_unit_seconds) or self.cost_unit_seconds <= 0 or self.reproducibility_repeats < 2:
            raise ValueError('invalid_cost_or_review_count')


@dataclass
class Stats:
    attempts: int = 0
    reviewed: int = 0
    reward_sum: float = 0.
    components: dict = field(default_factory=lambda: {k: 0. for k in COMPONENTS})


@dataclass(frozen=True)
class CoverageCell:
    target: str
    dimension: str
    value: str
    compiler_stage: str
    kernel_family: str
    guard_recompile_signature: str
    oracle_applicability: str

    @property
    def key(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))


class FeedbackScheduler:
    def __init__(self, candidates, config=None):
        self.candidates = tuple(sorted(set(candidates)))
        if not self.candidates:
            raise ValueError('empty_candidates')
        self.config = config or SchedulerConfig()
        dims = {c.dimension for c in self.candidates}
        if any(q > 0 and d not in dims for d, q in self.config.dimension_quotas.items()):
            raise ValueError('quota_dimension_has_no_candidates')
        self.stats = {c.key: Stats() for c in self.candidates}
        self.reservations, self.pending, self.reviews = {}, {}, {}
        self.decisions, self.kernels, self.cells = set(), set(), set()
        self.observed_cells = set()
        self.condition_cells = set()
        self.selection_audit = []

    @property
    def total(self):
        return len(self.reservations)

    def reserve(self, trial_id, candidate, audit=None):
        if trial_id in self.reservations:
            raise ValueError('duplicate_trial')
        if candidate not in self.candidates:
            raise ValueError('unknown_candidate')
        self.reservations[trial_id] = candidate.key
        self.stats[candidate.key].attempts += 1
        self.selection_audit.append({'trial_id': trial_id, 'candidate': candidate.key, **(audit or {'branch': 'external'})})
        return candidate

    def select(self, trial_id):
        n = self.total
        dim_counts = {d: sum(self.stats[c.key].attempts for c in self.candidates if c.dimension == d)
                      for d in {c.dimension for c in self.candidates}}
        deficits = {d: (n + 1) * q - dim_counts[d] for d, q in self.config.dimension_quotas.items() if q > 0}
        scores = {}
        for c in self.candidates:
            s = self.stats[c.key]
            mean = s.reward_sum / max(1, s.reviewed)
            uncertainty = self.config.ucb_scale * math.sqrt(2 * math.log(n + 2) / max(1, s.attempts))
            scores[c.key] = {'mean': mean, 'ucb': uncertainty, 'score': mean + uncertainty,
                             'attempts': s.attempts, 'reviewed': s.reviewed}
        # Quotas are minimum attempt shares, including rejected attempts.
        # Least-used candidates within a deficient dimension cannot be starved.
        deficient = [d for d in deficits if deficits[d] > 1e-12]
        if deficient:
            d = min(deficient, key=lambda x: (-deficits[x], x))
            pool = [c for c in self.candidates if c.dimension == d]
            chosen = min(pool, key=lambda c: (self.stats[c.key].attempts, c.key))
            branch = 'dimension_quota'
        elif math.floor((n + 1) * self.config.exploration_floor) > math.floor(n * self.config.exploration_floor):
            chosen = min(self.candidates, key=lambda c: (self.stats[c.key].attempts, c.key))
            branch = 'exploration_floor'
        else:
            chosen = min(self.candidates, key=lambda c: (-scores[c.key]['score'], self.stats[c.key].attempts, c.key))
            branch = 'ucb'
        return self.reserve(trial_id, chosen, {'branch': branch, 'deficits': deficits, 'scores': scores})

    def observe(self, trial_id, outcome):
        if trial_id not in self.reservations or trial_id in self.pending or trial_id in self.reviews:
            raise ValueError('unreserved_or_duplicate_observation')
        if not math.isfinite(outcome.get('cost_seconds', 0.)) or outcome.get('cost_seconds', 0.) < 0:
            raise ValueError('invalid_cost')
        self.pending[trial_id] = json.loads(json.dumps(outcome, allow_nan=False))

    def review(self, trial_id, *, repeats=(), disposition='accepted'):
        if trial_id in self.reviews:
            return self.reviews[trial_id]  # Exactly once, including resume.
        out = self.pending[trial_id]
        c = next(c for c in self.candidates if c.key == self.reservations[trial_id])
        applicable = out.get('valid', False) and out.get('oracle_applicability') == 'applicable'
        eligible = applicable and out.get('legal', False) and not out.get('timeout', False) and not out.get('rejection_reasons')
        eligible = bool(eligible and disposition == 'accepted')
        parts = {k: 0. for k in COMPONENTS}
        parts['cost'] = (out.get('cost_seconds', 0.) + out.get('review_cost_seconds', 0.)) / self.config.cost_unit_seconds
        category = out.get('category')
        rejections = out.get('generation_rejections', {})
        parts['illegal_generation'] = float(rejections.get('illegal_generation', int(category == 'illegal_generation')))
        parts['capability_rejection'] = float(rejections.get('expected_capability_rejection', int(category == 'expected_capability_rejection')))
        parts['semantic_rejection'] = float(rejections.get('semantic_inapplicable', int(category == 'semantic_inapplicable'))
                                            + rejections.get('disabled_extension', int(category == 'disabled_extension')))
        parts['timeout'] = float(out.get('timeout', False))
        accepted_cells = []
        observed_cells = []
        for stage in out.get('compiler_stages', []) or ['unobserved']:
            for family in out.get('kernel_families', []) or ['unobserved']:
                cell = CoverageCell(c.target, c.dimension, c.value, stage, family,
                                    out.get('guard_recompile_signature', 'unobserved'),
                                    out.get('oracle_applicability', 'unknown'))
                self.observed_cells.add(cell.key)
                observed_cells.append(cell.key)
        if eligible:
            decisions = set(out.get('decision_signatures', []))
            kernels = set(out.get('kernel_signatures', []))
            parts['decision_novelty'] = float(bool(decisions - self.decisions))
            parts['kernel_novelty'] = float(bool(kernels - self.kernels))
            parts['target_confirmation'] = float(bool(out.get('target_confirmed')))
            condition_cell = (c.target, c.dimension, c.value)
            parts['condition_coverage'] = float(condition_cell not in self.condition_cells)
            digest = out.get('anomaly_signature')
            reproduced = digest and len(repeats) >= self.config.reproducibility_repeats and all(
                r.get('valid') is True and not r.get('timeout', False) and
                r.get('oracle_applicability') == 'applicable' and r.get('legal') is True and
                not r.get('rejection_reasons') and r.get('anomaly_signature') == digest for r in repeats)
            parts['semantic_anomaly'] = float(bool(reproduced))
            for stage in out.get('compiler_stages', []) or ['unknown']:
                for family in out.get('kernel_families', []) or ['unobserved']:
                    cell = CoverageCell(c.target, c.dimension, c.value, stage, family,
                                        out.get('guard_recompile_signature', 'unobserved'), 'applicable')
                    self.cells.add(cell.key)
                    accepted_cells.append(cell.key)
            self.condition_cells.add(condition_cell)
            self.decisions.update(decisions)
            self.kernels.update(kernels)
        weighted = {k: v * self.config.weights[k] for k, v in parts.items()}
        reward = sum(weighted.values())
        stat = self.stats[c.key]
        stat.reviewed += 1
        stat.reward_sum += reward
        for k in COMPONENTS:
            stat.components[k] += parts[k]
        review = {'trial_id': trial_id, 'eligible': eligible, 'disposition': disposition,
                  'components': parts, 'weighted_components': weighted, 'reward': reward,
                  'coverage_cells': accepted_cells, 'observed_cells': observed_cells,
                  'outcome': out, 'reproducibility_repeats': list(repeats)}
        self.reviews[trial_id] = review
        del self.pending[trial_id]
        return review

    def to_dict(self):
        payload = {'schema': SCHEMA, 'config': asdict(self.config), 'candidates': [asdict(c) for c in self.candidates],
                   'stats': {k: asdict(s) for k, s in self.stats.items()}, 'reservations': self.reservations,
                   'pending': self.pending, 'reviews': self.reviews, 'decisions': sorted(self.decisions),
                   'kernels': sorted(self.kernels), 'cells': sorted(self.cells),
                   'observed_cells': sorted(self.observed_cells),
                   'condition_cells': sorted(self.condition_cells), 'selection_audit': self.selection_audit}
        payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, data):
        payload = dict(data)
        digest = payload.pop('sha256')
        if payload.get('schema') != SCHEMA or hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest() != digest:
            raise ValueError('checkpoint_schema_or_digest_mismatch')
        obj = cls([Candidate(**dict(c, family=tuple(c['family']))) for c in payload['candidates']], SchedulerConfig(**payload['config']))
        obj.stats = {k: Stats(**v) for k, v in payload['stats'].items()}
        for name in ('reservations', 'pending', 'reviews', 'selection_audit'):
            setattr(obj, name, payload[name])
        for name in ('decisions', 'kernels', 'cells', 'observed_cells'):
            setattr(obj, name, set(payload[name]))
        obj.condition_cells = {tuple(x) for x in payload['condition_cells']}
        if set(obj.stats) != {c.key for c in obj.candidates} or sum(s.attempts for s in obj.stats.values()) != obj.total:
            raise ValueError('checkpoint_accounting_mismatch')
        return obj

    def checkpoint(self, path):
        path = Path(path)
        EventStore(str(path.parent)).checkpoint(self.to_dict(), path.name)

    @classmethod
    def restore(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
