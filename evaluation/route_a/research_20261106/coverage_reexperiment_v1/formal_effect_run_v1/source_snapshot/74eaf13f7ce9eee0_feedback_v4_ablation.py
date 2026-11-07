"""Ablation scheduler for the post-review v4 experiment.

This module deliberately leaves feedback_v4.Scheduler unchanged. The ablation
keeps the v4 branch cycle, menu, quotas, graph/decision novelty, candidate
frontier, duplicate penalty, and cost penalty, while removing runtime behavior
novelty from both reward accumulation and adaptive behavior state.
"""

from collections import defaultdict

from .feedback_v4 import SCHEMA, Scheduler as FullScheduler


SCHEMA_NO_RUNTIME_NOVELTY = SCHEMA + ".no-runtime-novelty"


class SchedulerNoRuntimeNovelty(FullScheduler):
    """v4 scheduler with kernel/guard/lowering behavior feedback ablated."""

    def select(self, request_id):
        cell, selection = super().select(request_id)
        selection = dict(selection)
        selection["schema"] = SCHEMA_NO_RUNTIME_NOVELTY
        selection["ablation"] = "runtime_novelty_disabled"
        return cell, selection

    def observe(self, request_id, cell, selection, outcome, evidence=None):
        # Preserve the state before the full scheduler observes the row. The
        # full scheduler still computes and records behavior signatures, which
        # makes the ablation auditable, but those signatures must not influence
        # subsequent selection or reward.
        previous_behaviors = set(self.behaviors)
        previous_cell_behaviors = {
            key: set(values) for key, values in self.cell_behaviors.items()
        }

        row = super().observe(request_id, cell, selection, outcome, evidence)
        key = self._cell_keys[cell]
        removed_reward = float(row["components"]["runtime_novelty"])

        # Undo the runtime novelty contribution that super().observe added.
        self.rewards[key] -= removed_reward
        self.behaviors = previous_behaviors
        self.cell_behaviors = defaultdict(set, previous_cell_behaviors)

        row["schema"] = SCHEMA_NO_RUNTIME_NOVELTY
        row["novelty"]["behavior_new"] = False
        row["components"]["runtime_novelty"] = 0.0
        row["reward"] = sum(row["components"].values())
        self.rows[-1] = row
        return row

    def state(self):
        state = super().state()
        state["schema"] = SCHEMA_NO_RUNTIME_NOVELTY
        state["policy"]["runtime_novelty_enabled"] = False
        state["policy"]["behavior_state_updates"] = False
        return state
