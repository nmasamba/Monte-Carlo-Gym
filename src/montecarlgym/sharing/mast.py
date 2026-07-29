"""Move-Average Sampling Technique rollout sharing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random

from ..core.backup import BackupOperator, MeanBackup
from ..core.path import Evaluation, SearchPath
from ..types import Action


@dataclass(slots=True)
class MoveStatistics:
    visits: int = 0
    total_return: float = 0.0
    mean_value: float = 0.0

    def update(self, value: float) -> None:
        self.visits += 1
        self.total_return += value
        self.mean_value = self.total_return / self.visits


@dataclass(slots=True)
class MoveStatisticsTable:
    _statistics: dict[Action, MoveStatistics] = field(default_factory=dict)

    def get(self, action: Action) -> MoveStatistics:
        return self._statistics.setdefault(action, MoveStatistics())

    def update(self, action: Action, value: float) -> None:
        self.get(action).update(value)

    def snapshot(self) -> dict[Action, MoveStatistics]:
        return {
            action: MoveStatistics(stats.visits, stats.total_return, stats.mean_value)
            for action, stats in self._statistics.items()
        }


@dataclass(frozen=True, slots=True)
class MASTRolloutPolicy:
    """Softmax rollout selection from global move-return statistics."""

    table: MoveStatisticsTable
    temperature: float = 1.0
    epsilon: float = 0.05

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")

    def choose_action(
        self,
        actions: Sequence[Action],
        rng: Random,
    ) -> Action:
        if not actions:
            raise RuntimeError("MAST rollout state has no legal actions")
        if rng.random() < self.epsilon:
            return actions[rng.randrange(len(actions))]
        values = [self.table.get(action).mean_value for action in actions]
        maximum = max(values)
        weights = [
            math.exp((value - maximum) / self.temperature) for value in values
        ]
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        for action, weight in zip(actions, weights, strict=True):
            cumulative += weight
            if cumulative >= threshold:
                return action
        return actions[-1]


@dataclass(frozen=True, slots=True)
class MASTBackup:
    """Update a base backup and global statistics for simulated moves."""

    table: MoveStatisticsTable
    base: BackupOperator = MeanBackup()
    discount: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")

    def update(self, path: SearchPath, evaluation: Evaluation) -> None:
        value = evaluation.value
        for step in reversed(path.steps):
            value = step.reward + self.discount * value
        self.base.update(path, evaluation)
        for action in (
            tuple(step.edge.action for step in path.steps)
            + tuple(evaluation.rollout_actions)
        ):
            self.table.update(action, value)
