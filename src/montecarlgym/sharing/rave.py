"""Rapid Action Value Estimation statistics and selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Protocol

from ..core.backup import (
    BackupOperator,
    IdentityValuePerspective,
    MeanBackup,
    ValuePerspective,
)
from ..core.path import Evaluation, SearchPath
from ..core.tree import ActionEdge, StateNode


class RAVEBetaSchedule(Protocol):
    def beta(self, edge: ActionEdge) -> float:
        """Return the AMAF blend weight for one edge."""


@dataclass(frozen=True, slots=True)
class VisitRAVEBeta:
    """Decay AMAF influence as direct edge evidence grows."""

    equivalence: float = 300.0

    def __post_init__(self) -> None:
        if self.equivalence <= 0:
            raise ValueError("equivalence must be positive")

    def beta(self, edge: ActionEdge) -> float:
        if edge.amaf_visits == 0:
            return 0.0
        denominator = (
            edge.visits
            + edge.amaf_visits
            + 4.0 * edge.visits * edge.amaf_visits / self.equivalence
        )
        return edge.amaf_visits / denominator


@dataclass(frozen=True, slots=True)
class RAVETreePolicy:
    """Blend direct and AMAF means, then add ordinary UCT exploration."""

    exploration_constant: float = math.sqrt(2.0)
    beta_schedule: RAVEBetaSchedule = VisitRAVEBeta()

    def __post_init__(self) -> None:
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")

    def select(self, node: StateNode, rng: Random) -> ActionEdge:
        if not node.edges:
            raise RuntimeError("RAVE requires an expanded node")
        unvisited = [edge for edge in node.edges.values() if edge.visits == 0]
        if unvisited:
            return unvisited[rng.randrange(len(unvisited))]
        log_parent = math.log(max(1, node.visits))

        def score(edge: ActionEdge) -> float:
            beta = self.beta_schedule.beta(edge)
            blended = (
                (1.0 - beta) * edge.mean_value
                + beta * edge.amaf_mean_value
            )
            return blended + self.exploration_constant * math.sqrt(
                log_parent / edge.visits
            )

        scores = [(edge, score(edge)) for edge in node.edges.values()]
        best_score = max(value for _, value in scores)
        best = [edge for edge, value in scores if value == best_score]
        return best[rng.randrange(len(best))]


@dataclass(frozen=True, slots=True)
class RAVEBackup:
    """Apply direct backup plus AMAF updates for later equivalent moves."""

    base: BackupOperator = MeanBackup()
    discount: float = 1.0
    perspective: ValuePerspective = IdentityValuePerspective()

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")

    def update(self, path: SearchPath, evaluation: Evaluation) -> None:
        edge_returns = [0.0] * len(path.steps)
        value = evaluation.value
        for reverse_depth, index in enumerate(
            range(len(path.steps) - 1, -1, -1)
        ):
            step = path.steps[index]
            value = step.reward + self.discount * value
            edge_returns[index] = self.perspective.for_edge(
                value,
                step=step,
                depth_from_leaf=reverse_depth,
            )

        self.base.update(path, evaluation)
        rollout_moves = tuple(evaluation.rollout_actions)
        for index, step in enumerate(path.steps):
            later_moves = (
                tuple(item.edge.action for item in path.steps[index + 1 :])
                + rollout_moves
            )
            direct_action = step.edge.action
            for action in dict.fromkeys(later_moves):
                if action == direct_action:
                    continue
                edge = step.node.edges.get(action)
                if edge is not None:
                    edge.update_amaf(edge_returns[index])
