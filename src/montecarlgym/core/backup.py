"""Backup operators for search paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .path import Evaluation, PathStep, SearchPath
from .tree import StateNode


class ValuePerspective(Protocol):
    """Convert a return to the perspective represented by an action edge."""

    def for_edge(
        self,
        value: float,
        *,
        step: PathStep,
        depth_from_leaf: int,
    ) -> float:
        """Return the value stored on ``step.edge``."""


@dataclass(frozen=True, slots=True)
class IdentityValuePerspective:
    """Single-agent/default perspective: never flip a return's sign."""

    def for_edge(
        self,
        value: float,
        *,
        step: PathStep,
        depth_from_leaf: int,
    ) -> float:
        del step, depth_from_leaf
        return value


@dataclass(frozen=True, slots=True)
class AlternatingValuePerspective:
    """Two-player zero-sum perspective with explicit alternating sign."""

    def for_edge(
        self,
        value: float,
        *,
        step: PathStep,
        depth_from_leaf: int,
    ) -> float:
        del step
        return -value if depth_from_leaf % 2 else value


class BackupOperator(Protocol):
    def update(self, path: SearchPath, evaluation: Evaluation) -> None:
        """Apply one completed evaluation to an explicit path."""


@dataclass(frozen=True, slots=True)
class MeanBackup:
    """Maintain edge visit count, total return, and arithmetic mean."""

    discount: float = 1.0
    perspective: ValuePerspective = IdentityValuePerspective()

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")

    def update(self, path: SearchPath, evaluation: Evaluation) -> None:
        value = evaluation.value
        path.leaf.visits += 1
        for depth, step in enumerate(reversed(path.steps)):
            value = step.reward + self.discount * value
            edge_value = self.perspective.for_edge(
                value,
                step=step,
                depth_from_leaf=depth,
            )
            step.edge.update(edge_value)
            step.outcome.visits += 1
            step.node.visits += 1


@dataclass(frozen=True, slots=True)
class RobustBackup:
    """Propagate the value of each state's most-visited action edge."""

    discount: float = 1.0
    perspective: ValuePerspective = IdentityValuePerspective()

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")

    def update(self, path: SearchPath, evaluation: Evaluation) -> None:
        value = evaluation.value
        path.leaf.visits += 1
        for depth, step in enumerate(reversed(path.steps)):
            value = step.reward + self.discount * value
            edge_value = self.perspective.for_edge(
                value,
                step=step,
                depth_from_leaf=depth,
            )
            step.edge.update(edge_value)
            step.outcome.visits += 1
            step.node.visits += 1
            robust_edge = max(
                step.node.edges.values(),
                key=lambda edge: (edge.visits, repr(edge.action)),
            )
            value = robust_edge.mean_value


class MixWeightSchedule(Protocol):
    def weight(self, node: StateNode) -> float:
        """Return the robust component's weight for ``node``."""


@dataclass(frozen=True, slots=True)
class ConstantMixWeight:
    robust_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.robust_weight <= 1.0:
            raise ValueError("robust_weight must be between zero and one")

    def weight(self, node: StateNode) -> float:
        del node
        return self.robust_weight


@dataclass(frozen=True, slots=True)
class VisitMixWeight:
    """Increase robust influence as a node accumulates evidence."""

    half_saturation_visits: float = 32.0
    maximum_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.half_saturation_visits <= 0:
            raise ValueError("half_saturation_visits must be positive")
        if not 0.0 <= self.maximum_weight <= 1.0:
            raise ValueError("maximum_weight must be between zero and one")

    def weight(self, node: StateNode) -> float:
        return self.maximum_weight * node.visits / (
            node.visits + self.half_saturation_visits
        )


@dataclass(frozen=True, slots=True)
class MixBackup:
    """Propagate a convex mix of visit-weighted mean and robust value."""

    discount: float = 1.0
    schedule: MixWeightSchedule = ConstantMixWeight()
    perspective: ValuePerspective = IdentityValuePerspective()

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")

    def update(self, path: SearchPath, evaluation: Evaluation) -> None:
        value = evaluation.value
        path.leaf.visits += 1
        for depth, step in enumerate(reversed(path.steps)):
            value = step.reward + self.discount * value
            edge_value = self.perspective.for_edge(
                value,
                step=step,
                depth_from_leaf=depth,
            )
            step.edge.update(edge_value)
            step.outcome.visits += 1
            step.node.visits += 1

            visited = [edge for edge in step.node.edges.values() if edge.visits]
            total_visits = sum(edge.visits for edge in visited)
            mean_value = sum(edge.total_return for edge in visited) / total_visits
            robust_edge = max(
                visited,
                key=lambda edge: (edge.visits, repr(edge.action)),
            )
            robust_weight = self.schedule.weight(step.node)
            if not 0.0 <= robust_weight <= 1.0:
                raise ValueError("mix schedule returned a weight outside [0, 1]")
            value = (
                (1.0 - robust_weight) * mean_value
                + robust_weight * robust_edge.mean_value
            )
