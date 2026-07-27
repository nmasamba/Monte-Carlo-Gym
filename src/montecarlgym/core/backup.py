"""Backup operators for search paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .path import Evaluation, PathStep, SearchPath


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
