"""Iteration-local search paths and evaluation values."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Action, ResourceUsage
from .tree import ActionEdge, OutcomeLink, StateNode


@dataclass(frozen=True, slots=True)
class PathStep:
    """One sampled state-action-outcome transition in an iteration."""

    node: StateNode
    edge: ActionEdge
    outcome: OutcomeLink

    @property
    def reward(self) -> float:
        return self.outcome.reward


@dataclass(slots=True)
class SearchPath:
    """The exact edges and stochastic outcomes traversed in one iteration."""

    root: StateNode
    steps: list[PathStep] = field(default_factory=list)

    @property
    def leaf(self) -> StateNode:
        if not self.steps:
            return self.root
        return self.steps[-1].outcome.child

    def append(
        self,
        node: StateNode,
        edge: ActionEdge,
        outcome: OutcomeLink,
    ) -> None:
        self.steps.append(PathStep(node, edge, outcome))


@dataclass(frozen=True, slots=True)
class Evaluation:
    """A rollout or evaluator result from the current search frontier."""

    value: float
    depth: int = 0
    terminated: bool = False
    truncated: bool = False
    stop_reason: str = "evaluated"
    rollout_actions: tuple[Action, ...] = ()
    usage: ResourceUsage = ResourceUsage()
