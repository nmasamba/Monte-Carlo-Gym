"""Convex combinations of interchangeable evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from ..core.path import Evaluation
from ..core.tree import StateNode
from ..policies.rollout_policies import RolloutContext
from .base import Evaluator


@dataclass(frozen=True, slots=True)
class MixedEvaluator:
    """Blend two evaluator values while preserving rollout provenance."""

    first: Evaluator
    second: Evaluator
    first_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.first_weight <= 1.0:
            raise ValueError("first_weight must be between zero and one")

    def evaluate(
        self,
        frontier: StateNode,
        model: RolloutContext,
        rng: Random,
    ) -> Evaluation:
        first = self.first.evaluate(frontier, model, rng)
        second = self.second.evaluate(frontier, model, rng)
        weight = self.first_weight
        return Evaluation(
            value=weight * first.value + (1.0 - weight) * second.value,
            depth=max(first.depth, second.depth),
            terminated=first.terminated or second.terminated,
            truncated=first.truncated or second.truncated,
            stop_reason="mixed",
            rollout_actions=first.rollout_actions + second.rollout_actions,
        )
