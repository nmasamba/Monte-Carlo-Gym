"""Evaluator contracts shared by direct, mixed, and policy/value evaluation."""

from __future__ import annotations

from random import Random
from typing import Protocol

from ..core.path import Evaluation
from ..core.tree import StateNode
from ..policies.rollout_policies import RolloutContext


class Evaluator(Protocol):
    def evaluate(
        self,
        frontier: StateNode,
        model: RolloutContext,
        rng: Random,
    ) -> Evaluation:
        """Estimate return from ``frontier`` without real side effects."""
