"""Direct value evaluation without environment rollout."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import Any

from ..core.path import Evaluation
from ..core.tree import StateNode
from ..policies.rollout_policies import RolloutContext

ValueFunction = Callable[[Any], float]


@dataclass(frozen=True, slots=True)
class DirectValueEvaluator:
    """Evaluate a frontier with an injected scalar value function."""

    value_function: ValueFunction

    def evaluate(
        self,
        frontier: StateNode,
        model: RolloutContext,
        rng: Random,
    ) -> Evaluation:
        del model, rng
        if frontier.terminal:
            return Evaluation(
                0.0,
                terminated=frontier.terminated,
                truncated=frontier.truncated,
                stop_reason="terminal" if frontier.terminated else "truncated",
            )
        value = float(self.value_function(frontier.state))
        if not math.isfinite(value):
            raise ValueError("direct value function returned a non-finite value")
        frontier.value_estimate = value
        return Evaluation(value, stop_reason="direct_value")


@dataclass(frozen=True, slots=True)
class NoRolloutEvaluator(DirectValueEvaluator):
    """Explicit AlphaZero-style name for direct value-only evaluation."""
