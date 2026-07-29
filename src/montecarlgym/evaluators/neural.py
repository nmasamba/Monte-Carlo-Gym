"""Framework-neutral policy/value prediction for PUCT presets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol

from ..core.expansion import expand_legal_edges
from ..core.path import Evaluation
from ..core.tree import ActionEdge, StateNode
from ..policies.rollout_policies import RolloutContext
from ..types import Action


@dataclass(frozen=True, slots=True)
class PolicyValuePrediction:
    value: float
    priors: Mapping[Action, float]
    model_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PolicyValuePredictor(Protocol):
    def predict(
        self,
        state: Any,
        legal_actions: Sequence[Action],
    ) -> PolicyValuePrediction:
        """Return a scalar value and unnormalized non-negative priors."""


@dataclass(slots=True)
class PolicyValueEvaluator:
    """Combined expansion and no-rollout evaluation for PUCT search."""

    predictor: PolicyValuePredictor

    def expand(
        self,
        node: StateNode,
        legal_actions: Sequence[Action],
    ) -> tuple[ActionEdge, ...]:
        edges = expand_legal_edges(node, legal_actions)
        if node.value_estimate is not None and all(
            edge.prior is not None for edge in edges
        ):
            return edges

        prediction = self.predictor.predict(node.state, legal_actions)
        value = float(prediction.value)
        if not math.isfinite(value):
            raise ValueError("policy/value predictor returned a non-finite value")
        if set(prediction.priors) != set(legal_actions):
            raise ValueError("policy priors must exactly match the legal actions")
        raw_priors = {
            action: float(prediction.priors[action]) for action in legal_actions
        }
        if any(not math.isfinite(prior) or prior < 0 for prior in raw_priors.values()):
            raise ValueError("policy priors must be finite and non-negative")
        total = sum(raw_priors.values())
        if total <= 0:
            raise ValueError("at least one policy prior must be positive")
        if (
            node.policy_version is not None
            and prediction.model_version != node.policy_version
        ):
            raise ValueError("policy/value model version changed within a search tree")

        node.value_estimate = value
        node.policy_version = prediction.model_version
        node.statistics["policy_value_metadata"] = dict(prediction.metadata)
        for action, edge in node.edges.items():
            edge.prior = raw_priors[action] / total
        return edges

    def evaluate(
        self,
        frontier: StateNode,
        model: RolloutContext,
        rng: Random,
    ) -> Evaluation:
        del rng
        if frontier.terminal:
            return Evaluation(
                0.0,
                terminated=frontier.terminated,
                truncated=frontier.truncated,
                stop_reason="terminal" if frontier.terminated else "truncated",
            )
        self.expand(frontier, model.legal_actions(frontier.state))
        assert frontier.value_estimate is not None
        return Evaluation(frontier.value_estimate, stop_reason="policy_value")
