"""Learned expected-value-of-compute routing with randomized audits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from random import Random

from ..routing import RouterContext
from ..types import Action, ComputeAction
from .learning import LinearEVCModel


def router_features(context: RouterContext, action: Action) -> dict[str, float]:
    """Create the stable contextual feature schema used by Phase 4 models."""

    summary = context.summaries.get(action)
    means = [item.mean for item in context.summaries.values()]
    best_mean = max(means) if means else 0.0
    action_mean = 0.0 if summary is None else summary.mean
    return {
        "action_mean": action_mean,
        "action_uncertainty": 0.0 if summary is None else summary.variance**0.5,
        "action_risk": 0.0 if summary is None else summary.risk,
        "gap_to_best": max(0.0, best_mean - action_mean),
        "evidence_count": 0.0 if summary is None else float(summary.evidence_count),
        "search_depth": float(context.search_depth),
        "remaining_cost": context.remaining_budget.max_cost,
        "remaining_accurate_calls": float(
            context.remaining_budget.max_accurate_calls
        ),
    }


@dataclass(slots=True)
class LearnedEVCRouter:
    """Escalate when learned verification utility exceeds its resource price.

    ``audit_probability`` provides bounded epsilon-greedy overlap. Every chosen
    accurate route logs its exact behavior propensity and whether it came from
    the randomized audit component.
    """

    cheap_model_id: str
    accurate_model_id: str
    evc_model: LinearEVCModel
    accurate_cost: float
    cost_weight: float = 1.0
    minimum_net_evc: float = 0.0
    audit_probability: float = 0.05
    seed: int = 0
    cheap_token_budget: int = 0
    accurate_token_budget: int = 0
    cheap_rollout_depth: int = 1
    accurate_rollout_depth: int = 1
    cold_start_uncertainty_weight: float = 1.0
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.accurate_cost < 0 or self.cost_weight < 0:
            raise ValueError("EVC router costs and weights must be non-negative")
        if not 0.0 <= self.audit_probability <= 1.0:
            raise ValueError("audit_probability must be between zero and one")
        if self.cold_start_uncertainty_weight < 0:
            raise ValueError("cold_start_uncertainty_weight must be non-negative")
        self._rng = Random(self.seed)

    def choose(self, context: RouterContext) -> ComputeAction | None:
        if (
            context.feasible_model_ids
            and self.cheap_model_id not in context.feasible_model_ids
        ):
            return None
        for action in context.candidate_actions:
            if context.query_counts.get((action, self.cheap_model_id), 0) == 0:
                return ComputeAction(
                    state_id=context.state_id,
                    task_action=action,
                    model_id=self.cheap_model_id,
                    token_budget=self.cheap_token_budget,
                    rollout_depth=self.cheap_rollout_depth,
                    route_propensity=1.0,
                )

        if context.remaining_budget.max_accurate_calls <= 0:
            return None
        if (
            context.feasible_model_ids
            and self.accurate_model_id not in context.feasible_model_ids
        ):
            return None
        candidates = [
            action
            for action in context.candidate_actions
            if action not in context.verified_actions
        ]
        if not candidates:
            return None

        features: dict[Action, Mapping[str, float]] = {
            action: router_features(context, action) for action in candidates
        }
        gains: dict[Action, float] = {}
        for action in candidates:
            prediction = self.evc_model.predict(features[action])
            if prediction is None:
                gains[action] = (
                    features[action]["action_uncertainty"]
                    * self.cold_start_uncertainty_weight
                )
            else:
                gains[action] = prediction.mean
        net_values = {
            action: gain - self.cost_weight * self.accurate_cost
            for action, gain in gains.items()
        }
        best = max(candidates, key=lambda action: (net_values[action], repr(action)))
        exploit_enabled = net_values[best] > self.minimum_net_evc
        audit = self._rng.random() < self.audit_probability
        if audit:
            chosen = candidates[self._rng.randrange(len(candidates))]
        elif exploit_enabled:
            chosen = best
        else:
            return None

        uniform_probability = self.audit_probability / len(candidates)
        propensity = uniform_probability
        if exploit_enabled and chosen == best:
            propensity += 1.0 - self.audit_probability
        return ComputeAction(
            state_id=context.state_id,
            task_action=chosen,
            model_id=self.accurate_model_id,
            token_budget=self.accurate_token_budget,
            rollout_depth=self.accurate_rollout_depth,
            request_verification=True,
            route_propensity=propensity,
            audit=audit,
            expected_value_of_compute=gains[chosen],
        )


@dataclass(slots=True)
class RandomEscalationRouter:
    """Cheap-first randomized escalation baseline with exact propensities."""

    cheap_model_id: str
    accurate_model_id: str
    escalation_probability: float
    seed: int = 0
    cheap_rollout_depth: int = 1
    accurate_rollout_depth: int = 1
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.escalation_probability <= 1.0:
            raise ValueError(
                "escalation_probability must be between zero and one"
            )
        self._rng = Random(self.seed)

    def choose(self, context: RouterContext) -> ComputeAction | None:
        if (
            context.feasible_model_ids
            and self.cheap_model_id not in context.feasible_model_ids
        ):
            return None
        for action in context.candidate_actions:
            if context.query_counts.get((action, self.cheap_model_id), 0) == 0:
                return ComputeAction(
                    state_id=context.state_id,
                    task_action=action,
                    model_id=self.cheap_model_id,
                    rollout_depth=self.cheap_rollout_depth,
                    route_propensity=1.0,
                )
        if context.remaining_budget.max_accurate_calls <= 0 or (
            context.feasible_model_ids
            and self.accurate_model_id not in context.feasible_model_ids
        ):
            return None
        candidates = [
            action
            for action in context.candidate_actions
            if action not in context.verified_actions
        ]
        if not candidates or self._rng.random() >= self.escalation_probability:
            return None
        chosen = candidates[self._rng.randrange(len(candidates))]
        return ComputeAction(
            state_id=context.state_id,
            task_action=chosen,
            model_id=self.accurate_model_id,
            rollout_depth=self.accurate_rollout_depth,
            request_verification=True,
            route_propensity=self.escalation_probability / len(candidates),
            audit=True,
        )
