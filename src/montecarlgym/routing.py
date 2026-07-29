"""Meta-level policies for allocating search compute."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .types import Action, ComputeAction, ModelObservation, SearchBudget


@dataclass(frozen=True, slots=True)
class BranchSummary:
    """Router-facing aggregate for one candidate task action."""

    mean: float
    variance: float
    risk: float
    evidence_count: int
    verified: bool = False


@dataclass(frozen=True, slots=True)
class RouterContext:
    """Information exposed to a compute router at one search node."""

    state_id: str
    candidate_actions: tuple[Action, ...]
    evidence: Mapping[Action, tuple[ModelObservation, ...]]
    remaining_budget: SearchBudget
    search_depth: int
    summaries: Mapping[Action, BranchSummary] = field(default_factory=dict)
    query_counts: Mapping[tuple[Action, str], int] = field(
        default_factory=dict
    )
    verified_actions: frozenset[Action] = frozenset()
    feasible_model_ids: frozenset[str] = frozenset()

    @property
    def total_queries(self) -> int:
        return sum(self.query_counts.values())


class ComputeRouter(Protocol):
    """Choose the next simulation/model query, or stop gathering evidence."""

    def choose(self, context: RouterContext) -> ComputeAction | None:
        """Return ``None`` when current evidence is sufficient to act."""


def _count(context: RouterContext, action: Action, model_id: str) -> int:
    return context.query_counts.get((action, model_id), 0)


def _model_is_feasible(context: RouterContext, model_id: str) -> bool:
    return not context.feasible_model_ids or model_id in context.feasible_model_ids


@dataclass(frozen=True, slots=True)
class CheapOnlyRouter:
    """Allocate a fixed number of cheap queries to every task branch."""

    model_id: str
    samples_per_action: int = 1
    token_budget: int = 0
    rollout_depth: int = 1

    def choose(self, context: RouterContext) -> ComputeAction | None:
        if self.samples_per_action < 1:
            raise ValueError("samples_per_action must be positive")
        if not _model_is_feasible(context, self.model_id):
            return None
        for action in context.candidate_actions:
            if _count(context, action, self.model_id) < self.samples_per_action:
                return ComputeAction(
                    state_id=context.state_id,
                    task_action=action,
                    model_id=self.model_id,
                    token_budget=self.token_budget,
                    rollout_depth=self.rollout_depth,
                    route_propensity=1.0,
                )
        return None


@dataclass(frozen=True, slots=True)
class AccurateOnlyRouter:
    """Allocate only high-fidelity, verification-requesting queries."""

    model_id: str
    samples_per_action: int = 1
    token_budget: int = 0
    rollout_depth: int = 1

    def choose(self, context: RouterContext) -> ComputeAction | None:
        if self.samples_per_action < 1:
            raise ValueError("samples_per_action must be positive")
        if context.remaining_budget.max_accurate_calls <= 0:
            return None
        if not _model_is_feasible(context, self.model_id):
            return None
        for action in context.candidate_actions:
            if _count(context, action, self.model_id) < self.samples_per_action:
                return ComputeAction(
                    state_id=context.state_id,
                    task_action=action,
                    model_id=self.model_id,
                    token_budget=self.token_budget,
                    rollout_depth=self.rollout_depth,
                    request_verification=True,
                    route_propensity=1.0,
                )
        return None


@dataclass(frozen=True, slots=True)
class FixedCascadeRouter:
    """Query every branch cheaply, then verify the top-k candidates."""

    cheap_model_id: str
    accurate_model_id: str
    top_k: int = 2
    cheap_samples_per_action: int = 1
    cheap_token_budget: int = 0
    accurate_token_budget: int = 0
    cheap_rollout_depth: int = 1
    accurate_rollout_depth: int = 1

    def choose(self, context: RouterContext) -> ComputeAction | None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.cheap_samples_per_action < 1:
            raise ValueError("cheap_samples_per_action must be positive")
        if _model_is_feasible(context, self.cheap_model_id):
            for action in context.candidate_actions:
                if (
                    _count(context, action, self.cheap_model_id)
                    < self.cheap_samples_per_action
                ):
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
        if not _model_is_feasible(context, self.accurate_model_id):
            return None

        def cheap_mean(action: Action) -> float:
            observations = context.evidence.get(action, ())
            cheap_observations = observations[: self.cheap_samples_per_action]
            if not cheap_observations:
                return -math.inf
            return sum(item.value for item in cheap_observations) / len(
                cheap_observations
            )

        ranked = sorted(
            context.candidate_actions,
            key=lambda action: (cheap_mean(action), repr(action)),
            reverse=True,
        )
        for action in ranked[: self.top_k]:
            if action not in context.verified_actions:
                return ComputeAction(
                    state_id=context.state_id,
                    task_action=action,
                    model_id=self.accurate_model_id,
                    token_budget=self.accurate_token_budget,
                    rollout_depth=self.accurate_rollout_depth,
                    request_verification=True,
                    route_propensity=1.0,
                )
        return None


@dataclass(frozen=True, slots=True)
class ThresholdRouter:
    """Reference router that escalates statistically ambiguous branches.

    Production implementations should learn expected value of computation and
    account for discrepancy, latency, risk, and downstream branching.
    """

    cheap_model_id: str
    accurate_model_id: str
    z_score: float = 1.96
    accurate_token_budget: int = 0
    cheap_token_budget: int = 0
    cheap_rollout_depth: int = 1
    accurate_rollout_depth: int = 1

    def choose(self, context: RouterContext) -> ComputeAction | None:
        summaries: list[tuple[Action, float, float]] = []
        for action in context.candidate_actions:
            observations = context.evidence.get(action, ())
            if not observations:
                if not _model_is_feasible(context, self.cheap_model_id):
                    return None
                return ComputeAction(
                    state_id=context.state_id,
                    task_action=action,
                    model_id=self.cheap_model_id,
                    token_budget=self.cheap_token_budget,
                    rollout_depth=self.cheap_rollout_depth,
                    route_propensity=1.0,
                )
            summary = context.summaries.get(action)
            latest = observations[-1]
            summaries.append(
                (
                    action,
                    summary.mean if summary is not None else latest.value,
                    summary.variance
                    if summary is not None
                    else latest.variance,
                )
            )

        if context.remaining_budget.max_accurate_calls <= 0:
            return None
        if not _model_is_feasible(context, self.accurate_model_id):
            return None

        best_action, best_mean, best_var = max(summaries, key=lambda item: item[1])
        best_lower = best_mean - self.z_score * best_var**0.5
        ambiguous = [
            (action, mean + self.z_score * variance**0.5)
            for action, mean, variance in summaries
            if mean + self.z_score * variance**0.5 >= best_lower
        ]
        candidates = [
            item
            for item in ambiguous
            if item[0] not in context.verified_actions
        ]
        if not candidates:
            return None
        action = max(candidates, key=lambda item: item[1])[0]
        return ComputeAction(
            state_id=context.state_id,
            task_action=action,
            model_id=self.accurate_model_id,
            token_budget=self.accurate_token_budget,
            rollout_depth=self.accurate_rollout_depth,
            request_verification=True,
            route_propensity=1.0,
        )
