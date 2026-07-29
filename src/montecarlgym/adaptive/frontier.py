"""Budget-aware adaptive evaluation at classical MCTS frontiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from ..core.budget import BudgetExhausted
from ..core.expansion import expand_legal_edges
from ..core.path import Evaluation
from ..core.tree import StateNode
from ..models import ModelPortfolio
from ..policies.rollout_policies import RolloutContext
from ..types import SearchBudget
from .planner import AdaptiveComputePlanner, AdaptiveSearchReport


class BudgetedFrontierContext(RolloutContext, Protocol):
    """Classical rollout context extended with a nested hard-budget view."""

    def remaining_budget(self) -> SearchBudget: ...


def _minimum_optional(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _nested_budget(
    requested: SearchBudget,
    remaining: SearchBudget,
) -> SearchBudget:
    deadlines = [
        value
        for value in (requested.deadline_s, remaining.deadline_s)
        if value is not None
    ]
    return SearchBudget(
        max_cost=min(requested.max_cost, remaining.max_cost),
        max_tokens=min(requested.max_tokens, remaining.max_tokens),
        max_accurate_calls=min(
            requested.max_accurate_calls,
            remaining.max_accurate_calls,
        ),
        max_iterations=requested.max_iterations,
        deadline_s=min(deadlines) if deadlines else None,
        max_model_calls=_minimum_optional(
            requested.max_model_calls,
            remaining.max_model_calls,
        ),
        max_environment_calls=_minimum_optional(
            requested.max_environment_calls,
            remaining.max_environment_calls,
        ),
    )


@dataclass(slots=True)
class AdaptiveFrontierEvaluator:
    """Use a model portfolio to value and annotate each selected frontier.

    The inner planner derives its budget from the outer MCTS ledger. Its usage
    is returned on ``Evaluation`` and atomically absorbed before classical
    backup, so nested model calls cannot bypass the root search envelope.
    """

    planner: AdaptiveComputePlanner
    models: ModelPortfolio
    budget: SearchBudget
    reports: list[AdaptiveSearchReport] = field(default_factory=list)

    def evaluate(
        self,
        frontier: StateNode,
        model: RolloutContext,
        rng: Random,
    ) -> Evaluation:
        if frontier.terminal:
            return Evaluation(
                0.0,
                terminated=frontier.terminated,
                truncated=frontier.truncated,
                stop_reason="terminal" if frontier.terminated else "truncated",
            )
        remaining_method = getattr(model, "remaining_budget", None)
        if remaining_method is None:
            raise TypeError(
                "adaptive frontier evaluation requires a budget-aware "
                "classical rollout context"
            )
        remaining = remaining_method()
        actions = tuple(model.legal_actions(frontier.state))
        try:
            result = self.planner.plan_candidates(
                frontier.state,
                candidate_actions=actions,
                models=self.models,
                budget=_nested_budget(self.budget, remaining),
                seed=rng.getrandbits(64),
            )
        except BudgetExhausted:
            return Evaluation(0.0, stop_reason="adaptive_resource_budget")
        edges = expand_legal_edges(frontier, actions)
        estimates = {} if result.report is None else result.report.branch_estimates
        for edge in edges:
            estimate = estimates.get(edge.action)
            if estimate is not None:
                edge.evidence.append(estimate)
        frontier.value_estimate = result.predicted_value
        if result.report is not None:
            self.reports.append(result.report)
            frontier.statistics["last_adaptive_report"] = result.report
        return Evaluation(
            result.predicted_value,
            stop_reason="adaptive_frontier",
            rollout_actions=(result.action,),
            usage=result.usage,
        )
