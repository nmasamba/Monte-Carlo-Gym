"""Composable stopping policies for adaptive evidence gathering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..routing import RouterContext


class StopPolicy(Protocol):
    """Decide whether branch evidence is sufficient to select an action."""

    def should_stop(self, context: RouterContext) -> bool:
        """Return true without mutating planner or model state."""


@dataclass(frozen=True, slots=True)
class NeverStopPolicy:
    """Defer stopping to router exhaustion or hard resource budgets."""

    def should_stop(self, context: RouterContext) -> bool:
        del context
        return False


@dataclass(frozen=True, slots=True)
class FixedQueryStopPolicy:
    """Stop after an exact, deterministic number of compute actions."""

    max_queries: int

    def should_stop(self, context: RouterContext) -> bool:
        if self.max_queries < 1:
            raise ValueError("max_queries must be positive")
        return context.total_queries >= self.max_queries


@dataclass(frozen=True, slots=True)
class ConfidenceStopPolicy:
    """Stop when one branch's lower bound clears every competitor's upper."""

    z_score: float = 1.96
    min_evidence_per_action: int = 1
    require_verified_best: bool = False

    def should_stop(self, context: RouterContext) -> bool:
        if self.z_score < 0:
            raise ValueError("z_score must be non-negative")
        if self.min_evidence_per_action < 1:
            raise ValueError("min_evidence_per_action must be positive")
        if any(
            context.summaries.get(action) is None
            or context.summaries[action].evidence_count
            < self.min_evidence_per_action
            for action in context.candidate_actions
        ):
            return False
        summaries = [context.summaries[action] for action in context.candidate_actions]
        best = max(summaries, key=lambda summary: summary.mean)
        if self.require_verified_best and not best.verified:
            return False
        best_lower = best.mean - self.z_score * best.variance**0.5
        competitors = [summary for summary in summaries if summary is not best]
        return all(
            best_lower
            > summary.mean + self.z_score * summary.variance**0.5
            for summary in competitors
        )
