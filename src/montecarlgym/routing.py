"""Meta-level policies for allocating search compute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .types import Action, ComputeAction, ModelObservation, SearchBudget


@dataclass(frozen=True, slots=True)
class RouterContext:
    """Information exposed to a compute router at one search node."""

    state_id: str
    candidate_actions: tuple[Action, ...]
    evidence: Mapping[Action, tuple[ModelObservation, ...]]
    remaining_budget: SearchBudget
    search_depth: int


class ComputeRouter(Protocol):
    """Choose the next simulation/model query, or stop gathering evidence."""

    def choose(self, context: RouterContext) -> ComputeAction | None:
        """Return ``None`` when current evidence is sufficient to act."""


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

    def choose(self, context: RouterContext) -> ComputeAction | None:
        if context.remaining_budget.max_accurate_calls <= 0:
            return None

        summaries: list[tuple[Action, float, float]] = []
        for action in context.candidate_actions:
            observations = context.evidence.get(action, ())
            if not observations:
                return ComputeAction(
                    state_id=context.state_id,
                    task_action=action,
                    model_id=self.cheap_model_id,
                )
            latest = observations[-1]
            summaries.append((action, latest.value, latest.variance))

        best_action, best_mean, best_var = max(summaries, key=lambda item: item[1])
        best_lower = best_mean - self.z_score * best_var**0.5
        ambiguous = [
            (action, mean + self.z_score * variance**0.5)
            for action, mean, variance in summaries
            if mean + self.z_score * variance**0.5 >= best_lower
        ]
        already_verified = {
            action
            for action, observations in context.evidence.items()
            if len(observations) > 1
        }
        candidates = [
            item for item in ambiguous if item[0] not in already_verified
        ]
        if not candidates:
            return None
        action = max(candidates, key=lambda item: item[1])[0]
        return ComputeAction(
            state_id=context.state_id,
            task_action=action,
            model_id=self.accurate_model_id,
            token_budget=self.accurate_token_budget,
            request_verification=True,
        )
