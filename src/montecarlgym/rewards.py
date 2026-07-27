"""Reward and preference interfaces kept separate from search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class RewardSignal:
    value: float
    verified: bool
    source: str
    components: Mapping[str, float]


class RewardSource(Protocol):
    """Score an outcome using rules, humans, preferences, or learned models."""

    def score(
        self,
        *,
        initial_state: Any,
        action: Any,
        outcome: Any,
    ) -> RewardSignal:
        """Return a scalar objective and provenance."""


@dataclass(frozen=True, slots=True)
class WeightedReward:
    """Compose explicit objective components without hiding their provenance."""

    weights: Mapping[str, float]

    def combine(
        self,
        components: Mapping[str, float],
        *,
        verified: bool,
        source: str,
    ) -> RewardSignal:
        missing = set(self.weights) - set(components)
        if missing:
            raise ValueError(f"missing reward components: {sorted(missing)}")
        value = sum(self.weights[name] * components[name] for name in self.weights)
        return RewardSignal(value, verified, source, dict(components))
