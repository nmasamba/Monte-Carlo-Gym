"""Optional off-policy estimators for learning from routing logs.

Causal estimation is an evaluation and data-quality layer in v1, not a required
assumption of the world model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LoggedRoutingDecision:
    context_id: str
    chosen_route: str
    reward: float
    propensity: float
    baseline_prediction: float = 0.0
    target_probability: float = 1.0
    randomized_audit: bool = False
    feasible_routes: tuple[str, ...] = ()
    context_features: Mapping[str, float] = field(default_factory=dict)

    def importance_weight(self, min_propensity: float) -> float:
        if not 0.0 < self.propensity <= 1.0:
            raise ValueError("logged propensity must be in (0, 1]")
        if not 0.0 <= self.target_probability <= 1.0:
            raise ValueError("target_probability must be in [0, 1]")
        return self.target_probability / max(self.propensity, min_propensity)


class OffPolicyEstimator(Protocol):
    def estimate(self, records: Sequence[LoggedRoutingDecision]) -> float:
        """Estimate policy value from logged routing decisions."""


@dataclass(frozen=True, slots=True)
class InversePropensityEstimator:
    """Minimal IPS diagnostic; use clipping and doubly robust models in studies."""

    min_propensity: float = 0.05

    def estimate(self, records: Sequence[LoggedRoutingDecision]) -> float:
        if not records:
            raise ValueError("at least one record is required")
        _validate_clipping(self.min_propensity)
        weighted = [
            record.importance_weight(self.min_propensity) * record.reward
            for record in records
        ]
        return sum(weighted) / len(weighted)


@dataclass(frozen=True, slots=True)
class SelfNormalizedIPSEstimator:
    """Self-normalized IPS diagnostic with explicit overlap clipping."""

    min_propensity: float = 0.05

    def estimate(self, records: Sequence[LoggedRoutingDecision]) -> float:
        if not records:
            raise ValueError("at least one record is required")
        _validate_clipping(self.min_propensity)
        weights = [
            record.importance_weight(self.min_propensity) for record in records
        ]
        denominator = sum(weights)
        if denominator <= 0:
            raise ValueError("target policy has no overlap with logged routes")
        return sum(
            weight * record.reward
            for weight, record in zip(weights, records, strict=True)
        ) / denominator


@dataclass(frozen=True, slots=True)
class DoublyRobustEstimator:
    """Doubly robust value estimate using logged outcome-model predictions."""

    min_propensity: float = 0.05

    def estimate(self, records: Sequence[LoggedRoutingDecision]) -> float:
        if not records:
            raise ValueError("at least one record is required")
        _validate_clipping(self.min_propensity)
        estimates = [
            record.baseline_prediction
            + record.importance_weight(self.min_propensity)
            * (record.reward - record.baseline_prediction)
            for record in records
        ]
        if not all(math.isfinite(value) for value in estimates):
            raise ValueError("off-policy inputs must be finite")
        return sum(estimates) / len(estimates)


def _validate_clipping(min_propensity: float) -> None:
    if not 0.0 < min_propensity <= 1.0:
        raise ValueError("min_propensity must be in (0, 1]")
