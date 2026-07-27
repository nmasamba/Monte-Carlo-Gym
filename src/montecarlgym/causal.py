"""Optional off-policy estimators for learning from routing logs.

Causal estimation is an evaluation and data-quality layer in v1, not a required
assumption of the world model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class LoggedRoutingDecision:
    context_id: str
    chosen_route: str
    reward: float
    propensity: float
    baseline_prediction: float = 0.0


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
        weighted = [
            record.reward / max(record.propensity, self.min_propensity)
            for record in records
        ]
        return sum(weighted) / len(weighted)
