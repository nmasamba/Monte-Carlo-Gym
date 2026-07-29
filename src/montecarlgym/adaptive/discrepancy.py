"""Online discrepancy estimates learned only from verified model pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DiscrepancyEstimate:
    """Mean additive correction and uncertainty for a model pair."""

    mean: float = 0.0
    variance: float = 0.0
    count: int = 0


class DiscrepancyModel(Protocol):
    """Correct cheap evidence using paired verified observations."""

    def estimate(
        self,
        cheap_model_id: str,
        accurate_model_id: str,
    ) -> DiscrepancyEstimate:
        """Return the current additive accurate-minus-cheap estimate."""

    def update(
        self,
        cheap_model_id: str,
        accurate_model_id: str,
        *,
        cheap_value: float,
        verified_value: float,
    ) -> None:
        """Update from one verified cheap/accurate pair."""


@dataclass(slots=True)
class _RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def estimate(self) -> DiscrepancyEstimate:
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return DiscrepancyEstimate(self.mean, variance, self.count)


class RunningDiscrepancyModel:
    """Dependency-free Welford estimator keyed by ordered model pair."""

    def __init__(self) -> None:
        self._moments: dict[tuple[str, str], _RunningMoments] = {}

    def estimate(
        self,
        cheap_model_id: str,
        accurate_model_id: str,
    ) -> DiscrepancyEstimate:
        moments = self._moments.get((cheap_model_id, accurate_model_id))
        return moments.estimate() if moments is not None else DiscrepancyEstimate()

    def update(
        self,
        cheap_model_id: str,
        accurate_model_id: str,
        *,
        cheap_value: float,
        verified_value: float,
    ) -> None:
        key = (cheap_model_id, accurate_model_id)
        self._moments.setdefault(key, _RunningMoments()).update(
            verified_value - cheap_value
        )
