"""Verified replay records for discrepancy learning and self-improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class VerifiedTransition:
    """Pair cheap predictions with later high-fidelity observations."""

    state_id: str
    action: Any
    cheap_model_id: str
    cheap_prediction: float
    accurate_model_id: str
    verified_outcome: float
    router_propensity: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def discrepancy(self) -> float:
        return self.verified_outcome - self.cheap_prediction


class VerifiedReplayStore:
    """Small in-memory reference store; production backends may be distributed."""

    def __init__(self) -> None:
        self._records: list[VerifiedTransition] = []

    def append(self, record: VerifiedTransition) -> None:
        self._records.append(record)

    def extend(self, records: Iterable[VerifiedTransition]) -> None:
        self._records.extend(records)

    def snapshot(self) -> tuple[VerifiedTransition, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)
