"""Generic contracts for benchmarks, scoring, and episode records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..planner import PlanResult
from ..types import State


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Benchmark-owned quality and safety measurements."""

    success: bool
    return_value: float
    regret: float | None = None
    risk: float = 0.0
    constraint_violations: int = 0
    verifier_passed: bool | None = None
    extras: Mapping[str, float | int | bool | None] = field(
        default_factory=dict
    )


class Benchmark(Protocol):
    """Sample paired tasks and score a planner result."""

    @property
    def benchmark_id(self) -> str:
        """Stable benchmark and version identifier."""

    def sample(self, seed: int) -> State:
        """Return a task instance determined by ``seed``."""

    def task_id(self, task: State) -> str:
        """Return a stable task-instance identifier."""

    def score(self, task: State, result: PlanResult) -> EpisodeMetrics:
        """Score the selected task action using benchmark ground truth."""


class PlannerFactory(Protocol):
    def __call__(self, **settings: Any) -> Any:
        """Construct a planner from resolved settings."""
