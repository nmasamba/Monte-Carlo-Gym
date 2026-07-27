"""Shared immutable value objects for planning and accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping

State = Hashable
Action = Hashable


class Fidelity(str, Enum):
    """Coarse fidelity labels; models may expose additional numeric metadata."""

    CHEAP = "cheap"
    INTERMEDIATE = "intermediate"
    ACCURATE = "accurate"


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Hard limits applied to one planning call.

    Cost is a normalized research unit. Connectors may additionally map it to
    currency, GPU time, wall-clock time, or sandbox executions.
    """

    max_cost: float
    max_tokens: int
    max_accurate_calls: int
    max_iterations: int = 10_000
    deadline_s: float | None = None


@dataclass(frozen=True, slots=True)
class ComputeAction:
    """A meta-level decision about how to evaluate a task branch."""

    state_id: str
    task_action: Action
    model_id: str
    token_budget: int = 0
    rollout_depth: int = 1
    request_verification: bool = False


@dataclass(frozen=True, slots=True)
class ModelQuote:
    """Conservative resource reservation for a prospective model query."""

    cost: float
    tokens: int = 0
    accurate_calls: int = 0
    expected_latency_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelObservation:
    """One model's value/transition evidence and its measured resource use."""

    value: float
    variance: float
    cost: float
    tokens: int = 0
    latency_s: float = 0.0
    risk: float = 0.0
    next_state: State | None = None
    terminated: bool = False
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Accumulated resources for a completed planning call."""

    cost: float = 0.0
    tokens: int = 0
    accurate_calls: int = 0
    iterations: int = 0
    latency_s: float = 0.0
