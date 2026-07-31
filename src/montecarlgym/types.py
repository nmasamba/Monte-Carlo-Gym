"""Shared immutable value objects for planning and accounting."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Observations are not necessarily hashable.  Classical search uses an injected
# StateCodec to turn them into stable keys; the adaptive interfaces intentionally
# keep accepting arbitrary state objects.
State = Any
Action = Hashable


class Fidelity(str, Enum):
    """Coarse fidelity labels; models may expose additional numeric metadata."""

    CHEAP = "cheap"
    INTERMEDIATE = "intermediate"
    ACCURATE = "accurate"


class EvidenceProvenance(str, Enum):
    """Origin of model evidence, kept distinct in traces and replay.

    ``VERIFIED`` is intentionally not a provenance value. Verification is an
    independent property of evidence: an executable simulator may be verified
    for a controlled benchmark while a real observation may still be pending
    an external check.
    """

    SYNTHETIC = "synthetic"
    MODEL_PREDICTED = "model_predicted"
    LEARNED = "learned"
    EXECUTABLE = "executable"
    REAL = "real"


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
    max_model_calls: int | None = None
    max_environment_calls: int | None = None

    def __post_init__(self) -> None:
        if self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if self.max_accurate_calls < 0:
            raise ValueError("max_accurate_calls must be non-negative")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.deadline_s is not None and self.deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        for name, limit in (
            ("max_model_calls", self.max_model_calls),
            ("max_environment_calls", self.max_environment_calls),
        ):
            if limit is not None and limit < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class ComputeAction:
    """A meta-level decision about how to evaluate a task branch."""

    state_id: str
    task_action: Action
    model_id: str
    token_budget: int = 0
    rollout_depth: int = 1
    request_verification: bool = False
    route_propensity: float | None = None
    audit: bool = False
    expected_value_of_compute: float | None = None

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id cannot be empty")
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if self.token_budget < 0:
            raise ValueError("token_budget must be non-negative")
        if self.rollout_depth < 1:
            raise ValueError("rollout_depth must be positive")
        propensity = self.route_propensity
        if propensity is not None and not 0.0 < propensity <= 1.0:
            raise ValueError("route_propensity must be in (0, 1]")
        evc = self.expected_value_of_compute
        if evc is not None and not math.isfinite(evc):
            raise ValueError("expected_value_of_compute must be finite")


@dataclass(frozen=True, slots=True)
class ModelQuote:
    """Conservative resource reservation for a prospective model query."""

    cost: float
    tokens: int = 0
    accurate_calls: int = 0
    expected_latency_s: float = 0.0
    environment_calls: int = 0


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
    environment_calls: int = 0
    provenance: EvidenceProvenance = EvidenceProvenance.SYNTHETIC
    verified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Accumulated resources for a completed planning call."""

    cost: float = 0.0
    tokens: int = 0
    accurate_calls: int = 0
    iterations: int = 0
    latency_s: float = 0.0
    model_calls: int = 0
    environment_calls: int = 0
