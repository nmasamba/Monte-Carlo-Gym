"""Planner-level result contracts shared by MCTS and experiment baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import ModelPortfolio
from .types import Action, ResourceUsage, SearchBudget, State


@dataclass(frozen=True, slots=True)
class PlanResult:
    """An action recommendation with auditable accounting and trace metadata."""

    action: Action
    predicted_value: float
    usage: ResourceUsage
    trace: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


class Planner(Protocol):
    """Common contract for classical MCTS and adaptive-compute planners."""

    def plan(
        self,
        state: State,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        """Return the selected task action without mutating the live system."""
