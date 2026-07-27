"""Interfaces for cheap, intermediate, and accurate generative models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from random import Random
from typing import Protocol, runtime_checkable

from .types import Action, Fidelity, ModelObservation, ModelQuote, State


@runtime_checkable
class GenerativeModel(Protocol):
    """A branch evaluator or state-transition simulator.

    Implementations may wrap a learned world model, an LLM, a Gymnasium clone,
    a browser sandbox, a code/database environment, or a real system guarded by
    an approval boundary.
    """

    @property
    def model_id(self) -> str:
        """Stable identifier used in logs and replay."""

    @property
    def fidelity(self) -> Fidelity:
        """Relative fidelity class used for budgets and reporting."""

    def quote(
        self,
        *,
        token_budget: int,
        rollout_depth: int,
    ) -> ModelQuote:
        """Return a conservative reservation before the model is called."""

    def evaluate(
        self,
        state: State,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        """Return evidence about an action without mutating the live system."""


@dataclass(slots=True)
class ModelPortfolio:
    """Validated registry of interchangeable generative models."""

    _models: dict[str, GenerativeModel]

    @classmethod
    def from_models(cls, models: Iterable[GenerativeModel]) -> ModelPortfolio:
        registry: dict[str, GenerativeModel] = {}
        for model in models:
            if model.model_id in registry:
                raise ValueError(f"duplicate model_id: {model.model_id}")
            registry[model.model_id] = model
        if not registry:
            raise ValueError("a model portfolio cannot be empty")
        return cls(registry)

    def get(self, model_id: str) -> GenerativeModel:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model_id: {model_id}") from exc

    def by_fidelity(self, fidelity: Fidelity) -> tuple[GenerativeModel, ...]:
        return tuple(
            model
            for model in self._models.values()
            if model.fidelity is fidelity
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(self._models)
