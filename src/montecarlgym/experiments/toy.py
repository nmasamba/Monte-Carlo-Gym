"""Controlled multi-fidelity benchmark and reference planners.

This benchmark is intentionally small. It makes routing behavior, budget
accounting, and reproducibility testable before integrating a tree environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from ..adaptive.budget import AdaptiveResourceLedger
from ..models import ModelPortfolio
from ..planner import PlanResult
from ..types import (
    Action,
    Fidelity,
    ModelObservation,
    ModelQuote,
    ResourceUsage,
    SearchBudget,
)


@dataclass(frozen=True, slots=True)
class ToyBenchmarkConfig:
    actions: int = 5
    cheap_noise: float = 0.25
    accurate_noise: float = 0.03
    bias_scale: float = 0.35
    cheap_cost: float = 1.0
    accurate_cost: float = 12.0
    cheap_tokens: int = 16
    accurate_tokens: int = 96

    def __post_init__(self) -> None:
        if self.actions < 2:
            raise ValueError("actions must be at least 2")
        if self.cheap_noise < 0 or self.accurate_noise < 0:
            raise ValueError("noise values must be non-negative")
        if self.cheap_cost < 0 or self.accurate_cost < 0:
            raise ValueError("cost values must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToyBenchmarkConfig:
        allowed = {field for field in cls.__dataclass_fields__}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown benchmark settings: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ToyTask:
    task_id: str
    values: tuple[float, ...]
    cheap_bias: tuple[float, ...]

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(f"a{index}" for index in range(len(self.values)))

    def index(self, action: Action) -> int:
        if not isinstance(action, str) or not action.startswith("a"):
            raise ValueError(f"invalid toy action: {action!r}")
        index = int(action[1:])
        if index < 0 or index >= len(self.values):
            raise ValueError(f"invalid toy action: {action!r}")
        return index

    def true_value(self, action: Action) -> float:
        return self.values[self.index(action)]

    @property
    def optimal_action(self) -> str:
        index = max(range(len(self.values)), key=self.values.__getitem__)
        return f"a{index}"


def sample_task(config: ToyBenchmarkConfig, seed: int) -> ToyTask:
    rng = Random(seed)
    values = tuple(rng.uniform(-1.0, 1.0) for _ in range(config.actions))
    cheap_bias = tuple(
        rng.gauss(0.0, config.bias_scale) for _ in range(config.actions)
    )
    return ToyTask(f"toy-{seed:08d}", values, cheap_bias)


@dataclass(frozen=True, slots=True)
class ToyModel:
    """A cheap biased oracle or an expensive near-ground-truth oracle."""

    config: ToyBenchmarkConfig
    _model_id: str
    _fidelity: Fidelity

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def fidelity(self) -> Fidelity:
        return self._fidelity

    def quote(
        self,
        *,
        token_budget: int,
        rollout_depth: int,
    ) -> ModelQuote:
        del token_budget, rollout_depth
        if self.fidelity is Fidelity.ACCURATE:
            return ModelQuote(
                cost=self.config.accurate_cost,
                tokens=self.config.accurate_tokens,
                accurate_calls=1,
            )
        return ModelQuote(
            cost=self.config.cheap_cost,
            tokens=self.config.cheap_tokens,
        )

    def evaluate(
        self,
        state: ToyTask,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del token_budget, rollout_depth
        true_value = state.true_value(action)
        index = state.index(action)
        if self.fidelity is Fidelity.ACCURATE:
            noise = self.config.accurate_noise
            mean = true_value
            cost = self.config.accurate_cost
            tokens = self.config.accurate_tokens
        else:
            noise = self.config.cheap_noise
            mean = true_value + state.cheap_bias[index]
            cost = self.config.cheap_cost
            tokens = self.config.cheap_tokens
        return ModelObservation(
            value=rng.gauss(mean, noise) if noise else mean,
            variance=noise**2,
            cost=cost,
            tokens=tokens,
            metadata={
                "model_id": self.model_id,
                "fidelity": self.fidelity.value,
            },
        )


def make_portfolio(config: ToyBenchmarkConfig) -> ModelPortfolio:
    return ModelPortfolio.from_models(
        [
            ToyModel(config, "toy-cheap", Fidelity.CHEAP),
            ToyModel(config, "toy-accurate", Fidelity.ACCURATE),
        ]
    )


class _Ledger:
    def __init__(self, budget: SearchBudget) -> None:
        self._ledger = AdaptiveResourceLedger(budget)

    def query(
        self,
        model: ToyModel,
        task: ToyTask,
        action: Action,
        rng: Random,
    ) -> ModelObservation | None:
        quote = model.quote(token_budget=0, rollout_depth=1)
        reservation = self._ledger.reserve(quote)
        if reservation is None:
            return None
        try:
            observation = model.evaluate(
                task,
                action,
                token_budget=0,
                rollout_depth=1,
                rng=rng,
            )
        except Exception:
            self._ledger.fail(reservation)
            raise
        self._ledger.commit(reservation, observation)
        return observation

    def usage(self) -> ResourceUsage:
        return self._ledger.usage()


def _models(portfolio: ModelPortfolio) -> tuple[ToyModel, ToyModel]:
    cheap = portfolio.get("toy-cheap")
    accurate = portfolio.get("toy-accurate")
    if not isinstance(cheap, ToyModel) or not isinstance(accurate, ToyModel):
        raise TypeError("toy planners require the toy model portfolio")
    return cheap, accurate


def _trace(
    action: Action,
    observation: ModelObservation,
    model_id: str,
) -> dict[str, Any]:
    return {
        "task_action": action,
        "model_id": model_id,
        "value": observation.value,
        "variance": observation.variance,
        "cost": observation.cost,
        "tokens": observation.tokens,
    }


@dataclass(frozen=True, slots=True)
class CheapOnlyPlanner:
    name: str = "cheap_only"

    def plan(
        self,
        state: ToyTask,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        cheap, _ = _models(models)
        rng = Random(seed)
        ledger = _Ledger(budget)
        estimates: dict[str, float] = {}
        trace: list[dict[str, Any]] = []
        for action in state.actions:
            observation = ledger.query(cheap, state, action, rng)
            if observation is None:
                break
            estimates[action] = observation.value
            trace.append(_trace(action, observation, cheap.model_id))
        if not estimates:
            raise RuntimeError("budget cannot afford one cheap query")
        action = max(estimates, key=estimates.__getitem__)
        return PlanResult(action, estimates[action], ledger.usage(), tuple(trace))


@dataclass(frozen=True, slots=True)
class HighFidelityOnlyPlanner:
    name: str = "accurate_only"

    def plan(
        self,
        state: ToyTask,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        _, accurate = _models(models)
        rng = Random(seed)
        ledger = _Ledger(budget)
        estimates: dict[str, float] = {}
        trace: list[dict[str, Any]] = []
        for action in state.actions:
            observation = ledger.query(accurate, state, action, rng)
            if observation is None:
                break
            estimates[action] = observation.value
            trace.append(_trace(action, observation, accurate.model_id))
        if not estimates:
            raise RuntimeError("budget cannot afford one accurate query")
        action = max(estimates, key=estimates.__getitem__)
        return PlanResult(action, estimates[action], ledger.usage(), tuple(trace))


@dataclass(frozen=True, slots=True)
class FixedCascadePlanner:
    top_k: int = 2
    name: str = "fixed_cascade"

    def plan(
        self,
        state: ToyTask,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        cheap, accurate = _models(models)
        rng = Random(seed)
        ledger = _Ledger(budget)
        estimates: dict[str, float] = {}
        trace: list[dict[str, Any]] = []

        for action in state.actions:
            observation = ledger.query(cheap, state, action, rng)
            if observation is None:
                break
            estimates[action] = observation.value
            trace.append(_trace(action, observation, cheap.model_id))
        if not estimates:
            raise RuntimeError("budget cannot afford one cheap query")

        ranked = sorted(estimates, key=estimates.__getitem__, reverse=True)
        for action in ranked[: self.top_k]:
            observation = ledger.query(accurate, state, action, rng)
            if observation is None:
                break
            estimates[action] = observation.value
            trace.append(_trace(action, observation, accurate.model_id))

        action = max(estimates, key=estimates.__getitem__)
        return PlanResult(action, estimates[action], ledger.usage(), tuple(trace))


@dataclass(frozen=True, slots=True)
class AdaptiveFidelityPlanner:
    """Escalate branches whose cheap confidence intervals remain competitive."""

    z_score: float = 1.64
    name: str = "adaptive"

    def plan(
        self,
        state: ToyTask,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        cheap, accurate = _models(models)
        rng = Random(seed)
        ledger = _Ledger(budget)
        evidence: dict[str, ModelObservation] = {}
        trace: list[dict[str, Any]] = []

        for action in state.actions:
            observation = ledger.query(cheap, state, action, rng)
            if observation is None:
                break
            evidence[action] = observation
            trace.append(_trace(action, observation, cheap.model_id))
        if not evidence:
            raise RuntimeError("budget cannot afford one cheap query")

        best_action = max(evidence, key=lambda action: evidence[action].value)
        best = evidence[best_action]
        best_lower = best.value - self.z_score * best.variance**0.5
        candidates = [
            action
            for action, observation in evidence.items()
            if (
                observation.value + self.z_score * observation.variance**0.5
                >= best_lower
            )
        ]
        candidates.sort(
            key=lambda action: (
                evidence[action].value
                + self.z_score * evidence[action].variance**0.5
            ),
            reverse=True,
        )

        for action in candidates:
            observation = ledger.query(accurate, state, action, rng)
            if observation is None:
                break
            evidence[action] = observation
            trace.append(_trace(action, observation, accurate.model_id))

        action = max(evidence, key=lambda candidate: evidence[candidate].value)
        return PlanResult(
            action,
            evidence[action].value,
            ledger.usage(),
            tuple(trace),
        )


PLANNER_TYPES = {
    "cheap_only": CheapOnlyPlanner,
    "accurate_only": HighFidelityOnlyPlanner,
    "fixed_cascade": FixedCascadePlanner,
    "adaptive": AdaptiveFidelityPlanner,
}
