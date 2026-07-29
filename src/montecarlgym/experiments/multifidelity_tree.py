"""Deterministic learned-model/executable-model integration benchmark.

This is an engineering diagnostic for Phase 3. It is deliberately small and
must not be reported as evidence for the FidelityMCTS research claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from ..adaptive import AdaptiveComputePlanner
from ..models import ModelPortfolio
from ..planner import Planner
from ..routing import (
    AccurateOnlyRouter,
    CheapOnlyRouter,
    ComputeRouter,
    FixedCascadeRouter,
    ThresholdRouter,
)
from ..types import (
    Action,
    EvidenceProvenance,
    Fidelity,
    ModelObservation,
    ModelQuote,
)


@dataclass(frozen=True, slots=True)
class ShallowTreeConfig:
    actions: int = 5
    horizon: int = 3
    learned_cost: float = 0.25
    executable_step_cost: float = 2.0
    learned_tokens: int = 8
    learned_variance: float = 0.09

    def __post_init__(self) -> None:
        if self.actions < 2:
            raise ValueError("actions must be at least 2")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.learned_cost < 0 or self.executable_step_cost < 0:
            raise ValueError("model costs must be non-negative")
        if self.learned_tokens < 0:
            raise ValueError("learned_tokens must be non-negative")
        if self.learned_variance < 0:
            raise ValueError("learned_variance must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShallowTreeConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed - {"type"}
        if unknown:
            raise ValueError(f"unknown benchmark settings: {sorted(unknown)}")
        return cls(**{key: value for key, value in data.items() if key != "type"})


@dataclass(frozen=True, slots=True)
class ShallowTreeTask:
    """Task branches with observable features and hidden executable rewards."""

    task_id: str
    features: tuple[float, ...]
    rewards: tuple[tuple[float, ...], ...]

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(f"a{index}" for index in range(len(self.features)))

    def index(self, action: Action) -> int:
        if not isinstance(action, str) or not action.startswith("a"):
            raise ValueError(f"invalid shallow-tree action: {action!r}")
        index = int(action[1:])
        if index < 0 or index >= len(self.features):
            raise ValueError(f"invalid shallow-tree action: {action!r}")
        return index

    def true_value(self, action: Action) -> float:
        return sum(self.rewards[self.index(action)])

    @property
    def optimal_action(self) -> str:
        return max(self.actions, key=self.true_value)


def sample_shallow_tree(config: ShallowTreeConfig, seed: int) -> ShallowTreeTask:
    """Sample a paired deterministic task without touching global RNG state."""

    rng = Random(seed)
    features = tuple(rng.uniform(-1.0, 1.0) for _ in range(config.actions))
    reward_rows: list[tuple[float, ...]] = []
    for index, feature in enumerate(features):
        # The learned model can capture the dominant linear signal but not the
        # deterministic nonlinear/task-specific residual.
        total = 0.25 + 1.35 * feature + 0.45 * feature * feature
        total += rng.uniform(-0.35, 0.35) + 0.04 * index
        weights = [rng.uniform(0.5, 1.5) for _ in range(config.horizon)]
        scale = sum(weights)
        reward_rows.append(tuple(total * weight / scale for weight in weights))
    return ShallowTreeTask(
        task_id=f"phase3-tree-{seed:08d}",
        features=features,
        rewards=tuple(reward_rows),
    )


@dataclass(frozen=True, slots=True)
class ShallowTreeActionProvider:
    def legal_actions(self, state: ShallowTreeTask) -> tuple[str, ...]:
        return state.actions


@dataclass(frozen=True, slots=True)
class LearnedLinearValueModel:
    """A fitted cheap value model; coefficients come from training examples."""

    config: ShallowTreeConfig
    intercept: float
    slope: float

    @classmethod
    def fit_reference(cls, config: ShallowTreeConfig) -> LearnedLinearValueModel:
        training = tuple(
            (feature, 0.25 + 1.35 * feature + 0.45 * feature * feature)
            for feature in (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)
        )
        mean_x = sum(x for x, _ in training) / len(training)
        mean_y = sum(y for _, y in training) / len(training)
        denominator = sum((x - mean_x) ** 2 for x, _ in training)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in training)
        slope /= denominator
        return cls(config, mean_y - slope * mean_x, slope)

    @property
    def model_id(self) -> str:
        return "phase3-learned-linear"

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.CHEAP

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del rollout_depth
        return ModelQuote(
            cost=self.config.learned_cost,
            tokens=min(token_budget, self.config.learned_tokens),
        )

    def evaluate(
        self,
        state: ShallowTreeTask,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del rollout_depth, rng
        feature = state.features[state.index(action)]
        return ModelObservation(
            value=self.intercept + self.slope * feature,
            variance=self.config.learned_variance,
            cost=self.config.learned_cost,
            tokens=min(token_budget, self.config.learned_tokens),
            provenance=EvidenceProvenance.LEARNED,
            metadata={"feature": feature, "model_family": "linear_regression"},
        )


@dataclass(frozen=True, slots=True)
class ExecutableTreeModel:
    """High-fidelity evaluator that executes isolated deterministic rollouts."""

    config: ShallowTreeConfig

    @property
    def model_id(self) -> str:
        return "phase3-executable-tree"

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.ACCURATE

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del token_budget
        steps = min(rollout_depth, self.config.horizon)
        return ModelQuote(
            cost=steps * self.config.executable_step_cost,
            accurate_calls=1,
            environment_calls=steps,
        )

    def evaluate(
        self,
        state: ShallowTreeTask,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del token_budget, rng
        rewards = state.rewards[state.index(action)]
        steps = min(rollout_depth, len(rewards))
        completed = steps == len(rewards)
        return ModelObservation(
            value=sum(rewards[:steps]),
            variance=0.0,
            cost=steps * self.config.executable_step_cost,
            environment_calls=steps,
            provenance=EvidenceProvenance.EXECUTABLE,
            verified=completed,
            terminated=completed,
            truncated=not completed,
            metadata={"executed_steps": steps, "horizon": len(rewards)},
        )


def make_shallow_tree_portfolio(config: ShallowTreeConfig) -> ModelPortfolio:
    return ModelPortfolio.from_models(
        [
            LearnedLinearValueModel.fit_reference(config),
            ExecutableTreeModel(config),
        ]
    )


def make_shallow_tree_planner(
    method: str,
    config: ShallowTreeConfig,
    settings: dict[str, Any],
) -> Planner:
    """Construct Phase 3 diagnostic planners from explicit fixed policies."""

    cheap_id = "phase3-learned-linear"
    accurate_id = "phase3-executable-tree"
    learned_tokens = int(settings.pop("learned_tokens", config.learned_tokens))
    router: ComputeRouter
    if method == "phase3_learned_only":
        router = CheapOnlyRouter(
            cheap_id,
            token_budget=learned_tokens,
            **settings,
        )
    elif method == "phase3_executable_only":
        router = AccurateOnlyRouter(
            accurate_id,
            rollout_depth=config.horizon,
            **settings,
        )
    elif method == "phase3_fixed_cascade":
        router = FixedCascadeRouter(
            cheap_id,
            accurate_id,
            cheap_token_budget=learned_tokens,
            accurate_rollout_depth=config.horizon,
            **settings,
        )
    elif method == "phase3_threshold":
        router = ThresholdRouter(
            cheap_id,
            accurate_id,
            cheap_token_budget=learned_tokens,
            accurate_rollout_depth=config.horizon,
            **settings,
        )
    else:
        raise ValueError(f"unknown Phase 3 method: {method}")
    return AdaptiveComputePlanner(
        action_provider=ShallowTreeActionProvider(),
        router=router,
    )
