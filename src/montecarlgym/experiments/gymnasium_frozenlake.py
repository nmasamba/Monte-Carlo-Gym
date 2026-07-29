"""Real Gymnasium L1 pilot: learned values versus executable environment clones."""

from __future__ import annotations

import importlib
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Protocol, cast, runtime_checkable

from ..adaptive import (
    AdaptiveComputePlanner,
    AdaptiveFrontierEvaluator,
    FixedQueryStopPolicy,
    NeverStopPolicy,
    RandomEscalationRouter,
    RunningDiscrepancyModel,
    StopPolicy,
)
from ..adaptive.learning import (
    CalibratedLinearDiscrepancyModel,
    LinearEVCModel,
)
from ..adaptive.routing import LearnedEVCRouter
from ..agent import MCTSAgent
from ..config import MCTSConfig
from ..core.tree import DefaultStateCodec
from ..gym_wrapper import MCTSEnvWrapper
from ..models import ModelPortfolio
from ..replay import JsonlVerifiedReplayStore, VerifiedTransition
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
    ResourceUsage,
    SearchBudget,
)
from .artifacts import ArtifactWriter
from .metrics import aggregate_episode_records
from .preregistration import (
    ExperimentStage,
    FrozenProtocol,
    protocol_sha256,
    validate_confirmatory_output,
    validate_fresh_output,
    validate_preregistration_protocol,
)


def _gymnasium() -> Any:
    try:
        gymnasium = importlib.import_module("gymnasium")
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "the FrozenLake pilot requires the optional Gymnasium dependency; "
            "install montecarlgym[gym]"
        ) from exc
    return gymnasium


@runtime_checkable
class _Indexable(Protocol):
    def __index__(self) -> int: ...


def _discrete_index(value: object) -> int:
    """Normalize Python and NumPy-style discrete scalar values."""

    if isinstance(value, bool) or not isinstance(value, _Indexable):
        raise TypeError("expected a non-boolean integer-like discrete value")
    return value.__index__()


def _state_components(value: object) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        return _discrete_index(value[0]), _discrete_index(value[1])
    return _discrete_index(value), 0


@dataclass
class TimeAwareObservationEnv:
    """Expose elapsed steps so MCTS state identity includes the time limit."""

    env: Any
    elapsed_steps: int = 0

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any) -> tuple[tuple[int, int], Mapping[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.elapsed_steps = 0
        return (_discrete_index(observation), self.elapsed_steps), info

    def step(
        self, action: Action
    ) -> tuple[tuple[int, int], float, bool, bool, Mapping[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        return (
            (_discrete_index(observation), self.elapsed_steps),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        self.env.close()


@dataclass(frozen=True, slots=True)
class DiscreteActionProvider:
    actions: int

    def legal_actions(self, state: Any) -> tuple[int, ...]:
        del state
        return tuple(range(self.actions))


@dataclass(frozen=True, slots=True)
class TabularLearnedValueModel:
    """Offline Q-learning value table with empirical TD-error uncertainty."""

    q_values: tuple[tuple[float, ...], ...]
    variances: tuple[tuple[float, ...], ...]
    inference_cost: float = 0.05

    @property
    def model_id(self) -> str:
        return "frozenlake-tabular-q"

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.CHEAP

    @property
    def action_count(self) -> int:
        return len(self.q_values[0])

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del token_budget, rollout_depth
        return ModelQuote(cost=self.inference_cost)

    def evaluate(
        self,
        state: Any,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del token_budget, rollout_depth, rng
        state_index, _ = _state_components(state)
        action_index = _discrete_index(action)
        return ModelObservation(
            value=self.q_values[state_index][action_index],
            variance=self.variances[state_index][action_index],
            cost=self.inference_cost,
            provenance=EvidenceProvenance.LEARNED,
            metadata={"model_family": "offline_tabular_q_learning"},
        )

    def greedy_action(self, state: Any) -> int:
        state_index, _ = _state_components(state)
        values = self.q_values[state_index]
        return max(range(len(values)), key=lambda action: (values[action], -action))


@dataclass(frozen=True, slots=True)
class FrozenLakeCloneModel:
    """Execute isolated native FrozenLake rollouts from an observed state."""

    env_id: str
    map_name: str
    is_slippery: bool
    policy: TabularLearnedValueModel
    discount: float = 0.99
    step_cost: float = 1.0
    maximum_rollout_depth: int = 32

    @property
    def model_id(self) -> str:
        return "frozenlake-native-clone"

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.ACCURATE

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del token_budget
        steps = min(rollout_depth, self.maximum_rollout_depth)
        return ModelQuote(
            cost=steps * self.step_cost,
            accurate_calls=1,
            environment_calls=steps,
        )

    def evaluate(
        self,
        state: Any,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del token_budget
        gymnasium = _gymnasium()
        env = gymnasium.make(
            self.env_id,
            map_name=self.map_name,
            is_slippery=self.is_slippery,
            max_episode_steps=self.maximum_rollout_depth,
        )
        steps = 0
        value = 0.0
        factor = 1.0
        terminated = False
        truncated = False
        observation: Any = state
        try:
            env.reset(seed=rng.randrange(2**32))
            unwrapped = env.unwrapped
            if not hasattr(unwrapped, "s"):
                raise RuntimeError(
                    "FrozenLake clone adapter requires the native discrete state"
                )
            state_index, elapsed_steps = _state_components(state)
            unwrapped.s = state_index
            if hasattr(env, "_elapsed_steps"):
                env._elapsed_steps = elapsed_steps
            if hasattr(unwrapped, "lastaction"):
                unwrapped.lastaction = None
            current_action = _discrete_index(action)
            depth = min(rollout_depth, self.maximum_rollout_depth)
            for _ in range(depth):
                observation, reward, terminated, truncated, _ = env.step(
                    current_action
                )
                value += factor * float(reward)
                factor *= self.discount
                steps += 1
                if terminated or truncated:
                    break
                current_action = self.policy.greedy_action(observation)
        finally:
            env.close()
        return ModelObservation(
            value=value,
            variance=0.0,
            cost=steps * self.step_cost,
            next_state=observation,
            terminated=terminated,
            truncated=truncated or not terminated,
            environment_calls=steps,
            provenance=EvidenceProvenance.EXECUTABLE,
            verified=True,
            metadata={
                "adapter": "gymnasium_frozenlake_clone",
                "executed_steps": steps,
                "requested_depth": rollout_depth,
            },
        )


def train_tabular_value_model(
    benchmark: Mapping[str, Any],
    training: Mapping[str, Any],
) -> TabularLearnedValueModel:
    """Train on declared training episodes only; never consume test seeds."""

    gymnasium = _gymnasium()
    env = gymnasium.make(
        str(benchmark["env_id"]),
        map_name=str(benchmark["map_name"]),
        is_slippery=bool(benchmark["is_slippery"]),
        max_episode_steps=int(training["maximum_steps"]),
    )
    state_count = int(env.observation_space.n)
    action_count = int(env.action_space.n)
    q_values = [[0.0] * action_count for _ in range(state_count)]
    squared_error = [[0.0] * action_count for _ in range(state_count)]
    counts = [[0] * action_count for _ in range(state_count)]
    episodes = int(training["episodes"])
    alpha = float(training["alpha"])
    discount = float(training["discount"])
    epsilon_start = float(training["epsilon_start"])
    epsilon_end = float(training["epsilon_end"])
    maximum_steps = int(training["maximum_steps"])
    rng = Random(int(training["seed"]))
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=rng.randrange(2**32))
            fraction = episode / max(1, episodes - 1)
            epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
            for _ in range(maximum_steps):
                state = int(observation)
                if rng.random() < epsilon:
                    action = rng.randrange(action_count)
                else:
                    action = max(
                        range(action_count),
                        key=lambda item: (q_values[state][item], -item),
                    )
                next_observation, reward, terminated, truncated, _ = env.step(action)
                bootstrap = (
                    0.0
                    if terminated or truncated
                    else max(q_values[int(next_observation)])
                )
                target = float(reward) + discount * bootstrap
                error = target - q_values[state][action]
                q_values[state][action] += alpha * error
                counts[state][action] += 1
                squared_error[state][action] += error * error
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        env.close()
    variances = [
        [
            max(1e-4, squared_error[state][action] / max(1, counts[state][action]))
            for action in range(action_count)
        ]
        for state in range(state_count)
    ]
    return TabularLearnedValueModel(
        q_values=tuple(tuple(row) for row in q_values),
        variances=tuple(tuple(row) for row in variances),
        inference_cost=float(training.get("inference_cost", 0.05)),
    )


def _calibration_replay(
    cheap: TabularLearnedValueModel,
    accurate: FrozenLakeCloneModel,
    *,
    budget: SearchBudget,
    seed: int,
) -> tuple[VerifiedTransition, ...]:
    rng = Random(seed)
    records: list[VerifiedTransition] = []
    for state, action_values in enumerate(cheap.q_values):
        best = max(action_values)
        for action, cheap_value in enumerate(action_values):
            observation = accurate.evaluate(
                state,
                action,
                token_budget=0,
                rollout_depth=accurate.maximum_rollout_depth,
                rng=rng,
            )
            features = {
                "action_mean": cheap_value,
                "action_uncertainty": cheap.variances[state][action] ** 0.5,
                "action_risk": 0.0,
                "gap_to_best": max(0.0, best - cheap_value),
                "evidence_count": 1.0,
                "search_depth": 0.0,
                "remaining_cost": budget.max_cost,
                "remaining_accurate_calls": float(budget.max_accurate_calls),
            }
            records.append(
                VerifiedTransition(
                    state_id=f"frozenlake-calibration-{state}",
                    action=action,
                    cheap_model_id=cheap.model_id,
                    cheap_prediction=cheap_value,
                    accurate_model_id=accurate.model_id,
                    verified_outcome=observation.value,
                    router_propensity=1.0,
                    cheap_provenance=EvidenceProvenance.LEARNED,
                    accurate_provenance=EvidenceProvenance.EXECUTABLE,
                    context_features=features,
                    metadata={"source": "exhaustive_native_clone", "state": state},
                )
            )
    return tuple(records)


def _sum_usage(left: ResourceUsage, right: ResourceUsage) -> ResourceUsage:
    return ResourceUsage(
        cost=left.cost + right.cost,
        tokens=left.tokens + right.tokens,
        accurate_calls=left.accurate_calls + right.accurate_calls,
        iterations=left.iterations + right.iterations,
        latency_s=left.latency_s + right.latency_s,
        model_calls=left.model_calls + right.model_calls,
        environment_calls=left.environment_calls + right.environment_calls,
    )


def _router_diagnostics(
    evc_model: LinearEVCModel,
    discrepancy: CalibratedLinearDiscrepancyModel,
    records: Sequence[VerifiedTransition],
) -> dict[str, float | int]:
    if not records:
        raise ValueError("router diagnostics require held-out records")
    evc_errors: list[float] = []
    discrepancy_errors: list[float] = []
    evc_covered = 0
    discrepancy_covered = 0
    for record in records:
        evc_prediction = evc_model.predict(record.context_features)
        discrepancy_prediction = discrepancy.predict_contextual(
            record.cheap_model_id,
            record.accurate_model_id,
            record.context_features,
        )
        if evc_prediction is None or discrepancy_prediction is None:
            raise RuntimeError("fitted router models returned no prediction")
        evc_error = abs(record.discrepancy) - evc_prediction.mean
        discrepancy_error = record.discrepancy - discrepancy_prediction.mean
        evc_errors.append(evc_error)
        discrepancy_errors.append(discrepancy_error)
        evc_covered += int(
            abs(evc_error) <= evc_prediction.interval_half_width
        )
        discrepancy_covered += int(
            abs(discrepancy_error)
            <= discrepancy_prediction.interval_half_width
        )
    count = len(records)
    return {
        "calibration_records": count,
        "evc_proxy_calibration_mae": (
            sum(abs(error) for error in evc_errors) / count
        ),
        "evc_proxy_calibration_rmse": math.sqrt(
            sum(error * error for error in evc_errors) / count
        ),
        "evc_calibration_interval_coverage": evc_covered / count,
        "discrepancy_calibration_mae": (
            sum(abs(error) for error in discrepancy_errors) / count
        ),
        "discrepancy_calibration_rmse": math.sqrt(
            sum(error * error for error in discrepancy_errors) / count
        ),
        "discrepancy_calibration_interval_coverage": (
            discrepancy_covered / count
        ),
    }


def _search_agent(
    method: str,
    *,
    cheap: TabularLearnedValueModel,
    accurate: FrozenLakeCloneModel,
    evc_model: LinearEVCModel,
    discrepancy: RunningDiscrepancyModel,
    replay: JsonlVerifiedReplayStore,
    outer_budget: SearchBudget,
    frontier_budget: SearchBudget,
    settings: Mapping[str, Any],
    seed: int,
) -> MCTSAgent:
    if method == "classical_uct":
        return MCTSAgent(
            budget=outer_budget,
            seed=seed,
            config=MCTSConfig(max_tree_depth=int(settings.get("tree_depth", 16))),
        )
    router: ComputeRouter
    effective_evc_model = evc_model
    effective_discrepancy = discrepancy
    stop_policy: StopPolicy = NeverStopPolicy()
    if method == "frontier_learned":
        router = CheapOnlyRouter(cheap.model_id)
    elif method == "frontier_accurate":
        router = AccurateOnlyRouter(
            accurate.model_id,
            rollout_depth=accurate.maximum_rollout_depth,
        )
    elif method == "frontier_fixed":
        router = FixedCascadeRouter(
            cheap.model_id,
            accurate.model_id,
            top_k=int(settings.get("top_k", 2)),
            accurate_rollout_depth=accurate.maximum_rollout_depth,
        )
    elif method == "frontier_threshold":
        router = ThresholdRouter(
            cheap.model_id,
            accurate.model_id,
            z_score=float(settings.get("z_score", 1.64)),
            accurate_rollout_depth=accurate.maximum_rollout_depth,
        )
    elif method == "frontier_random":
        router = RandomEscalationRouter(
            cheap.model_id,
            accurate.model_id,
            escalation_probability=float(
                settings.get("escalation_probability", 0.25)
            ),
            accurate_rollout_depth=accurate.maximum_rollout_depth,
            seed=seed,
        )
    elif method in {
        "frontier_evc",
        "frontier_evc_without_contextual_discrepancy",
        "frontier_evc_without_audit_traffic",
        "frontier_evc_with_fixed_stopping",
        "frontier_routing_without_verified_replay",
    }:
        audit_probability = float(settings.get("audit_probability", 0.05))
        if method == "frontier_evc_without_contextual_discrepancy":
            effective_discrepancy = RunningDiscrepancyModel()
        elif method == "frontier_evc_without_audit_traffic":
            audit_probability = 0.0
        elif method == "frontier_evc_with_fixed_stopping":
            stop_policy = FixedQueryStopPolicy(
                int(settings.get("fixed_query_limit", 5))
            )
        elif method == "frontier_routing_without_verified_replay":
            effective_evc_model = LinearEVCModel()
            effective_discrepancy = RunningDiscrepancyModel()
        router = LearnedEVCRouter(
            cheap.model_id,
            accurate.model_id,
            effective_evc_model,
            accurate_cost=float(settings.get("accurate_cost", accurate.step_cost)),
            cost_weight=float(settings.get("cost_weight", 0.05)),
            minimum_net_evc=float(settings.get("minimum_net_evc", 0.0)),
            audit_probability=audit_probability,
            accurate_rollout_depth=accurate.maximum_rollout_depth,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown FrozenLake method: {method}")
    portfolio = ModelPortfolio.from_models((cheap, accurate))
    planner = AdaptiveComputePlanner(
        action_provider=DiscreteActionProvider(cheap.action_count),
        router=router,
        stop_policy=stop_policy,
        discrepancy=effective_discrepancy,
        replay=replay,
    )
    evaluator = AdaptiveFrontierEvaluator(planner, portfolio, frontier_budget)
    return MCTSAgent(
        budget=outer_budget,
        seed=seed,
        evaluator=evaluator,
        config=MCTSConfig(max_tree_depth=int(settings.get("tree_depth", 16))),
        state_codec=DefaultStateCodec(),
    )


def _episode(
    method: str,
    *,
    benchmark: Mapping[str, Any],
    cheap: TabularLearnedValueModel,
    accurate: FrozenLakeCloneModel,
    evc_model: LinearEVCModel,
    discrepancy: RunningDiscrepancyModel,
    replay: JsonlVerifiedReplayStore,
    outer_budget: SearchBudget,
    frontier_budget: SearchBudget,
    settings: Mapping[str, Any],
    seed: int,
) -> tuple[float, int, ResourceUsage]:
    gymnasium = _gymnasium()
    env = TimeAwareObservationEnv(
        gymnasium.make(
            str(benchmark["env_id"]),
            map_name=str(benchmark["map_name"]),
            is_slippery=bool(benchmark["is_slippery"]),
            max_episode_steps=int(benchmark["maximum_episode_steps"]),
        )
    )
    observation, _ = env.reset(seed=seed)
    maximum_steps = int(benchmark["maximum_episode_steps"])
    total_return = 0.0
    usage = ResourceUsage()
    steps = 0
    if method == "direct_learned":
        agent = None
        wrapper = None
    else:
        agent = _search_agent(
            method,
            cheap=cheap,
            accurate=accurate,
            evc_model=evc_model,
            discrepancy=discrepancy,
            replay=replay,
            outer_budget=outer_budget,
            frontier_budget=frontier_budget,
            settings=settings,
            seed=seed,
        )
        wrapper = MCTSEnvWrapper(env)
    try:
        for _ in range(maximum_steps):
            action: Action
            if agent is None:
                action = cheap.greedy_action(observation)
                usage = _sum_usage(
                    usage,
                    ResourceUsage(cost=cheap.inference_cost, model_calls=1),
                )
            else:
                assert wrapper is not None
                action = agent.compute_action(wrapper, observation)
                assert agent.last_report is not None
                usage = _sum_usage(usage, agent.last_report.usage)
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            steps += 1
            if agent is not None:
                agent.observe(
                    action=action,
                    observation=next_observation,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                )
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    return total_return, steps, usage


def run_frozenlake_study(
    protocol: Mapping[str, Any],
    output: Path,
    *,
    stage: ExperimentStage,
    frozen: FrozenProtocol | None = None,
    revision_verified: bool = False,
) -> dict[str, Any]:
    """Run a labelled pilot or a fingerprint-guarded confirmatory suite."""

    validate_preregistration_protocol(protocol)
    if protocol.get("stage") != stage.value:
        raise ValueError("requested stage differs from the protocol stage")
    if stage is ExperimentStage.CONFIRMATORY:
        if frozen is None:
            raise ValueError("confirmatory runs require a frozen protocol")
        if not revision_verified:
            raise ValueError(
                "confirmatory caller must verify the clean source revision"
            )
        if protocol_sha256(protocol) != frozen.sha256:
            raise ValueError("confirmatory protocol differs from frozen manifest")
        validate_confirmatory_output(output)
        seeds = tuple(int(seed) for seed in protocol["confirmatory_seeds"])
    else:
        validate_fresh_output(output)
        pilot_seeds = protocol.get("pilot_seeds")
        if not isinstance(pilot_seeds, Sequence) or not pilot_seeds:
            raise ValueError("exploratory protocols require non-empty pilot_seeds")
        seeds = tuple(int(seed) for seed in pilot_seeds)

    benchmark = cast(Mapping[str, Any], protocol["benchmark"])
    training = cast(Mapping[str, Any], protocol["training"])
    cheap = train_tabular_value_model(benchmark, training)
    accurate = FrozenLakeCloneModel(
        env_id=str(benchmark["env_id"]),
        map_name=str(benchmark["map_name"]),
        is_slippery=bool(benchmark["is_slippery"]),
        policy=cheap,
        discount=float(training["discount"]),
        step_cost=float(benchmark.get("clone_step_cost", 1.0)),
        maximum_rollout_depth=int(benchmark["maximum_episode_steps"]),
    )
    budget_grid = [SearchBudget(**item) for item in protocol["budget_grid"]]
    calibration_records = _calibration_replay(
        cheap,
        accurate,
        budget=budget_grid[-1],
        seed=int(training["router_training_seed"]),
    )
    calibration_states = {
        int(state) for state in training["router_calibration_states"]
    }
    router_training = tuple(
        record
        for record in calibration_records
        if int(record.metadata["state"]) not in calibration_states
    )
    router_calibration = tuple(
        record
        for record in calibration_records
        if int(record.metadata["state"]) in calibration_states
    )
    if not router_training or not router_calibration:
        raise ValueError(
            "router_calibration_states must create non-empty state-grouped splits"
        )
    evc_model = LinearEVCModel(ridge=float(training.get("router_ridge", 1.0)))
    evc_model.fit(
        router_training,
        calibration_records=router_calibration,
        coverage=float(training.get("calibration_coverage", 0.9)),
    )
    diagnostic_discrepancy = CalibratedLinearDiscrepancyModel(
        ridge=float(training.get("discrepancy_ridge", 1.0))
    )
    diagnostic_discrepancy.fit(
        router_training,
        calibration_records=router_calibration,
        coverage=float(training.get("calibration_coverage", 0.9)),
    )
    router_diagnostics = _router_diagnostics(
        evc_model,
        diagnostic_discrepancy,
        router_calibration,
    )
    writer = ArtifactWriter(output)
    writer.write_json("resolved_protocol.json", protocol)
    writer.write_json(
        "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "gymnasium": _gymnasium().__version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage.value,
            "protocol_sha256": protocol_sha256(protocol),
            "code_revision": None if frozen is None else frozen.code_revision,
        },
    )
    writer.write_json("router_diagnostics.json", router_diagnostics)
    checkpoints = output / "checkpoints"
    evc_model.save(checkpoints / "evc.json")
    training_store = JsonlVerifiedReplayStore(
        output / "replay" / "router_training.jsonl"
    )
    training_store.extend(calibration_records)

    primary_methods = tuple(str(method) for method in protocol["methods"])
    ablations = tuple(str(method) for method in protocol["ablations"])
    methods = tuple(dict.fromkeys((*primary_methods, *ablations)))
    method_settings = cast(
        Mapping[str, Mapping[str, Any]], protocol.get("method_settings", {})
    )
    frontier_settings = cast(
        Mapping[str, Any], protocol["resource_accounting"]
    )
    frontier_budget = SearchBudget(**frontier_settings["frontier_budget"])
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for budget_index, budget in enumerate(budget_grid):
        for seed in seeds:
            for method in methods:
                discrepancy = CalibratedLinearDiscrepancyModel(
                    ridge=float(training.get("discrepancy_ridge", 1.0))
                )
                discrepancy.fit(
                    router_training,
                    calibration_records=router_calibration,
                    coverage=float(training.get("calibration_coverage", 0.9)),
                )
                replay = JsonlVerifiedReplayStore(
                    output
                    / "replay"
                    / f"budget-{budget_index}-{method}-{seed}.jsonl"
                )
                try:
                    return_value, steps, usage = _episode(
                        method,
                        benchmark=benchmark,
                        cheap=cheap,
                        accurate=accurate,
                        evc_model=evc_model,
                        discrepancy=discrepancy,
                        replay=replay,
                        outer_budget=budget,
                        frontier_budget=frontier_budget,
                        settings=(
                            method_settings.get("frontier_evc", {})
                            if method in ablations
                            else method_settings.get(method, {})
                        ),
                        seed=seed,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "schema_version": 1,
                            "stage": stage.value,
                            "method": method,
                            "seed": seed,
                            "budget_index": budget_index,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    if protocol["failure_policy"].get("abort_on_error", True):
                        writer.write_jsonl("failures.jsonl", failures)
                        raise
                    continue
                records.append(
                    {
                        "schema_version": 1,
                        "stage": stage.value,
                        "benchmark": "Gymnasium/FrozenLake-v1",
                        "method": method,
                        "task_id": f"frozenlake-{seed}",
                        "seed": seed,
                        "budget_index": budget_index,
                        "action": None,
                        "optimal_action": None,
                        "success": return_value > 0,
                        "return": return_value,
                        "regret": 1.0 - return_value,
                        "predicted_value": None,
                        "cost": usage.cost,
                        "tokens": usage.tokens,
                        "accurate_calls": usage.accurate_calls,
                        "iterations": usage.iterations,
                        "latency_s": usage.latency_s,
                        "model_calls": usage.model_calls,
                        "environment_calls": usage.environment_calls,
                        "episode_steps": steps,
                        "risk": 0.0,
                        "verified_pairs": len(replay),
                    }
                )
    writer.write_jsonl("runs.jsonl", records)
    writer.write_jsonl("failures.jsonl", failures)
    by_budget = {
        str(budget_index): aggregate_episode_records(
            record
            for record in records
            if record["budget_index"] == budget_index
        )
        for budget_index in range(len(budget_grid))
    }
    summary = {
        "schema_version": 1,
        "stage": stage.value,
        "benchmark": "Gymnasium/FrozenLake-v1",
        "protocol_sha256": protocol_sha256(protocol),
        "records": len(records),
        "failures": len(failures),
        "primary_methods": primary_methods,
        "ablations": ablations,
        "router_diagnostics": router_diagnostics,
        "methods": aggregate_episode_records(records),
        "methods_by_budget": by_budget,
        "notice": (
            "Exploratory pilot; not a paper result."
            if stage is ExperimentStage.EXPLORATORY
            else "Confirmatory run governed by the frozen preregistration."
        ),
    }
    writer.write_json("summary.json", summary)
    return summary
