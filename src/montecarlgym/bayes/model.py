"""Tabular posterior generative model for MCBRL-style root sampling."""

from __future__ import annotations

import copy
import math
from collections.abc import Hashable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from random import Random
from typing import Any

from ..core.tree import DefaultStateCodec, StateCodec
from ..types import Action
from .conjugate import DirichletTransitionPosterior, NormalGammaPosterior


@dataclass(frozen=True, slots=True)
class PosteriorOutcome:
    """One known support point of a tabular transition posterior."""

    next_state: Any
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateActionBelief:
    """Reward and transition beliefs for one state-action pair."""

    outcomes: Mapping[Hashable, PosteriorOutcome]
    reward: NormalGammaPosterior = field(default_factory=NormalGammaPosterior)
    transitions: DirichletTransitionPosterior = field(
        default_factory=DirichletTransitionPosterior
    )

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("a state-action belief needs at least one outcome")
        unknown = set(self.transitions.concentrations) - set(self.outcomes)
        if unknown:
            raise ValueError(
                "transition posterior contains unknown outcomes: "
                f"{sorted(unknown, key=repr)!r}"
            )
        for outcome_key in self.outcomes:
            self.transitions.concentrations.setdefault(
                outcome_key,
                self.transitions.prior_concentration,
            )

    def observe(self, *, outcome_key: Hashable, reward: float) -> None:
        """Apply one explicit real observation to this belief."""

        if outcome_key not in self.outcomes:
            raise KeyError(f"unknown posterior outcome: {outcome_key!r}")
        self.reward.update(reward)
        self.transitions.observe(outcome_key)


@dataclass(slots=True)
class TabularRootBelief:
    """Real posterior over finite dynamics, separate from an MCTS tree."""

    state_actions: Mapping[Hashable, Mapping[Action, StateActionBelief]]
    state_codec: StateCodec = field(default_factory=DefaultStateCodec)

    def __post_init__(self) -> None:
        if not self.state_actions:
            raise ValueError("root belief must contain at least one state")
        for state_key, actions in self.state_actions.items():
            try:
                hash(state_key)
            except TypeError as exc:
                raise TypeError("root-belief state keys must be hashable") from exc
            if not actions:
                raise ValueError(
                    f"non-terminal belief state {state_key!r} has no actions"
                )

    def actions(self, state: Any) -> tuple[Action, ...]:
        key = self.state_codec.key(state)
        try:
            return tuple(self.state_actions[key])
        except KeyError as exc:
            raise KeyError(f"state {key!r} is absent from the root belief") from exc

    def observe(
        self,
        *,
        state: Any,
        action: Action,
        outcome_key: Hashable,
        reward: float,
    ) -> None:
        """Update the real belief only when the caller supplies real evidence."""

        key = self.state_codec.key(state)
        try:
            state_action = self.state_actions[key][action]
        except KeyError as exc:
            raise KeyError(
                f"unknown state-action pair in root belief: {(key, action)!r}"
            ) from exc
        state_action.observe(outcome_key=outcome_key, reward=reward)


@dataclass(frozen=True, slots=True)
class _SampledAction:
    reward: float
    outcomes: tuple[tuple[float, PosteriorOutcome], ...]


@dataclass(frozen=True, slots=True)
class _ModelSnapshot:
    state: Any
    terminated: bool
    truncated: bool
    rng_state: tuple[Any, ...]
    sampled_dynamics: dict[Hashable, dict[Action, _SampledAction]] | None


class TabularRootSamplingModel:
    """Sample one finite MDP per simulation from a frozen real root belief.

    ``MCTSEngine`` calls :meth:`seed_simulation` once per iteration. That call
    draws reward parameters and transition probabilities for every known
    state-action pair. The sampled dynamics remain fixed for the full
    simulation, while ordinary tree backup updates only imaginary search
    statistics. Real belief updates require an explicit
    :meth:`TabularRootBelief.observe` call by the application.
    """

    def __init__(
        self,
        initial_state: Any,
        belief: TabularRootBelief,
        *,
        max_call_cost: float = 1.0,
        default_call_cost: float = 1.0,
        cost_key: str = "normalized_cost",
    ) -> None:
        if not math.isfinite(max_call_cost) or max_call_cost < 0:
            raise ValueError("max_call_cost must be finite and non-negative")
        if not math.isfinite(default_call_cost) or default_call_cost < 0:
            raise ValueError("default_call_cost must be finite and non-negative")
        if default_call_cost > max_call_cost:
            raise ValueError("default_call_cost cannot exceed max_call_cost")
        # Validate the initial state against the supplied real belief.
        belief.actions(initial_state)
        self.initial_state = copy.deepcopy(initial_state)
        self.belief = belief
        self.max_call_cost = float(max_call_cost)
        self.default_call_cost = float(default_call_cost)
        self.cost_key = cost_key
        self._state = copy.deepcopy(initial_state)
        self._terminated = False
        self._truncated = False
        self._rng = Random()
        self._sampled_dynamics: dict[
            Hashable, dict[Action, _SampledAction]
        ] | None = None
        self._in_transaction = False

    def snapshot(self) -> _ModelSnapshot:
        return _ModelSnapshot(
            copy.deepcopy(self._state),
            self._terminated,
            self._truncated,
            self._rng.getstate(),
            copy.deepcopy(self._sampled_dynamics),
        )

    def restore(self, snapshot: _ModelSnapshot) -> None:
        self._state = copy.deepcopy(snapshot.state)
        self._terminated = snapshot.terminated
        self._truncated = snapshot.truncated
        self._rng.setstate(snapshot.rng_state)
        self._sampled_dynamics = copy.deepcopy(snapshot.sampled_dynamics)

    @contextmanager
    def transaction(self) -> Iterator[TabularRootSamplingModel]:
        if self._in_transaction:
            raise RuntimeError("nested root-sampling transactions are unsupported")
        snapshot = self.snapshot()
        self._in_transaction = True
        try:
            yield self
        finally:
            self.restore(snapshot)
            self._in_transaction = False

    def seed_simulation(self, seed: int) -> None:
        if not self._in_transaction:
            raise RuntimeError("root sampling requires an active transaction")
        self._rng.seed(seed)
        sampled: dict[Hashable, dict[Action, _SampledAction]] = {}
        for state_key, actions in self.belief.state_actions.items():
            sampled_actions: dict[Action, _SampledAction] = {}
            for action, state_action in actions.items():
                probabilities = state_action.transitions.sample(self._rng)
                cumulative = 0.0
                outcomes: list[tuple[float, PosteriorOutcome]] = []
                for outcome_key, outcome in state_action.outcomes.items():
                    cumulative += probabilities[outcome_key]
                    outcomes.append((cumulative, outcome))
                outcomes[-1] = (1.0, outcomes[-1][1])
                sampled_actions[action] = _SampledAction(
                    state_action.reward.sample_mean(self._rng),
                    tuple(outcomes),
                )
            sampled[state_key] = sampled_actions
        self._sampled_dynamics = sampled

    def legal_actions(self, observation: Any | None = None) -> tuple[Action, ...]:
        state = self._state if observation is None else observation
        return self.belief.actions(state)

    def step(
        self,
        action: Action,
    ) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
        if self._sampled_dynamics is None:
            raise RuntimeError("seed_simulation must sample an MDP before step")
        if self._terminated or self._truncated:
            raise RuntimeError("step after sampled episode end")
        state_key = self.belief.state_codec.key(self._state)
        try:
            sampled_action = self._sampled_dynamics[state_key][action]
        except KeyError as exc:
            raise ValueError(
                f"illegal sampled state-action pair: {(state_key, action)!r}"
            ) from exc
        threshold = self._rng.random()
        outcome = sampled_action.outcomes[-1][1]
        for cumulative, candidate in sampled_action.outcomes:
            if threshold <= cumulative:
                outcome = candidate
                break
        self._state = copy.deepcopy(outcome.next_state)
        self._terminated = outcome.terminated
        self._truncated = outcome.truncated
        info = dict(outcome.info)
        info.setdefault(self.cost_key, self.default_call_cost)
        return (
            copy.deepcopy(self._state),
            sampled_action.reward,
            self._terminated,
            self._truncated,
            info,
        )

    def transition_cost(self, info: Mapping[str, Any]) -> float:
        raw_cost = info.get(self.cost_key, self.default_call_cost)
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition normalized cost must be numeric") from exc
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("transition cost must be finite and non-negative")
        return cost
