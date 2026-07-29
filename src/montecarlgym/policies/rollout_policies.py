"""Rollout action policies and the classical random-rollout evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Protocol

from ..core.path import Evaluation
from ..core.tree import StateNode
from ..types import Action

TransitionTuple = tuple[Any, float, bool, bool, Mapping[str, Any]]


class RolloutContext(Protocol):
    def legal_actions(self, observation: Any | None = None) -> tuple[Action, ...]: ...

    def try_step(self, action: Action) -> TransitionTuple | None: ...


class RolloutPolicy(Protocol):
    def choose_action(
        self,
        actions: Sequence[Action],
        rng: Random,
    ) -> Action:
        """Select a legal simulation action."""


@dataclass(frozen=True, slots=True)
class RandomRolloutPolicy:
    """Uniform random simulation over currently legal actions."""

    def choose_action(
        self,
        actions: Sequence[Action],
        rng: Random,
    ) -> Action:
        if not actions:
            raise RuntimeError("rollout state has no legal actions")
        return actions[rng.randrange(len(actions))]


@dataclass(frozen=True, slots=True)
class RandomRolloutEvaluator:
    """Evaluate a frontier by a bounded uniform-random rollout."""

    policy: RolloutPolicy = RandomRolloutPolicy()
    max_depth: int = 100
    discount: float = 1.0

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("rollout max_depth must be non-negative")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")

    def evaluate(
        self,
        frontier: StateNode,
        model: RolloutContext,
        rng: Random,
    ) -> Evaluation:
        if frontier.terminal:
            return Evaluation(
                0.0,
                terminated=frontier.terminated,
                truncated=frontier.truncated,
                stop_reason="terminal" if frontier.terminated else "truncated",
            )
        value = 0.0
        factor = 1.0
        observation = frontier.state
        rollout_actions: list[Action] = []
        for depth in range(self.max_depth):
            actions = model.legal_actions(observation)
            action = self.policy.choose_action(actions, rng)
            rollout_actions.append(action)
            transition = model.try_step(action)
            if transition is None:
                return Evaluation(
                    value,
                    depth,
                    stop_reason="resource_budget",
                    rollout_actions=tuple(rollout_actions[:-1]),
                )
            observation, reward, terminated, truncated, _ = transition
            value += factor * reward
            factor *= self.discount
            if terminated or truncated:
                return Evaluation(
                    value,
                    depth + 1,
                    terminated=terminated,
                    truncated=truncated,
                    stop_reason="terminal" if terminated else "truncated",
                    rollout_actions=tuple(rollout_actions),
                )
        return Evaluation(
            value,
            self.max_depth,
            stop_reason="rollout_depth",
            rollout_actions=tuple(rollout_actions),
        )
