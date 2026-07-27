"""Dependency-free Gymnasium-style fixtures for classical MCTS tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class DiscreteSpace:
    n: int


class StatefulRandomEnv:
    """Small reversible environment with environment-local randomness."""

    def __init__(self) -> None:
        self.action_space = DiscreteSpace(2)
        self.state = 0
        self.steps = 0
        self.rng = random.Random(0)

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self.state = 0
        self.steps = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        self.state += action + 1
        reward = self.rng.random()
        terminated = self.state >= 4
        truncated = not terminated and self.steps >= 3
        return self.state, reward, terminated, truncated, {
            "normalized_cost": 1.0
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "steps": self.steps,
            "rng": self.rng.getstate(),
        }

    def set_state(self, snapshot: dict[str, Any]) -> None:
        self.state = int(snapshot["state"])
        self.steps = int(snapshot["steps"])
        self.rng.setstate(snapshot["rng"])


class OneStepBanditEnv:
    def __init__(self, rewards: tuple[float, ...] = (0.0, 1.0)) -> None:
        self.action_space = DiscreteSpace(len(rewards))
        self.rewards = rewards
        self.done = False
        self.calls = 0

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.done = False
        self.calls = 0
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step after episode end")
        self.done = True
        self.calls += 1
        return action + 1, self.rewards[action], True, False, {
            "normalized_cost": 1.0
        }


class TerminationModeEnv:
    """Action zero terminates; action one truncates."""

    def __init__(self) -> None:
        self.action_space = DiscreteSpace(2)
        self.done = False

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.done = False
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step after episode end")
        self.done = True
        return action + 1, 0.0, action == 0, action == 1, {
            "normalized_cost": 1.0
        }


class StochasticOneStepEnv:
    def __init__(self) -> None:
        self.action_space = DiscreteSpace(1)
        self.rng = random.Random(0)
        self.done = False

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self.done = False
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        del action
        if self.done:
            raise RuntimeError("step after episode end")
        self.done = True
        return 1, self.rng.random(), True, False, {"normalized_cost": 1.0}


class TwoStepEnv:
    def __init__(self) -> None:
        self.action_space = DiscreteSpace(2)
        self.state = 0

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.state = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.state += 1
        terminated = self.state >= 2
        reward = float(action)
        return self.state, reward, terminated, False, {
            "normalized_cost": 1.0
        }


class ExplodingEnv(StatefulRandomEnv):
    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        self.state += action + 1
        self.rng.random()
        raise RuntimeError("fixture simulation failed")


class UnsafeCopyEnv(StatefulRandomEnv):
    def __deepcopy__(self, memo: dict[int, Any]) -> UnsafeCopyEnv:
        del memo
        return self
