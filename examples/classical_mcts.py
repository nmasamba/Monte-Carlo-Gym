"""Minimal dependency-free UCT example.

Run from a source checkout with:

    PYTHONPATH=src python examples/classical_mcts.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from montecarlgym import MCTSAgent, MCTSEnvWrapper, SearchBudget
from montecarlgym.config import MCTSConfig
from montecarlgym.core import (
    DefaultStateCodec,
    LegalActionExpander,
    ListTraceSink,
    MeanBackup,
)
from montecarlgym.policies import (
    MostVisitedActionSelector,
    RandomRolloutEvaluator,
    UCTTreePolicy,
)


@dataclass
class Discrete:
    n: int


class TinyLineEnv:
    """A two-action, finite-horizon Gymnasium-style fixture."""

    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.position = 0
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.position = 0
        self.steps = 0
        return self.position, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        self.position += action + 1
        terminated = self.position >= 4
        truncated = not terminated and self.steps >= 3
        reward = 1.0 if terminated else -0.05
        return self.position, reward, terminated, truncated, {
            "normalized_cost": 1.0
        }


def main() -> None:
    env = TinyLineEnv()
    sim_env = MCTSEnvWrapper(env)  # safe whole-environment copy per search
    trace = ListTraceSink()
    config = MCTSConfig(discount=0.99, max_tree_depth=8)
    agent = MCTSAgent(
        budget=SearchBudget(
            max_cost=48.0,
            max_tokens=0,
            max_accurate_calls=0,
            max_iterations=24,
            max_environment_calls=48,
        ),
        seed=7,
        tree_policy=UCTTreePolicy(exploration_constant=1.2),
        expander=LegalActionExpander(),
        evaluator=RandomRolloutEvaluator(max_depth=8, discount=config.discount),
        backup=MeanBackup(discount=config.discount),
        action_selector=MostVisitedActionSelector(),
        state_codec=DefaultStateCodec(),
        trace_sink=trace,
        config=config,
    )

    observation, info = env.reset(seed=7)
    del info
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.compute_action(sim_env, observation)
        report = agent.last_report
        assert report is not None
        print(
            f"state={observation} action={action} "
            f"iterations={report.usage.iterations} "
            f"calls={report.usage.environment_calls} "
            f"cost={report.usage.cost:.0f}"
        )
        observation, reward, terminated, truncated, info = env.step(action)
        agent.observe(
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    ending = "terminated" if terminated else "truncated"
    print(f"episode {ending}; structured iteration traces={len(trace.records)}")


if __name__ == "__main__":
    main()
