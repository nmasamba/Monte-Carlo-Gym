"""Framework-neutral PUCT example using an injected policy/value fixture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from montecarlgym import (
    MCTSEnvWrapper,
    SearchBudget,
    alphago_zero_preset,
)
from montecarlgym.evaluators import PolicyValuePrediction
from montecarlgym.types import Action


class TinyPlanningEnv:
    def __init__(self) -> None:
        self.state = 0
        self.action_space = type("Discrete", (), {"n": 2})()

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.state = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.state += 1
        return self.state, float(action), self.state >= 2, False, {}


class PolicyValueFixture:
    def predict(
        self,
        state: Any,
        legal_actions: Sequence[Action],
    ) -> PolicyValuePrediction:
        return PolicyValuePrediction(
            value=0.25 * float(state),
            priors={action: (0.8 if action == 1 else 0.2) for action in legal_actions},
            model_version="offline-fixture-v1",
        )


def main() -> None:
    env = TinyPlanningEnv()
    observation, _ = env.reset(seed=7)
    agent = alphago_zero_preset(
        budget=SearchBudget(
            max_cost=40.0,
            max_tokens=0,
            max_accurate_calls=0,
            max_iterations=24,
            max_environment_calls=40,
        ),
        predictor=PolicyValueFixture(),
        seed=7,
    )
    action = agent.compute_action(MCTSEnvWrapper(env), observation)
    assert agent.last_report is not None
    print(
        f"selected={action} visits={dict(agent.last_report.action_visits)} "
        f"values={dict(agent.last_report.action_values)}"
    )


if __name__ == "__main__":
    main()
