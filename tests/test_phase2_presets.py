"""End-to-end conformance tests for reversible-environment presets."""

from __future__ import annotations

import unittest
from collections.abc import Callable, Sequence
from typing import Any

from fixture_envs import TwoStepEnv

from montecarlgym import MCTSAgent, MCTSEnvWrapper
from montecarlgym.core.backup import ConstantMixWeight
from montecarlgym.evaluators import PolicyValuePrediction
from montecarlgym.presets import (
    alphago_apv_preset,
    crazy_stone_mix_preset,
    crazy_stone_robust_preset,
    dng_mcts_preset,
    mast_preset,
    rave_mast_preset,
    rave_preset,
    uct_preset,
)
from montecarlgym.types import Action, SearchBudget


class FixturePredictor:
    def predict(
        self,
        state: Any,
        legal_actions: Sequence[Action],
    ) -> PolicyValuePrediction:
        del state
        priors = {
            action: float(index + 1)
            for index, action in enumerate(legal_actions)
        }
        return PolicyValuePrediction(
            value=0.25,
            priors=priors,
            model_version="phase2-fixture",
        )


def budget() -> SearchBudget:
    return SearchBudget(
        max_cost=64.0,
        max_tokens=0,
        max_accurate_calls=0,
        max_iterations=12,
        max_environment_calls=64,
    )


class Phase2PresetTests(unittest.TestCase):
    def test_reversible_environment_compositions_return_legal_actions(
        self,
    ) -> None:
        factories: dict[str, Callable[[], MCTSAgent]] = {
            "uct": lambda: uct_preset(budget=budget(), seed=1),
            "alphago_apv": lambda: alphago_apv_preset(
                budget=budget(),
                predictor=FixturePredictor(),
                seed=1,
            ),
            "dng": lambda: dng_mcts_preset(budget=budget(), seed=1),
            "crazy_stone_robust": lambda: crazy_stone_robust_preset(
                budget=budget(),
                seed=1,
            ),
            "crazy_stone_mix": lambda: crazy_stone_mix_preset(
                budget=budget(),
                schedule=ConstantMixWeight(0.5),
                seed=1,
            ),
            "rave": lambda: rave_preset(budget=budget(), seed=1),
            "mast": lambda: mast_preset(budget=budget(), seed=1),
            "rave_mast": lambda: rave_mast_preset(budget=budget(), seed=1),
        }

        for name, factory in factories.items():
            with self.subTest(preset=name):
                env = TwoStepEnv()
                observation, _ = env.reset()
                agent = factory()

                action = agent.compute_action(
                    MCTSEnvWrapper(env),
                    observation,
                )

                self.assertIn(action, (0, 1))
                self.assertEqual(env.state, 0)
                assert agent.last_report is not None
                self.assertEqual(agent.last_report.usage.iterations, 12)


if __name__ == "__main__":
    unittest.main()
