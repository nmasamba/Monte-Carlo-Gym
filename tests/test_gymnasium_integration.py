from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from montecarlgym import MCTSEnvWrapper
from montecarlgym.config import MCTSConfig
from montecarlgym.experiments.gymnasium_frozenlake import run_frozenlake_study
from montecarlgym.experiments.preregistration import ExperimentStage
from montecarlgym.presets import uct_preset
from montecarlgym.types import SearchBudget

try:
    import gymnasium as gym
except ImportError:
    gym = None


@unittest.skipIf(gym is None, "Gymnasium optional dependency is not installed")
class GymnasiumIntegrationTests(unittest.TestCase):
    def test_frozen_lake_search_preserves_live_gymnasium_state(self) -> None:
        assert gym is not None
        env = gym.make("FrozenLake-v1", is_slippery=False)
        try:
            observation, _ = env.reset(seed=7)
            initial_state = env.unwrapped.s
            agent = uct_preset(
                budget=SearchBudget(
                    max_cost=48.0,
                    max_tokens=0,
                    max_accurate_calls=0,
                    max_iterations=8,
                    max_environment_calls=48,
                ),
                seed=7,
                config=MCTSConfig(max_tree_depth=12),
            )

            action = agent.compute_action(MCTSEnvWrapper(env), observation)

            self.assertTrue(env.action_space.contains(action))
            self.assertEqual(env.unwrapped.s, initial_state)
            self.assertEqual(env._elapsed_steps, 0)
        finally:
            env.close()

    def test_real_frozenlake_phase4_smoke_protocol_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads(
            (
                root
                / "experiments"
                / "pilots"
                / "frozenlake_smoke.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = run_frozenlake_study(
                protocol,
                Path(directory) / "pilot",
                stage=ExperimentStage.EXPLORATORY,
            )

        self.assertEqual(summary["stage"], "exploratory")
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["records"], 3)
        self.assertIn("0", summary["methods_by_budget"])
        self.assertIn("not a paper result", summary["notice"])


if __name__ == "__main__":
    unittest.main()
