from __future__ import annotations

import random
import unittest

from fixture_envs import (
    ExplodingEnv,
    StatefulRandomEnv,
    TerminationModeEnv,
    UnsafeCopyEnv,
)
from montecarlgym import MCTSEnvWrapper, NativeSnapshotStrategy, SnapshotError
from montecarlgym.agent import MCTSAgent
from montecarlgym.types import SearchBudget


def budget(**overrides: object) -> SearchBudget:
    values: dict[str, object] = {
        "max_cost": 20.0,
        "max_tokens": 0,
        "max_accurate_calls": 0,
        "max_iterations": 5,
        "max_model_calls": 20,
    }
    values.update(overrides)
    return SearchBudget(**values)  # type: ignore[arg-type]


class GymWrapperTests(unittest.TestCase):
    def test_snapshot_restore_reproduces_state_and_rng(self) -> None:
        env = StatefulRandomEnv()
        env.reset(seed=17)
        wrapper = MCTSEnvWrapper(env)
        snapshot = wrapper.snapshot()
        expected_rng = random.Random()
        expected_rng.setstate(env.rng.getstate())
        expected_draw = expected_rng.random()

        env.step(1)
        wrapper.restore(snapshot)

        self.assertEqual((env.state, env.steps), (0, 0))
        self.assertEqual(env.rng.random(), expected_draw)

    def test_transaction_restores_global_and_environment_rng(self) -> None:
        env = StatefulRandomEnv()
        env.reset(seed=5)
        wrapper = MCTSEnvWrapper(env)
        env_state = env.get_state()
        random.seed(1234)
        global_state = random.getstate()

        with wrapper.transaction():
            random.random()
            wrapper.step(1)

        self.assertEqual(env.get_state(), env_state)
        self.assertEqual(random.getstate(), global_state)

    def test_compute_action_does_not_mutate_live_environment(self) -> None:
        env = StatefulRandomEnv()
        observation, _ = env.reset(seed=11)
        before = env.get_state()
        agent = MCTSAgent(budget=budget(max_iterations=8), seed=4)

        action = agent.compute_action(MCTSEnvWrapper(env), observation)

        self.assertIn(action, (0, 1))
        self.assertEqual(env.get_state(), before)

    def test_terminated_and_truncated_remain_distinct(self) -> None:
        env = TerminationModeEnv()
        wrapper = MCTSEnvWrapper(env)
        with wrapper.transaction():
            root = wrapper.snapshot()
            transition = wrapper.step(0)
            self.assertEqual(transition[2:4], (True, False))
            wrapper.restore(root)
            transition = wrapper.step(1)
            self.assertEqual(transition[2:4], (False, True))

    def test_exception_during_native_simulation_restores_live_state(self) -> None:
        env = ExplodingEnv()
        observation, _ = env.reset(seed=19)
        before = env.get_state()
        strategy = NativeSnapshotStrategy(
            lambda current: current.get_state(),
            lambda current, state: current.set_state(state),
        )
        wrapper = MCTSEnvWrapper(env, strategy=strategy)
        agent = MCTSAgent(budget=budget(max_iterations=1), seed=2)

        with self.assertRaisesRegex(RuntimeError, "fixture simulation failed"):
            agent.compute_action(wrapper, observation)

        self.assertEqual(env.get_state(), before)

    def test_unsafe_partial_deepcopy_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "live object"):
            MCTSEnvWrapper(UnsafeCopyEnv())


if __name__ == "__main__":
    unittest.main()
