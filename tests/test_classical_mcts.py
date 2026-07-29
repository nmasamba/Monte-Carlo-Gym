from __future__ import annotations

import unittest
from random import Random

from fixture_envs import (
    OneStepBanditEnv,
    StochasticOneStepEnv,
    TerminationModeEnv,
    TwoStepEnv,
)

from montecarlgym import MCTSEnvWrapper
from montecarlgym.agent import MCTSAgent
from montecarlgym.core.backup import MeanBackup
from montecarlgym.core.path import Evaluation, SearchPath
from montecarlgym.core.tree import ActionEdge, OutcomeLink, StateNode
from montecarlgym.policies.tree_policies import UCTTreePolicy
from montecarlgym.types import SearchBudget


def budget(**overrides: object) -> SearchBudget:
    values: dict[str, object] = {
        "max_cost": 200.0,
        "max_tokens": 0,
        "max_accurate_calls": 0,
        "max_iterations": 40,
        "max_model_calls": 200,
    }
    values.update(overrides)
    return SearchBudget(**values)  # type: ignore[arg-type]


class PolicyAndBackupTests(unittest.TestCase):
    def test_uct_prioritizes_unvisited_actions(self) -> None:
        node = StateNode("root", 0, visits=11)
        visited = ActionEdge("visited", visits=10, total_return=100.0, mean_value=10.0)
        unvisited = ActionEdge("new")
        node.edges = {visited.action: visited, unvisited.action: unvisited}

        selected = UCTTreePolicy().select(node, Random(1))

        self.assertIs(selected, unvisited)

    def test_uct_prefers_higher_value_at_equal_sufficient_visits(self) -> None:
        node = StateNode("root", 0, visits=40)
        low = ActionEdge("low", visits=20, total_return=0.0, mean_value=0.0)
        high = ActionEdge("high", visits=20, total_return=20.0, mean_value=1.0)
        node.edges = {low.action: low, high.action: high}

        selected = UCTTreePolicy().select(node, Random(1))

        self.assertIs(selected, high)

    def test_mean_backup_updates_n_w_q(self) -> None:
        root = StateNode("root", "root")
        child = StateNode("child", "child")
        edge = ActionEdge("go")
        outcome = OutcomeLink("outcome", reward=1.0, child=child)
        root.edges[edge.action] = edge
        edge.outcomes[outcome.outcome_key] = outcome
        path = SearchPath(root)
        path.append(root, edge, outcome)
        backup = MeanBackup(discount=0.5)

        backup.update(path, Evaluation(3.0))
        backup.update(path, Evaluation(1.0))

        self.assertEqual(edge.N, 2)
        self.assertAlmostEqual(edge.W, 4.0)
        self.assertAlmostEqual(edge.Q, 2.0)
        self.assertEqual(outcome.visits, 2)


class ClassicalSearchTests(unittest.TestCase):
    def test_bandit_converges_and_returns_legal_action(self) -> None:
        env = OneStepBanditEnv()
        observation, _ = env.reset()
        agent = MCTSAgent(budget=budget(), seed=7)

        action = agent.compute_action(MCTSEnvWrapper(env), observation)

        self.assertEqual(action, 1)
        self.assertIn(action, range(env.action_space.n))
        self.assertEqual(env.calls, 0)

    def test_root_edge_visits_equal_completed_backups(self) -> None:
        env = OneStepBanditEnv()
        observation, _ = env.reset()
        agent = MCTSAgent(budget=budget(max_iterations=17), seed=3)
        agent.compute_action(MCTSEnvWrapper(env), observation)
        assert agent.tree is not None
        assert agent.last_report is not None

        edge_visits = sum(edge.visits for edge in agent.tree.root.edges.values())
        self.assertEqual(edge_visits, 17)
        self.assertEqual(edge_visits, agent.tree.root.visits)
        self.assertEqual(edge_visits, agent.last_report.usage.iterations)

    def test_hard_iteration_and_call_budgets_are_not_exceeded(self) -> None:
        env = OneStepBanditEnv()
        observation, _ = env.reset()
        agent = MCTSAgent(
            budget=budget(
                max_iterations=50,
                max_model_calls=3,
                max_environment_calls=4,
                max_cost=3.0,
            ),
            seed=5,
        )

        agent.compute_action(MCTSEnvWrapper(env), observation)
        assert agent.last_report is not None
        usage = agent.last_report.usage

        self.assertLessEqual(usage.iterations, 50)
        self.assertLessEqual(usage.model_calls, 3)
        self.assertLessEqual(usage.environment_calls, 4)
        self.assertLessEqual(usage.cost, 3.0)

    def test_search_graph_preserves_termination_modes(self) -> None:
        env = TerminationModeEnv()
        observation, _ = env.reset()
        agent = MCTSAgent(budget=budget(max_iterations=2), seed=9)

        agent.compute_action(MCTSEnvWrapper(env), observation)
        assert agent.tree is not None
        terminated = next(iter(agent.tree.root.edges[0].outcomes.values()))
        truncated = next(iter(agent.tree.root.edges[1].outcomes.values()))

        self.assertTrue(terminated.terminated)
        self.assertFalse(terminated.truncated)
        self.assertFalse(truncated.terminated)
        self.assertTrue(truncated.truncated)

    def test_stochastic_rewards_create_distinct_outcome_links(self) -> None:
        env = StochasticOneStepEnv()
        observation, _ = env.reset(seed=23)
        before_rng = env.rng.getstate()
        agent = MCTSAgent(budget=budget(max_iterations=12), seed=29)

        agent.compute_action(MCTSEnvWrapper(env), observation)
        assert agent.tree is not None
        edge = agent.tree.root.edges[0]

        self.assertGreater(len(edge.outcomes), 1)
        self.assertEqual(sum(link.visits for link in edge.outcomes.values()), 12)
        self.assertEqual(env.rng.getstate(), before_rng)

    def test_subtree_reuse_after_matching_real_transition(self) -> None:
        env = TwoStepEnv()
        observation, _ = env.reset()
        wrapper = MCTSEnvWrapper(env)
        agent = MCTSAgent(budget=budget(max_iterations=20), seed=13)
        action = agent.compute_action(wrapper, observation)
        assert agent.tree is not None
        old_tree = agent.tree
        real_transition = env.step(action)
        next_observation, reward, terminated, truncated, info = real_transition
        expected = old_tree.matching_outcome(
            action=action,
            observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        self.assertIsNotNone(expected)

        agent.observe(
            action=action,
            observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        assert expected is not None
        self.assertIs(agent.tree, old_tree)
        self.assertIs(agent.tree.root, expected.child)
        self.assertTrue(agent.last_transition_reused)

    def test_episode_reset_does_not_reuse_stale_state(self) -> None:
        env = OneStepBanditEnv()
        observation, _ = env.reset()
        wrapper = MCTSEnvWrapper(env)
        agent = MCTSAgent(budget=budget(max_iterations=8), seed=17)
        action = agent.compute_action(wrapper, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.observe(
            action=action,
            observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        stale_root = agent.tree.root if agent.tree is not None else None

        reset_observation, _ = env.reset()
        agent.compute_action(wrapper, reset_observation)

        assert agent.tree is not None
        self.assertIsNot(agent.tree.root, stale_root)
        self.assertFalse(agent.last_transition_reused)


if __name__ == "__main__":
    unittest.main()
