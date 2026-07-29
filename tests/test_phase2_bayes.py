from __future__ import annotations

import unittest
from random import Random

from fixture_envs import OneStepBanditEnv

from montecarlgym import MCTSEnvWrapper
from montecarlgym.bayes import (
    REWARD_POSTERIOR,
    TRANSITION_POSTERIOR,
    BayesianBackup,
    DirichletTransitionPosterior,
    NormalGammaPosterior,
    PosteriorOutcome,
    RootSamplingTreePolicy,
    StateActionBelief,
    TabularRootBelief,
    TabularRootSamplingModel,
    ThompsonTreePolicy,
)
from montecarlgym.core.path import Evaluation, SearchPath
from montecarlgym.core.tree import ActionEdge, OutcomeLink, StateNode
from montecarlgym.presets import dng_mcts_preset, mcbrl_root_sampling_preset
from montecarlgym.types import SearchBudget


def budget(iterations: int = 20) -> SearchBudget:
    return SearchBudget(
        max_cost=100.0,
        max_tokens=0,
        max_accurate_calls=0,
        max_iterations=iterations,
        max_environment_calls=100,
    )


class ConjugatePosteriorTests(unittest.TestCase):
    def test_normal_gamma_update_matches_analytic_fixture(self) -> None:
        posterior = NormalGammaPosterior()

        posterior.update(2.0)

        self.assertEqual(posterior.observations, 1)
        self.assertAlmostEqual(posterior.mean, 1.0)
        self.assertAlmostEqual(posterior.precision_scale, 2.0)
        self.assertAlmostEqual(posterior.shape, 1.5)
        self.assertAlmostEqual(posterior.rate, 2.0)

    def test_dirichlet_sample_is_normalized_and_reproducible(self) -> None:
        posterior = DirichletTransitionPosterior()
        posterior.observe("left")
        posterior.observe("right")
        posterior.observe("right")

        first = posterior.sample(Random(7))
        second = posterior.sample(Random(7))

        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first.values()), 1.0)
        self.assertGreater(
            posterior.concentrations["right"],
            posterior.concentrations["left"],
        )

    def test_bayesian_backup_updates_local_edge_posteriors(self) -> None:
        root = StateNode("root", 0)
        leaf = StateNode("leaf", 1)
        edge = ActionEdge("go")
        outcome = OutcomeLink("outcome", 1.0, leaf)
        root.edges[edge.action] = edge
        path = SearchPath(root)
        path.append(root, edge, outcome)

        BayesianBackup().update(path, Evaluation(2.0))

        reward = edge.statistics[REWARD_POSTERIOR]
        transition = edge.statistics[TRANSITION_POSTERIOR]
        self.assertIsInstance(reward, NormalGammaPosterior)
        self.assertIsInstance(transition, DirichletTransitionPosterior)
        self.assertEqual(reward.observations, 1)
        self.assertEqual(reward.mean, 1.5)
        self.assertIn("outcome", transition.concentrations)


class PosteriorPolicyTests(unittest.TestCase):
    def test_thompson_prefers_separated_concentrated_posterior(self) -> None:
        node = StateNode("root", 0)
        low = ActionEdge("low", visits=5)
        high = ActionEdge("high", visits=5)
        low.statistics[REWARD_POSTERIOR] = NormalGammaPosterior(
            mean=-2.0,
            precision_scale=1_000.0,
            shape=100.0,
            rate=1.0,
        )
        high.statistics[REWARD_POSTERIOR] = NormalGammaPosterior(
            mean=2.0,
            precision_scale=1_000.0,
            shape=100.0,
            rate=1.0,
        )
        node.edges = {low.action: low, high.action: high}

        selected = ThompsonTreePolicy().select(node, Random(3))

        self.assertIs(selected, high)

    def test_root_sample_is_cached_only_within_iteration(self) -> None:
        edge = ActionEdge("a", visits=1)
        policy = RootSamplingTreePolicy(prioritize_unvisited=False)
        rng = Random(11)
        policy.start_iteration(StateNode("root", 0), rng)

        first = policy.sample_edge(edge, rng)
        cached = policy.sample_edge(edge, rng)
        policy.start_iteration(StateNode("root", 0), rng)
        next_iteration = policy.sample_edge(edge, rng)

        self.assertEqual(first, cached)
        self.assertNotEqual(first, next_iteration)

    def test_dng_updates_imaginary_tree_beliefs(self) -> None:
        env = OneStepBanditEnv()
        observation, _ = env.reset()
        dng = dng_mcts_preset(budget=budget(), seed=5)
        dng.compute_action(MCTSEnvWrapper(env), observation)
        assert dng.tree is not None
        for edge in dng.tree.root.edges.values():
            posterior = edge.statistics[REWARD_POSTERIOR]
            self.assertEqual(posterior.observations, edge.visits)

    def test_root_sampled_model_keeps_real_belief_frozen(self) -> None:
        state_actions = {
            0: {
                action: StateActionBelief(
                    outcomes={
                        "done": PosteriorOutcome(
                            action + 1,
                            terminated=True,
                        )
                    },
                    reward=NormalGammaPosterior(
                        mean=float(action),
                        precision_scale=1_000.0,
                        shape=100.0,
                        rate=1.0,
                    ),
                )
                for action in (0, 1)
            }
        }
        belief = TabularRootBelief(state_actions)
        model = TabularRootSamplingModel(0, belief)
        before = {
            action: (
                state_action.reward.copy(),
                state_action.transitions.copy(),
            )
            for action, state_action in state_actions[0].items()
        }
        root_sampled = mcbrl_root_sampling_preset(budget=budget(), seed=5)

        action = root_sampled.compute_action(model, 0)

        self.assertEqual(action, 1)
        for candidate, state_action in state_actions[0].items():
            old_reward, old_transitions = before[candidate]
            self.assertEqual(state_action.reward, old_reward)
            self.assertEqual(state_action.transitions, old_transitions)

        belief.observe(
            state=0,
            action=1,
            outcome_key="done",
            reward=1.0,
        )
        self.assertEqual(state_actions[0][1].reward.observations, 1)

    def test_sampled_mdp_is_fixed_within_one_simulation(self) -> None:
        belief = TabularRootBelief(
            {
                0: {
                    0: StateActionBelief(
                        outcomes={
                            "done": PosteriorOutcome(1, terminated=True)
                        }
                    )
                }
            }
        )
        model = TabularRootSamplingModel(0, belief)

        with model.transaction():
            model.seed_simulation(19)
            sampled_snapshot = model.snapshot()
            first = model.step(0)
            model.restore(sampled_snapshot)
            second = model.step(0)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
