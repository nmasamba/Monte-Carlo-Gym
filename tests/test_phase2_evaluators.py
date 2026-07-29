from __future__ import annotations

import unittest
from collections.abc import Sequence
from random import Random
from typing import Any

from fixture_envs import TwoStepEnv

from montecarlgym import MCTSEnvWrapper
from montecarlgym.core.path import Evaluation
from montecarlgym.core.tree import ActionEdge, StateNode
from montecarlgym.evaluators import (
    DirectValueEvaluator,
    MixedEvaluator,
    PolicyValueEvaluator,
    PolicyValuePrediction,
)
from montecarlgym.policies.tree_policies import PUCTTreePolicy
from montecarlgym.presets import alphago_zero_preset
from montecarlgym.types import Action, SearchBudget


class StaticPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self,
        state: Any,
        legal_actions: Sequence[Action],
    ) -> PolicyValuePrediction:
        self.calls += 1
        priors = {action: (9.0 if action == 1 else 1.0) for action in legal_actions}
        return PolicyValuePrediction(
            value=float(state),
            priors=priors,
            model_version="fixture-v1",
        )


class NoStepContext:
    def __init__(self) -> None:
        self.steps = 0

    def legal_actions(self, observation: Any | None = None) -> tuple[int, ...]:
        del observation
        return (0, 1)

    def try_step(self, action: Action) -> None:
        del action
        self.steps += 1
        raise AssertionError("direct evaluation must not step the model")


class ConstantEvaluator:
    def __init__(self, value: float, actions: tuple[Action, ...] = ()) -> None:
        self.value = value
        self.actions = actions

    def evaluate(
        self,
        frontier: StateNode,
        model: NoStepContext,
        rng: Random,
    ) -> Evaluation:
        del frontier, model, rng
        return Evaluation(self.value, rollout_actions=self.actions)


def search_budget(iterations: int = 30) -> SearchBudget:
    return SearchBudget(
        max_cost=100.0,
        max_tokens=0,
        max_accurate_calls=0,
        max_iterations=iterations,
        max_environment_calls=100,
    )


class PUCTAndEvaluatorTests(unittest.TestCase):
    def test_puct_responds_monotonically_to_prior(self) -> None:
        node = StateNode("root", 0, visits=20)
        low = ActionEdge("low", visits=10, prior=0.1)
        high = ActionEdge("high", visits=10, prior=0.9)
        node.edges = {low.action: low, high.action: high}

        selected = PUCTTreePolicy().select(node, Random(2))

        self.assertIs(selected, high)

    def test_policy_value_expansion_normalizes_and_caches_priors(self) -> None:
        predictor = StaticPredictor()
        evaluator = PolicyValueEvaluator(predictor)
        node = StateNode("root", 3)

        evaluator.expand(node, (0, 1))
        evaluator.expand(node, (0, 1))

        self.assertEqual(predictor.calls, 1)
        self.assertAlmostEqual(node.edges[0].prior or 0.0, 0.1)
        self.assertAlmostEqual(node.edges[1].prior or 0.0, 0.9)
        self.assertEqual(node.value_estimate, 3.0)

    def test_direct_value_evaluator_performs_no_rollout(self) -> None:
        context = NoStepContext()
        evaluator = DirectValueEvaluator(lambda state: float(state) + 0.5)

        result = evaluator.evaluate(StateNode("s", 2), context, Random(1))

        self.assertEqual(result.value, 2.5)
        self.assertEqual(result.stop_reason, "direct_value")
        self.assertEqual(context.steps, 0)

    def test_mixed_evaluator_is_convex_and_retains_actions(self) -> None:
        evaluator = MixedEvaluator(
            ConstantEvaluator(4.0, ("a",)),
            ConstantEvaluator(0.0, ("b",)),
            first_weight=0.25,
        )

        result = evaluator.evaluate(
            StateNode("s", 0),
            NoStepContext(),
            Random(1),
        )

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.rollout_actions, ("a", "b"))

    def test_alphago_zero_preset_runs_without_algorithm_branch(self) -> None:
        env = TwoStepEnv()
        observation, _ = env.reset()
        predictor = StaticPredictor()
        agent = alphago_zero_preset(
            budget=search_budget(),
            predictor=predictor,
            seed=4,
        )

        action = agent.compute_action(MCTSEnvWrapper(env), observation)

        self.assertEqual(action, 1)
        assert agent.tree is not None
        self.assertGreater(agent.tree.root.edges[1].visits, 0)
        self.assertEqual(env.state, 0)


if __name__ == "__main__":
    unittest.main()
