from __future__ import annotations

import unittest
from dataclasses import dataclass
from random import Random
from typing import Any

from montecarlgym import MCTSEnvWrapper
from montecarlgym.adaptive import AdaptiveComputePlanner, AdaptiveFrontierEvaluator
from montecarlgym.agent import MCTSAgent
from montecarlgym.models import ModelPortfolio
from montecarlgym.routing import CheapOnlyRouter
from montecarlgym.types import (
    Action,
    EvidenceProvenance,
    Fidelity,
    ModelObservation,
    ModelQuote,
    SearchBudget,
)
from tests.fixture_envs import TwoStepEnv


@dataclass(frozen=True, slots=True)
class TwoActionProvider:
    def legal_actions(self, state: Any) -> tuple[int, int]:
        del state
        return (0, 1)


@dataclass(slots=True)
class FrontierValueModel:
    calls: int = 0

    @property
    def model_id(self) -> str:
        return "frontier-values"

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.CHEAP

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del token_budget, rollout_depth
        return ModelQuote(cost=0.1)

    def evaluate(
        self,
        state: Any,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del state, token_budget, rollout_depth, rng
        self.calls += 1
        return ModelObservation(
            value=float(action),
            variance=0.01,
            cost=0.1,
            provenance=EvidenceProvenance.LEARNED,
        )


def search_budget(**overrides: object) -> SearchBudget:
    values: dict[str, object] = {
        "max_cost": 10.0,
        "max_tokens": 0,
        "max_accurate_calls": 0,
        "max_iterations": 3,
        "max_model_calls": 9,
        "max_environment_calls": 3,
    }
    values.update(overrides)
    return SearchBudget(**values)  # type: ignore[arg-type]


class AdaptiveFrontierTests(unittest.TestCase):
    def _agent(self, budget: SearchBudget) -> tuple[MCTSAgent, FrontierValueModel]:
        value_model = FrontierValueModel()
        planner = AdaptiveComputePlanner(
            action_provider=TwoActionProvider(),
            router=CheapOnlyRouter(value_model.model_id),
        )
        evaluator = AdaptiveFrontierEvaluator(
            planner,
            ModelPortfolio.from_models((value_model,)),
            SearchBudget(
                max_cost=1.0,
                max_tokens=0,
                max_accurate_calls=0,
                max_iterations=2,
                max_model_calls=2,
                max_environment_calls=0,
            ),
        )
        return MCTSAgent(budget=budget, evaluator=evaluator, seed=19), value_model

    def test_adaptive_evidence_is_attached_to_frontier_edges(self) -> None:
        env = TwoStepEnv()
        observation, _ = env.reset()
        agent, value_model = self._agent(search_budget())

        action = agent.compute_action(MCTSEnvWrapper(env), observation)

        self.assertIn(action, (0, 1))
        self.assertEqual(env.state, 0)
        self.assertEqual(value_model.calls, 4)
        assert agent.last_report is not None
        self.assertEqual(agent.last_report.usage.iterations, 3)
        self.assertEqual(agent.last_report.usage.model_calls, 7)
        self.assertEqual(agent.last_report.usage.environment_calls, 3)
        assert agent.tree is not None
        frontier_children = [
            link.child
            for edge in agent.tree.root.edges.values()
            for link in edge.outcomes.values()
        ]
        self.assertTrue(
            all(
                child.statistics.get("last_adaptive_report")
                for child in frontier_children
            )
        )
        self.assertTrue(
            all(
                child_edge.evidence
                for child in frontier_children
                for child_edge in child.edges.values()
            )
        )

    def test_nested_planning_never_exceeds_outer_call_budget(self) -> None:
        env = TwoStepEnv()
        observation, _ = env.reset()
        budget = search_budget(max_iterations=20, max_model_calls=2)
        agent, value_model = self._agent(budget)

        agent.compute_action(MCTSEnvWrapper(env), observation)

        assert agent.last_report is not None
        self.assertLessEqual(
            agent.last_report.usage.model_calls,
            budget.max_model_calls or 0,
        )
        self.assertEqual(value_model.calls, 1)
        self.assertEqual(env.state, 0)


if __name__ == "__main__":
    unittest.main()
