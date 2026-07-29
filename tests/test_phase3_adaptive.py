from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from montecarlgym.adaptive import (
    AdaptiveComputePlanner,
    AdaptiveResourceLedger,
    FixedQueryStopPolicy,
    ModelEvaluationError,
    RunningDiscrepancyModel,
    VerificationError,
)
from montecarlgym.core.budget import ResourceQuoteExceeded
from montecarlgym.experiments.multifidelity_tree import (
    ExecutableTreeModel,
    ShallowTreeConfig,
    make_shallow_tree_planner,
    make_shallow_tree_portfolio,
    sample_shallow_tree,
)
from montecarlgym.experiments.runner import run_experiment
from montecarlgym.models import ModelPortfolio
from montecarlgym.routing import (
    AccurateOnlyRouter,
    CheapOnlyRouter,
    FixedCascadeRouter,
)
from montecarlgym.types import (
    Action,
    EvidenceProvenance,
    Fidelity,
    ModelObservation,
    ModelQuote,
    SearchBudget,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class FixtureState:
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class FixtureActionProvider:
    def legal_actions(self, state: FixtureState) -> tuple[str, ...]:
        return tuple(state.values)


@dataclass(frozen=True, slots=True)
class FixtureCodec:
    def key(self, state: FixtureState) -> tuple[tuple[str, float], ...]:
        return tuple(state.values.items())


@dataclass(slots=True)
class FixtureModel:
    _model_id: str
    _fidelity: Fidelity
    predictions: dict[str, float]
    cost: float
    tokens: int = 0
    environment_calls: int = 0
    verified: bool = False
    provenance: EvidenceProvenance = EvidenceProvenance.SYNTHETIC
    fail: bool = False
    calls: list[tuple[Action, int, int]] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def fidelity(self) -> Fidelity:
        return self._fidelity

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del token_budget, rollout_depth
        return ModelQuote(
            cost=self.cost,
            tokens=self.tokens,
            accurate_calls=int(self.fidelity is Fidelity.ACCURATE),
            environment_calls=self.environment_calls,
        )

    def evaluate(
        self,
        state: FixtureState,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del state, rng
        self.calls.append((action, token_budget, rollout_depth))
        if self.fail:
            raise RuntimeError("fixture failure")
        return ModelObservation(
            value=self.predictions[str(action)],
            variance=0.04 if self.fidelity is Fidelity.CHEAP else 0.0,
            cost=self.cost,
            tokens=self.tokens,
            environment_calls=self.environment_calls,
            provenance=self.provenance,
            verified=self.verified,
        )


def fixture_budget(**overrides: Any) -> SearchBudget:
    settings: dict[str, Any] = {
        "max_cost": 20.0,
        "max_tokens": 20,
        "max_accurate_calls": 2,
        "max_iterations": 8,
        "max_model_calls": 8,
        "max_environment_calls": 4,
    }
    settings.update(overrides)
    return SearchBudget(**settings)


class AdaptiveLedgerTests(unittest.TestCase):
    def test_every_hard_resource_limit_is_enforced(self) -> None:
        budget = fixture_budget(
            max_cost=2.0,
            max_tokens=3,
            max_accurate_calls=1,
            max_iterations=1,
            max_model_calls=1,
            max_environment_calls=2,
        )
        ledger = AdaptiveResourceLedger(budget)
        quote = ModelQuote(
            cost=2.0,
            tokens=3,
            accurate_calls=1,
            environment_calls=2,
        )
        reservation = ledger.reserve(quote)
        self.assertIsNotNone(reservation)
        assert reservation is not None
        ledger.commit(
            reservation,
            ModelObservation(
                value=1.0,
                variance=0.0,
                cost=1.5,
                tokens=2,
                environment_calls=2,
            ),
        )

        self.assertIsNone(ledger.reserve(ModelQuote(cost=0.0)))
        usage = ledger.usage()
        self.assertEqual(usage.iterations, 1)
        self.assertEqual(usage.model_calls, 1)
        self.assertEqual(usage.environment_calls, 2)
        self.assertEqual(usage.accurate_calls, 1)
        self.assertEqual(usage.tokens, 2)
        self.assertEqual(usage.cost, 1.5)

    def test_quote_overrun_fails_instead_of_exceeding_search_budget(self) -> None:
        ledger = AdaptiveResourceLedger(fixture_budget(max_cost=1.0))
        reservation = ledger.reserve(ModelQuote(cost=1.0, tokens=1))
        assert reservation is not None
        with self.assertRaises(ResourceQuoteExceeded):
            ledger.commit(
                reservation,
                ModelObservation(
                    value=0.0,
                    variance=0.0,
                    cost=1.1,
                    tokens=1,
                ),
            )
        self.assertEqual(ledger.usage().cost, 1.0)
        self.assertEqual(ledger.stop_reason, "quote_overrun")

    def test_deadline_blocks_a_query_that_cannot_finish_in_time(self) -> None:
        now = [10.0]
        ledger = AdaptiveResourceLedger(
            fixture_budget(deadline_s=0.5),
            clock=lambda: now[0],
        )
        self.assertIsNone(
            ledger.reserve(ModelQuote(cost=0.0, expected_latency_s=0.6))
        )
        self.assertEqual(ledger.usage().model_calls, 0)
        self.assertEqual(ledger.stop_reason, "deadline")

    def test_invalid_negative_quote_fails_before_model_execution(self) -> None:
        ledger = AdaptiveResourceLedger(fixture_budget())
        with self.assertRaisesRegex(ValueError, "quote.cost"):
            ledger.reserve(ModelQuote(cost=-1.0))


class AdaptivePlannerTests(unittest.TestCase):
    def _portfolio(self) -> tuple[ModelPortfolio, FixtureModel, FixtureModel]:
        cheap = FixtureModel(
            "cheap",
            Fidelity.CHEAP,
            {"left": 0.4, "right": 0.2},
            cost=1.0,
            tokens=2,
            provenance=EvidenceProvenance.LEARNED,
        )
        accurate = FixtureModel(
            "accurate",
            Fidelity.ACCURATE,
            {"left": 0.9, "right": 0.1},
            cost=4.0,
            environment_calls=2,
            verified=True,
            provenance=EvidenceProvenance.EXECUTABLE,
        )
        return ModelPortfolio.from_models([cheap, accurate]), cheap, accurate

    def test_compute_queries_do_not_mutate_or_execute_task_state(self) -> None:
        portfolio, cheap, _ = self._portfolio()
        state = FixtureState({"left": 0.9, "right": 0.1})
        before = copy.deepcopy(state)
        planner = AdaptiveComputePlanner(
            action_provider=FixtureActionProvider(),
            router=CheapOnlyRouter("cheap", token_budget=7, rollout_depth=3),
            codec=FixtureCodec(),
        )
        result = planner.plan(
            state,
            models=portfolio,
            budget=fixture_budget(),
            seed=11,
        )

        self.assertEqual(state, before)
        self.assertIn(result.action, state.values)
        self.assertEqual(
            cheap.calls,
            [("left", 7, 3), ("right", 7, 3)],
        )
        self.assertTrue(
            all(record["task_action"] in state.values for record in result.trace)
        )

    def test_fixed_cascade_updates_verified_replay_and_discrepancy(self) -> None:
        portfolio, _, _ = self._portfolio()
        discrepancy = RunningDiscrepancyModel()
        planner = AdaptiveComputePlanner(
            action_provider=FixtureActionProvider(),
            router=FixedCascadeRouter(
                "cheap",
                "accurate",
                top_k=1,
                cheap_token_budget=2,
                accurate_rollout_depth=2,
            ),
            discrepancy=discrepancy,
            codec=FixtureCodec(),
        )
        result = planner.plan(
            FixtureState({"left": 0.9, "right": 0.1}),
            models=portfolio,
            budget=fixture_budget(),
            seed=3,
        )

        self.assertEqual(result.action, "left")
        self.assertEqual(result.usage.model_calls, 3)
        self.assertEqual(result.usage.environment_calls, 2)
        assert result.report is not None
        self.assertEqual(result.report.replay_records_added, 1)
        record = planner.replay.snapshot()[0]
        self.assertEqual(record.cheap_provenance, EvidenceProvenance.LEARNED)
        self.assertEqual(
            record.accurate_provenance,
            EvidenceProvenance.EXECUTABLE,
        )
        estimate = discrepancy.estimate("cheap", "accurate")
        self.assertEqual(estimate.count, 1)
        self.assertAlmostEqual(estimate.mean, 0.5)

    def test_fixed_stop_policy_stops_after_exact_query_count(self) -> None:
        portfolio, _, _ = self._portfolio()
        planner = AdaptiveComputePlanner(
            action_provider=FixtureActionProvider(),
            router=CheapOnlyRouter("cheap", samples_per_action=4),
            stop_policy=FixedQueryStopPolicy(3),
            codec=FixtureCodec(),
        )
        result = planner.plan(
            FixtureState({"left": 0.9, "right": 0.1}),
            models=portfolio,
            budget=fixture_budget(),
            seed=5,
        )
        self.assertEqual(result.usage.iterations, 3)
        assert result.report is not None
        self.assertEqual(result.report.stop_reason, "stop_policy")

    def test_failed_model_call_is_charged_and_not_converted_to_zero(self) -> None:
        failing = FixtureModel(
            "failing",
            Fidelity.CHEAP,
            {"left": 1.0},
            cost=2.5,
            tokens=3,
            fail=True,
        )
        planner = AdaptiveComputePlanner(
            action_provider=FixtureActionProvider(),
            router=CheapOnlyRouter("failing"),
            codec=FixtureCodec(),
        )
        with self.assertRaises(ModelEvaluationError) as caught:
            planner.plan(
                FixtureState({"left": 1.0}),
                models=ModelPortfolio.from_models([failing]),
                budget=fixture_budget(),
                seed=7,
            )
        self.assertEqual(caught.exception.usage.cost, 2.5)
        self.assertEqual(caught.exception.usage.tokens, 3)
        self.assertEqual(caught.exception.usage.model_calls, 1)

    def test_unverified_response_cannot_satisfy_verification_route(self) -> None:
        unverified = FixtureModel(
            "accurate",
            Fidelity.ACCURATE,
            {"left": 1.0},
            cost=2.0,
            verified=False,
            provenance=EvidenceProvenance.EXECUTABLE,
        )
        planner = AdaptiveComputePlanner(
            action_provider=FixtureActionProvider(),
            router=AccurateOnlyRouter("accurate"),
            codec=FixtureCodec(),
        )
        with self.assertRaises(VerificationError):
            planner.plan(
                FixtureState({"left": 1.0}),
                models=ModelPortfolio.from_models([unverified]),
                budget=fixture_budget(),
                seed=9,
            )
        self.assertEqual(len(planner.replay), 0)

    def test_planner_stops_before_an_unaffordable_accurate_call(self) -> None:
        portfolio, _, accurate = self._portfolio()
        planner = AdaptiveComputePlanner(
            action_provider=FixtureActionProvider(),
            router=FixedCascadeRouter("cheap", "accurate", top_k=1),
            codec=FixtureCodec(),
        )
        budget = fixture_budget(max_cost=2.0, max_environment_calls=0)
        result = planner.plan(
            FixtureState({"left": 0.9, "right": 0.1}),
            models=portfolio,
            budget=budget,
            seed=13,
        )
        self.assertLessEqual(result.usage.cost, budget.max_cost)
        self.assertEqual(result.usage.environment_calls, 0)
        self.assertEqual(accurate.calls, [])


class Phase3BenchmarkTests(unittest.TestCase):
    def test_executable_rollout_preserves_termination_and_truncation(self) -> None:
        config = ShallowTreeConfig(actions=2, horizon=3)
        task = sample_shallow_tree(config, 17)
        model = ExecutableTreeModel(config)
        partial = model.evaluate(
            task,
            "a0",
            token_budget=0,
            rollout_depth=2,
            rng=Random(1),
        )
        complete = model.evaluate(
            task,
            "a0",
            token_budget=0,
            rollout_depth=3,
            rng=Random(1),
        )
        self.assertFalse(partial.terminated)
        self.assertTrue(partial.truncated)
        self.assertFalse(partial.verified)
        self.assertTrue(complete.terminated)
        self.assertFalse(complete.truncated)
        self.assertTrue(complete.verified)

    def test_executable_only_selects_optimum_deterministically(self) -> None:
        config = ShallowTreeConfig(actions=4, horizon=3)
        task = sample_shallow_tree(config, 23)
        planner = make_shallow_tree_planner(
            "phase3_executable_only", config, {}
        )
        result = planner.plan(
            task,
            models=make_shallow_tree_portfolio(config),
            budget=SearchBudget(
                max_cost=24.0,
                max_tokens=0,
                max_accurate_calls=4,
                max_iterations=4,
                max_model_calls=4,
                max_environment_calls=12,
            ),
            seed=29,
        )
        self.assertEqual(result.action, task.optimal_action)
        self.assertEqual(result.usage.environment_calls, 12)

    def test_phase3_experiment_is_reproducible_and_labelled_diagnostic(self) -> None:
        config = json.loads(
            (ROOT / "experiments" / "configs" / "phase3_tree.json").read_text(
                encoding="utf-8"
            )
        )
        config["run"]["repetitions"] = 5
        with tempfile.TemporaryDirectory() as first_dir:
            first = run_experiment(config, Path(first_dir))
        with tempfile.TemporaryDirectory() as second_dir:
            second = run_experiment(config, Path(second_dir))

        self.assertEqual(first, second)
        self.assertIn("diagnostic", first["notice"])
        self.assertNotIn("paper result", first["notice"].lower())
        for method in first["methods"].values():
            self.assertLessEqual(method["mean_cost"], config["budget"]["max_cost"])


if __name__ == "__main__":
    unittest.main()
