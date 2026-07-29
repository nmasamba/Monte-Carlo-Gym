from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from montecarlgym.adaptive import (
    BranchEvidence,
    CalibratedLinearDiscrepancyModel,
    DiscrepancyAwareAggregator,
    LinearEVCModel,
)
from montecarlgym.adaptive.routing import (
    LearnedEVCRouter,
    RandomEscalationRouter,
)
from montecarlgym.causal import (
    DoublyRobustEstimator,
    InversePropensityEstimator,
    LoggedRoutingDecision,
    SelfNormalizedIPSEstimator,
)
from montecarlgym.replay import JsonlVerifiedReplayStore, VerifiedTransition
from montecarlgym.routing import BranchSummary, RouterContext
from montecarlgym.types import (
    ComputeAction,
    EvidenceProvenance,
    Fidelity,
    ModelObservation,
    SearchBudget,
)


def features(value: float) -> dict[str, float]:
    return {
        "action_mean": value,
        "action_uncertainty": 0.2,
        "action_risk": 0.0,
        "gap_to_best": max(0.0, 1.0 - value),
        "evidence_count": 1.0,
        "search_depth": 1.0,
        "remaining_cost": 10.0,
        "remaining_accurate_calls": 2.0,
    }


def verified_record(value: float, discrepancy: float) -> VerifiedTransition:
    return VerifiedTransition(
        state_id=f"state-{value}",
        action=("branch", value),
        cheap_model_id="cheap",
        cheap_prediction=value,
        accurate_model_id="accurate",
        verified_outcome=value + discrepancy,
        router_propensity=0.25,
        cheap_provenance=EvidenceProvenance.LEARNED,
        accurate_provenance=EvidenceProvenance.EXECUTABLE,
        context_features=features(value),
        predicted_evc=abs(discrepancy),
        randomized_audit=True,
        metadata={"split": "router_training"},
    )


def router_context() -> RouterContext:
    budget = SearchBudget(
        max_cost=10.0,
        max_tokens=0,
        max_accurate_calls=2,
        max_iterations=10,
        max_model_calls=10,
        max_environment_calls=10,
    )
    observation = ModelObservation(value=0.5, variance=0.04, cost=0.1)
    return RouterContext(
        state_id="root",
        candidate_actions=("left", "right"),
        evidence={"left": (observation,), "right": (observation,)},
        remaining_budget=budget,
        search_depth=1,
        summaries={
            "left": BranchSummary(0.2, 0.04, 0.0, 1),
            "right": BranchSummary(0.8, 0.04, 0.0, 1),
        },
        query_counts={("left", "cheap"): 1, ("right", "cheap"): 1},
        feasible_model_ids=frozenset({"cheap", "accurate"}),
    )


class PersistentReplayTests(unittest.TestCase):
    def test_jsonl_round_trip_preserves_auditable_training_fields(self) -> None:
        record = verified_record(0.25, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verified.jsonl"
            store = JsonlVerifiedReplayStore(path, fsync=True)
            store.append(record)

            restored = JsonlVerifiedReplayStore(path).snapshot()

        self.assertEqual(restored, (record,))

    def test_jsonl_rejects_unsupported_actions_and_non_finite_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verified.jsonl"
            store = JsonlVerifiedReplayStore(path)
            invalid_action = verified_record(0.25, 0.5)
            invalid_action = VerifiedTransition(
                **{
                    field: getattr(invalid_action, field)
                    for field in invalid_action.__dataclass_fields__
                    if field != "action"
                },
                action={"not": "codec-safe"},
            )
            with self.assertRaises(TypeError):
                store.append(invalid_action)
            path.write_text('{"schema_version": 1, "value": NaN}\n')
            with self.assertRaisesRegex(ValueError, "line 1"):
                JsonlVerifiedReplayStore(path)


class ContextualLearningTests(unittest.TestCase):
    def test_evc_checkpoint_is_reproducible_and_calibrated(self) -> None:
        training = (
            verified_record(0.0, 0.1),
            verified_record(0.3, 0.4),
            verified_record(0.6, 0.7),
            verified_record(0.9, 1.0),
        )
        calibration = (verified_record(0.45, 0.55),)
        model = LinearEVCModel(ridge=0.01)
        model.fit(training, calibration_records=calibration, coverage=0.9)
        prediction = model.predict(features(0.5))
        assert prediction is not None
        self.assertGreaterEqual(prediction.mean, 0.0)
        self.assertGreater(prediction.variance, 0.0)
        self.assertEqual(prediction.training_examples, len(training))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evc.json"
            model.save(path)
            restored = LinearEVCModel.load(path).predict(features(0.5))
        self.assertEqual(restored, prediction)

    def test_contextual_discrepancy_changes_with_router_features(self) -> None:
        records = tuple(
            verified_record(value, 1.0 - value)
            for value in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        model = CalibratedLinearDiscrepancyModel(ridge=1e-6)
        model.fit(records)

        low = model.estimate_contextual("cheap", "accurate", features(0.9))
        high = model.estimate_contextual("cheap", "accurate", features(0.1))

        self.assertGreater(high.mean, low.mean)
        self.assertEqual(high.count, len(records))
        self.assertGreater(high.variance, 0.0)

    def test_branch_aggregation_uses_contextual_correction(self) -> None:
        records = tuple(
            verified_record(value, 1.0 - value)
            for value in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        model = CalibratedLinearDiscrepancyModel(ridge=1e-6)
        model.fit(records)
        aggregator = DiscrepancyAwareAggregator(model)
        evidence = BranchEvidence(
            compute_action=ComputeAction("state", "left", "cheap"),
            fidelity=Fidelity.CHEAP,
            provenance=EvidenceProvenance.LEARNED,
            observation=ModelObservation(value=0.1, variance=0.01, cost=0.1),
            query_index=1,
            context_features=features(0.1),
        )

        estimate = aggregator.estimate(
            "left",
            (evidence,),
            accurate_model_ids=("accurate",),
        )

        assert estimate is not None
        self.assertGreater(estimate.mean, 0.5)

    def test_evc_router_logs_exact_randomized_audit_propensity(self) -> None:
        model = LinearEVCModel(ridge=0.01)
        model.fit(
            tuple(
                verified_record(value, 0.5 + value)
                for value in (0.0, 0.3, 0.6, 0.9)
            )
        )
        router = LearnedEVCRouter(
            "cheap",
            "accurate",
            model,
            accurate_cost=1.0,
            cost_weight=0.0,
            audit_probability=1.0,
            seed=4,
        )

        decision = router.choose(router_context())

        assert decision is not None
        self.assertTrue(decision.audit)
        self.assertEqual(decision.route_propensity, 0.5)
        self.assertIsNotNone(decision.expected_value_of_compute)

    def test_evc_router_stops_when_compute_value_is_below_cost(self) -> None:
        model = LinearEVCModel(ridge=0.01)
        model.fit(tuple(verified_record(value, 0.01) for value in (0.0, 1.0)))
        router = LearnedEVCRouter(
            "cheap",
            "accurate",
            model,
            accurate_cost=10.0,
            cost_weight=1.0,
            audit_probability=0.0,
        )
        self.assertIsNone(router.choose(router_context()))

    def test_random_escalation_baseline_logs_its_probability(self) -> None:
        router = RandomEscalationRouter(
            "cheap",
            "accurate",
            escalation_probability=1.0,
            seed=3,
        )

        decision = router.choose(router_context())

        assert decision is not None
        self.assertTrue(decision.audit)
        self.assertEqual(decision.route_propensity, 0.5)
        self.assertEqual(decision.model_id, "accurate")


class OffPolicyEstimatorTests(unittest.TestCase):
    def test_ips_self_normalized_and_doubly_robust_match_fixture(self) -> None:
        records = (
            LoggedRoutingDecision(
                "a",
                "verify",
                reward=2.0,
                propensity=0.5,
                target_probability=0.25,
                baseline_prediction=1.0,
            ),
            LoggedRoutingDecision(
                "b",
                "skip",
                reward=0.0,
                propensity=0.25,
                target_probability=0.25,
                baseline_prediction=0.5,
            ),
        )

        self.assertAlmostEqual(InversePropensityEstimator().estimate(records), 0.5)
        self.assertAlmostEqual(
            SelfNormalizedIPSEstimator().estimate(records),
            2.0 / 3.0,
        )
        self.assertAlmostEqual(DoublyRobustEstimator().estimate(records), 0.75)


if __name__ == "__main__":
    unittest.main()
