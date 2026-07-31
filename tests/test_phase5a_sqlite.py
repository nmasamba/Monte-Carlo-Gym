from __future__ import annotations

import json
import stat
import tempfile
import unittest
import warnings
from pathlib import Path
from random import Random

from montecarlgym.causal import LoggedRoutingDecision
from montecarlgym.core.tree import StateNode
from montecarlgym.experiments.analysis import (
    HypervolumeReference,
    calibration_metrics,
    hypervolume_2d,
    off_policy_diagnostics,
)
from montecarlgym.experiments.preregistration import (
    validate_preregistration_protocol,
)
from montecarlgym.experiments.sqlite_l2 import (
    FUTURE_CONFIRMATORY_RESERVATION,
    PRIMARY_SQLITE_METHODS,
    SQLCandidate,
    SQLiteCheapModel,
    SQLiteExecutableModel,
    SQLiteExecutionStatus,
    SQLitePartition,
    SQLiteSandbox,
    SQLiteTask,
    load_sqlite_partition,
    make_sqlite_frontier_evaluator,
    make_sqlite_planner,
    make_sqlite_portfolio,
    paired_sqlite_task,
    partition_manifest,
    score_sqlite_action,
    train_sqlite_components,
)
from montecarlgym.experiments.sqlite_study import (
    matched_budget_diagnostic,
    run_sqlite_study,
    validate_exploratory_output,
    validate_raw_record,
)
from montecarlgym.types import EvidenceProvenance, SearchBudget

ROOT = Path(__file__).resolve().parents[1]


def budget(**overrides: object) -> SearchBudget:
    values: dict[str, object] = {
        "max_cost": 9.0,
        "max_tokens": 32,
        "max_accurate_calls": 2,
        "max_iterations": 6,
        "deadline_s": 2.0,
        "max_model_calls": 6,
        "max_environment_calls": 2,
    }
    values.update(overrides)
    return SearchBudget(**values)  # type: ignore[arg-type]


class SQLiteVerifierTests(unittest.TestCase):
    def test_every_materialized_fixture_has_one_objective_solution(self) -> None:
        for partition in (
            SQLitePartition.DEVELOPMENT,
            SQLitePartition.CALIBRATION,
            SQLitePartition.EXPLORATORY,
        ):
            for task in load_sqlite_partition(partition):
                passing = [
                    action
                    for action in task.actions
                    if score_sqlite_action(task, action).verifier_passed
                ]
                self.assertEqual(passing, ["q0"], task.task_id)

    def test_transaction_restores_template_after_success_and_rejection(self) -> None:
        task = load_sqlite_partition(SQLitePartition.EXPLORATORY)[0]
        sandbox = SQLiteSandbox(task)
        checksum = sandbox.fixture_checksum()
        try:
            success = sandbox.execute(task.candidate("q0").sql)
            rejected = sandbox.execute("DELETE FROM customers")
            self.assertTrue(success.verifier_passed)
            self.assertEqual(rejected.status, SQLiteExecutionStatus.REJECTED)
            self.assertEqual(sandbox.fixture_checksum(), checksum)
            self.assertTrue(sandbox.fixture_intact)
        finally:
            sandbox.close()

    def test_timeout_is_truncated_negative_evidence_and_fully_charged(self) -> None:
        task = SQLiteTask(
            task_id="timeout-fixture",
            partition=SQLitePartition.DEVELOPMENT,
            template_family="timeout_fixture",
            prompt="Return one row.",
            setup_sql="CREATE TABLE x(value INTEGER); INSERT INTO x VALUES (1);",
            candidates=(
                SQLCandidate(
                    "loop",
                    "WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM t) "
                    "SELECT SUM(x) FROM t",
                ),
                SQLCandidate("safe", "SELECT value FROM x"),
            ),
            expected_columns=("value",),
            expected_rows=((1,),),
            semantic_tokens=("select",),
        )
        model = SQLiteExecutableModel(
            cost=4.0,
            maximum_vm_steps=100,
            timeout_s=0.1,
        )
        observation = model.evaluate(
            task,
            "loop",
            token_budget=0,
            rollout_depth=1,
            rng=Random(1),
        )
        self.assertEqual(observation.value, -1.0)
        self.assertEqual(observation.cost, 4.0)
        self.assertTrue(observation.verified)
        self.assertTrue(observation.truncated)
        self.assertFalse(observation.terminated)
        self.assertEqual(
            observation.metadata["execution_status"],
            SQLiteExecutionStatus.TIMEOUT.value,
        )

    def test_model_provenance_is_explicit(self) -> None:
        task = load_sqlite_partition(SQLitePartition.EXPLORATORY)[0]
        cheap = SQLiteCheapModel().evaluate(
            task,
            "q0",
            token_budget=8,
            rollout_depth=1,
            rng=Random(1),
        )
        accurate = SQLiteExecutableModel().evaluate(
            task,
            "q0",
            token_budget=0,
            rollout_depth=1,
            rng=Random(1),
        )
        self.assertEqual(cheap.provenance, EvidenceProvenance.MODEL_PREDICTED)
        self.assertFalse(cheap.verified)
        self.assertEqual(accurate.provenance, EvidenceProvenance.EXECUTABLE)
        self.assertTrue(accurate.verified)
        self.assertTrue(accurate.metadata["verifier_passed"])


class SQLitePartitionTests(unittest.TestCase):
    def test_splits_are_family_disjoint_and_confirmatory_is_unmaterialized(
        self,
    ) -> None:
        partitions = [
            load_sqlite_partition(SQLitePartition.DEVELOPMENT),
            load_sqlite_partition(SQLitePartition.CALIBRATION),
            load_sqlite_partition(SQLitePartition.EXPLORATORY),
        ]
        families = [{task.template_family for task in tasks} for tasks in partitions]
        self.assertFalse(families[0] & families[1])
        self.assertFalse(families[0] & families[2])
        self.assertFalse(families[1] & families[2])
        with self.assertRaises(PermissionError):
            load_sqlite_partition(SQLitePartition.FUTURE_CONFIRMATORY)
        manifest = partition_manifest()[SQLitePartition.FUTURE_CONFIRMATORY.value]
        self.assertIsNone(manifest["task_ids"])
        self.assertIsNone(manifest["seeds"])
        self.assertEqual(
            FUTURE_CONFIRMATORY_RESERVATION.status, "reserved_unmaterialized"
        )

    def test_paired_task_permutation_is_deterministic(self) -> None:
        task = load_sqlite_partition(SQLitePartition.EXPLORATORY)[0]
        first = paired_sqlite_task(task, 501)
        second = paired_sqlite_task(task, 501)
        self.assertEqual(first, second)
        self.assertEqual(first.fixture_sha256, second.fixture_sha256)
        self.assertEqual(set(first.actions), set(task.actions))

    def test_training_rejects_pilot_and_never_consumes_confirmatory(self) -> None:
        components = train_sqlite_components(budget=budget())
        self.assertTrue(components.training_records)
        self.assertTrue(components.calibration_records)
        used = {
            record.metadata["partition"]
            for record in (
                *components.training_records,
                *components.calibration_records,
            )
        }
        self.assertEqual(
            used,
            {
                SQLitePartition.DEVELOPMENT.value,
                SQLitePartition.CALIBRATION.value,
            },
        )


class SQLiteAdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.components = train_sqlite_components(budget=budget())

    def test_task_and_compute_actions_remain_distinct(self) -> None:
        task = paired_sqlite_task(
            load_sqlite_partition(SQLitePartition.EXPLORATORY)[0],
            501,
        )
        planner = make_sqlite_planner(
            "fidelity_mcts",
            components=self.components,
            matched_accurate_calls=2,
            seed=3,
        )
        result = planner.plan(
            task,
            models=make_sqlite_portfolio(),
            budget=budget(),
            seed=3,
        )
        self.assertIn(result.action, task.actions)
        self.assertTrue(result.trace)
        self.assertTrue(
            all(trace["task_action"] in task.actions for trace in result.trace)
        )
        self.assertTrue(
            all(trace["model_id"] not in task.actions for trace in result.trace)
        )

    def test_classical_adaptive_frontier_interface_executes_l2_portfolio(self) -> None:
        task = load_sqlite_partition(SQLitePartition.EXPLORATORY)[0]
        evaluator = make_sqlite_frontier_evaluator(
            components=self.components,
            budget=budget(),
            seed=13,
        )

        class Context:
            def legal_actions(self, state: SQLiteTask) -> tuple[str, ...]:
                return state.actions

            def remaining_budget(self) -> SearchBudget:
                return budget()

        node = StateNode((task.task_id, task.fixture_sha256), task)
        evaluation = evaluator.evaluate(node, Context(), Random(13))
        self.assertIn(evaluation.rollout_actions[0], task.actions)
        self.assertGreater(evaluation.usage.model_calls, 0)
        self.assertEqual(len(node.edges), len(task.actions))
        self.assertTrue(all(edge.evidence for edge in node.edges.values()))

    def test_hard_budget_and_matched_random_quota(self) -> None:
        task = paired_sqlite_task(
            load_sqlite_partition(SQLitePartition.EXPLORATORY)[0],
            502,
        )
        planner = make_sqlite_planner(
            "random_matched",
            components=self.components,
            matched_accurate_calls=2,
            seed=7,
        )
        limit = budget()
        result = planner.plan(
            task,
            models=make_sqlite_portfolio(),
            budget=limit,
            seed=7,
        )
        self.assertEqual(result.usage.accurate_calls, 2)
        self.assertEqual(result.usage.cost, 9.0)
        self.assertEqual(result.usage.tokens, 32)
        self.assertLessEqual(result.usage.model_calls, limit.max_model_calls or 0)
        self.assertLessEqual(
            result.usage.environment_calls,
            limit.max_environment_calls or 0,
        )
        accurate = [trace for trace in result.trace if trace["fidelity"] == "accurate"]
        self.assertTrue(all(trace["randomized_audit"] for trace in accurate))
        self.assertTrue(
            all(0.0 < trace["route_propensity"] <= 1.0 for trace in accurate)
        )

    def test_deterministic_planner_rerun(self) -> None:
        task = paired_sqlite_task(
            load_sqlite_partition(SQLitePartition.EXPLORATORY)[1],
            503,
        )
        outcomes = []
        for _ in range(2):
            planner = make_sqlite_planner(
                "cheap_only",
                components=self.components,
                matched_accurate_calls=0,
                seed=11,
            )
            result = planner.plan(
                task,
                models=make_sqlite_portfolio(),
                budget=budget(max_accurate_calls=0, max_environment_calls=0),
                seed=11,
            )
            outcomes.append((result.action, result.predicted_value, result.usage))
        self.assertEqual(outcomes[0], outcomes[1])


class Phase5AnalysisTests(unittest.TestCase):
    def test_hypervolume_fixture(self) -> None:
        reference = HypervolumeReference(success=0.0, cost=10.0)
        value = hypervolume_2d(
            (
                {"success": 0.5, "cost": 2.0},
                {"success": 0.8, "cost": 6.0},
            ),
            reference=reference,
        )
        self.assertAlmostEqual(value, 5.2)

    def test_calibration_metric_fixture(self) -> None:
        metrics = calibration_metrics(
            (0.1, 0.8, 0.6, 0.2),
            (0.0, 1.0, 1.0, 0.0),
            variances=(0.04, 0.04, 0.04, 0.04),
            bins=2,
        )
        self.assertEqual(metrics.count, 4)
        self.assertAlmostEqual(metrics.brier or 0.0, 0.0625)
        self.assertIsNotNone(metrics.nll)
        self.assertIsNotNone(metrics.ece)
        self.assertAlmostEqual(metrics.interval_coverage or 0.0, 0.75)

    def test_ope_fixture_and_poor_overlap_warning(self) -> None:
        records = (
            LoggedRoutingDecision(
                "a",
                "verify",
                reward=1.0,
                propensity=0.01,
                baseline_prediction=0.6,
                target_probability=1.0,
            ),
            LoggedRoutingDecision(
                "b",
                "verify",
                reward=0.0,
                propensity=0.5,
                baseline_prediction=0.4,
                target_probability=1.0,
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnostics = off_policy_diagnostics(
                records,
                online_randomized=0.5,
                min_propensity=0.05,
            )
        self.assertTrue(diagnostics.poor_overlap)
        self.assertLess(diagnostics.effective_sample_size, 2.0)
        self.assertTrue(caught)
        self.assertEqual(
            set(diagnostics.absolute_error_vs_online),
            {"naive", "ips", "snips", "doubly_robust"},
        )


class SQLiteStudyTests(unittest.TestCase):
    def _smoke_protocol(self) -> dict[str, object]:
        return json.loads(
            (ROOT / "experiments" / "pilots" / "sqlite_l2_smoke.json").read_text(
                encoding="utf-8"
            )
        )

    def test_candidate_and_smoke_protocols_validate_without_confirmatory_seeds(
        self,
    ) -> None:
        for path in (
            ROOT / "experiments" / "pilots" / "sqlite_l2_smoke.json",
            ROOT / "experiments" / "protocols" / "sqlite_l2_phase5a_candidate.json",
        ):
            protocol = json.loads(path.read_text(encoding="utf-8"))
            validate_preregistration_protocol(protocol)
            self.assertEqual(protocol["confirmatory_seeds"], [])
            self.assertFalse(protocol["confirmatory_seed_policy"]["materialized"])

    def test_confirmatory_and_preregistered_output_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "confirmatory"):
                validate_exploratory_output(root / "confirmatory" / "run")
            with self.assertRaisesRegex(ValueError, "preregistered"):
                validate_exploratory_output(root / "preregistered" / "run")

    def test_smoke_runs_all_baselines_at_all_budgets_and_raw_is_immutable(self) -> None:
        protocol = self._smoke_protocol()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilots" / "sqlite-smoke"
            summary = run_sqlite_study(protocol, output)
            expected_methods = len(protocol["methods"]) + len(protocol["ablations"])
            expected_records = (
                expected_methods
                * len(protocol["budget_grid"])
                * len(protocol["pilot_seeds"])
                * len(protocol["benchmark"]["exploratory_task_ids"])
            )
            self.assertEqual(summary["records"], expected_records)
            self.assertEqual(summary["failures"], 0)
            self.assertTrue(summary["baseline_budget_coverage_complete"])
            self.assertEqual(
                set(PRIMARY_SQLITE_METHODS),
                set(protocol["methods"]),
            )
            for budget_methods in summary["methods_by_budget"].values():
                self.assertEqual(set(budget_methods), set(summary["methods"]))
            episodes_path = output / "raw" / "episodes.jsonl"
            self.assertFalse(episodes_path.stat().st_mode & stat.S_IWUSR)
            first = json.loads(episodes_path.read_text().splitlines()[0])
            validate_raw_record(first, kind="episode")
            tampered = dict(first)
            tampered["success"] = not tampered["success"]
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_raw_record(tampered, kind="episode")
            manifest = json.loads(
                (output / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["raw_files_read_only"])
            self.assertTrue((output / "analysis.json").is_file())
            self.assertTrue((output / "per_task_differences.jsonl").is_file())
            decision_records = [
                json.loads(line)
                for line in (output / "raw" / "decisions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            sql_failures = [
                record
                for record in decision_records
                if record["execution_status"] == "sql_error"
                and record["action_kind"] == "compute"
            ]
            self.assertTrue(sql_failures)
            self.assertTrue(all(record["cost"] == 4.0 for record in sql_failures))
            self.assertTrue(all(record["value"] == -1.0 for record in sql_failures))
            gaps = matched_budget_diagnostic(summary)
            self.assertTrue(gaps)
            self.assertTrue(all(value == 0.0 for value in gaps.values()))
            self.assertIn("not paper results", summary["notice"].lower())


if __name__ == "__main__":
    unittest.main()
