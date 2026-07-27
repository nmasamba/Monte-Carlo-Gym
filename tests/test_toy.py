from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from montecarlgym.experiments.runner import run_experiment
from montecarlgym.experiments.toy import (
    AdaptiveFidelityPlanner,
    ToyBenchmarkConfig,
    make_portfolio,
    sample_task,
)
from montecarlgym.types import SearchBudget


ROOT = Path(__file__).resolve().parents[1]


class ToyHarnessTests(unittest.TestCase):
    def test_adaptive_planner_obeys_budget(self) -> None:
        config = ToyBenchmarkConfig(actions=4)
        task = sample_task(config, 13)
        budget = SearchBudget(
            max_cost=16.0,
            max_tokens=160,
            max_accurate_calls=1,
            max_iterations=10,
        )
        result = AdaptiveFidelityPlanner().plan(
            task,
            models=make_portfolio(config),
            budget=budget,
            seed=99,
        )

        self.assertLessEqual(result.usage.cost, budget.max_cost)
        self.assertLessEqual(result.usage.tokens, budget.max_tokens)
        self.assertEqual(result.usage.accurate_calls, 1)
        self.assertIn(result.action, task.actions)

    def test_experiment_is_reproducible(self) -> None:
        config_path = ROOT / "experiments" / "configs" / "toy.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["run"]["repetitions"] = 8

        with tempfile.TemporaryDirectory() as first_dir:
            first = run_experiment(config, Path(first_dir))
        with tempfile.TemporaryDirectory() as second_dir:
            second = run_experiment(config, Path(second_dir))

        self.assertEqual(first, second)
        self.assertEqual(set(first["methods"]), set(config["methods"]))

    def test_accurate_call_limit_is_never_exceeded(self) -> None:
        config_path = ROOT / "experiments" / "configs" / "toy.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["run"]["repetitions"] = 5
        config["budget"]["max_accurate_calls"] = 2

        with tempfile.TemporaryDirectory() as output_dir:
            run_experiment(config, Path(output_dir))
            records = [
                json.loads(line)
                for line in (Path(output_dir) / "runs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertTrue(records)
        self.assertTrue(
            all(record["accurate_calls"] <= 2 for record in records)
        )


if __name__ == "__main__":
    unittest.main()
