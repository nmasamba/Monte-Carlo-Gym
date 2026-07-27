from __future__ import annotations

import unittest

from montecarlgym.experiments.metrics import (
    paired_bootstrap_difference,
    pareto_frontier,
)
from montecarlgym.experiments.registry import FactoryRegistry


class ExperimentUtilityTests(unittest.TestCase):
    def test_pareto_frontier(self) -> None:
        points = {
            "cheap": {"success": 0.6, "cost": 1.0},
            "adaptive": {"success": 0.9, "cost": 3.0},
            "dominated": {"success": 0.5, "cost": 4.0},
        }
        frontier = pareto_frontier(
            points,
            maximize=("success",),
            minimize=("cost",),
        )
        self.assertEqual(frontier, ("adaptive", "cheap"))

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = [1.0, 1.0, 0.0, 1.0]
        second = [0.0, 1.0, 0.0, 0.0]
        one = paired_bootstrap_difference(
            first, second, seed=7, samples=200
        )
        two = paired_bootstrap_difference(
            first, second, seed=7, samples=200
        )
        self.assertEqual(one, two)
        self.assertAlmostEqual(one.estimate, 0.5)

    def test_registry_rejects_duplicates(self) -> None:
        registry: FactoryRegistry[object] = FactoryRegistry()
        registry.register("thing", object)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.register("thing", object)
        self.assertEqual(registry.names(), ("thing",))


if __name__ == "__main__":
    unittest.main()
