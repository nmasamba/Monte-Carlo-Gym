from __future__ import annotations

import unittest
from random import Random

from montecarlgym.models import ModelPortfolio
from montecarlgym.types import Fidelity, ModelObservation, ModelQuote


class DummyModel:
    def __init__(self, model_id: str, fidelity: Fidelity) -> None:
        self._model_id = model_id
        self._fidelity = fidelity

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def fidelity(self) -> Fidelity:
        return self._fidelity

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        return ModelQuote(1.0)

    def evaluate(
        self,
        state: object,
        action: object,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        return ModelObservation(value=0.0, variance=1.0, cost=1.0)


class ModelPortfolioTests(unittest.TestCase):
    def test_registry_and_fidelity_lookup(self) -> None:
        cheap = DummyModel("cheap", Fidelity.CHEAP)
        accurate = DummyModel("accurate", Fidelity.ACCURATE)
        portfolio = ModelPortfolio.from_models([cheap, accurate])

        self.assertIs(portfolio.get("cheap"), cheap)
        self.assertEqual(portfolio.by_fidelity(Fidelity.ACCURATE), (accurate,))

    def test_duplicate_ids_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate model_id"):
            ModelPortfolio.from_models(
                [
                    DummyModel("same", Fidelity.CHEAP),
                    DummyModel("same", Fidelity.ACCURATE),
                ]
            )

    def test_empty_portfolio_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            ModelPortfolio.from_models([])


if __name__ == "__main__":
    unittest.main()
