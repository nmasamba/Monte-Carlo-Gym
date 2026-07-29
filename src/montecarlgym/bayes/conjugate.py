"""Standard-library conjugate posteriors for Bayesian MCTS fixtures."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from random import Random


@dataclass(slots=True)
class NormalGammaPosterior:
    """Normal-Gamma posterior over an unknown Gaussian mean and precision."""

    mean: float = 0.0
    precision_scale: float = 1.0
    shape: float = 1.0
    rate: float = 1.0
    observations: int = 0

    def __post_init__(self) -> None:
        if self.precision_scale <= 0:
            raise ValueError("precision_scale must be positive")
        if self.shape <= 0 or self.rate <= 0:
            raise ValueError("Normal-Gamma shape and rate must be positive")

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("posterior observations must be finite")
        old_scale = self.precision_scale
        new_scale = old_scale + 1.0
        delta = value - self.mean
        self.mean += delta / new_scale
        self.precision_scale = new_scale
        self.shape += 0.5
        self.rate += 0.5 * old_scale * delta * delta / new_scale
        self.observations += 1

    @property
    def predictive_variance(self) -> float:
        """Variance of the posterior predictive Student-t distribution."""

        if self.shape <= 1.0:
            return math.inf
        return self.rate * (self.precision_scale + 1.0) / (
            (self.shape - 1.0) * self.precision_scale
        )

    def sample_mean(self, rng: Random) -> float:
        precision = rng.gammavariate(self.shape, 1.0 / self.rate)
        standard_deviation = math.sqrt(
            1.0 / (self.precision_scale * precision)
        )
        return rng.gauss(self.mean, standard_deviation)

    def copy(self) -> NormalGammaPosterior:
        return NormalGammaPosterior(
            self.mean,
            self.precision_scale,
            self.shape,
            self.rate,
            self.observations,
        )


@dataclass(slots=True)
class DirichletTransitionPosterior:
    """Sparse Dirichlet counts over observed stochastic outcome identities."""

    prior_concentration: float = 1.0
    concentrations: dict[Hashable, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prior_concentration <= 0:
            raise ValueError("prior_concentration must be positive")
        if any(value <= 0 for value in self.concentrations.values()):
            raise ValueError("Dirichlet concentrations must be positive")

    def observe(self, outcome: Hashable) -> None:
        self.concentrations[outcome] = (
            self.concentrations.get(outcome, self.prior_concentration) + 1.0
        )

    def sample(self, rng: Random) -> Mapping[Hashable, float]:
        if not self.concentrations:
            raise ValueError("cannot sample an empty transition posterior")
        draws = {
            outcome: rng.gammavariate(concentration, 1.0)
            for outcome, concentration in self.concentrations.items()
        }
        total = sum(draws.values())
        return {outcome: draw / total for outcome, draw in draws.items()}

    def copy(self) -> DirichletTransitionPosterior:
        return DirichletTransitionPosterior(
            self.prior_concentration,
            dict(self.concentrations),
        )
