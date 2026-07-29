"""Tree-local Bayesian statistic updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.backup import (
    BackupOperator,
    IdentityValuePerspective,
    MeanBackup,
    ValuePerspective,
)
from ..core.path import Evaluation, SearchPath
from .conjugate import DirichletTransitionPosterior, NormalGammaPosterior

REWARD_POSTERIOR = "normal_gamma_reward"
TRANSITION_POSTERIOR = "dirichlet_transition"


@dataclass(frozen=True, slots=True)
class BayesianBackup:
    """Update local reward/transition posteriors alongside a base backup.

    These posteriors belong to the search tree. They are deliberately separate
    from any real agent belief supplied to a root-sampling preset.
    """

    base: BackupOperator = MeanBackup()
    discount: float = 1.0
    perspective: ValuePerspective = IdentityValuePerspective()
    reward_factory: Callable[[], NormalGammaPosterior] = NormalGammaPosterior
    transition_factory: Callable[
        [], DirichletTransitionPosterior
    ] = DirichletTransitionPosterior

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")

    def update(self, path: SearchPath, evaluation: Evaluation) -> None:
        values = [0.0] * len(path.steps)
        value = evaluation.value
        for reverse_depth, index in enumerate(
            range(len(path.steps) - 1, -1, -1)
        ):
            step = path.steps[index]
            value = step.reward + self.discount * value
            values[index] = self.perspective.for_edge(
                value,
                step=step,
                depth_from_leaf=reverse_depth,
            )

        self.base.update(path, evaluation)
        for step, edge_value in zip(path.steps, values, strict=True):
            reward_posterior = step.edge.statistics.setdefault(
                REWARD_POSTERIOR,
                self.reward_factory(),
            )
            transition_posterior = step.edge.statistics.setdefault(
                TRANSITION_POSTERIOR,
                self.transition_factory(),
            )
            if not isinstance(reward_posterior, NormalGammaPosterior):
                raise TypeError("edge reward posterior has an incompatible type")
            if not isinstance(
                transition_posterior,
                DirichletTransitionPosterior,
            ):
                raise TypeError(
                    "edge transition posterior has an incompatible type"
                )
            reward_posterior.update(edge_value)
            transition_posterior.observe(step.outcome.outcome_key)
