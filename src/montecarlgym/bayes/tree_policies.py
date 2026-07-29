"""Posterior-sampling tree policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from ..core.tree import ActionEdge, StateNode
from .backup import REWARD_POSTERIOR
from .conjugate import NormalGammaPosterior


def _posterior(
    edge: ActionEdge,
    factory: Callable[[], NormalGammaPosterior],
) -> NormalGammaPosterior:
    posterior = edge.statistics.setdefault(REWARD_POSTERIOR, factory())
    if not isinstance(posterior, NormalGammaPosterior):
        raise TypeError("edge reward posterior has an incompatible type")
    return posterior


@dataclass(frozen=True, slots=True)
class ThompsonTreePolicy:
    """Sample each action's local reward-mean posterior at selection time."""

    posterior_factory: Callable[[], NormalGammaPosterior] = NormalGammaPosterior
    prioritize_unvisited: bool = True

    def select(self, node: StateNode, rng: Random) -> ActionEdge:
        if not node.edges:
            raise RuntimeError("Thompson selection requires an expanded node")
        if self.prioritize_unvisited:
            unvisited = [edge for edge in node.edges.values() if edge.visits == 0]
            if unvisited:
                return unvisited[rng.randrange(len(unvisited))]
        samples = [
            (edge, _posterior(edge, self.posterior_factory).sample_mean(rng))
            for edge in node.edges.values()
        ]
        best_value = max(value for _, value in samples)
        best = [edge for edge, value in samples if value == best_value]
        return best[rng.randrange(len(best))]
