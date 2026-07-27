"""Root decision rules."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from ..core.tree import ActionEdge, StateNode


class RootActionSelector(Protocol):
    def select(self, root: StateNode, rng: Random) -> ActionEdge:
        """Select the real task action after search."""


@dataclass(frozen=True, slots=True)
class MostVisitedActionSelector:
    """Robust child selection, with mean value as a deterministic tiebreaker."""

    def select(self, root: StateNode, rng: Random) -> ActionEdge:
        del rng
        if not root.edges:
            raise RuntimeError("search did not discover a legal root action")
        return max(
            root.edges.values(),
            key=lambda edge: (edge.visits, edge.mean_value, repr(edge.action)),
        )
