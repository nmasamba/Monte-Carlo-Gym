"""Tree-selection policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Protocol

from ..core.tree import ActionEdge, StateNode


class TreePolicy(Protocol):
    def select(self, node: StateNode, rng: Random) -> ActionEdge:
        """Select one already-expanded legal action edge."""


@dataclass(frozen=True, slots=True)
class UCTTreePolicy:
    """UCB1 selection over action-edge means and visit counts."""

    exploration_constant: float = math.sqrt(2.0)

    def __post_init__(self) -> None:
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")

    def select(self, node: StateNode, rng: Random) -> ActionEdge:
        if not node.edges:
            raise RuntimeError("UCT requires an expanded node with legal actions")
        unvisited = [edge for edge in node.edges.values() if edge.visits == 0]
        if unvisited:
            return unvisited[rng.randrange(len(unvisited))]

        log_parent = math.log(max(1, node.visits))

        def score(edge: ActionEdge) -> float:
            return edge.mean_value + self.exploration_constant * math.sqrt(
                log_parent / edge.visits
            )

        scores = [(edge, score(edge)) for edge in node.edges.values()]
        best_score = max(item[1] for item in scores)
        best = [edge for edge, value in scores if value == best_score]
        return best[rng.randrange(len(best))]


@dataclass(frozen=True, slots=True)
class PUCTTreePolicy:
    """Prior-guided UCT used by policy/value search presets."""

    exploration_constant: float = 1.5

    def __post_init__(self) -> None:
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")

    def select(self, node: StateNode, rng: Random) -> ActionEdge:
        if not node.edges:
            raise RuntimeError("PUCT requires an expanded node with legal actions")
        edges = tuple(node.edges.values())
        if any(edge.prior is None for edge in edges):
            raise RuntimeError("PUCT requires a prior on every legal action edge")

        def prior(edge: ActionEdge) -> float:
            if edge.prior is None:  # guarded above; retains local type narrowing
                raise RuntimeError("PUCT edge prior disappeared during selection")
            return edge.prior

        total_visits = sum(edge.visits for edge in edges)
        if total_visits == 0:
            best_prior = max(prior(edge) for edge in edges)
            best = [edge for edge in edges if prior(edge) == best_prior]
            return best[rng.randrange(len(best))]

        parent_scale = math.sqrt(total_visits)

        def score(edge: ActionEdge) -> float:
            return edge.mean_value + (
                self.exploration_constant
                * prior(edge)
                * parent_scale
                / (1 + edge.visits)
            )

        scores = [(edge, score(edge)) for edge in edges]
        best_score = max(item[1] for item in scores)
        best = [edge for edge, value in scores if value == best_score]
        return best[rng.randrange(len(best))]
