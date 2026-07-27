"""Configuration helpers for the classical MCTS preset.

The engine itself receives concrete policy objects.  ``MCTSConfig`` only holds
the small set of orchestration parameters shared by those objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Algorithm-independent controls for one classical search engine."""

    discount: float = 1.0
    max_tree_depth: int = 100

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be positive")
