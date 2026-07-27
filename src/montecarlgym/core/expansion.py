"""Legal-action expansion components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..types import Action
from .tree import ActionEdge, StateNode


@dataclass(frozen=True, slots=True)
class LegalActionExpander:
    """Attach every currently legal action to a state node exactly once."""

    def expand(
        self,
        node: StateNode,
        legal_actions: Sequence[Action],
    ) -> tuple[ActionEdge, ...]:
        if node.terminal:
            node.expanded = True
            return ()
        seen: set[Action] = set()
        for action in legal_actions:
            try:
                duplicate = action in seen
            except TypeError as exc:
                raise TypeError("MCTS actions must be hashable") from exc
            if duplicate:
                raise ValueError(f"duplicate legal action: {action!r}")
            seen.add(action)
            node.edges.setdefault(action, ActionEdge(action))
        stale = set(node.edges) - seen
        if stale:
            raise ValueError(
                "state identity produced an incompatible legal-action set; "
                f"stale actions: {sorted(stale, key=repr)!r}"
            )
        if not node.edges:
            raise RuntimeError("non-terminal state exposes no legal actions")
        node.expanded = True
        return tuple(node.edges.values())
