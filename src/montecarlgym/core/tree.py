"""State/action/outcome graph used by every MCTS policy preset."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..types import Action


class StateCodec(Protocol):
    """Map an observation or information state to a stable identity."""

    def key(self, state: Any) -> Hashable:
        """Return a hashable identity compatible with tree reuse."""


def _freeze(value: Any) -> Hashable:
    if value is None or isinstance(value, (str, bytes, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        return value
    if isinstance(value, Mapping):
        pairs = [(_freeze(key), _freeze(item)) for key, item in value.items()]
        return ("mapping", tuple(sorted(pairs, key=repr)))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_freeze(item) for item in value), key=repr)))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            "state is not hashable; inject a StateCodec that defines its "
            "information-state identity"
        ) from exc
    return value


@dataclass(frozen=True, slots=True)
class DefaultStateCodec:
    """A conservative codec for scalar and nested container observations."""

    def key(self, state: Any) -> Hashable:
        return _freeze(state)


@dataclass(slots=True)
class StateNode:
    """One state or information state; it has no canonical parent."""

    state_key: Hashable
    state: Any
    terminated: bool = False
    truncated: bool = False
    visits: int = 0
    edges: dict[Action, ActionEdge] = field(default_factory=dict)
    expanded: bool = False

    @property
    def terminal(self) -> bool:
        return self.terminated or self.truncated


@dataclass(slots=True)
class OutcomeLink:
    """One sampled reward/next-state outcome of an action edge."""

    outcome_key: Hashable
    reward: float
    child: StateNode
    terminated: bool = False
    truncated: bool = False
    visits: int = 0


@dataclass(slots=True)
class ActionEdge:
    """Action statistics live here, independently of successor states."""

    action: Action
    visits: int = 0
    total_return: float = 0.0
    mean_value: float = 0.0
    prior: float | None = None
    evidence: list[Any] = field(default_factory=list)
    outcomes: dict[Hashable, OutcomeLink] = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.visits

    @property
    def W(self) -> float:
        return self.total_return

    @property
    def Q(self) -> float:
        return self.mean_value

    def update(self, value: float) -> None:
        self.visits += 1
        self.total_return += value
        self.mean_value = self.total_return / self.visits


class SearchTree:
    """A rooted DAG with optional transposition sharing by state key."""

    def __init__(
        self,
        state: Any,
        *,
        codec: StateCodec | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        self.codec = codec or DefaultStateCodec()
        key = self.codec.key(state)
        self.root = StateNode(key, state, terminated, truncated)
        self._nodes: dict[Hashable, StateNode] = {key: self.root}

    @property
    def nodes(self) -> tuple[StateNode, ...]:
        return tuple(self._nodes.values())

    def get_or_create(
        self,
        state: Any,
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> tuple[StateNode, bool]:
        key = self.codec.key(state)
        existing = self._nodes.get(key)
        if existing is not None:
            # A codec that aliases incompatible terminal semantics is unsafe for
            # reuse.  Fail explicitly rather than corrupting the graph.
            if (
                existing.terminated != terminated
                or existing.truncated != truncated
            ):
                raise ValueError(
                    "state codec aliased states with incompatible terminal flags"
                )
            return existing, False
        node = StateNode(key, state, terminated, truncated)
        self._nodes[key] = node
        return node, True

    def outcome_key(
        self,
        state: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Hashable:
        if not math.isfinite(reward):
            raise ValueError("environment rewards must be finite")
        return (self.codec.key(state), reward, terminated, truncated)

    def link_outcome(
        self,
        edge: ActionEdge,
        *,
        state: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> tuple[OutcomeLink, bool]:
        key = self.outcome_key(state, reward, terminated, truncated)
        existing = edge.outcomes.get(key)
        if existing is not None:
            return existing, False
        child, _ = self.get_or_create(
            state,
            terminated=terminated,
            truncated=truncated,
        )
        outcome = OutcomeLink(
            key,
            reward,
            child,
            terminated,
            truncated,
        )
        edge.outcomes[key] = outcome
        return outcome, True

    def matching_outcome(
        self,
        *,
        action: Action,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> OutcomeLink | None:
        edge = self.root.edges.get(action)
        if edge is None:
            return None
        key = self.outcome_key(observation, reward, terminated, truncated)
        return edge.outcomes.get(key)

    def reroot(self, node: StateNode) -> None:
        """Make ``node`` the root and drop the index of unreachable branches."""

        reachable: dict[Hashable, StateNode] = {}
        pending = [node]
        while pending:
            current = pending.pop()
            if current.state_key in reachable:
                continue
            reachable[current.state_key] = current
            for edge in current.edges.values():
                pending.extend(link.child for link in edge.outcomes.values())
        self.root = node
        self._nodes = reachable

    def legal_actions(self) -> Sequence[Action]:
        return tuple(self.root.edges)
