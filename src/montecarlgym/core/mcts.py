"""Algorithm-agnostic orchestration for classical Monte Carlo tree search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol

from ..config import MCTSConfig
from ..policies.action_selection import RootActionSelector
from ..policies.rollout_policies import RolloutContext, TransitionTuple
from ..policies.tree_policies import TreePolicy
from ..types import Action, ResourceUsage, SearchBudget
from .backup import BackupOperator
from .budget import ResourceLedger
from .path import Evaluation, SearchPath
from .tree import ActionEdge, SearchTree, StateCodec, StateNode


class Expander(Protocol):
    def expand(
        self,
        node: StateNode,
        legal_actions: Sequence[Action],
    ) -> tuple[ActionEdge, ...]:
        """Attach legal action edges to ``node``."""


class Evaluator(Protocol):
    def evaluate(
        self,
        frontier: StateNode,
        model: RolloutContext,
        rng: Random,
    ) -> Evaluation:
        """Evaluate future return from a selected frontier."""


class SimulationModel(Protocol):
    """Reversible step model consumed by the classical kernel.

    ``MCTSEnvWrapper`` is one implementation. Standalone generative models can
    implement this protocol without importing or pretending to be Gymnasium.
    """

    max_call_cost: float

    def transaction(self) -> AbstractContextManager[Any]: ...

    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...

    def seed_simulation(self, seed: int) -> None: ...

    def legal_actions(self, observation: Any | None = None) -> tuple[Action, ...]: ...

    def step(self, action: Action) -> TransitionTuple: ...

    def transition_cost(self, info: Mapping[str, Any]) -> float: ...


@dataclass(frozen=True, slots=True)
class IterationTrace:
    iteration: int
    actions: tuple[Action, ...]
    rewards: tuple[float, ...]
    evaluation: float
    evaluation_depth: int
    stop_reason: str
    model_calls: int
    normalized_cost: float


class TraceSink(Protocol):
    def record(self, trace: IterationTrace) -> None:
        """Consume one structured, non-chain-of-thought iteration trace."""


@dataclass(frozen=True, slots=True)
class NullTraceSink:
    def record(self, trace: IterationTrace) -> None:
        del trace


@dataclass(slots=True)
class ListTraceSink:
    records: list[IterationTrace] = field(default_factory=list)

    def record(self, trace: IterationTrace) -> None:
        self.records.append(trace)


@dataclass(frozen=True, slots=True)
class MCTSSearchReport:
    usage: ResourceUsage
    stop_reason: str
    root_visits: int
    action_visits: Mapping[Action, int]
    action_values: Mapping[Action, float]


@dataclass(frozen=True, slots=True)
class MCTSSearchResult:
    action: Action
    predicted_value: float
    tree: SearchTree
    report: MCTSSearchReport


class _BudgetedModel:
    def __init__(self, env: SimulationModel, ledger: ResourceLedger) -> None:
        self.env = env
        self.ledger = ledger

    def legal_actions(self, observation: Any | None = None) -> tuple[Action, ...]:
        return self.env.legal_actions(observation)

    def try_step(self, action: Action) -> TransitionTuple | None:
        reservation = self.ledger.reserve_call(self.env.max_call_cost)
        if reservation is None:
            return None
        # Exceptions deliberately propagate.  The reservation remains charged
        # and the environment transaction restores the live state.
        transition = self.env.step(action)
        measured_cost = self.env.transition_cost(transition[4])
        self.ledger.commit_call(reservation, measured_cost=measured_cost)
        return transition

    def remaining_budget(self) -> SearchBudget:
        return self.ledger.remaining_budget()

    def absorb_usage(self, usage: ResourceUsage) -> None:
        self.ledger.absorb_usage(usage)


@dataclass(slots=True)
class MCTSEngine:
    """Own selection/expansion/evaluation/backup mechanics, not equations."""

    tree_policy: TreePolicy
    expander: Expander
    evaluator: Evaluator
    backup: BackupOperator
    action_selector: RootActionSelector
    state_codec: StateCodec
    config: MCTSConfig = MCTSConfig()
    trace_sink: TraceSink = NullTraceSink()

    def search(
        self,
        tree: SearchTree,
        model: SimulationModel,
        *,
        budget: SearchBudget,
        rng: Random,
    ) -> MCTSSearchResult:
        """Run a transactionally isolated search and select a root action."""

        if self.state_codec.key(tree.root.state) != tree.root.state_key:
            raise ValueError("search tree root is incompatible with the state codec")
        ledger = ResourceLedger(budget)
        with model.transaction():
            root_snapshot = model.snapshot()
            while ledger.can_start_iteration():
                model.restore(root_snapshot)
                model.seed_simulation(rng.getrandbits(64))
                start_iteration = getattr(
                    self.tree_policy,
                    "start_iteration",
                    None,
                )
                if start_iteration is not None:
                    start_iteration(tree.root, rng)
                context = _BudgetedModel(model, ledger)
                path, evaluation = self._run_iteration(tree, context, rng)
                if path is None or evaluation is None:
                    break
                self.backup.update(path, evaluation)
                ledger.complete_iteration()
                self.trace_sink.record(
                    IterationTrace(
                        iteration=ledger.iterations,
                        actions=tuple(step.edge.action for step in path.steps),
                        rewards=tuple(step.reward for step in path.steps),
                        evaluation=evaluation.value,
                        evaluation_depth=evaluation.depth,
                        stop_reason=evaluation.stop_reason,
                        model_calls=ledger.model_calls,
                        normalized_cost=ledger.cost,
                    )
                )

            selected = self.action_selector.select(tree.root, rng)

        if ledger.stop_reason == "not_started":
            ledger.stop_reason = "search_complete"
        usage = ledger.usage()
        report = MCTSSearchReport(
            usage=usage,
            stop_reason=ledger.stop_reason,
            root_visits=tree.root.visits,
            action_visits={
                action: edge.visits for action, edge in tree.root.edges.items()
            },
            action_values={
                action: edge.mean_value for action, edge in tree.root.edges.items()
            },
        )
        return MCTSSearchResult(
            selected.action,
            selected.mean_value,
            tree,
            report,
        )

    def _run_iteration(
        self,
        tree: SearchTree,
        model: _BudgetedModel,
        rng: Random,
    ) -> tuple[SearchPath | None, Evaluation | None]:
        path = SearchPath(tree.root)
        node = tree.root
        for _ in range(self.config.max_tree_depth):
            if node.terminal:
                return path, Evaluation(
                    0.0,
                    terminated=node.terminated,
                    truncated=node.truncated,
                    stop_reason="terminal" if node.terminated else "truncated",
                )
            actions = model.legal_actions(node.state)
            self.expander.expand(node, actions)
            edge = self.tree_policy.select(node, rng)
            unvisited = edge.visits == 0
            transition = model.try_step(edge.action)
            if transition is None:
                if not path.steps:
                    return None, None
                return path, Evaluation(0.0, stop_reason="resource_budget")
            observation, reward, terminated, truncated, _ = transition
            outcome, new_outcome = tree.link_outcome(
                edge,
                state=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            path.append(node, edge, outcome)
            node = outcome.child
            if terminated or truncated:
                return path, Evaluation(
                    0.0,
                    terminated=terminated,
                    truncated=truncated,
                    stop_reason="terminal" if terminated else "truncated",
                )
            if unvisited or new_outcome:
                evaluation = self.evaluator.evaluate(node, model, rng)
                model.absorb_usage(evaluation.usage)
                return path, evaluation

        return path, Evaluation(0.0, stop_reason="tree_depth")
