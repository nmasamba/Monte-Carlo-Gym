"""User-facing stateful agent for classical MCTS and subtree reuse."""

from __future__ import annotations

from random import Random
from typing import Any

from .config import MCTSConfig
from .core.backup import BackupOperator, MeanBackup
from .core.expansion import LegalActionExpander
from .core.mcts import (
    Evaluator,
    Expander,
    MCTSEngine,
    MCTSSearchReport,
    MCTSSearchResult,
    NullTraceSink,
    SimulationModel,
    TraceSink,
)
from .core.tree import DefaultStateCodec, SearchTree, StateCodec
from .policies.action_selection import (
    MostVisitedActionSelector,
    RootActionSelector,
)
from .policies.rollout_policies import RandomRolloutEvaluator
from .policies.tree_policies import TreePolicy, UCTTreePolicy
from .types import Action, SearchBudget


class MCTSAgent:
    """Classical MCTS agent whose real transitions enter only via ``observe``.

    All algorithm behavior is injected.  The defaults assemble the Phase 1 UCT
    preset: UCT selection, legal-action expansion, random rollout, mean backup,
    and most-visited root action selection.
    """

    def __init__(
        self,
        *,
        budget: SearchBudget,
        seed: int = 0,
        tree_policy: TreePolicy | None = None,
        expander: Expander | None = None,
        evaluator: Evaluator | None = None,
        backup: BackupOperator | None = None,
        action_selector: RootActionSelector | None = None,
        state_codec: StateCodec | None = None,
        trace_sink: TraceSink | None = None,
        config: MCTSConfig | None = None,
    ) -> None:
        self.budget = budget
        self.state_codec = state_codec or DefaultStateCodec()
        self.config = config or MCTSConfig()
        self._rng = Random(seed)
        self.engine = MCTSEngine(
            tree_policy=tree_policy or UCTTreePolicy(),
            expander=expander or LegalActionExpander(),
            evaluator=evaluator
            or RandomRolloutEvaluator(discount=self.config.discount),
            backup=backup or MeanBackup(discount=self.config.discount),
            action_selector=action_selector or MostVisitedActionSelector(),
            state_codec=self.state_codec,
            config=self.config,
            trace_sink=trace_sink or NullTraceSink(),
        )
        self._tree: SearchTree | None = None
        self._last_result: MCTSSearchResult | None = None
        self._episode_done = False
        self._last_reused = False

    @property
    def tree(self) -> SearchTree | None:
        return self._tree

    @property
    def last_result(self) -> MCTSSearchResult | None:
        return self._last_result

    @property
    def last_report(self) -> MCTSSearchReport | None:
        return None if self._last_result is None else self._last_result.report

    @property
    def last_transition_reused(self) -> bool:
        return self._last_reused

    def compute_action(
        self,
        sim_env: SimulationModel,
        observation: Any,
        *,
        budget: SearchBudget | None = None,
    ) -> Action:
        """Search without advancing or corrupting the live simulation model."""

        key = self.state_codec.key(observation)
        if (
            self._tree is None
            or self._episode_done
            or self._tree.root.state_key != key
        ):
            self._tree = SearchTree(observation, codec=self.state_codec)
            self._episode_done = False
            self._last_reused = False
        else:
            # Preserve the latest observation object for legal-action adapters
            # while retaining all compatible statistics.
            self._tree.root.state = observation

        result = self.engine.search(
            self._tree,
            sim_env,
            budget=budget or self.budget,
            rng=self._rng,
        )
        self._last_result = result
        return result.action

    def observe(
        self,
        *,
        action: Action,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Any | None = None,
    ) -> None:
        """Synchronize the tree with one transition already taken by the user."""

        del info
        reused = False
        if self._tree is not None:
            outcome = self._tree.matching_outcome(
                action=action,
                observation=observation,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            if outcome is not None:
                self._tree.reroot(outcome.child)
                self._tree.root.state = observation
                reused = True
            else:
                self._tree = SearchTree(
                    observation,
                    codec=self.state_codec,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
        else:
            self._tree = SearchTree(
                observation,
                codec=self.state_codec,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        self._last_reused = reused
        self._episode_done = bool(terminated) or bool(truncated)

    def reset(self) -> None:
        """Explicitly invalidate all episode-specific search state."""

        self._tree = None
        self._last_result = None
        self._episode_done = False
        self._last_reused = False
