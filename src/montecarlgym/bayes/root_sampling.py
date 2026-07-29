"""Root-sampled posterior selection without imaginary belief updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random

from ..core.tree import ActionEdge, StateNode
from .conjugate import NormalGammaPosterior
from .tree_policies import _posterior


@dataclass(slots=True)
class RootSamplingTreePolicy:
    """Draw one reward-model sample per edge and reuse it for an iteration.

    Pair this policy with an ordinary ``MeanBackup`` to keep the supplied root
    belief frozen while planning. Call ``start_iteration`` before each root
    simulation; ``MCTSEngine`` does this through its generic lifecycle hook.
    """

    posterior_factory: Callable[[], NormalGammaPosterior] = NormalGammaPosterior
    prioritize_unvisited: bool = True
    _samples: dict[int, float] = field(default_factory=dict, init=False)

    def start_iteration(self, root: StateNode, rng: Random) -> None:
        del root, rng
        self._samples.clear()

    def sample_edge(self, edge: ActionEdge, rng: Random) -> float:
        identity = id(edge)
        if identity not in self._samples:
            self._samples[identity] = _posterior(
                edge,
                self.posterior_factory,
            ).sample_mean(rng)
        return self._samples[identity]

    def select(self, node: StateNode, rng: Random) -> ActionEdge:
        if not node.edges:
            raise RuntimeError("root sampling requires an expanded node")
        if self.prioritize_unvisited:
            unvisited = [edge for edge in node.edges.values() if edge.visits == 0]
            if unvisited:
                return unvisited[rng.randrange(len(unvisited))]
        samples = [(edge, self.sample_edge(edge, rng)) for edge in node.edges.values()]
        best_value = max(value for _, value in samples)
        best = [edge for edge, value in samples if value == best_value]
        return best[rng.randrange(len(best))]
