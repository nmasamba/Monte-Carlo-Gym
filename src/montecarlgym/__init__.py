"""MonteCarloGym public research interfaces.

The package separates task actions from compute actions. A task action changes
the planned environment; a compute action chooses how the planner gathers
evidence about a branch.
"""

from .agent import MCTSAgent
from .config import MCTSConfig
from .gym_wrapper import (
    DeepCopySnapshotStrategy,
    MCTSEnvWrapper,
    NativeSnapshotStrategy,
    SnapshotError,
)
from .models import GenerativeModel, ModelPortfolio
from .planner import PlanResult, Planner
from .routing import ComputeRouter, RouterContext
from .types import (
    ComputeAction,
    Fidelity,
    ModelObservation,
    ModelQuote,
    SearchBudget,
)

__all__ = [
    "ComputeAction",
    "ComputeRouter",
    "Fidelity",
    "GenerativeModel",
    "DeepCopySnapshotStrategy",
    "MCTSAgent",
    "MCTSConfig",
    "MCTSEnvWrapper",
    "ModelObservation",
    "ModelPortfolio",
    "ModelQuote",
    "NativeSnapshotStrategy",
    "PlanResult",
    "Planner",
    "RouterContext",
    "SearchBudget",
    "SnapshotError",
]

__version__ = "0.1.0a0"
