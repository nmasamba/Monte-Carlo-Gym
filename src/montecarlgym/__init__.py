"""MonteCarloGym public research interfaces.

The package separates task actions from compute actions. A task action changes
the planned environment; a compute action chooses how the planner gathers
evidence about a branch.
"""

from .adaptive import (
    AdaptiveComputePlanner,
    AdaptiveFrontierEvaluator,
    AdaptivePlanResult,
    CalibratedLinearDiscrepancyModel,
    LearnedEVCRouter,
    LinearEVCModel,
    RandomEscalationRouter,
)
from .agent import MCTSAgent
from .config import MCTSConfig
from .gym_wrapper import (
    DeepCopySnapshotStrategy,
    MCTSEnvWrapper,
    NativeSnapshotStrategy,
    SnapshotError,
)
from .models import GenerativeModel, ModelPortfolio
from .planner import Planner, PlanResult
from .presets import (
    alphago_apv_preset,
    alphago_zero_preset,
    crazy_stone_mix_preset,
    crazy_stone_robust_preset,
    dng_mcts_preset,
    mast_preset,
    mcbrl_root_sampling_preset,
    rave_mast_preset,
    rave_preset,
    uct_preset,
)
from .replay import JsonlVerifiedReplayStore, VerifiedReplayStore, VerifiedTransition
from .routing import (
    AccurateOnlyRouter,
    CheapOnlyRouter,
    ComputeRouter,
    FixedCascadeRouter,
    RouterContext,
    ThresholdRouter,
)
from .types import (
    ComputeAction,
    EvidenceProvenance,
    Fidelity,
    ModelObservation,
    ModelQuote,
    SearchBudget,
)

__all__ = [
    "AccurateOnlyRouter",
    "AdaptiveComputePlanner",
    "AdaptiveFrontierEvaluator",
    "AdaptivePlanResult",
    "CalibratedLinearDiscrepancyModel",
    "CheapOnlyRouter",
    "ComputeAction",
    "ComputeRouter",
    "EvidenceProvenance",
    "Fidelity",
    "FixedCascadeRouter",
    "GenerativeModel",
    "DeepCopySnapshotStrategy",
    "MCTSAgent",
    "MCTSConfig",
    "MCTSEnvWrapper",
    "LearnedEVCRouter",
    "LinearEVCModel",
    "ModelObservation",
    "ModelPortfolio",
    "ModelQuote",
    "NativeSnapshotStrategy",
    "PlanResult",
    "Planner",
    "RouterContext",
    "RandomEscalationRouter",
    "SearchBudget",
    "SnapshotError",
    "ThresholdRouter",
    "JsonlVerifiedReplayStore",
    "VerifiedReplayStore",
    "VerifiedTransition",
    "alphago_apv_preset",
    "alphago_zero_preset",
    "crazy_stone_mix_preset",
    "crazy_stone_robust_preset",
    "dng_mcts_preset",
    "mast_preset",
    "mcbrl_root_sampling_preset",
    "rave_mast_preset",
    "rave_preset",
    "uct_preset",
]

__version__ = "0.2.0a2"
