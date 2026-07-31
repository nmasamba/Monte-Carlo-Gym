"""Adaptive multi-fidelity planning primitives.

This layer allocates compute actions to gather branch evidence. It never
executes the selected task action against a live environment.
"""

from .budget import AdaptiveResourceLedger, QueryReservation
from .discrepancy import (
    DiscrepancyEstimate,
    DiscrepancyModel,
    RunningDiscrepancyModel,
)
from .evidence import (
    BranchEstimate,
    BranchEvidence,
    DiscrepancyAwareAggregator,
    EvidenceAggregator,
)
from .frontier import AdaptiveFrontierEvaluator, BudgetedFrontierContext
from .learning import (
    ROUTER_FEATURE_NAMES,
    CalibratedLinearDiscrepancyModel,
    ContextualPrediction,
    LinearEVCModel,
)
from .planner import (
    ActionProvider,
    AdaptiveComputePlanner,
    AdaptivePlanResult,
    AdaptiveQueryTrace,
    AdaptiveSearchReport,
    AdaptiveTraceSink,
    ListAdaptiveTraceSink,
    ModelEvaluationError,
    NullAdaptiveTraceSink,
    VerificationError,
)
from .routing import (
    LearnedEVCRouter,
    MatchedRandomEscalationRouter,
    RandomEscalationRouter,
    router_features,
)
from .stopping import (
    ConfidenceStopPolicy,
    FixedQueryStopPolicy,
    NeverStopPolicy,
    StopPolicy,
)

__all__ = [
    "ActionProvider",
    "AdaptiveComputePlanner",
    "AdaptiveFrontierEvaluator",
    "AdaptivePlanResult",
    "AdaptiveResourceLedger",
    "AdaptiveQueryTrace",
    "AdaptiveSearchReport",
    "AdaptiveTraceSink",
    "BranchEstimate",
    "BranchEvidence",
    "BudgetedFrontierContext",
    "CalibratedLinearDiscrepancyModel",
    "ConfidenceStopPolicy",
    "ContextualPrediction",
    "DiscrepancyAwareAggregator",
    "DiscrepancyEstimate",
    "DiscrepancyModel",
    "EvidenceAggregator",
    "FixedQueryStopPolicy",
    "ListAdaptiveTraceSink",
    "LearnedEVCRouter",
    "MatchedRandomEscalationRouter",
    "LinearEVCModel",
    "ModelEvaluationError",
    "NeverStopPolicy",
    "NullAdaptiveTraceSink",
    "QueryReservation",
    "RandomEscalationRouter",
    "ROUTER_FEATURE_NAMES",
    "RunningDiscrepancyModel",
    "StopPolicy",
    "VerificationError",
    "router_features",
]
