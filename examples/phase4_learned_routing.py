"""Fit a dependency-free EVC proxy and inspect one randomized audit route."""

from __future__ import annotations

from montecarlgym import LinearEVCModel, VerifiedTransition
from montecarlgym.adaptive.routing import LearnedEVCRouter
from montecarlgym.routing import BranchSummary, RouterContext
from montecarlgym.types import ModelObservation, SearchBudget


def features(mean: float) -> dict[str, float]:
    return {
        "action_mean": mean,
        "action_uncertainty": 0.2,
        "action_risk": 0.0,
        "gap_to_best": 1.0 - mean,
        "evidence_count": 1.0,
        "search_depth": 1.0,
        "remaining_cost": 8.0,
        "remaining_accurate_calls": 1.0,
    }


records = tuple(
    VerifiedTransition(
        state_id=f"training-{index}",
        action=index,
        cheap_model_id="cheap",
        cheap_prediction=mean,
        accurate_model_id="accurate",
        verified_outcome=mean + discrepancy,
        context_features=features(mean),
    )
    for index, (mean, discrepancy) in enumerate(
        ((0.1, 0.8), (0.3, 0.6), (0.6, 0.3), (0.9, 0.1))
    )
)
evc = LinearEVCModel(ridge=0.1)
evc.fit(records)
router = LearnedEVCRouter(
    "cheap",
    "accurate",
    evc,
    accurate_cost=1.0,
    cost_weight=0.1,
    audit_probability=1.0,
    seed=7,
)
observation = ModelObservation(value=0.5, variance=0.04, cost=0.05)
context = RouterContext(
    state_id="example-root",
    candidate_actions=("left", "right"),
    evidence={"left": (observation,), "right": (observation,)},
    remaining_budget=SearchBudget(8.0, 0, 1, max_model_calls=2),
    search_depth=1,
    summaries={
        "left": BranchSummary(0.2, 0.04, 0.0, 1),
        "right": BranchSummary(0.8, 0.04, 0.0, 1),
    },
    query_counts={("left", "cheap"): 1, ("right", "cheap"): 1},
    feasible_model_ids=frozenset({"cheap", "accurate"}),
)
decision = router.choose(context)
assert decision is not None
print(
    {
        "task_action": decision.task_action,
        "compute_model": decision.model_id,
        "audit": decision.audit,
        "propensity": decision.route_propensity,
        "evc_proxy": decision.expected_value_of_compute,
    }
)
