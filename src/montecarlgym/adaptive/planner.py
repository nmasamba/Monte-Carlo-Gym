"""Branch-level adaptive compute planner with verified replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from random import Random
from typing import Any, Protocol

from ..core.budget import BudgetExhausted
from ..core.tree import DefaultStateCodec, StateCodec
from ..models import ModelPortfolio
from ..planner import PlanResult
from ..replay import VerifiedReplayStore, VerifiedTransition
from ..routing import BranchSummary, ComputeRouter, RouterContext
from ..types import (
    Action,
    ComputeAction,
    Fidelity,
    ResourceUsage,
    SearchBudget,
    State,
)
from .budget import AdaptiveResourceLedger
from .discrepancy import RunningDiscrepancyModel
from .evidence import (
    BranchEstimate,
    BranchEvidence,
    DiscrepancyAwareAggregator,
    EvidenceAggregator,
    aggregate_branches,
)
from .routing import router_features
from .stopping import NeverStopPolicy, StopPolicy


class ActionProvider(Protocol):
    """Return legal task actions without mutating the live system."""

    def legal_actions(self, state: State) -> Sequence[Action]:
        """Return the candidate task actions for ``state``."""


@dataclass(frozen=True, slots=True)
class AdaptiveQueryTrace:
    """Structured audit record for one successful compute action."""

    query_index: int
    state_id: str
    task_action: Action
    model_id: str
    fidelity: str
    provenance: str
    verified: bool
    request_verification: bool
    route_propensity: float | None
    token_budget: int
    rollout_depth: int
    value: float
    variance: float
    risk: float
    cost: float
    tokens: int
    environment_calls: int
    randomized_audit: bool
    expected_value_of_compute: float | None


class AdaptiveTraceSink(Protocol):
    """Consume structured routing traces without receiving private reasoning."""

    def record(self, trace: AdaptiveQueryTrace) -> None:
        """Persist or stream one completed query record."""


@dataclass(frozen=True, slots=True)
class NullAdaptiveTraceSink:
    def record(self, trace: AdaptiveQueryTrace) -> None:
        del trace


@dataclass(slots=True)
class ListAdaptiveTraceSink:
    records: list[AdaptiveQueryTrace] = field(default_factory=list)

    def record(self, trace: AdaptiveQueryTrace) -> None:
        self.records.append(trace)


@dataclass(frozen=True, slots=True)
class AdaptiveSearchReport:
    """Auditable outcome of one adaptive planning call."""

    usage: ResourceUsage
    stop_reason: str
    state_id: str
    branch_estimates: Mapping[Action, BranchEstimate]
    query_counts: Mapping[tuple[Action, str], int]
    replay_records_added: int
    randomized_audit_queries: int = 0


@dataclass(frozen=True, slots=True)
class AdaptivePlanResult(PlanResult):
    """Plan result extended with branch and routing diagnostics."""

    report: AdaptiveSearchReport | None = None


class ModelEvaluationError(RuntimeError):
    """A model call failed after its conservative quote was charged."""

    def __init__(
        self,
        model_id: str,
        *,
        usage: ResourceUsage,
    ) -> None:
        super().__init__(f"model evaluation failed: {model_id}")
        self.model_id = model_id
        self.usage = usage


class VerificationError(RuntimeError):
    """A route requested verification but the model did not provide it."""

    def __init__(
        self,
        model_id: str,
        *,
        usage: ResourceUsage,
    ) -> None:
        super().__init__(
            f"model {model_id!r} did not satisfy a verification request"
        )
        self.model_id = model_id
        self.usage = usage


@dataclass(slots=True)
class AdaptiveComputePlanner:
    """Allocate model queries across task branches, then recommend an action.

    Compute actions only change planner evidence. The returned task action is
    never executed by this class, preserving the live-system side-effect
    boundary used by the classical Gymnasium agent.
    """

    action_provider: ActionProvider
    router: ComputeRouter
    stop_policy: StopPolicy = field(default_factory=NeverStopPolicy)
    codec: StateCodec = field(default_factory=DefaultStateCodec)
    discrepancy: RunningDiscrepancyModel = field(
        default_factory=RunningDiscrepancyModel
    )
    aggregator: EvidenceAggregator | None = None
    replay: VerifiedReplayStore = field(default_factory=VerifiedReplayStore)
    trace_sink: AdaptiveTraceSink = field(
        default_factory=NullAdaptiveTraceSink
    )
    risk_aversion: float = 0.0
    search_depth: int = 0

    def __post_init__(self) -> None:
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if self.search_depth < 0:
            raise ValueError("search_depth must be non-negative")
        if self.aggregator is None:
            self.aggregator = DiscrepancyAwareAggregator(self.discrepancy)

    def plan(
        self,
        state: State,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> AdaptivePlanResult:
        return self.plan_candidates(
            state,
            candidate_actions=self.action_provider.legal_actions(state),
            models=models,
            budget=budget,
            seed=seed,
        )

    def plan_candidates(
        self,
        state: State,
        *,
        candidate_actions: Sequence[Action],
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> AdaptivePlanResult:
        """Plan over an explicit frontier-local legal-action set."""

        candidates = tuple(candidate_actions)
        if not candidates:
            raise ValueError("adaptive planning requires at least one legal action")
        if len(set(candidates)) != len(candidates):
            raise ValueError("action provider returned duplicate task actions")

        state_id = repr(self.codec.key(state))
        evidence: dict[Action, list[BranchEvidence]] = {
            action: [] for action in candidates
        }
        query_counts: dict[tuple[Action, str], int] = {}
        traces: list[Mapping[str, Any]] = []
        ledger = AdaptiveResourceLedger(budget)
        rng = Random(seed)
        replay_before = len(self.replay)
        accurate_models = models.by_fidelity(Fidelity.ACCURATE)
        accurate_model_ids = tuple(model.model_id for model in accurate_models)
        stop_reason = "router_exhausted"

        while True:
            estimates = self._estimates(
                evidence,
                accurate_model_ids=accurate_model_ids,
            )
            context = self._context(
                state_id=state_id,
                candidates=candidates,
                evidence=evidence,
                estimates=estimates,
                query_counts=query_counts,
                remaining_budget=ledger.remaining_budget(),
                model_ids=models.ids(),
            )
            if estimates and self.stop_policy.should_stop(context):
                stop_reason = "stop_policy"
                break
            compute_action = self.router.choose(context)
            if compute_action is None:
                stop_reason = "router_exhausted"
                break
            self._validate_compute_action(
                compute_action,
                state_id=state_id,
                candidates=candidates,
                models=models,
            )
            context_features = router_features(
                context,
                compute_action.task_action,
            )
            model = models.get(compute_action.model_id)
            if (
                compute_action.request_verification
                and model.fidelity is not Fidelity.ACCURATE
            ):
                raise ValueError(
                    "verification requests require an accurate-fidelity model"
                )
            quote = model.quote(
                token_budget=compute_action.token_budget,
                rollout_depth=compute_action.rollout_depth,
            )
            if model.fidelity is Fidelity.ACCURATE and quote.accurate_calls < 1:
                raise ValueError(
                    "accurate-fidelity models must reserve an accurate call"
                )
            reservation = ledger.reserve(quote)
            if reservation is None:
                stop_reason = ledger.stop_reason
                break
            try:
                observation = model.evaluate(
                    state,
                    compute_action.task_action,
                    token_budget=compute_action.token_budget,
                    rollout_depth=compute_action.rollout_depth,
                    rng=rng,
                )
            except Exception as exc:
                ledger.fail(reservation)
                raise ModelEvaluationError(
                    model.model_id,
                    usage=ledger.usage(),
                ) from exc
            ledger.commit(reservation, observation)
            if compute_action.request_verification and not observation.verified:
                raise VerificationError(
                    model.model_id,
                    usage=ledger.usage(),
                )

            item = BranchEvidence(
                compute_action=compute_action,
                fidelity=model.fidelity,
                provenance=observation.provenance,
                observation=observation,
                query_index=ledger.iterations,
                context_features=dict(context_features),
            )
            branch = evidence[compute_action.task_action]
            branch.append(item)
            count_key = (compute_action.task_action, model.model_id)
            query_counts[count_key] = query_counts.get(count_key, 0) + 1
            self._record_verified_pair(
                state_id=state_id,
                current=item,
                branch=branch,
                context_features=context_features,
            )
            trace = AdaptiveQueryTrace(
                query_index=item.query_index,
                state_id=state_id,
                task_action=compute_action.task_action,
                model_id=model.model_id,
                fidelity=model.fidelity.value,
                provenance=observation.provenance.value,
                verified=observation.verified,
                request_verification=compute_action.request_verification,
                route_propensity=compute_action.route_propensity,
                token_budget=compute_action.token_budget,
                rollout_depth=compute_action.rollout_depth,
                value=observation.value,
                variance=observation.variance,
                risk=observation.risk,
                cost=observation.cost,
                tokens=observation.tokens,
                environment_calls=observation.environment_calls,
                randomized_audit=compute_action.audit,
                expected_value_of_compute=(
                    compute_action.expected_value_of_compute
                ),
            )
            self.trace_sink.record(trace)
            traces.append(asdict(trace))

        estimates = self._estimates(
            evidence,
            accurate_model_ids=accurate_model_ids,
        )
        if not estimates:
            raise BudgetExhausted(
                "budget could not afford one model query for any task branch"
            )
        selected = max(
            (estimates[action] for action in candidates if action in estimates),
            key=lambda estimate: (
                estimate.mean
                - self.risk_aversion
                * (estimate.risk + estimate.variance**0.5)
            ),
        )
        usage = ledger.usage()
        report = AdaptiveSearchReport(
            usage=usage,
            stop_reason=stop_reason,
            state_id=state_id,
            branch_estimates=dict(estimates),
            query_counts=dict(query_counts),
            replay_records_added=len(self.replay) - replay_before,
            randomized_audit_queries=sum(
                bool(trace["randomized_audit"]) for trace in traces
            ),
        )
        return AdaptivePlanResult(
            action=selected.action,
            predicted_value=selected.mean,
            usage=usage,
            trace=tuple(traces),
            report=report,
        )

    def _estimates(
        self,
        evidence: Mapping[Action, Sequence[BranchEvidence]],
        *,
        accurate_model_ids: Sequence[str],
    ) -> dict[Action, BranchEstimate]:
        aggregator = self.aggregator
        if aggregator is None:  # Defensive for custom post-init mutation.
            raise RuntimeError("adaptive planner has no evidence aggregator")
        return aggregate_branches(
            aggregator,
            evidence,
            accurate_model_ids=accurate_model_ids,
        )

    def _context(
        self,
        *,
        state_id: str,
        candidates: tuple[Action, ...],
        evidence: Mapping[Action, Sequence[BranchEvidence]],
        estimates: Mapping[Action, BranchEstimate],
        query_counts: Mapping[tuple[Action, str], int],
        remaining_budget: SearchBudget,
        model_ids: Sequence[str],
    ) -> RouterContext:
        return RouterContext(
            state_id=state_id,
            candidate_actions=candidates,
            evidence={
                action: tuple(item.observation for item in branch)
                for action, branch in evidence.items()
            },
            remaining_budget=remaining_budget,
            search_depth=self.search_depth,
            summaries={
                action: BranchSummary(
                    mean=estimate.mean,
                    variance=estimate.variance,
                    risk=estimate.risk,
                    evidence_count=estimate.evidence_count,
                    verified=estimate.verified,
                )
                for action, estimate in estimates.items()
            },
            query_counts=dict(query_counts),
            verified_actions=frozenset(
                action for action, estimate in estimates.items() if estimate.verified
            ),
            feasible_model_ids=frozenset(model_ids),
        )

    @staticmethod
    def _validate_compute_action(
        compute_action: ComputeAction,
        *,
        state_id: str,
        candidates: Sequence[Action],
        models: ModelPortfolio,
    ) -> None:
        if compute_action.state_id != state_id:
            raise ValueError("router returned a compute action for another state")
        if compute_action.task_action not in candidates:
            raise ValueError("router returned an illegal task-action branch")
        models.get(compute_action.model_id)

    def _record_verified_pair(
        self,
        *,
        state_id: str,
        current: BranchEvidence,
        branch: Sequence[BranchEvidence],
        context_features: Mapping[str, float],
    ) -> None:
        if current.fidelity is not Fidelity.ACCURATE or not current.verified:
            return
        cheap = next(
            (
                item
                for item in reversed(branch[:-1])
                if item.fidelity is not Fidelity.ACCURATE
            ),
            None,
        )
        if cheap is None:
            return
        contextual_update = getattr(self.discrepancy, "update_contextual", None)
        if contextual_update is None:
            self.discrepancy.update(
                cheap.compute_action.model_id,
                current.compute_action.model_id,
                cheap_value=cheap.observation.value,
                verified_value=current.observation.value,
            )
        else:
            contextual_update(
                cheap.compute_action.model_id,
                current.compute_action.model_id,
                cheap_value=cheap.observation.value,
                verified_value=current.observation.value,
                features=context_features,
            )
        self.replay.append(
            VerifiedTransition(
                state_id=state_id,
                action=current.compute_action.task_action,
                cheap_model_id=cheap.compute_action.model_id,
                cheap_prediction=cheap.observation.value,
                accurate_model_id=current.compute_action.model_id,
                verified_outcome=current.observation.value,
                router_propensity=current.compute_action.route_propensity,
                cheap_provenance=cheap.provenance,
                accurate_provenance=current.provenance,
                context_features=dict(context_features),
                predicted_evc=(
                    current.compute_action.expected_value_of_compute
                ),
                randomized_audit=current.compute_action.audit,
                metadata={
                    "cheap_query_index": cheap.query_index,
                    "accurate_query_index": current.query_index,
                },
            )
        )
