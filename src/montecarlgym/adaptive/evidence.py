"""Typed branch evidence and discrepancy-aware aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..types import (
    Action,
    ComputeAction,
    EvidenceProvenance,
    Fidelity,
    ModelObservation,
)
from .discrepancy import DiscrepancyModel


@dataclass(frozen=True, slots=True)
class BranchEvidence:
    """One auditable compute result attached to a task-action branch."""

    compute_action: ComputeAction
    fidelity: Fidelity
    provenance: EvidenceProvenance
    observation: ModelObservation
    query_index: int
    context_features: Mapping[str, float] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.observation.verified


@dataclass(frozen=True, slots=True)
class BranchEstimate:
    """Aggregated value, uncertainty, and risk for one task action."""

    action: Action
    mean: float
    variance: float
    risk: float
    evidence_count: int
    verified: bool
    provenance: tuple[EvidenceProvenance, ...]


class EvidenceAggregator(Protocol):
    """Aggregate branch-local evidence without choosing a task action."""

    def estimate(
        self,
        action: Action,
        evidence: Sequence[BranchEvidence],
        *,
        accurate_model_ids: Sequence[str],
    ) -> BranchEstimate | None:
        """Return ``None`` when a branch has no usable evidence."""


@dataclass(frozen=True, slots=True)
class DiscrepancyAwareAggregator:
    """Prefer verified evidence; otherwise correct cheap model estimates."""

    discrepancy: DiscrepancyModel

    def estimate(
        self,
        action: Action,
        evidence: Sequence[BranchEvidence],
        *,
        accurate_model_ids: Sequence[str],
    ) -> BranchEstimate | None:
        if not evidence:
            return None
        verified = [item for item in evidence if item.verified]
        selected = verified or list(evidence)
        values: list[float] = []
        variances: list[float] = []
        risks: list[float] = []
        provenances: list[EvidenceProvenance] = []
        accurate_id = accurate_model_ids[0] if accurate_model_ids else None

        for item in selected:
            value = item.observation.value
            variance = item.observation.variance
            if not verified and item.fidelity is not Fidelity.ACCURATE:
                if accurate_id is not None:
                    contextual_estimate = getattr(
                        self.discrepancy,
                        "estimate_contextual",
                        None,
                    )
                    if contextual_estimate is None:
                        correction = self.discrepancy.estimate(
                            item.compute_action.model_id,
                            accurate_id,
                        )
                    else:
                        correction = contextual_estimate(
                            item.compute_action.model_id,
                            accurate_id,
                            item.context_features,
                        )
                    value += correction.mean
                    variance += correction.variance
            values.append(value)
            variances.append(variance)
            risks.append(item.observation.risk)
            provenances.append(item.provenance)

        mean = sum(values) / len(values)
        within_variance = sum(variances) / (len(variances) ** 2)
        between_variance = (
            sum((value - mean) ** 2 for value in values) / len(values)
            if len(values) > 1
            else 0.0
        )
        variance = within_variance + between_variance
        if not math.isfinite(mean) or not math.isfinite(variance):
            raise ValueError("aggregated branch estimates must be finite")
        return BranchEstimate(
            action=action,
            mean=mean,
            variance=variance,
            risk=sum(risks) / len(risks),
            evidence_count=len(evidence),
            verified=bool(verified),
            provenance=tuple(dict.fromkeys(provenances)),
        )


def aggregate_branches(
    aggregator: EvidenceAggregator,
    evidence: Mapping[Action, Sequence[BranchEvidence]],
    *,
    accurate_model_ids: Sequence[str],
) -> dict[Action, BranchEstimate]:
    """Aggregate every non-empty branch in deterministic mapping order."""

    estimates: dict[Action, BranchEstimate] = {}
    for action, branch_evidence in evidence.items():
        estimate = aggregator.estimate(
            action,
            branch_evidence,
            accurate_model_ids=accurate_model_ids,
        )
        if estimate is not None:
            estimates[action] = estimate
    return estimates
