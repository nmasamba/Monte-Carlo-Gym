"""Reproducible exploratory analysis for matched-budget benchmark records."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist, fmean
from typing import Any

from ..causal import (
    DoublyRobustEstimator,
    InversePropensityEstimator,
    LoggedRoutingDecision,
    SelfNormalizedIPSEstimator,
)
from ..types import Fidelity
from .metrics import (
    aggregate_episode_records,
    paired_bootstrap_difference,
    pareto_frontier,
)


@dataclass(frozen=True, slots=True)
class HypervolumeReference:
    success: float
    cost: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.success <= 1.0:
            raise ValueError("reference success must be in [0, 1]")
        if self.cost < 0 or not math.isfinite(self.cost):
            raise ValueError("reference cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    count: int
    rmse: float
    brier: float | None
    nll: float | None
    interval_coverage: float | None
    ece: float | None


@dataclass(frozen=True, slots=True)
class OPEDiagnostics:
    count: int
    naive: float
    ips: float
    snips: float
    doubly_robust: float
    online_randomized: float | None
    absolute_error_vs_online: Mapping[str, float]
    effective_sample_size: float
    effective_sample_fraction: float
    minimum_propensity: float
    maximum_importance_weight: float
    poor_overlap: bool
    warnings: tuple[str, ...]


def hypervolume_2d(
    points: Sequence[Mapping[str, float]],
    *,
    reference: HypervolumeReference,
) -> float:
    """Exact union area for success-maximizing, cost-minimizing points.

    Points outside the fixed reference rectangle are rejected instead of
    silently changing the reference point after outcomes are observed.
    """

    if not points:
        return 0.0
    normalized: list[tuple[float, float]] = []
    for point in points:
        success = float(point["success"])
        cost = float(point["cost"])
        if not math.isfinite(success) or not math.isfinite(cost):
            raise ValueError("hypervolume points must be finite")
        if success < reference.success or cost > reference.cost:
            raise ValueError("hypervolume point lies outside the fixed reference")
        normalized.append((cost, success))
    # At each cost retain the best observed success, then integrate increasing
    # success slices to the fixed worse-cost reference.
    best_by_cost: dict[float, float] = {}
    for cost, success in normalized:
        best_by_cost[cost] = max(best_by_cost.get(cost, -math.inf), success)
    area = 0.0
    previous_success = reference.success
    for cost, success in sorted(best_by_cost.items()):
        if success <= previous_success:
            continue
        area += (reference.cost - cost) * (success - previous_success)
        previous_success = success
    return area


def calibration_metrics(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    *,
    variances: Sequence[float] | None = None,
    bins: int = 10,
) -> CalibrationMetrics:
    """Compute regression and probability calibration diagnostics."""

    if len(predictions) != len(outcomes) or not predictions:
        raise ValueError("calibration inputs must have equal non-zero length")
    if bins < 1:
        raise ValueError("calibration bins must be positive")
    if variances is not None and len(variances) != len(predictions):
        raise ValueError("calibration variances must align with predictions")
    values = (*predictions, *outcomes)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("calibration inputs must be finite")
    errors = [
        float(prediction) - float(outcome)
        for prediction, outcome in zip(predictions, outcomes, strict=True)
    ]
    rmse = math.sqrt(fmean(error * error for error in errors))
    probabilistic = all(0.0 <= value <= 1.0 for value in predictions) and all(
        value in (0.0, 1.0) for value in outcomes
    )
    brier: float | None = None
    nll: float | None = None
    ece: float | None = None
    if probabilistic:
        brier = fmean(error * error for error in errors)
        clipped = [min(1.0 - 1e-12, max(1e-12, value)) for value in predictions]
        nll = -fmean(
            outcome * math.log(prediction)
            + (1.0 - outcome) * math.log(1.0 - prediction)
            for prediction, outcome in zip(clipped, outcomes, strict=True)
        )
        bin_errors: list[float] = []
        for bin_index in range(bins):
            lower = bin_index / bins
            upper = (bin_index + 1) / bins
            indices = [
                index
                for index, value in enumerate(predictions)
                if lower <= value < upper or (bin_index == bins - 1 and value == 1.0)
            ]
            if not indices:
                continue
            mean_prediction = fmean(predictions[index] for index in indices)
            mean_outcome = fmean(outcomes[index] for index in indices)
            bin_errors.append(
                len(indices) / len(predictions) * abs(mean_prediction - mean_outcome)
            )
        ece = sum(bin_errors)
    coverage: float | None = None
    if variances is not None:
        if any(value < 0 or not math.isfinite(value) for value in variances):
            raise ValueError("calibration variances must be finite and non-negative")
        coverage = fmean(
            abs(outcome - prediction) <= 1.96 * variance**0.5
            for prediction, outcome, variance in zip(
                predictions,
                outcomes,
                variances,
                strict=True,
            )
        )
    return CalibrationMetrics(
        len(predictions),
        rmse,
        brier,
        nll,
        coverage,
        ece,
    )


def off_policy_diagnostics(
    records: Sequence[LoggedRoutingDecision],
    *,
    online_randomized: float | None,
    min_propensity: float = 0.05,
) -> OPEDiagnostics:
    """Compute naive/IPS/SNIPS/DR estimates plus overlap and ESS."""

    if not records:
        raise ValueError("off-policy diagnostics require logged decisions")
    naive = fmean(record.reward for record in records)
    ips = InversePropensityEstimator(min_propensity).estimate(records)
    snips = SelfNormalizedIPSEstimator(min_propensity).estimate(records)
    dr = DoublyRobustEstimator(min_propensity).estimate(records)
    weights = [record.importance_weight(min_propensity) for record in records]
    squared_sum = sum(weight * weight for weight in weights)
    ess = (sum(weights) ** 2 / squared_sum) if squared_sum else 0.0
    minimum = min(record.propensity for record in records)
    overlap_warnings: list[str] = []
    if minimum < min_propensity:
        overlap_warnings.append("behavior propensity falls below clipping threshold")
    if ess / len(records) < 0.25:
        overlap_warnings.append(
            "effective sample size is below 25% of logged decisions"
        )
    if any(
        record.target_probability > 0 and record.propensity <= 0 for record in records
    ):
        overlap_warnings.append("target policy has zero behavior-policy overlap")
    poor_overlap = bool(overlap_warnings)
    if poor_overlap:
        warnings.warn("; ".join(overlap_warnings), RuntimeWarning, stacklevel=2)
    estimates = {
        "naive": naive,
        "ips": ips,
        "snips": snips,
        "doubly_robust": dr,
    }
    errors = (
        {}
        if online_randomized is None
        else {name: abs(value - online_randomized) for name, value in estimates.items()}
    )
    return OPEDiagnostics(
        count=len(records),
        naive=naive,
        ips=ips,
        snips=snips,
        doubly_robust=dr,
        online_randomized=online_randomized,
        absolute_error_vs_online=errors,
        effective_sample_size=ess,
        effective_sample_fraction=ess / len(records),
        minimum_propensity=minimum,
        maximum_importance_weight=max(weights),
        poor_overlap=poor_overlap,
        warnings=tuple(overlap_warnings),
    )


def exploratory_power_diagnostic(
    paired_differences: Sequence[float],
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, float | int | str | None]:
    """Normal-approximation sample-size diagnostic, explicitly exploratory."""

    if not paired_differences:
        raise ValueError("power diagnostics require paired differences")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be between zero and one")
    mean = fmean(paired_differences)
    if len(paired_differences) > 1:
        variance = sum((value - mean) ** 2 for value in paired_differences) / (
            len(paired_differences) - 1
        )
        standard_deviation = variance**0.5
    else:
        standard_deviation = 0.0
    required: int | None
    if abs(mean) < 1e-12 or standard_deviation == 0.0:
        required = None
    else:
        normal = NormalDist()
        z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
        z_power = normal.inv_cdf(power)
        required = math.ceil(
            ((z_alpha + z_power) * standard_deviation / abs(mean)) ** 2
        )
    return {
        "label": "exploratory_pilot_based_normal_approximation",
        "pilot_pairs": len(paired_differences),
        "observed_mean_paired_effect": mean,
        "observed_standard_deviation": standard_deviation,
        "alpha_two_sided": alpha,
        "target_power": power,
        "required_pairs_unadjusted": required,
        "warning": (
            "Not a confirmatory sample-size decision; repeat after the benchmark, "
            "budget grid, and endpoint family are approved."
        ),
    }


def _paired_values(
    episodes: Sequence[Mapping[str, Any]],
    method: str,
    reference_method: str,
    metric: str,
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    indexed = {
        (
            str(record["task_id"]),
            int(record["paired_seed"]),
            int(record["budget_index"]),
            str(record["method"]),
        ): record
        for record in episodes
    }
    left: list[float] = []
    right: list[float] = []
    per_task: list[dict[str, Any]] = []
    units = sorted(
        {
            (
                str(record["task_id"]),
                int(record["paired_seed"]),
                int(record["budget_index"]),
            )
            for record in episodes
            if record["method"] == reference_method
        }
    )
    for task_id, seed, budget_index in units:
        reference = indexed.get((task_id, seed, budget_index, reference_method))
        candidate = indexed.get((task_id, seed, budget_index, method))
        if reference is None or candidate is None:
            continue
        candidate_value = float(candidate[metric])
        reference_value = float(reference[metric])
        left.append(reference_value)
        right.append(candidate_value)
        per_task.append(
            {
                "task_id": task_id,
                "paired_seed": seed,
                "budget_index": budget_index,
                "method": method,
                "reference_method": reference_method,
                "metric": metric,
                "reference_value": reference_value,
                "method_value": candidate_value,
                "paired_difference": reference_value - candidate_value,
            }
        )
    return left, right, per_task


def _budget_method_aggregates(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    budget_indices = sorted({int(record["budget_index"]) for record in episodes})
    return {
        str(budget_index): aggregate_episode_records(
            record for record in episodes if int(record["budget_index"]) == budget_index
        )
        for budget_index in budget_indices
    }


def analyze_sqlite_records(
    episodes: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Produce every Phase 5A aggregate from immutable raw records."""

    if not episodes:
        raise ValueError("analysis requires completed episode records")
    bootstrap_samples = int(protocol["analysis"].get("bootstrap_samples", 2_000))
    confidence = float(protocol["analysis"]["confidence_level"])
    reference_method = "fidelity_mcts"
    methods = sorted({str(record["method"]) for record in episodes})
    effects: dict[str, Any] = {}
    effects_by_budget: dict[str, dict[str, Any]] = {}
    per_task_records: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        if method == reference_method:
            continue
        method_effects: dict[str, Any] = {}
        for metric_index, metric in enumerate(("success", "return", "cost")):
            first, second, per_task = _paired_values(
                episodes,
                method,
                reference_method,
                metric,
            )
            per_task_records.extend(per_task)
            if first:
                interval = paired_bootstrap_difference(
                    first,
                    second,
                    seed=71_000 + method_index * 100 + metric_index,
                    samples=bootstrap_samples,
                    confidence=confidence,
                )
                method_effects[metric] = asdict(interval)
        effects[method] = method_effects

    for budget_index in sorted({int(record["budget_index"]) for record in episodes}):
        budget_episodes = [
            record for record in episodes if int(record["budget_index"]) == budget_index
        ]
        budget_effects: dict[str, Any] = {}
        for method_index, method in enumerate(methods):
            if method == reference_method:
                continue
            method_effects = {}
            for metric_index, metric in enumerate(("success", "return", "cost")):
                first, second, _ = _paired_values(
                    budget_episodes,
                    method,
                    reference_method,
                    metric,
                )
                if not first:
                    continue
                interval = paired_bootstrap_difference(
                    first,
                    second,
                    seed=(
                        91_000
                        + budget_index * 1_000
                        + method_index * 100
                        + metric_index
                    ),
                    samples=bootstrap_samples,
                    confidence=confidence,
                )
                method_effects[metric] = asdict(interval)
            budget_effects[method] = method_effects
        effects_by_budget[str(budget_index)] = budget_effects

    by_budget = _budget_method_aggregates(episodes)
    point_map: dict[str, dict[str, float]] = {}
    for budget_key, method_data in by_budget.items():
        for method, metrics in method_data.items():
            point_map[f"{method}@{budget_key}"] = {
                "success": float(metrics["success_rate"]),
                "cost": float(metrics["mean_cost"]),
            }
    frontier_names = pareto_frontier(
        point_map,
        maximize=("success",),
        minimize=("cost",),
    )
    reference_payload = protocol["analysis"]["hypervolume_reference"]
    reference = HypervolumeReference(
        success=float(reference_payload["success"]),
        cost=float(reference_payload["cost"]),
    )
    hypervolume_by_method: dict[str, float] = {}
    for method in methods:
        method_points = [
            point for name, point in point_map.items() if name.startswith(f"{method}@")
        ]
        hypervolume_by_method[method] = hypervolume_2d(
            method_points,
            reference=reference,
        )

    prediction_decisions = [
        record
        for record in decisions
        if record.get("model_predicted")
        and record.get("counterfactual_label") is not None
        and record.get("value") is not None
    ]
    calibration: dict[str, Any]
    if prediction_decisions:
        calibration_result = calibration_metrics(
            [float(record["value"]) for record in prediction_decisions],
            [float(record["counterfactual_label"]) for record in prediction_decisions],
            variances=[
                float(record.get("variance", 0.0)) for record in prediction_decisions
            ],
            bins=int(protocol["analysis"].get("calibration_bins", 10)),
        )
        calibration = asdict(calibration_result)
    else:
        calibration = {"count": 0, "warning": "no prediction decisions available"}

    random_decisions = [
        record
        for record in decisions
        if record["method"] == "random_matched"
        and record.get("fidelity") == Fidelity.ACCURATE.value
        and record.get("route_propensity") is not None
        and record.get("counterfactual_label") is not None
    ]
    online_randomized = (
        fmean(float(record["counterfactual_label"]) for record in random_decisions)
        if random_decisions
        else None
    )
    ope: dict[str, Any]
    if random_decisions:
        logged = tuple(
            LoggedRoutingDecision(
                context_id=str(record["decision_id"]),
                chosen_route="verify",
                reward=float(record["counterfactual_label"]),
                propensity=float(record["route_propensity"]),
                baseline_prediction=float(record.get("cheap_prediction", 0.5)),
                target_probability=1.0,
                randomized_audit=bool(record.get("randomized_audit", False)),
                feasible_routes=("skip", "verify"),
                context_features={
                    str(key): float(value)
                    for key, value in cast_mapping(
                        record.get("calibration_features", {})
                    ).items()
                },
            )
            for record in random_decisions
        )
        ope = asdict(
            off_policy_diagnostics(
                logged,
                online_randomized=online_randomized,
                min_propensity=float(
                    protocol["analysis"].get("ope_min_propensity", 0.05)
                ),
            )
        )
        ope["online_randomized_estimand"] = (
            "mean verifier pass outcome among randomized online escalation decisions"
        )
    else:
        ope = {"count": 0, "warning": "no randomized audit decisions available"}

    matched_budget = {
        budget_index: {
            "fidelity_mcts_accurate_calls": data.get("fidelity_mcts", {}).get(
                "mean_accurate_calls"
            ),
            "random_matched_accurate_calls": data.get("random_matched", {}).get(
                "mean_accurate_calls"
            ),
            "fidelity_mcts_cost": data.get("fidelity_mcts", {}).get("mean_cost"),
            "random_matched_cost": data.get("random_matched", {}).get("mean_cost"),
            "matching_basis": "calibration-partition accurate-call quota",
            "pilot_accurate_call_gap": (
                None
                if "fidelity_mcts" not in data or "random_matched" not in data
                else abs(
                    float(data["fidelity_mcts"]["mean_accurate_calls"])
                    - float(data["random_matched"]["mean_accurate_calls"])
                )
            ),
            "pilot_normalized_cost_gap": (
                None
                if "fidelity_mcts" not in data or "random_matched" not in data
                else abs(
                    float(data["fidelity_mcts"]["mean_cost"])
                    - float(data["random_matched"]["mean_cost"])
                )
            ),
        }
        for budget_index, data in by_budget.items()
    }

    ablation_names = [str(name) for name in protocol["ablations"]]
    ablations = {
        name: effects.get(name, {}) for name in ablation_names if name in effects
    }
    nonpositive = sorted(
        method
        for method, metrics in effects.items()
        if isinstance(metrics.get("success"), Mapping)
        and float(metrics["success"]["estimate"]) <= 0.0
    )
    power_first, power_second, _ = _paired_values(
        episodes,
        "learned_direct",
        reference_method,
        "success",
    )
    paired_power = [
        first - second for first, second in zip(power_first, power_second, strict=True)
    ]
    power = exploratory_power_diagnostic(paired_power)

    report = {
        "schema_version": 1,
        "stage": "exploratory",
        "notice": "Exploratory diagnostics only; these are not paper results.",
        "records": len(episodes),
        "decisions": len(decisions),
        "failures": len(failures),
        "methods_by_budget": by_budget,
        "matched_budget": matched_budget,
        "paired_effects_fidelity_minus_method": effects,
        "paired_effects_by_budget_fidelity_minus_method": effects_by_budget,
        "pareto": {
            "fixed_reference": asdict(reference),
            "frontier_points": list(frontier_names),
            "points": point_map,
            "hypervolume_by_method": hypervolume_by_method,
        },
        "router_calibration": calibration,
        "off_policy_evaluation": ope,
        "ablations": ablations,
        "power_diagnostic": power,
        "negative_or_null_success_findings": nonpositive,
        "failure_records": [dict(record) for record in failures],
    }
    return report, per_task_records


def cast_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping in raw analysis record")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(payload)
    return records


def analyze_sqlite_output(output: Path) -> dict[str, Any]:
    """Reproduce aggregates in memory without modifying an existing run."""

    protocol = json.loads((output / "resolved_protocol.json").read_text())
    episodes = read_jsonl(output / "raw" / "episodes.jsonl")
    decisions = read_jsonl(output / "raw" / "decisions.jsonl")
    failures = read_jsonl(output / "raw" / "failures.jsonl")
    report, _ = analyze_sqlite_records(
        episodes,
        decisions,
        failures,
        protocol=protocol,
    )
    return report
