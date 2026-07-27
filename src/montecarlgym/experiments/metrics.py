"""Dependency-free aggregation, Pareto, and paired-bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


def aggregate_episode_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate the stable core metrics by method."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["method"]), []).append(record)

    result: dict[str, dict[str, float | int]] = {}
    for method, method_records in sorted(grouped.items()):
        result[method] = {
            "episodes": len(method_records),
            "success_rate": fmean(
                float(record["success"]) for record in method_records
            ),
            "mean_return": fmean(
                float(record["return"]) for record in method_records
            ),
            "mean_regret": fmean(
                float(record["regret"]) for record in method_records
            ),
            "mean_cost": fmean(
                float(record["cost"]) for record in method_records
            ),
            "mean_tokens": fmean(
                float(record["tokens"]) for record in method_records
            ),
            "mean_accurate_calls": fmean(
                float(record["accurate_calls"]) for record in method_records
            ),
        }
    return result


def dominates(
    candidate: Mapping[str, float],
    other: Mapping[str, float],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> bool:
    """Return whether ``candidate`` weakly improves every objective and one strictly."""

    weak = all(candidate[key] >= other[key] for key in maximize) and all(
        candidate[key] <= other[key] for key in minimize
    )
    strict = any(candidate[key] > other[key] for key in maximize) or any(
        candidate[key] < other[key] for key in minimize
    )
    return weak and strict


def pareto_frontier(
    points: Mapping[str, Mapping[str, float]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> tuple[str, ...]:
    """Return non-dominated method identifiers in deterministic order."""

    frontier = []
    for name, point in sorted(points.items()):
        if not any(
            other_name != name
            and dominates(other, point, maximize=maximize, minimize=minimize)
            for other_name, other in points.items()
        ):
            frontier.append(name)
    return tuple(frontier)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    seed: int,
    samples: int = 2_000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap the paired mean difference ``first - second``."""

    if len(first) != len(second) or not first:
        raise ValueError("paired samples must have equal non-zero length")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    differences = [left - right for left, right in zip(first, second)]
    rng = Random(seed)
    bootstrapped = sorted(
        fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, int(tail * samples))
    upper_index = min(samples - 1, int((1.0 - tail) * samples) - 1)
    return BootstrapInterval(
        estimate=fmean(differences),
        lower=bootstrapped[lower_index],
        upper=bootstrapped[upper_index],
        confidence=confidence,
    )
