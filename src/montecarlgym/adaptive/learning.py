"""Dependency-free contextual calibration and expected-compute learning."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..replay import VerifiedTransition, validate_transition
from .discrepancy import (
    DiscrepancyEstimate,
    RunningDiscrepancyModel,
)

ROUTER_FEATURE_NAMES = (
    "action_mean",
    "action_uncertainty",
    "action_risk",
    "gap_to_best",
    "evidence_count",
    "search_depth",
    "remaining_cost",
    "remaining_accurate_calls",
)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with partial-pivot Gaussian elimination."""

    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("contextual regression design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                item - factor * pivot_item
                for item, pivot_item in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[max(0, index)]


@dataclass(frozen=True, slots=True)
class ContextualPrediction:
    mean: float
    variance: float
    interval_half_width: float
    training_examples: int


@dataclass(slots=True)
class _CalibratedRegressor:
    feature_names: tuple[str, ...]
    ridge: float
    means: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    residual_variance: float = 0.0
    interval_half_width: float = 0.0
    training_examples: int = 0

    def fit(
        self,
        examples: Sequence[tuple[Mapping[str, float], float]],
        *,
        calibration: Sequence[tuple[Mapping[str, float], float]] = (),
        coverage: float = 0.9,
    ) -> None:
        if not examples:
            raise ValueError("at least one contextual training example is required")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be between zero and one")
        rows = [
            [float(features.get(name, 0.0)) for name in self.feature_names]
            for features, _ in examples
        ]
        labels = [float(label) for _, label in examples]
        if not all(
            math.isfinite(item) for row in rows for item in row
        ) or not all(math.isfinite(label) for label in labels):
            raise ValueError("contextual training data must be finite")
        means = tuple(
            sum(row[column] for row in rows) / len(rows)
            for column in range(len(self.feature_names))
        )
        scales = tuple(
            max(
                1e-8,
                (
                    sum(
                        (row[column] - means[column]) ** 2 for row in rows
                    )
                    / len(rows)
                )
                ** 0.5,
            )
            for column in range(len(self.feature_names))
        )
        design = [
            [1.0]
            + [
                (row[column] - means[column]) / scales[column]
                for column in range(len(self.feature_names))
            ]
            for row in rows
        ]
        width = len(self.feature_names) + 1
        gram = [[0.0 for _ in range(width)] for _ in range(width)]
        target = [0.0 for _ in range(width)]
        for row, label in zip(design, labels, strict=True):
            for left in range(width):
                target[left] += row[left] * label
                for right in range(width):
                    gram[left][right] += row[left] * row[right]
        for index in range(1, width):
            gram[index][index] += self.ridge
        # Stabilize the intercept for tiny cold-start datasets.
        gram[0][0] += 1e-12
        weights = tuple(_solve(gram, target))
        self.means = means
        self.scales = scales
        self.weights = weights
        self.training_examples = len(examples)
        train_errors = [
            label - self._predict_with(row, means, scales, weights)
            for row, label in examples
        ]
        self.residual_variance = max(
            1e-12,
            sum(error * error for error in train_errors) / len(train_errors),
        )
        calibration_source = calibration or examples
        absolute_errors = [
            abs(
                float(label)
                - self._predict_with(features, means, scales, weights)
            )
            for features, label in calibration_source
        ]
        self.interval_half_width = _quantile(absolute_errors, coverage)

    def predict(self, features: Mapping[str, float]) -> ContextualPrediction | None:
        if not self.weights:
            return None
        mean = self._predict_with(features, self.means, self.scales, self.weights)
        calibrated_variance = max(
            self.residual_variance,
            (self.interval_half_width / 1.6448536269514722) ** 2,
        )
        return ContextualPrediction(
            mean=mean,
            variance=calibrated_variance,
            interval_half_width=self.interval_half_width,
            training_examples=self.training_examples,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "ridge": self.ridge,
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "residual_variance": self.residual_variance,
            "interval_half_width": self.interval_half_width,
            "training_examples": self.training_examples,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> _CalibratedRegressor:
        feature_names = _string_sequence(payload, "feature_names")
        model = cls(
            feature_names,
            _finite_number(payload, "ridge"),
        )
        model.means = _number_sequence(payload, "means")
        model.scales = _number_sequence(payload, "scales")
        model.weights = _number_sequence(payload, "weights")
        model.residual_variance = _finite_number(payload, "residual_variance")
        model.interval_half_width = _finite_number(
            payload, "interval_half_width"
        )
        training_examples = payload.get("training_examples")
        if not isinstance(training_examples, int) or isinstance(
            training_examples, bool
        ):
            raise ValueError("regressor training_examples must be an integer")
        model.training_examples = training_examples
        expected_width = len(feature_names)
        if (
            len(model.means) != expected_width
            or len(model.scales) != expected_width
            or len(model.weights) != expected_width + 1
        ):
            raise ValueError("regressor checkpoint has inconsistent dimensions")
        if any(scale <= 0.0 for scale in model.scales):
            raise ValueError("regressor checkpoint scales must be positive")
        return model

    def _predict_with(
        self,
        features: Mapping[str, float] | Sequence[float],
        means: Sequence[float],
        scales: Sequence[float],
        weights: Sequence[float],
    ) -> float:
        if isinstance(features, Mapping):
            values = [float(features.get(name, 0.0)) for name in self.feature_names]
        else:
            values = [float(item) for item in features]
        standardized = [
            (value - means[index]) / scales[index]
            for index, value in enumerate(values)
        ]
        return weights[0] + sum(
            weight * value
            for weight, value in zip(weights[1:], standardized, strict=True)
        )


def _finite_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"regressor field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"regressor field {key!r} must be finite")
    return result


def _number_sequence(
    payload: Mapping[str, object], key: str
) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"regressor field {key!r} must be an array")
    return tuple(
        _finite_number({"value": item}, "value") for item in value
    )


def _string_sequence(
    payload: Mapping[str, object], key: str
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"regressor field {key!r} must be a non-empty array")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"regressor field {key!r} must contain strings")
    return tuple(value)


class LinearEVCModel:
    """Fit a contextual verification-utility proxy from verified replay.

    The supervised target is absolute cheap-versus-verified discrepancy. It is
    a transparent expected-value-of-compute proxy, not a causal effect estimate.
    """

    def __init__(
        self,
        *,
        feature_names: Sequence[str] = ROUTER_FEATURE_NAMES,
        ridge: float = 1.0,
    ) -> None:
        self._regressor = _CalibratedRegressor(tuple(feature_names), ridge)

    @property
    def training_examples(self) -> int:
        return self._regressor.training_examples

    def fit(
        self,
        records: Sequence[VerifiedTransition],
        *,
        calibration_records: Sequence[VerifiedTransition] = (),
        coverage: float = 0.9,
    ) -> None:
        for record in (*records, *calibration_records):
            validate_transition(record)
        examples = [
            (record.context_features, abs(record.discrepancy))
            for record in records
        ]
        calibration = [
            (record.context_features, abs(record.discrepancy))
            for record in calibration_records
        ]
        self._regressor.fit(
            examples,
            calibration=calibration,
            coverage=coverage,
        )

    def predict(self, features: Mapping[str, float]) -> ContextualPrediction | None:
        prediction = self._regressor.predict(features)
        if prediction is None:
            return None
        return ContextualPrediction(
            mean=max(0.0, prediction.mean),
            variance=prediction.variance,
            interval_half_width=prediction.interval_half_width,
            training_examples=prediction.training_examples,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"schema_version": 1, "regressor": self._regressor.to_dict()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LinearEVCModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("EVC checkpoint must be a JSON object")
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported EVC checkpoint schema")
        regressor = payload.get("regressor")
        if not isinstance(regressor, Mapping):
            raise ValueError("EVC checkpoint regressor must be an object")
        model = cls()
        model._regressor = _CalibratedRegressor.from_dict(regressor)
        return model


class CalibratedLinearDiscrepancyModel(RunningDiscrepancyModel):
    """Model-pair-specific contextual correction with calibrated intervals."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str] = ROUTER_FEATURE_NAMES,
        ridge: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_names = tuple(feature_names)
        self.ridge = ridge
        self._models: dict[tuple[str, str], _CalibratedRegressor] = {}

    def fit(
        self,
        records: Sequence[VerifiedTransition],
        *,
        calibration_records: Sequence[VerifiedTransition] = (),
        coverage: float = 0.9,
    ) -> None:
        grouped: dict[tuple[str, str], list[VerifiedTransition]] = {}
        calibration_grouped: dict[tuple[str, str], list[VerifiedTransition]] = {}
        for record in records:
            validate_transition(record)
            grouped.setdefault(
                (record.cheap_model_id, record.accurate_model_id), []
            ).append(record)
        for record in calibration_records:
            validate_transition(record)
            calibration_grouped.setdefault(
                (record.cheap_model_id, record.accurate_model_id), []
            ).append(record)
        for key, pair_records in grouped.items():
            regressor = _CalibratedRegressor(self.feature_names, self.ridge)
            regressor.fit(
                [
                    (record.context_features, record.discrepancy)
                    for record in pair_records
                ],
                calibration=[
                    (record.context_features, record.discrepancy)
                    for record in calibration_grouped.get(key, [])
                ],
                coverage=coverage,
            )
            self._models[key] = regressor
            for record in pair_records:
                super().update(
                    record.cheap_model_id,
                    record.accurate_model_id,
                    cheap_value=record.cheap_prediction,
                    verified_value=record.verified_outcome,
                )

    def estimate_contextual(
        self,
        cheap_model_id: str,
        accurate_model_id: str,
        features: Mapping[str, float],
    ) -> DiscrepancyEstimate:
        prediction = self.predict_contextual(
            cheap_model_id,
            accurate_model_id,
            features,
        )
        if prediction is None:
            return super().estimate(cheap_model_id, accurate_model_id)
        return DiscrepancyEstimate(
            mean=prediction.mean,
            variance=prediction.variance,
            count=prediction.training_examples,
        )

    def predict_contextual(
        self,
        cheap_model_id: str,
        accurate_model_id: str,
        features: Mapping[str, float],
    ) -> ContextualPrediction | None:
        """Return the calibrated prediction, including its empirical interval."""

        regressor = self._models.get((cheap_model_id, accurate_model_id))
        return None if regressor is None else regressor.predict(features)

    def update_contextual(
        self,
        cheap_model_id: str,
        accurate_model_id: str,
        *,
        cheap_value: float,
        verified_value: float,
        features: Mapping[str, float],
    ) -> None:
        del features
        super().update(
            cheap_model_id,
            accurate_model_id,
            cheap_value=cheap_value,
            verified_value=verified_value,
        )
