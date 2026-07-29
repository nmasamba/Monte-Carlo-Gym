"""Verified replay records for discrepancy learning and self-improvement."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import EvidenceProvenance


@dataclass(frozen=True, slots=True)
class VerifiedTransition:
    """Pair cheap predictions with later high-fidelity observations."""

    state_id: str
    action: Any
    cheap_model_id: str
    cheap_prediction: float
    accurate_model_id: str
    verified_outcome: float
    router_propensity: float | None = None
    cheap_provenance: EvidenceProvenance = EvidenceProvenance.SYNTHETIC
    accurate_provenance: EvidenceProvenance = EvidenceProvenance.EXECUTABLE
    context_features: Mapping[str, float] = field(default_factory=dict)
    predicted_evc: float | None = None
    randomized_audit: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def discrepancy(self) -> float:
        return self.verified_outcome - self.cheap_prediction


class VerifiedReplayStore:
    """Small in-memory reference store; production backends may be distributed."""

    def __init__(self) -> None:
        self._records: list[VerifiedTransition] = []

    def append(self, record: VerifiedTransition) -> None:
        self._records.append(record)

    def extend(self, records: Iterable[VerifiedTransition]) -> None:
        self._records.extend(records)

    def snapshot(self) -> tuple[VerifiedTransition, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)


def _encode_action(action: Any) -> Mapping[str, Any]:
    if action is None or isinstance(action, (bool, int, float, str)):
        return {"kind": "scalar", "value": action}
    if isinstance(action, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_action(item) for item in action],
        }
    raise TypeError(
        "persistent replay actions must be JSON scalars or nested tuples; "
        "provide a serializable action codec before persisting this action"
    )


def _decode_action(value: Mapping[str, Any]) -> Any:
    kind = value.get("kind")
    if kind == "scalar":
        result = value.get("value")
        if result is not None and not isinstance(
            result, (bool, int, float, str)
        ):
            raise ValueError("invalid scalar action in replay record")
        return result
    if kind == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("invalid tuple action in replay record")
        if not all(isinstance(item, Mapping) for item in items):
            raise ValueError("invalid nested action in replay record")
        return tuple(_decode_action(item) for item in items)
    raise ValueError("invalid action encoding in replay record")


def _record_to_json(record: VerifiedTransition) -> dict[str, Any]:
    validate_transition(record)
    payload = {
        "schema_version": 1,
        "state_id": record.state_id,
        "action": _encode_action(record.action),
        "cheap_model_id": record.cheap_model_id,
        "cheap_prediction": record.cheap_prediction,
        "accurate_model_id": record.accurate_model_id,
        "verified_outcome": record.verified_outcome,
        "router_propensity": record.router_propensity,
        "cheap_provenance": record.cheap_provenance.value,
        "accurate_provenance": record.accurate_provenance.value,
        "context_features": dict(record.context_features),
        "predicted_evc": record.predicted_evc,
        "randomized_audit": record.randomized_audit,
        "metadata": dict(record.metadata),
    }
    # Validate every nested metadata value before opening the append handle.
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload


def _record_from_json(payload: Mapping[str, Any]) -> VerifiedTransition:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported verified replay schema version")
    features = payload.get("context_features", {})
    metadata = payload.get("metadata", {})
    action = payload.get("action")
    if not isinstance(features, Mapping):
        raise ValueError("replay context_features must be an object")
    if not isinstance(metadata, Mapping):
        raise ValueError("replay metadata must be an object")
    if not isinstance(action, Mapping):
        raise ValueError("replay action must use the tagged action encoding")
    record = VerifiedTransition(
        state_id=str(payload["state_id"]),
        action=_decode_action(action),
        cheap_model_id=str(payload["cheap_model_id"]),
        cheap_prediction=float(payload["cheap_prediction"]),
        accurate_model_id=str(payload["accurate_model_id"]),
        verified_outcome=float(payload["verified_outcome"]),
        router_propensity=(
            None
            if payload.get("router_propensity") is None
            else float(payload["router_propensity"])
        ),
        cheap_provenance=EvidenceProvenance(
            str(payload["cheap_provenance"])
        ),
        accurate_provenance=EvidenceProvenance(
            str(payload["accurate_provenance"])
        ),
        context_features={str(key): float(item) for key, item in features.items()},
        predicted_evc=(
            None
            if payload.get("predicted_evc") is None
            else float(payload["predicted_evc"])
        ),
        randomized_audit=bool(payload.get("randomized_audit", False)),
        metadata=dict(metadata),
    )
    validate_transition(record)
    return record


class JsonlVerifiedReplayStore(VerifiedReplayStore):
    """Append-only, reloadable JSONL replay with strict schema validation."""

    def __init__(self, path: Path, *, fsync: bool = False) -> None:
        super().__init__()
        self.path = path
        self.fsync = fsync
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(
                            line,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                ValueError(f"invalid JSON constant: {value}")
                            ),
                        )
                        if not isinstance(payload, Mapping):
                            raise ValueError("record must be a JSON object")
                        self._records.append(_record_from_json(payload))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"invalid replay record at line {line_number}"
                        ) from exc

    def append(self, record: VerifiedTransition) -> None:
        payload = _record_to_json(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
        self._records.append(record)

    def extend(self, records: Iterable[VerifiedTransition]) -> None:
        for record in records:
            self.append(record)


def validate_transition(record: VerifiedTransition) -> None:
    """Validate finite training values before fitting learned components."""

    values = (
        record.cheap_prediction,
        record.verified_outcome,
        record.discrepancy,
        *record.context_features.values(),
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("verified replay contains a non-finite training value")
    if record.router_propensity is not None and not (
        0.0 < record.router_propensity <= 1.0
    ):
        raise ValueError("verified replay propensity must be in (0, 1]")
    if record.predicted_evc is not None and not math.isfinite(
        record.predicted_evc
    ):
        raise ValueError("verified replay predicted EVC must be finite")
