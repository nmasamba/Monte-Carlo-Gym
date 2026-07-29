"""Immutable protocol fingerprints and confirmatory-run guards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..types import SearchBudget


class ExperimentStage(str, Enum):
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


def canonical_protocol_bytes(protocol: Mapping[str, Any]) -> bytes:
    """Return the stable JSON representation used for protocol identity."""

    return json.dumps(
        protocol,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_protocol_bytes(protocol)).hexdigest()


def validate_protocol_artifacts(
    protocol: Mapping[str, Any],
    root: Path,
) -> None:
    """Verify every repository-local artifact locked by the protocol."""

    artifacts = protocol.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be an object of paths to SHA-256 values")
    resolved_root = root.resolve()
    for name, expected in artifacts.items():
        relative = Path(str(name))
        if relative.is_absolute():
            raise ValueError("preregistered artifact paths must be relative")
        path = (resolved_root / relative).resolve()
        if path != resolved_root and resolved_root not in path.parents:
            raise ValueError(f"artifact path escapes repository root: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(f"preregistered artifact is missing: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != str(expected).lower():
            raise ValueError(f"preregistered artifact hash differs: {name}")


def _require_nonempty_sequence(
    protocol: Mapping[str, Any],
    key: str,
) -> Sequence[Any]:
    value = protocol.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"preregistration field {key!r} must be an array")
    if not value:
        raise ValueError(f"preregistration field {key!r} cannot be empty")
    return value


def validate_preregistration_protocol(protocol: Mapping[str, Any]) -> None:
    """Require the decisions that prevent post-result analytic flexibility."""

    required_objects = {
        "benchmark",
        "training",
        "analysis",
        "failure_policy",
        "resource_accounting",
    }
    required_sequences = {
        "hypotheses",
        "primary_endpoints",
        "secondary_endpoints",
        "methods",
        "ablations",
        "budget_grid",
        "confirmatory_seeds",
        "exclusions",
    }
    if protocol.get("protocol_version") != 1:
        raise ValueError("only preregistration protocol_version 1 is supported")
    try:
        ExperimentStage(str(protocol.get("stage")))
    except ValueError as exc:
        raise ValueError(
            "preregistration stage must be 'exploratory' or 'confirmatory'"
        ) from exc
    if not str(protocol.get("study_id", "")).strip():
        raise ValueError("preregistration study_id cannot be empty")
    for key in sorted(required_objects):
        if not isinstance(protocol.get(key), Mapping):
            raise ValueError(f"preregistration field {key!r} must be an object")
    for key in sorted(required_sequences):
        _require_nonempty_sequence(protocol, key)
    for key in ("methods", "ablations"):
        names = [
            str(item).strip()
            for item in _require_nonempty_sequence(protocol, key)
        ]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"preregistration field {key!r} must be unique names")
    method_names = {
        str(item) for item in _require_nonempty_sequence(protocol, "methods")
    }
    ablation_names = {
        str(item) for item in _require_nonempty_sequence(protocol, "ablations")
    }
    if method_names & ablation_names:
        raise ValueError("methods and ablations must use distinct identifiers")
    seeds = [
        int(seed)
        for seed in _require_nonempty_sequence(protocol, "confirmatory_seeds")
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("confirmatory_seeds must be unique")
    pilot_seeds = protocol.get("pilot_seeds", ())
    if pilot_seeds:
        if not isinstance(pilot_seeds, Sequence) or isinstance(
            pilot_seeds, (str, bytes)
        ):
            raise ValueError("pilot_seeds must be an array")
        pilot_seed_values = [int(seed) for seed in pilot_seeds]
        if set(seeds) & set(pilot_seed_values):
            raise ValueError("pilot and confirmatory seeds must be disjoint")
    for item in _require_nonempty_sequence(protocol, "budget_grid"):
        if not isinstance(item, Mapping):
            raise ValueError("every budget_grid entry must be an object")
        try:
            SearchBudget(**item)
        except (TypeError, ValueError) as exc:
            raise ValueError("budget_grid contains an invalid hard budget") from exc
    analysis = protocol["analysis"]
    assert isinstance(analysis, Mapping)
    for key in (
        "confidence_level",
        "interval_method",
        "multiple_comparison_correction",
        "stopping_rule",
    ):
        if key not in analysis:
            raise ValueError(f"analysis must freeze {key!r}")
    if analysis["stopping_rule"] != "run_all_declared_units":
        raise ValueError(
            "confirmatory stopping_rule must be 'run_all_declared_units'"
        )
    confidence = analysis["confidence_level"]
    if not isinstance(confidence, (int, float)) or not 0.0 < confidence < 1.0:
        raise ValueError("analysis confidence_level must be between zero and one")
    failure_policy = protocol["failure_policy"]
    assert isinstance(failure_policy, Mapping)
    for key in ("abort_on_error", "exclude_after_run", "retry_policy"):
        if key not in failure_policy:
            raise ValueError(f"failure_policy must freeze {key!r}")
    artifacts = protocol.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be an object of names to SHA-256 values")
    for name, digest in artifacts.items():
        if len(str(digest)) != 64 or any(
            char not in "0123456789abcdef" for char in str(digest).lower()
        ):
            raise ValueError(f"artifact {name!r} does not have a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class FrozenProtocol:
    protocol: Mapping[str, Any]
    sha256: str
    frozen_at: str
    code_revision: str
    registration_url: str | None = None


def freeze_protocol(
    protocol: Mapping[str, Any],
    destination: Path,
    *,
    code_revision: str,
    registration_url: str | None = None,
) -> FrozenProtocol:
    """Write a new immutable manifest; existing paths are never overwritten."""

    validate_preregistration_protocol(protocol)
    if protocol.get("stage") != ExperimentStage.CONFIRMATORY.value:
        raise ValueError("only confirmatory protocols may be frozen")
    if not code_revision.strip():
        raise ValueError("code_revision cannot be empty")
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen preregistration: {destination}"
        )
    digest = protocol_sha256(protocol)
    frozen_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "protocol_sha256": digest,
        "frozen_at": frozen_at,
        "code_revision": code_revision,
        "registration_url": registration_url,
        "protocol": protocol,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FrozenProtocol(
        protocol=dict(protocol),
        sha256=digest,
        frozen_at=frozen_at,
        code_revision=code_revision,
        registration_url=registration_url,
    )


def load_frozen_protocol(path: Path) -> FrozenProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "frozen":
        raise ValueError("file is not a frozen preregistration manifest")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("frozen manifest protocol must be an object")
    validate_preregistration_protocol(protocol)
    digest = protocol_sha256(protocol)
    if digest != payload.get("protocol_sha256"):
        raise ValueError("frozen preregistration fingerprint does not match content")
    return FrozenProtocol(
        protocol=dict(protocol),
        sha256=digest,
        frozen_at=str(payload["frozen_at"]),
        code_revision=str(payload["code_revision"]),
        registration_url=(
            None
            if payload.get("registration_url") is None
            else str(payload["registration_url"])
        ),
    )


def validate_confirmatory_output(output: Path) -> None:
    """Keep confirmatory artifacts separate and refuse accidental overwrite."""

    if "confirmatory" not in output.parts:
        raise ValueError(
            "confirmatory outputs must live under a directory named "
            "'confirmatory'"
        )
    validate_fresh_output(output)


def validate_fresh_output(output: Path) -> None:
    """Refuse to merge a study with artifacts from an earlier invocation."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"experiment output directory is not empty: {output}"
        )
