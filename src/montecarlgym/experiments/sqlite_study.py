"""Guarded exploratory runner for the offline SQLite L2 benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from ..planner import Planner
from ..types import Fidelity, SearchBudget
from .analysis import analyze_sqlite_records
from .metrics import aggregate_episode_records
from .preregistration import (
    ExperimentStage,
    protocol_sha256,
    validate_fresh_output,
    validate_preregistration_protocol,
)
from .sqlite_l2 import (
    BENCHMARK_ID,
    PRIMARY_SQLITE_METHODS,
    SQLITE_ABLATIONS,
    SQLitePartition,
    calibrate_matched_random_calls,
    get_sqlite_task,
    make_sqlite_frontier_evaluator,
    make_sqlite_planner,
    make_sqlite_portfolio,
    paired_sqlite_task,
    partition_manifest,
    planner_report_fields,
    score_sqlite_action,
    sqlite_fixture_artifact_hash,
    train_sqlite_components,
)

RAW_SCHEMA_VERSION = 1


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _record_with_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    if "record_sha256" in payload:
        raise ValueError("raw record already contains record_sha256")
    payload["record_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def validate_raw_record(record: Mapping[str, Any], *, kind: str) -> None:
    """Validate the immutable raw-record envelope and embedded fingerprint."""

    common = {
        "schema_version",
        "record_kind",
        "stage",
        "benchmark",
        "method",
        "task_id",
        "paired_seed",
        "budget_index",
        "record_sha256",
    }
    required = {
        "decision": common
        | {
            "decision_id",
            "action_kind",
            "task_action",
            "provenance",
            "synthetic",
            "model_predicted",
            "executable",
            "verified",
            "execution_status",
        },
        "episode": common
        | {
            "success",
            "return",
            "normalized_cost",
            "calls_by_fidelity",
            "execution_failures",
            "latency_s",
            "stopping_reason",
            "route_propensities",
            "calibration_features",
            "provenance",
        },
        "failure": common | {"error_type", "message"},
    }
    if kind not in required:
        raise ValueError(f"unknown raw record kind: {kind}")
    missing = required[kind] - set(record)
    if missing:
        raise ValueError(f"raw {kind} record is missing fields: {sorted(missing)}")
    if record["schema_version"] != RAW_SCHEMA_VERSION:
        raise ValueError("unsupported raw record schema version")
    if record["record_kind"] != kind:
        raise ValueError("raw record kind differs from validator kind")
    digest = record["record_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("raw record SHA-256 is invalid")
    payload = dict(record)
    del payload["record_sha256"]
    if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != digest:
        raise ValueError("raw record SHA-256 does not match its content")


def validate_exploratory_output(output: Path) -> None:
    """Reject output locations that could overlap protected study surfaces."""

    protected = {"confirmatory", "preregistered"}
    if protected & {part.lower() for part in output.parts}:
        raise ValueError(
            "exploratory outputs cannot use confirmatory or preregistered paths"
        )
    validate_fresh_output(output)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )


def _write_jsonl_immutable(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            validate_raw_record(record, kind=kind)
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_method_seed(task_id: str, paired_seed: int, method: str) -> int:
    payload = f"{task_id}:{paired_seed}:{method}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _protocol_methods(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    methods = tuple(str(method) for method in protocol["methods"])
    ablations = tuple(str(method) for method in protocol["ablations"])
    combined = tuple(dict.fromkeys((*methods, *ablations)))
    unknown = set(combined) - set((*PRIMARY_SQLITE_METHODS, *SQLITE_ABLATIONS))
    if unknown:
        raise ValueError(f"unknown SQLite methods: {sorted(unknown)}")
    missing = set(PRIMARY_SQLITE_METHODS) - set(methods)
    if missing:
        raise ValueError(
            "SQLite exploratory protocols must include every required baseline: "
            + ", ".join(sorted(missing))
        )
    return combined


def _validate_sqlite_protocol(protocol: Mapping[str, Any]) -> None:
    validate_preregistration_protocol(protocol)
    if protocol.get("stage") != ExperimentStage.EXPLORATORY.value:
        raise PermissionError(
            "Phase 5A may run exploratory SQLite protocols only; future "
            "confirmatory fixtures are intentionally unavailable"
        )
    benchmark = protocol["benchmark"]
    if not isinstance(benchmark, Mapping):
        raise ValueError("benchmark must be an object")
    if benchmark.get("type") != BENCHMARK_ID:
        raise ValueError("protocol does not target the SQLite L2 benchmark")
    task_ids = benchmark.get("exploratory_task_ids")
    if not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes)):
        raise ValueError("benchmark.exploratory_task_ids must be an array")
    allowed = {
        task.task_id
        for task in (get_sqlite_task(str(task_id)) for task_id in task_ids)
        if task.partition is SQLitePartition.EXPLORATORY
    }
    if allowed != {str(task_id) for task_id in task_ids}:
        raise PermissionError("exploratory runs may access exploratory task IDs only")
    budgets = protocol["budget_grid"]
    if not isinstance(budgets, Sequence) or len(budgets) < 5:
        raise ValueError("SQLite exploratory protocols require at least five budgets")
    _protocol_methods(protocol)


def _normalize_compute_decisions(
    *,
    result_trace: Sequence[Mapping[str, Any]],
    method: str,
    task_id: str,
    paired_seed: int,
    budget_index: int,
    truth_by_action: Mapping[str, bool],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    cheap_by_action: dict[str, float] = {}
    for index, trace in enumerate(result_trace, start=1):
        action = str(trace.get("task_action"))
        provenance = str(trace.get("provenance", "synthetic"))
        model_predicted = bool(
            trace.get(
                "model_predicted",
                provenance in {"learned", "model_predicted"},
            )
        )
        executable = bool(trace.get("executable", provenance == "executable"))
        action_kind = (
            "task" if trace.get("decision_type") == "task_action" else "compute"
        )
        value = trace.get("value")
        if model_predicted and value is not None:
            cheap_by_action[action] = float(value)
        record = _record_with_hash(
            {
                "schema_version": RAW_SCHEMA_VERSION,
                "record_kind": "decision",
                "stage": ExperimentStage.EXPLORATORY.value,
                "benchmark": BENCHMARK_ID,
                "decision_id": (
                    f"{task_id}:{paired_seed}:{budget_index}:{method}:compute:{index}"
                ),
                "method": method,
                "task_id": task_id,
                "paired_seed": paired_seed,
                "budget_index": budget_index,
                "query_index": int(trace.get("query_index", index)),
                "action_kind": action_kind,
                "task_action": action,
                "model_id": trace.get("model_id"),
                "fidelity": trace.get("fidelity"),
                "provenance": provenance,
                "synthetic": bool(trace.get("synthetic", provenance == "synthetic")),
                "model_predicted": model_predicted,
                "executable": executable,
                "verified": bool(trace.get("verified", False)),
                "verifier_passed": trace.get("verifier_passed"),
                "execution_status": str(trace.get("execution_status", "completed")),
                "failure_type": trace.get("failure_type"),
                "terminated": bool(trace.get("terminated", False)),
                "truncated": bool(trace.get("truncated", False)),
                "route_propensity": trace.get("route_propensity"),
                "randomized_audit": bool(
                    trace.get("randomized_audit", trace.get("audit", False))
                ),
                "expected_value_of_compute": trace.get("expected_value_of_compute"),
                "value": value,
                "variance": trace.get("variance"),
                "cost": float(trace.get("cost", 0.0)),
                "tokens": int(trace.get("tokens", 0)),
                "environment_calls": int(trace.get("environment_calls", 0)),
                "latency_s": float(trace.get("latency_s", 0.0)),
                "calibration_features": dict(
                    cast(
                        Mapping[str, float],
                        trace.get("calibration_features", {}),
                    )
                ),
                "counterfactual_label": (
                    None if action not in truth_by_action else truth_by_action[action]
                ),
                "cheap_prediction": cheap_by_action.get(action),
                "analysis_counterfactual": action in truth_by_action,
            }
        )
        normalized.append(record)
    # Accurate decisions occur after their cheap partner in every routed method.
    for record in normalized:
        if record.get("fidelity") == Fidelity.ACCURATE.value:
            action = str(record["task_action"])
            if action in cheap_by_action:
                mutable = dict(record)
                del mutable["record_sha256"]
                mutable["cheap_prediction"] = cheap_by_action[action]
                record.clear()
                record.update(_record_with_hash(mutable))
    return normalized


def _task_execution_decision(
    *,
    method: str,
    task_id: str,
    paired_seed: int,
    budget_index: int,
    action: str,
    result: Any,
) -> dict[str, Any]:
    return _record_with_hash(
        {
            "schema_version": RAW_SCHEMA_VERSION,
            "record_kind": "decision",
            "stage": ExperimentStage.EXPLORATORY.value,
            "benchmark": BENCHMARK_ID,
            "decision_id": (
                f"{task_id}:{paired_seed}:{budget_index}:{method}:task:final"
            ),
            "method": method,
            "task_id": task_id,
            "paired_seed": paired_seed,
            "budget_index": budget_index,
            "query_index": None,
            "action_kind": "task",
            "task_action": action,
            "model_id": None,
            "fidelity": None,
            "provenance": "executable",
            "synthetic": False,
            "model_predicted": False,
            "executable": True,
            "verified": True,
            "verifier_passed": result.verifier_passed,
            "execution_status": result.status.value,
            "failure_type": result.error_type,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "route_propensity": 1.0,
            "randomized_audit": False,
            "expected_value_of_compute": None,
            "value": 1.0 if result.verifier_passed else -1.0,
            "variance": 0.0,
            "cost": 0.0,
            "tokens": 0,
            "environment_calls": 0,
            "latency_s": result.latency_s,
            "calibration_features": {},
            "counterfactual_label": result.verifier_passed,
            "cheap_prediction": None,
            "analysis_counterfactual": False,
        }
    )


def _failure_record(
    *,
    method: str,
    task_id: str,
    paired_seed: int,
    budget_index: int,
    exc: Exception,
) -> dict[str, Any]:
    return _record_with_hash(
        {
            "schema_version": RAW_SCHEMA_VERSION,
            "record_kind": "failure",
            "stage": ExperimentStage.EXPLORATORY.value,
            "benchmark": BENCHMARK_ID,
            "method": method,
            "task_id": task_id,
            "paired_seed": paired_seed,
            "budget_index": budget_index,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    )


def run_sqlite_study(
    protocol: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Run the paired exploratory protocol and persist raw records first."""

    _validate_sqlite_protocol(protocol)
    validate_exploratory_output(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(output / "resolved_protocol.json", protocol)
    _write_json_exclusive(output / "partitions.json", partition_manifest())
    _write_json_exclusive(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "sqlite": sqlite3_version(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": ExperimentStage.EXPLORATORY.value,
            "protocol_sha256": protocol_sha256(protocol),
            "notice": "Exploratory diagnostics only; not paper results.",
        },
    )
    budgets = tuple(SearchBudget(**item) for item in protocol["budget_grid"])
    components = train_sqlite_components(budget=budgets[-1])
    settings = cast(
        Mapping[str, Mapping[str, Any]],
        protocol.get("method_settings", {}),
    )
    fidelity_settings = settings.get("fidelity_mcts", {})
    matched_calls = calibrate_matched_random_calls(
        budgets,
        components=components,
        method_settings=fidelity_settings,
    )
    methods = _protocol_methods(protocol)
    benchmark = cast(Mapping[str, Any], protocol["benchmark"])
    task_ids = tuple(str(task_id) for task_id in benchmark["exploratory_task_ids"])
    paired_seeds = tuple(int(seed) for seed in protocol["pilot_seeds"])
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    abort_on_error = bool(protocol["failure_policy"].get("abort_on_error", True))

    for task_id in task_ids:
        base_task = get_sqlite_task(task_id)
        for paired_seed in paired_seeds:
            task = paired_sqlite_task(base_task, paired_seed)
            truth_by_action = {
                action: score_sqlite_action(task, action).verifier_passed
                for action in task.actions
            }
            for budget_index, budget in enumerate(budgets):
                for method in methods:
                    seed = _stable_method_seed(task_id, paired_seed, method)
                    method_settings = (
                        fidelity_settings
                        if method in SQLITE_ABLATIONS
                        else settings.get(method, {})
                    )
                    try:
                        portfolio = make_sqlite_portfolio(
                            cheap_cost=float(benchmark.get("cheap_query_cost", 0.25)),
                            executable_cost=float(
                                benchmark.get("executable_query_cost", 4.0)
                            ),
                            timeout_s=float(benchmark.get("execution_timeout_s", 0.25)),
                            maximum_vm_steps=int(
                                benchmark.get("maximum_vm_steps", 100_000)
                            ),
                        )
                        planner: Planner
                        if method == "fidelity_mcts":
                            frontier_evaluator = make_sqlite_frontier_evaluator(
                                components=components,
                                budget=budget,
                                seed=seed,
                                method_settings=method_settings,
                                portfolio=portfolio,
                            )
                            planner = frontier_evaluator.planner
                        else:
                            planner = make_sqlite_planner(
                                method,
                                components=components,
                                matched_accurate_calls=matched_calls[budget_index],
                                seed=seed,
                                method_settings=method_settings,
                            )
                        result = planner.plan(
                            task,
                            models=portfolio,
                            budget=budget,
                            seed=seed,
                        )
                        final_execution = score_sqlite_action(task, result.action)
                    except Exception as exc:
                        failures.append(
                            _failure_record(
                                method=method,
                                task_id=task_id,
                                paired_seed=paired_seed,
                                budget_index=budget_index,
                                exc=exc,
                            )
                        )
                        if abort_on_error:
                            _persist_raw(output, episodes, decisions, failures)
                            raise
                        continue

                    compute_decisions = _normalize_compute_decisions(
                        result_trace=result.trace,
                        method=method,
                        task_id=task_id,
                        paired_seed=paired_seed,
                        budget_index=budget_index,
                        truth_by_action=truth_by_action,
                    )
                    decisions.extend(compute_decisions)
                    decisions.append(
                        _task_execution_decision(
                            method=method,
                            task_id=task_id,
                            paired_seed=paired_seed,
                            budget_index=budget_index,
                            action=str(result.action),
                            result=final_execution,
                        )
                    )
                    compute_only = [
                        record
                        for record in compute_decisions
                        if record["action_kind"] == "compute"
                    ]
                    calls_by_fidelity = {
                        fidelity.value: sum(
                            record.get("fidelity") == fidelity.value
                            for record in compute_only
                        )
                        for fidelity in Fidelity
                    }
                    execution_failures = sum(
                        record["execution_status"]
                        in {"sql_error", "timeout", "rejected"}
                        for record in (*compute_only, decisions[-1])
                    )
                    route_propensities = [
                        float(record["route_propensity"])
                        for record in compute_only
                        if record.get("route_propensity") is not None
                    ]
                    calibration_features = [
                        dict(cast(Mapping[str, float], record["calibration_features"]))
                        for record in compute_only
                        if record["calibration_features"]
                    ]
                    stop_reason, report_fields = planner_report_fields(result)
                    episode = _record_with_hash(
                        {
                            "schema_version": RAW_SCHEMA_VERSION,
                            "record_kind": "episode",
                            "stage": ExperimentStage.EXPLORATORY.value,
                            "benchmark": BENCHMARK_ID,
                            "method": method,
                            "task_id": task_id,
                            "paired_seed": paired_seed,
                            "seed": seed,
                            "budget_index": budget_index,
                            "fixture_sha256": task.fixture_sha256,
                            "template_family": task.template_family,
                            "action": result.action,
                            "success": final_execution.verifier_passed,
                            "return": 1.0 if final_execution.verifier_passed else 0.0,
                            "regret": 0.0 if final_execution.verifier_passed else 1.0,
                            "predicted_value": result.predicted_value,
                            "normalized_cost": result.usage.cost,
                            "cost": result.usage.cost,
                            "tokens": result.usage.tokens,
                            "accurate_calls": result.usage.accurate_calls,
                            "iterations": result.usage.iterations,
                            "model_calls": result.usage.model_calls,
                            "environment_calls": result.usage.environment_calls,
                            "calls_by_fidelity": calls_by_fidelity,
                            "execution_failures": execution_failures,
                            "latency_s": (
                                result.usage.latency_s + final_execution.latency_s
                            ),
                            "task_execution_latency_s": final_execution.latency_s,
                            "task_execution_status": final_execution.status.value,
                            "stopping_reason": stop_reason,
                            "route_propensities": route_propensities,
                            "calibration_features": calibration_features,
                            "provenance": sorted(
                                {str(record["provenance"]) for record in compute_only}
                                | {"executable"}
                            ),
                            "randomized_audit_queries": sum(
                                bool(record["randomized_audit"])
                                for record in compute_only
                            ),
                            "matched_accurate_call_target": matched_calls[budget_index],
                            "risk": 0.0,
                            "terminated": final_execution.terminated,
                            "truncated": final_execution.truncated,
                            "report": dict(report_fields),
                            "tree_reuse_applicable": False,
                            "notice": "Exploratory diagnostic; not a paper result.",
                        }
                    )
                    episodes.append(episode)

    _persist_raw(output, episodes, decisions, failures)
    required_coverage = {
        (method, budget_index)
        for method in methods
        for budget_index in range(len(budgets))
    }
    observed_coverage = {
        (str(record["method"]), int(record["budget_index"])) for record in episodes
    }
    missing_coverage = required_coverage - observed_coverage
    if missing_coverage:
        raise RuntimeError(
            "one or more methods did not complete at every budget point: "
            + repr(sorted(missing_coverage))
        )
    analysis, per_task = analyze_sqlite_records(
        episodes,
        decisions,
        failures,
        protocol=protocol,
    )
    _write_json_exclusive(output / "analysis.json", analysis)
    _write_derived_jsonl(output / "per_task_differences.jsonl", per_task)
    by_budget = analysis["methods_by_budget"]
    summary = {
        "schema_version": 1,
        "stage": ExperimentStage.EXPLORATORY.value,
        "benchmark": BENCHMARK_ID,
        "protocol_sha256": protocol_sha256(protocol),
        "records": len(episodes),
        "decisions": len(decisions),
        "failures": len(failures),
        "budget_points": len(budgets),
        "methods": list(methods),
        "baseline_budget_coverage_complete": True,
        "matched_random_accurate_call_targets": list(matched_calls),
        "methods_overall": aggregate_episode_records(episodes),
        "methods_by_budget": by_budget,
        "raw_artifacts": {
            "episodes": "raw/episodes.jsonl",
            "decisions": "raw/decisions.jsonl",
            "failures": "raw/failures.jsonl",
        },
        "notice": "Exploratory diagnostics only; these are not paper results.",
    }
    _write_json_exclusive(output / "summary.json", summary)
    raw_paths = (
        output / "raw" / "episodes.jsonl",
        output / "raw" / "decisions.jsonl",
        output / "raw" / "failures.jsonl",
    )
    _write_json_exclusive(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "fixture_artifact_sha256": sqlite_fixture_artifact_hash(),
            "raw_sha256": {
                str(path.relative_to(output)): _file_sha256(path) for path in raw_paths
            },
            "protocol_sha256": protocol_sha256(protocol),
            "raw_files_read_only": all(
                not (path.stat().st_mode & 0o222) for path in raw_paths
            ),
        },
    )
    return summary


def _persist_raw(
    output: Path,
    episodes: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    raw = output / "raw"
    _write_jsonl_immutable(raw / "episodes.jsonl", episodes, kind="episode")
    _write_jsonl_immutable(raw / "decisions.jsonl", decisions, kind="decision")
    _write_jsonl_immutable(raw / "failures.jsonl", failures, kind="failure")


def _write_derived_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def sqlite3_version() -> str:
    """Small indirection that keeps environment metadata easy to fixture."""

    import sqlite3

    return sqlite3.sqlite_version


def matched_budget_diagnostic(summary: Mapping[str, Any]) -> dict[str, float]:
    """Return absolute random-versus-adaptive call/cost gaps for quick checks."""

    gaps: dict[str, float] = {}
    by_budget = cast(
        Mapping[str, Mapping[str, Mapping[str, Any]]], summary["methods_by_budget"]
    )
    for budget, methods in by_budget.items():
        adaptive = methods["fidelity_mcts"]
        random = methods["random_matched"]
        gaps[f"budget_{budget}_accurate_calls"] = abs(
            float(adaptive["mean_accurate_calls"])
            - float(random["mean_accurate_calls"])
        )
        gaps[f"budget_{budget}_normalized_cost"] = abs(
            float(adaptive["mean_cost"]) - float(random["mean_cost"])
        )
    return gaps


def pilot_failure_rate(summary: Mapping[str, Any]) -> float:
    records = int(summary["records"])
    failures = int(summary["failures"])
    return failures / max(1, records + failures)


def mean_success(summary: Mapping[str, Any], method: str) -> float:
    methods = cast(Mapping[str, Mapping[str, Any]], summary["methods_overall"])
    return float(methods[method]["success_rate"])


def mean_cost(summary: Mapping[str, Any], method: str) -> float:
    methods = cast(Mapping[str, Mapping[str, Any]], summary["methods_overall"])
    return float(methods[method]["mean_cost"])


def average_route_propensity(records: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        float(record["route_propensity"])
        for record in records
        if record.get("route_propensity") is not None
    ]
    return None if not values else fmean(values)
