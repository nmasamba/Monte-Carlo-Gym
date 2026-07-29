"""Command-line and programmatic runner for the controlled benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ..planner import Planner
from ..types import SearchBudget, State
from .artifacts import ArtifactWriter
from .metrics import aggregate_episode_records
from .multifidelity_tree import (
    ShallowTreeConfig,
    make_shallow_tree_planner,
    make_shallow_tree_portfolio,
    sample_shallow_tree,
)
from .toy import (
    PLANNER_TYPES,
    ToyBenchmarkConfig,
    make_portfolio,
    sample_task,
)


def _method_seed(task_seed: int, method: str) -> int:
    stable_name = sum((index + 1) * ord(char) for index, char in enumerate(method))
    return task_seed * 1_000_003 + stable_name


def _planner(method: str, settings: dict[str, Any]) -> Planner:
    try:
        planner_type = PLANNER_TYPES[method]
    except KeyError as exc:
        raise ValueError(f"unknown method: {method}") from exc
    return cast(Planner, planner_type(**settings))


def _validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "benchmark", "run", "budget", "methods"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("only schema_version 1 is supported")
    if config["run"]["repetitions"] < 1:
        raise ValueError("run.repetitions must be positive")
    if not config["methods"]:
        raise ValueError("at least one method is required")


def run_experiment(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run paired seeds, persist episode records, and return aggregate metrics."""

    _validate_config(config)
    benchmark_data = dict(config["benchmark"])
    benchmark_type = str(benchmark_data.get("type", "toy_multifidelity"))
    if benchmark_type == "toy_multifidelity":
        benchmark_data.pop("type", None)
        toy_config = ToyBenchmarkConfig.from_dict(benchmark_data)
        tree_config = None
    elif benchmark_type == "phase3_shallow_tree":
        toy_config = None
        tree_config = ShallowTreeConfig.from_dict(benchmark_data)
    else:
        raise ValueError(f"unknown benchmark type: {benchmark_type}")
    budget = SearchBudget(**config["budget"])
    planner_settings = config.get("planner_settings", {})
    repetitions = int(config["run"]["repetitions"])
    base_seed = int(config["run"]["base_seed"])

    writer = ArtifactWriter(output_dir)
    writer.write_json("resolved_config.json", config)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    writer.write_json("environment.json", environment)

    records: list[dict[str, Any]] = []
    for offset in range(repetitions):
        task_seed = base_seed + offset
        task: State
        if toy_config is not None:
            task = sample_task(toy_config, task_seed)
        elif tree_config is not None:
            task = sample_shallow_tree(tree_config, task_seed)
        else:  # pragma: no cover - guarded by benchmark validation above
            raise AssertionError("benchmark configuration was not resolved")
        for method in config["methods"]:
            settings = dict(planner_settings.get(method, {}))
            if toy_config is not None:
                planner = _planner(method, settings)
                portfolio = make_portfolio(toy_config)
            else:
                if tree_config is None:  # pragma: no cover - type narrowing
                    raise AssertionError("missing shallow-tree configuration")
                planner = make_shallow_tree_planner(method, tree_config, settings)
                portfolio = make_shallow_tree_portfolio(tree_config)
            result = planner.plan(
                task,
                models=portfolio,
                budget=budget,
                seed=_method_seed(task_seed, method),
            )
            true_value = task.true_value(result.action)
            optimal_value = task.true_value(task.optimal_action)
            record = {
                "schema_version": 1,
                "method": method,
                "task_id": task.task_id,
                "seed": task_seed,
                "action": result.action,
                "optimal_action": task.optimal_action,
                "success": result.action == task.optimal_action,
                "return": true_value,
                "regret": optimal_value - true_value,
                "predicted_value": result.predicted_value,
                "cost": result.usage.cost,
                "tokens": result.usage.tokens,
                "accurate_calls": result.usage.accurate_calls,
                "iterations": result.usage.iterations,
                "latency_s": result.usage.latency_s,
                "model_calls": result.usage.model_calls,
                "environment_calls": result.usage.environment_calls,
                "risk": 0.0,
                "trace": list(result.trace),
            }
            records.append(record)

    writer.write_jsonl("runs.jsonl", records)
    writer.write_jsonl("failures.jsonl", [])
    methods = aggregate_episode_records(records)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": benchmark_type,
        "repetitions": repetitions,
        "budget": asdict(budget),
        "methods": methods,
        "notice": (
            "Controlled engineering diagnostic only; not empirical evidence for "
            "the FidelityMCTS research claims."
        ),
    }
    writer.write_json("summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a MonteCarloGym controlled diagnostic benchmark."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_experiment(config, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
