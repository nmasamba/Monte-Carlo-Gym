"""Phase 3 learned-model/executable-model routing diagnostic."""

from __future__ import annotations

from montecarlgym.adaptive import AdaptiveComputePlanner
from montecarlgym.experiments.multifidelity_tree import (
    ShallowTreeActionProvider,
    ShallowTreeConfig,
    make_shallow_tree_portfolio,
    sample_shallow_tree,
)
from montecarlgym.routing import ThresholdRouter
from montecarlgym.types import SearchBudget


def main() -> None:
    config = ShallowTreeConfig(actions=5, horizon=3)
    task = sample_shallow_tree(config, seed=73)
    planner = AdaptiveComputePlanner(
        action_provider=ShallowTreeActionProvider(),
        router=ThresholdRouter(
            cheap_model_id="phase3-learned-linear",
            accurate_model_id="phase3-executable-tree",
            z_score=1.64,
            cheap_token_budget=config.learned_tokens,
            accurate_rollout_depth=config.horizon,
        ),
    )
    result = planner.plan(
        task,
        models=make_shallow_tree_portfolio(config),
        budget=SearchBudget(
            max_cost=19.25,
            max_tokens=40,
            max_accurate_calls=3,
            max_iterations=8,
            max_model_calls=8,
            max_environment_calls=9,
        ),
        seed=101,
    )

    assert result.report is not None
    print(f"selected task action: {result.action}")
    print(f"diagnostic optimal action: {task.optimal_action}")
    print(f"stop reason: {result.report.stop_reason}")
    print(f"resource usage: {result.usage}")
    print(f"compute queries: {len(result.trace)}")
    print(f"verified replay pairs: {result.report.replay_records_added}")
    for query in result.trace:
        print(
            "  "
            f"branch={query['task_action']} model={query['model_id']} "
            f"tokens={query['tokens']} depth={query['rollout_depth']} "
            f"verified={query['verified']}"
        )
    print("Engineering diagnostic only; this is not a paper result.")


if __name__ == "__main__":
    main()
