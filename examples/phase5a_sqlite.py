"""Quick offline SQLite query-repair example with no optional dependencies."""

from __future__ import annotations

from montecarlgym.experiments.sqlite_l2 import (
    SQLitePartition,
    load_sqlite_partition,
    make_sqlite_planner,
    make_sqlite_portfolio,
    score_sqlite_action,
    train_sqlite_components,
)
from montecarlgym.types import SearchBudget


def main() -> None:
    budget = SearchBudget(
        max_cost=17.0,
        max_tokens=32,
        max_accurate_calls=4,
        max_iterations=8,
        deadline_s=2.0,
        max_model_calls=8,
        max_environment_calls=4,
    )
    task = load_sqlite_partition(SQLitePartition.EXPLORATORY)[0]
    components = train_sqlite_components(budget=budget)
    planner = make_sqlite_planner(
        "fidelity_mcts",
        components=components,
        matched_accurate_calls=1,
        seed=0,
    )
    result = planner.plan(
        task,
        models=make_sqlite_portfolio(),
        budget=budget,
        seed=0,
    )
    verified = score_sqlite_action(task, result.action)
    print(
        {
            "task_id": task.task_id,
            "selected_candidate": result.action,
            "verifier_passed": verified.verifier_passed,
            "normalized_cost": result.usage.cost,
            "accurate_calls": result.usage.accurate_calls,
            "notice": "offline exploratory example; not a paper result",
        }
    )


if __name__ == "__main__":
    main()
