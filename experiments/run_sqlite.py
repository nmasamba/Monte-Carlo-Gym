"""Run the guarded offline SQLite L2 exploratory benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from montecarlgym.experiments.preregistration import ExperimentStage  # noqa: E402
from montecarlgym.experiments.sqlite_study import run_sqlite_study  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline executable SQLite L2 exploratory study."
    )
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in ExperimentStage],
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage != ExperimentStage.EXPLORATORY.value:
        parser.error(
            "Phase 5A has no materialized confirmatory SQLite fixtures; only "
            "--stage exploratory is permitted"
        )
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_sqlite_study(protocol, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
