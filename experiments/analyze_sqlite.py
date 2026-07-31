"""Reproduce SQLite L2 aggregates from immutable raw exploratory records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from montecarlgym.experiments.analysis import (  # noqa: E402
    analyze_sqlite_records,
    read_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute SQLite L2 analysis without modifying the run."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    protocol = json.loads(
        (args.input / "resolved_protocol.json").read_text(encoding="utf-8")
    )
    report, differences = analyze_sqlite_records(
        read_jsonl(args.input / "raw" / "episodes.jsonl"),
        read_jsonl(args.input / "raw" / "decisions.jsonl"),
        read_jsonl(args.input / "raw" / "failures.jsonl"),
        protocol=protocol,
    )
    payload = {"analysis": report, "per_task_differences": differences}
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
