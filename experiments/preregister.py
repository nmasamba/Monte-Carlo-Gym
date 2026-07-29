"""Freeze a complete confirmatory protocol from a clean source revision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from montecarlgym.experiments.preregistration import (  # noqa: E402
    freeze_protocol,
    protocol_sha256,
    validate_preregistration_protocol,
    validate_protocol_artifacts,
)


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a MonteCarloGym confirmatory protocol."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registration-url")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_preregistration_protocol(protocol)
    validate_protocol_artifacts(protocol, ROOT)
    if args.validate_only:
        print(f"valid protocol sha256={protocol_sha256(protocol)}")
        return 0
    if args.output is None:
        parser.error("freezing requires --output")
    if _git_output("status", "--porcelain"):
        raise SystemExit(
            "refusing to preregister from a dirty worktree; commit and validate "
            "the exact protocol implementation first"
        )
    revision = _git_output("rev-parse", "HEAD")
    frozen = freeze_protocol(
        protocol,
        args.output,
        code_revision=revision,
        registration_url=args.registration_url,
    )
    print(f"frozen protocol sha256={frozen.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
