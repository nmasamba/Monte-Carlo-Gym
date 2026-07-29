"""Run exploratory or fingerprint-guarded Gymnasium studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from montecarlgym.experiments.gymnasium_frozenlake import (  # noqa: E402
    run_frozenlake_study,
)
from montecarlgym.experiments.preregistration import (  # noqa: E402
    ExperimentStage,
    load_frozen_protocol,
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


def _verify_confirmatory_revision(code_revision: str) -> None:
    if _git_output("status", "--porcelain"):
        raise SystemExit("confirmatory runs require a clean worktree")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", code_revision, "HEAD"),
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SystemExit(
            "frozen source revision is not an ancestor of the current checkout"
        )
    changed = _git_output("diff", "--name-only", f"{code_revision}..HEAD")
    disallowed = [
        name
        for name in changed.splitlines()
        if name and not name.startswith("experiments/preregistered/")
    ]
    if disallowed:
        raise SystemExit(
            "implementation changed after preregistration: "
            + ", ".join(disallowed)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a real Gymnasium multi-fidelity study."
    )
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in ExperimentStage],
        required=True,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = ExperimentStage(args.stage)
    if stage is ExperimentStage.EXPLORATORY:
        if args.config is None or args.preregistration is not None:
            parser.error("exploratory runs require --config only")
        protocol = json.loads(args.config.read_text(encoding="utf-8"))
        frozen = None
    else:
        if args.preregistration is None or args.config is not None:
            parser.error("confirmatory runs require --preregistration only")
        frozen = load_frozen_protocol(args.preregistration)
        protocol = frozen.protocol
        validate_protocol_artifacts(protocol, ROOT)
        _verify_confirmatory_revision(frozen.code_revision)
    summary = run_frozenlake_study(
        protocol,
        args.output,
        stage=stage,
        frozen=frozen,
        revision_verified=stage is ExperimentStage.CONFIRMATORY,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
