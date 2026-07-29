from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from montecarlgym.experiments.preregistration import (
    ExperimentStage,
    freeze_protocol,
    load_frozen_protocol,
    validate_confirmatory_output,
    validate_preregistration_protocol,
    validate_protocol_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


def protocol(*, stage: ExperimentStage) -> dict[str, object]:
    payload = json.loads(
        (ROOT / "experiments" / "pilots" / "frozenlake_v1.json").read_text()
    )
    payload["stage"] = stage.value
    return payload


class PreregistrationTests(unittest.TestCase):
    def test_complete_protocol_freezes_and_detects_tampering(self) -> None:
        candidate = protocol(stage=ExperimentStage.CONFIRMATORY)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            frozen = freeze_protocol(
                candidate,
                path,
                code_revision="abc123",
            )
            loaded = load_frozen_protocol(path)
            self.assertEqual(loaded.sha256, frozen.sha256)
            with self.assertRaises(FileExistsError):
                freeze_protocol(candidate, path, code_revision="abc123")

            payload = json.loads(path.read_text())
            payload["protocol"]["study_id"] = "changed-after-freeze"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_frozen_protocol(path)

    def test_exploratory_protocol_cannot_be_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "confirmatory"):
                freeze_protocol(
                    protocol(stage=ExperimentStage.EXPLORATORY),
                    Path(directory) / "invalid.json",
                    code_revision="abc123",
                )

    def test_required_analysis_decisions_are_enforced(self) -> None:
        candidate = protocol(stage=ExperimentStage.CONFIRMATORY)
        analysis = candidate["analysis"]
        assert isinstance(analysis, dict)
        del analysis["stopping_rule"]
        with self.assertRaisesRegex(ValueError, "stopping_rule"):
            validate_preregistration_protocol(candidate)

    def test_confirmatory_outputs_are_separate_and_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "confirmatory"):
                validate_confirmatory_output(root / "ordinary" / "run-1")
            destination = root / "output" / "confirmatory" / "study" / "run-1"
            validate_confirmatory_output(destination)
            destination.mkdir(parents=True)
            (destination / "existing.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                validate_confirmatory_output(destination)

    def test_locked_artifact_hashes_are_verified(self) -> None:
        candidate = protocol(stage=ExperimentStage.CONFIRMATORY)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "checkpoint.bin"
            artifact.write_bytes(b"locked checkpoint")
            candidate["artifacts"] = {
                "checkpoint.bin": (
                    "863b17ebd636ef553142f53623e49b37ec3ed10d497747b5"
                    "771ae744cb9cb3ff"
                )
            }
            validate_protocol_artifacts(candidate, root)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                validate_protocol_artifacts(candidate, root)


if __name__ == "__main__":
    unittest.main()
