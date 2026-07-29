"""Append-safe artifact helpers for reproducible experiment runs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactWriter:
    """Write the stable on-disk contract consumed by analysis tooling."""

    root: Path

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, value: Mapping[str, Any]) -> Path:
        self.initialize()
        path = self.root / name
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def write_jsonl(
        self,
        name: str,
        records: Iterable[Mapping[str, Any]],
    ) -> Path:
        self.initialize()
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return path
