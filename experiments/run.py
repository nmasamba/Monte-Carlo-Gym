"""Run from a source checkout without requiring an editable installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from montecarlgym.experiments.runner import main  # noqa: E402

raise SystemExit(main())
