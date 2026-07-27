"""Native callback snapshot strategy."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable


class SnapshotError(RuntimeError):
    """An environment cannot satisfy the transactional restore contract."""


@dataclass(frozen=True, slots=True)
class NativeSnapshotStrategy:
    """Use environment-specific full-state snapshot and restore callbacks.

    The callbacks receive the outer environment object.  They are responsible
    for all environment and Gym wrapper state; ``MCTSEnvWrapper`` additionally
    protects its own flags and discoverable RNG streams.
    """

    get_state: Callable[[Any], Any]
    set_state: Callable[[Any, Any], None]
    isolates_live: bool = False
    name: str = "native"

    def validate(self, env: Any) -> None:
        try:
            state = copy.deepcopy(self.get_state(env))
            self.set_state(env, copy.deepcopy(state))
        except Exception as exc:
            raise SnapshotError(
                "native snapshot callbacks failed validation"
            ) from exc

    def capture(self, env: Any) -> Any:
        try:
            return copy.deepcopy(self.get_state(env))
        except Exception as exc:
            raise SnapshotError("native environment snapshot failed") from exc

    def restore(self, env: Any, state: Any) -> None:
        try:
            self.set_state(env, copy.deepcopy(state))
        except Exception as exc:
            raise SnapshotError("native environment restore failed") from exc

    def fork(self, env: Any) -> Any:
        return env
