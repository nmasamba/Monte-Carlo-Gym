"""Verified whole-environment deep-copy simulation."""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .snapshot import SnapshotError

_IMMUTABLE = (type(None), bool, int, float, complex, str, bytes, range, Enum)
_CALLABLE_ATOMS = (
    types.BuiltinFunctionType,
    types.FunctionType,
    types.MethodType,
    types.ModuleType,
    type,
)


def _assert_independent(original: Any, clone: Any) -> None:
    """Reject custom deepcopy implementations that retain mutable objects."""

    pending = [(original, clone)]
    seen: set[tuple[int, int]] = set()
    while pending:
        left, right = pending.pop()
        pair = (id(left), id(right))
        if pair in seen:
            continue
        seen.add(pair)
        if isinstance(left, _IMMUTABLE + _CALLABLE_ATOMS):
            continue
        if left is right:
            if isinstance(left, tuple):
                pending.extend(zip(left, right, strict=True))
                continue
            if isinstance(left, frozenset):
                # A shared immutable container is safe; its elements cannot be
                # aligned reliably and mutable elements are uncommon here.
                continue
            # Extension-level scalar objects without state are normally
            # immutable atoms (for example a NumPy dtype).
            if not hasattr(left, "__dict__") and not isinstance(
                left, (dict, list, set, bytearray)
            ):
                continue
            raise SnapshotError(
                "deepcopy retained shared mutable state of type "
                f"{type(left).__name__}"
            )
        if type(left) is not type(right):
            raise SnapshotError("deepcopy changed an environment object's type")
        if isinstance(left, dict):
            if left.keys() != right.keys():
                raise SnapshotError("deepcopy changed dictionary keys")
            pending.extend((value, right[key]) for key, value in left.items())
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise SnapshotError("deepcopy changed a sequence length")
            pending.extend(zip(left, right, strict=True))
        elif hasattr(left, "__dict__"):
            left_vars = vars(left)
            right_vars = vars(right)
            if left_vars.keys() != right_vars.keys():
                raise SnapshotError("deepcopy changed object attributes")
            pending.extend(
                (value, right_vars[name]) for name, value in left_vars.items()
            )


def _copy_environment(env: Any) -> Any:
    try:
        clone = copy.deepcopy(env)
    except Exception as exc:
        raise SnapshotError(
            "environment cannot be deep-copied safely; provide native "
            "snapshot/restore callbacks"
        ) from exc
    if clone is env:
        raise SnapshotError("environment deepcopy returned the live object")
    _assert_independent(env, clone)
    return clone


def _restore_object(target: Any, snapshot: Any) -> None:
    if type(target) is not type(snapshot) or not hasattr(target, "__dict__"):
        raise SnapshotError(
            "deep-copy restore requires an environment with a mutable __dict__"
        )
    replacement = _copy_environment(snapshot)
    target_vars = vars(target)
    target_vars.clear()
    target_vars.update(vars(replacement))


@dataclass(frozen=True, slots=True)
class DeepCopySnapshotStrategy:
    """Run search on a verified independent clone of the live environment."""

    isolates_live: bool = True
    name: str = "deepcopy"

    def validate(self, env: Any) -> None:
        if not hasattr(env, "__dict__"):
            raise SnapshotError(
                "deep-copy strategy requires a mutable environment __dict__"
            )
        _copy_environment(env)

    def capture(self, env: Any) -> Any:
        return _copy_environment(env)

    def restore(self, env: Any, state: Any) -> None:
        _restore_object(env, state)

    def fork(self, env: Any) -> Any:
        return _copy_environment(env)
