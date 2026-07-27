"""Transactional adapter for Gymnasium-style environments.

Gymnasium is deliberately not imported.  Compatible objects only need the
standard ``step`` five-tuple and an enumerable legal-action surface.
"""

from __future__ import annotations

import copy
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from ..types import Action
from .snapshot import SnapshotError


class SnapshotStrategy(Protocol):
    isolates_live: bool
    name: str

    def validate(self, env: Any) -> None: ...

    def capture(self, env: Any) -> Any: ...

    def restore(self, env: Any, state: Any) -> None: ...

    def fork(self, env: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class _RNGRecord:
    target: Any
    kind: str
    state: Any


@dataclass(frozen=True, slots=True)
class EnvSnapshot:
    """Opaque complete snapshot produced by ``MCTSEnvWrapper.snapshot``."""

    strategy_name: str
    environment_state: Any
    python_random_state: object
    numpy_random_state: object | None
    rng_records: tuple[_RNGRecord, ...]
    terminated: bool
    truncated: bool


def _environment_objects(env: Any) -> tuple[Any, ...]:
    pending = [env]
    objects: list[Any] = []
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        objects.append(item)
        for name in ("env", "unwrapped", "action_space", "observation_space"):
            try:
                child = getattr(item, name, None)
            except Exception:
                child = None
            if child is not None and child is not item:
                pending.append(child)
    return tuple(objects)


def _capture_rngs(env: Any) -> tuple[_RNGRecord, ...]:
    records: list[_RNGRecord] = []
    seen: set[int] = set()
    for owner in _environment_objects(env):
        values = [owner]
        try:
            values.extend(vars(owner).values())
        except TypeError:
            pass
        for rng in values:
            if id(rng) in seen:
                continue
            if isinstance(rng, random.Random):
                seen.add(id(rng))
                records.append(_RNGRecord(rng, "python", rng.getstate()))
                continue
            bit_generator = getattr(rng, "bit_generator", None)
            if bit_generator is not None and hasattr(bit_generator, "state"):
                seen.add(id(rng))
                records.append(
                    _RNGRecord(
                        rng,
                        "numpy_generator",
                        copy.deepcopy(bit_generator.state),
                    )
                )
                continue
            if (
                type(rng).__module__.startswith("numpy.random")
                and hasattr(rng, "get_state")
                and hasattr(rng, "set_state")
            ):
                seen.add(id(rng))
                records.append(
                    _RNGRecord(rng, "numpy_random_state", rng.get_state())
                )
    return tuple(records)


def _restore_rngs(records: tuple[_RNGRecord, ...]) -> None:
    for record in records:
        if record.kind == "python":
            record.target.setstate(record.state)
        elif record.kind == "numpy_generator":
            record.target.bit_generator.state = copy.deepcopy(record.state)
        elif record.kind == "numpy_random_state":
            record.target.set_state(copy.deepcopy(record.state))


def _seed_rngs(env: Any, seed: int) -> None:
    """Give one simulation iteration a reproducible, non-live RNG stream."""

    random.seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(seed % (2**32))
    for index, record in enumerate(_capture_rngs(env), start=1):
        stream_seed = (seed + index * 1_000_003) % (2**64)
        if record.kind == "python":
            record.target.seed(stream_seed)
        elif record.kind == "numpy_generator":
            bit_generator = record.target.bit_generator
            try:
                fresh = type(bit_generator)(stream_seed)
                bit_generator.state = fresh.state
            except Exception as exc:
                raise SnapshotError(
                    "environment NumPy generator cannot be reseeded safely"
                ) from exc
        elif record.kind == "numpy_random_state":
            record.target.seed(stream_seed % (2**32))


def _numpy_global_state() -> object | None:
    numpy = sys.modules.get("numpy")
    if numpy is None:
        return None
    try:
        return copy.deepcopy(numpy.random.get_state())
    except AttributeError:
        return None


def _restore_numpy_global(state: object | None) -> None:
    if state is None:
        return
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.set_state(copy.deepcopy(state))


class MCTSEnvWrapper:
    """Protect a live Gymnasium-style environment during speculative search."""

    def __init__(
        self,
        env: Any,
        *,
        strategy: SnapshotStrategy | None = None,
        legal_actions: Callable[[Any, Any | None], Sequence[Action]] | None = None,
        max_call_cost: float = 1.0,
        default_call_cost: float = 1.0,
        cost_key: str = "normalized_cost",
    ) -> None:
        if max_call_cost < 0 or default_call_cost < 0:
            raise ValueError("call costs must be non-negative")
        if default_call_cost > max_call_cost:
            raise ValueError("default_call_cost cannot exceed max_call_cost")
        if strategy is None:
            from .deepcopy import DeepCopySnapshotStrategy

            strategy = DeepCopySnapshotStrategy()
        strategy.validate(env)
        self.env = env
        self.strategy = strategy
        self._active_env = env
        self._legal_actions_fn = legal_actions
        self.max_call_cost = float(max_call_cost)
        self.default_call_cost = float(default_call_cost)
        self.cost_key = cost_key
        self._terminated = False
        self._truncated = False
        self._in_transaction = False

    @property
    def active_env(self) -> Any:
        return self._active_env

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    def snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            strategy_name=self.strategy.name,
            environment_state=self.strategy.capture(self._active_env),
            python_random_state=random.getstate(),
            numpy_random_state=_numpy_global_state(),
            rng_records=_capture_rngs(self._active_env),
            terminated=self._terminated,
            truncated=self._truncated,
        )

    def restore(self, snapshot: EnvSnapshot) -> None:
        if snapshot.strategy_name != self.strategy.name:
            raise SnapshotError(
                "snapshot strategy does not match this environment wrapper"
            )
        self.strategy.restore(self._active_env, snapshot.environment_state)
        random.setstate(snapshot.python_random_state)
        _restore_numpy_global(snapshot.numpy_random_state)
        _restore_rngs(snapshot.rng_records)
        self._terminated = snapshot.terminated
        self._truncated = snapshot.truncated

    def seed_simulation(self, seed: int) -> None:
        """Fork reproducible stochastic outcomes without changing live RNGs."""

        if not self._in_transaction:
            raise RuntimeError("simulation RNGs may only be seeded in a transaction")
        _seed_rngs(self._active_env, seed)

    @contextmanager
    def transaction(self) -> Iterator[MCTSEnvWrapper]:
        """Run arbitrary simulations and restore all live/process state."""

        if self._in_transaction:
            raise RuntimeError("nested environment transactions are not supported")
        live_snapshot = self.snapshot()
        self._in_transaction = True
        try:
            self._active_env = self.strategy.fork(self.env)
            self._terminated = False
            self._truncated = False
            yield self
        finally:
            if self.strategy.isolates_live:
                self._active_env = self.env
                random.setstate(live_snapshot.python_random_state)
                _restore_numpy_global(live_snapshot.numpy_random_state)
                _restore_rngs(live_snapshot.rng_records)
                self._terminated = live_snapshot.terminated
                self._truncated = live_snapshot.truncated
            else:
                self._active_env = self.env
                self.restore(live_snapshot)
            self._in_transaction = False

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, Mapping[str, Any]]:
        result = self._active_env.reset(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Gymnasium reset() must return (observation, info)")
        observation, info = result
        if not isinstance(info, Mapping):
            raise TypeError("Gymnasium reset info must be a mapping")
        self._terminated = False
        self._truncated = False
        return observation, info

    def step(
        self,
        action: Action,
    ) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
        result = self._active_env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError(
                "Gymnasium step() must return observation, reward, terminated, "
                "truncated, info"
            )
        observation, reward, terminated, truncated, info = result
        if not isinstance(info, Mapping):
            raise TypeError("Gymnasium step info must be a mapping")
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        return (
            observation,
            float(reward),
            self._terminated,
            self._truncated,
            info,
        )

    def legal_actions(self, observation: Any | None = None) -> tuple[Action, ...]:
        env = self._active_env
        if self._legal_actions_fn is not None:
            actions = self._legal_actions_fn(env, observation)
        else:
            provider = getattr(env, "legal_actions", None)
            if callable(provider):
                actions = provider()
            else:
                available = getattr(env, "available_actions", None)
                actions = available() if callable(available) else available
                if actions is None:
                    space = getattr(env, "action_space", None)
                    count = getattr(space, "n", None)
                    if not isinstance(count, int):
                        raise SnapshotError(
                            "cannot enumerate legal actions; provide legal_actions="
                        )
                    actions = range(count)
        return tuple(actions)

    def transition_cost(self, info: Mapping[str, Any]) -> float:
        raw = info.get(self.cost_key, info.get("cost", self.default_call_cost))
        try:
            cost = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition normalized cost must be numeric") from exc
        return cost
