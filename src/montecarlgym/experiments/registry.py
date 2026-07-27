"""Explicit local registries; callers may construct isolated registries in tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class FactoryRegistry(Generic[T]):
    _factories: dict[str, Callable[..., T]] = field(default_factory=dict)

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if not name:
            raise ValueError("registry name cannot be empty")
        if name in self._factories:
            raise ValueError(f"duplicate registry name: {name}")
        self._factories[name] = factory

    def create(self, name: str, **settings: Any) -> T:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"unknown registry name: {name}") from exc
        return factory(**settings)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
