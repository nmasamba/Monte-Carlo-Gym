"""Strict resource reservations for adaptive model routing."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..core.budget import BudgetExhausted, ResourceQuoteExceeded
from ..types import ModelObservation, ModelQuote, ResourceUsage, SearchBudget


@dataclass(frozen=True, slots=True)
class QueryReservation:
    """Resources conservatively charged before one model invocation."""

    quote: ModelQuote
    started_at: float


class AdaptiveResourceLedger:
    """Enforce all adaptive-search budgets before invoking a model.

    Reservations are charged immediately. A successful call releases unused
    quoted cost, tokens, and environment calls. A failed call retains the full
    reservation because its actual resource use is not reliably observable.
    """

    def __init__(
        self,
        budget: SearchBudget,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.budget = budget
        self._clock = clock
        self._started_at = clock()
        self.cost = 0.0
        self.tokens = 0
        self.accurate_calls = 0
        self.iterations = 0
        self.latency_s = 0.0
        self.model_calls = 0
        self.environment_calls = 0
        self.stop_reason = "not_started"

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    @property
    def deadline_reached(self) -> bool:
        deadline = self.budget.deadline_s
        return deadline is not None and self.elapsed_s >= deadline

    def reserve(self, quote: ModelQuote) -> QueryReservation | None:
        """Reserve a quoted query, returning ``None`` when a limit blocks it."""

        self._validate_quote(quote)
        if self.iterations >= self.budget.max_iterations:
            self.stop_reason = "iteration_budget"
            return None
        if self.deadline_reached:
            self.stop_reason = "deadline"
            return None
        deadline = self.budget.deadline_s
        if (
            deadline is not None
            and self.elapsed_s + quote.expected_latency_s > deadline
        ):
            self.stop_reason = "deadline"
            return None
        if (
            self.budget.max_model_calls is not None
            and self.model_calls + 1 > self.budget.max_model_calls
        ):
            self.stop_reason = "model_call_budget"
            return None
        if self.cost + quote.cost > self.budget.max_cost:
            self.stop_reason = "cost_budget"
            return None
        if self.tokens + quote.tokens > self.budget.max_tokens:
            self.stop_reason = "token_budget"
            return None
        if (
            self.accurate_calls + quote.accurate_calls
            > self.budget.max_accurate_calls
        ):
            self.stop_reason = "accurate_call_budget"
            return None
        if (
            self.budget.max_environment_calls is not None
            and self.environment_calls + quote.environment_calls
            > self.budget.max_environment_calls
        ):
            self.stop_reason = "environment_call_budget"
            return None

        self.cost += quote.cost
        self.tokens += quote.tokens
        self.accurate_calls += quote.accurate_calls
        self.iterations += 1
        self.model_calls += 1
        self.environment_calls += quote.environment_calls
        self.stop_reason = "running"
        return QueryReservation(quote, self._clock())

    def commit(
        self,
        reservation: QueryReservation,
        observation: ModelObservation,
    ) -> None:
        """Commit measured use and release unused parts of a reservation."""

        self._validate_observation(observation)
        quote = reservation.quote
        overruns: list[str] = []
        if observation.cost > quote.cost:
            overruns.append("cost")
        if observation.tokens > quote.tokens:
            overruns.append("tokens")
        if observation.environment_calls > quote.environment_calls:
            overruns.append("environment_calls")
        if overruns:
            self.stop_reason = "quote_overrun"
            raise ResourceQuoteExceeded(
                "model observation exceeded its conservative quote for: "
                + ", ".join(overruns)
            )

        self.cost -= quote.cost - observation.cost
        self.tokens -= quote.tokens - observation.tokens
        self.environment_calls -= (
            quote.environment_calls - observation.environment_calls
        )
        self.latency_s += observation.latency_s
        if self.deadline_reached:
            self.stop_reason = "deadline"
            raise BudgetExhausted(
                "model call completed after the hard planning deadline"
            )

    def fail(self, reservation: QueryReservation) -> None:
        """Retain a failed call's reservation and account observed wall time."""

        self.latency_s += max(0.0, self._clock() - reservation.started_at)
        self.stop_reason = "model_error"

    def remaining_budget(self) -> SearchBudget:
        """Return a non-negative view of resources still available."""

        deadline = self.budget.deadline_s
        remaining_deadline = (
            None
            if deadline is None
            else max(0.0, deadline - self.elapsed_s)
        )
        max_model_calls = self.budget.max_model_calls
        max_environment_calls = self.budget.max_environment_calls
        return SearchBudget(
            max_cost=max(0.0, self.budget.max_cost - self.cost),
            max_tokens=max(0, self.budget.max_tokens - self.tokens),
            max_accurate_calls=max(
                0, self.budget.max_accurate_calls - self.accurate_calls
            ),
            max_iterations=max(
                0, self.budget.max_iterations - self.iterations
            ),
            deadline_s=remaining_deadline,
            max_model_calls=(
                None
                if max_model_calls is None
                else max(0, max_model_calls - self.model_calls)
            ),
            max_environment_calls=(
                None
                if max_environment_calls is None
                else max(
                    0,
                    max_environment_calls - self.environment_calls,
                )
            ),
        )

    def usage(self) -> ResourceUsage:
        return ResourceUsage(
            cost=self.cost,
            tokens=self.tokens,
            accurate_calls=self.accurate_calls,
            iterations=self.iterations,
            latency_s=self.latency_s,
            model_calls=self.model_calls,
            environment_calls=self.environment_calls,
        )

    @staticmethod
    def _validate_quote(quote: ModelQuote) -> None:
        AdaptiveResourceLedger._validate_float("quote.cost", quote.cost)
        AdaptiveResourceLedger._validate_float(
            "quote.expected_latency_s", quote.expected_latency_s
        )
        if quote.cost < 0:
            raise ValueError("quote.cost must be non-negative")
        if quote.expected_latency_s < 0:
            raise ValueError("quote.expected_latency_s must be non-negative")
        for name, value in (
            ("quote.tokens", quote.tokens),
            ("quote.accurate_calls", quote.accurate_calls),
            ("quote.environment_calls", quote.environment_calls),
        ):
            AdaptiveResourceLedger._validate_count(name, value)

    @staticmethod
    def _validate_observation(observation: ModelObservation) -> None:
        for name, value in (
            ("observation.value", observation.value),
            ("observation.variance", observation.variance),
            ("observation.cost", observation.cost),
            ("observation.latency_s", observation.latency_s),
            ("observation.risk", observation.risk),
        ):
            AdaptiveResourceLedger._validate_float(name, value)
        if observation.variance < 0:
            raise ValueError("observation.variance must be non-negative")
        if observation.cost < 0:
            raise ValueError("observation.cost must be non-negative")
        if observation.latency_s < 0:
            raise ValueError("observation.latency_s must be non-negative")
        if observation.risk < 0:
            raise ValueError("observation.risk must be non-negative")
        AdaptiveResourceLedger._validate_count(
            "observation.tokens", observation.tokens
        )
        AdaptiveResourceLedger._validate_count(
            "observation.environment_calls", observation.environment_calls
        )

    @staticmethod
    def _validate_float(name: str, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    @staticmethod
    def _validate_count(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
