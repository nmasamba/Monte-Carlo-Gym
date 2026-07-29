"""Hard resource accounting for classical search."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..types import ResourceUsage, SearchBudget


class BudgetExhausted(RuntimeError):
    """Raised only when a caller asks to exceed a hard resource limit."""


class ResourceQuoteExceeded(RuntimeError):
    """A model call reported more cost than was reserved before execution."""


@dataclass(frozen=True, slots=True)
class CallReservation:
    cost: float


class ResourceLedger:
    """Reserve before calls and commit measured normalized costs afterward."""

    def __init__(self, budget: SearchBudget) -> None:
        self.budget = budget
        self._started_at = time.monotonic()
        self.cost = 0.0
        self.tokens = 0
        self.accurate_calls = 0
        self.iterations = 0
        self.model_calls = 0
        self.environment_calls = 0
        self.stop_reason = "not_started"

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started_at

    @property
    def deadline_reached(self) -> bool:
        deadline = self.budget.deadline_s
        return deadline is not None and self.elapsed_s >= deadline

    def can_start_iteration(self) -> bool:
        if self.iterations >= self.budget.max_iterations:
            self.stop_reason = "iteration_budget"
            return False
        if self.deadline_reached:
            self.stop_reason = "deadline"
            return False
        return True

    def can_reserve_call(self, quoted_cost: float) -> bool:
        self._validate_cost(quoted_cost)
        if self.deadline_reached:
            self.stop_reason = "deadline"
            return False
        if (
            self.budget.max_model_calls is not None
            and self.model_calls >= self.budget.max_model_calls
        ):
            self.stop_reason = "model_call_budget"
            return False
        if (
            self.budget.max_environment_calls is not None
            and self.environment_calls >= self.budget.max_environment_calls
        ):
            self.stop_reason = "environment_call_budget"
            return False
        if self.cost + quoted_cost > self.budget.max_cost:
            self.stop_reason = "cost_budget"
            return False
        return True

    def reserve_call(self, quoted_cost: float) -> CallReservation | None:
        """Reserve a call conservatively; failed calls remain accounted for."""

        if not self.can_reserve_call(quoted_cost):
            return None
        self.cost += quoted_cost
        self.model_calls += 1
        self.environment_calls += 1
        return CallReservation(quoted_cost)

    def commit_call(
        self,
        reservation: CallReservation,
        *,
        measured_cost: float,
    ) -> None:
        self._validate_cost(measured_cost)
        if measured_cost > reservation.cost:
            self.stop_reason = "quote_overrun"
            raise ResourceQuoteExceeded(
                "environment/model call exceeded its conservative cost quote: "
                f"measured {measured_cost}, reserved {reservation.cost}"
            )
        self.cost -= reservation.cost - measured_cost
        if self.deadline_reached:
            self.stop_reason = "deadline"
            raise BudgetExhausted("model call completed after the hard deadline")

    def complete_iteration(self) -> None:
        if self.iterations >= self.budget.max_iterations:
            raise BudgetExhausted("iteration budget exhausted")
        self.iterations += 1

    def remaining_budget(self) -> SearchBudget:
        """Return the resources available to a nested budget-aware evaluator."""

        model_calls = self.budget.max_model_calls
        environment_calls = self.budget.max_environment_calls
        deadline = self.budget.deadline_s
        return SearchBudget(
            max_cost=max(0.0, self.budget.max_cost - self.cost),
            max_tokens=max(0, self.budget.max_tokens - self.tokens),
            max_accurate_calls=max(
                0, self.budget.max_accurate_calls - self.accurate_calls
            ),
            max_iterations=max(
                0, self.budget.max_iterations - self.iterations
            ),
            deadline_s=(
                None
                if deadline is None
                else max(0.0, deadline - self.elapsed_s)
            ),
            max_model_calls=(
                None
                if model_calls is None
                else max(0, model_calls - self.model_calls)
            ),
            max_environment_calls=(
                None
                if environment_calls is None
                else max(0, environment_calls - self.environment_calls)
            ),
        )

    def absorb_usage(self, usage: ResourceUsage) -> None:
        """Commit already-budgeted nested evaluator use without double counting.

        A nested evaluator must first derive its hard envelope from
        ``remaining_budget``. This method verifies that contract again before
        adding model, token, accurate-call, environment-call, and cost usage.
        Nested evaluation iterations do not consume outer MCTS iterations.
        """

        self._validate_usage(usage)
        if self.deadline_reached:
            self.stop_reason = "deadline"
            raise BudgetExhausted("nested evaluator completed after the deadline")
        if self.cost + usage.cost > self.budget.max_cost:
            self.stop_reason = "cost_budget"
            raise BudgetExhausted("nested evaluator exceeded the cost budget")
        if self.tokens + usage.tokens > self.budget.max_tokens:
            self.stop_reason = "token_budget"
            raise BudgetExhausted("nested evaluator exceeded the token budget")
        if (
            self.accurate_calls + usage.accurate_calls
            > self.budget.max_accurate_calls
        ):
            self.stop_reason = "accurate_call_budget"
            raise BudgetExhausted(
                "nested evaluator exceeded the accurate-call budget"
            )
        if (
            self.budget.max_model_calls is not None
            and self.model_calls + usage.model_calls
            > self.budget.max_model_calls
        ):
            self.stop_reason = "model_call_budget"
            raise BudgetExhausted("nested evaluator exceeded the model-call budget")
        if (
            self.budget.max_environment_calls is not None
            and self.environment_calls + usage.environment_calls
            > self.budget.max_environment_calls
        ):
            self.stop_reason = "environment_call_budget"
            raise BudgetExhausted(
                "nested evaluator exceeded the environment-call budget"
            )
        self.cost += usage.cost
        self.tokens += usage.tokens
        self.accurate_calls += usage.accurate_calls
        self.model_calls += usage.model_calls
        self.environment_calls += usage.environment_calls

    def usage(self) -> ResourceUsage:
        return ResourceUsage(
            cost=self.cost,
            tokens=self.tokens,
            accurate_calls=self.accurate_calls,
            iterations=self.iterations,
            latency_s=self.elapsed_s,
            model_calls=self.model_calls,
            environment_calls=self.environment_calls,
        )

    @staticmethod
    def _validate_cost(cost: float) -> None:
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(
                "normalized model-call cost must be finite and non-negative"
            )

    @classmethod
    def _validate_usage(cls, usage: ResourceUsage) -> None:
        cls._validate_cost(usage.cost)
        for name, value in (
            ("tokens", usage.tokens),
            ("accurate_calls", usage.accurate_calls),
            ("model_calls", usage.model_calls),
            ("environment_calls", usage.environment_calls),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"nested usage {name} must be non-negative")
