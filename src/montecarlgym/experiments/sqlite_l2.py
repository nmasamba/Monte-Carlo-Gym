"""Offline executable L2 benchmark for SQL query construction and repair.

The benchmark is intentionally self-contained.  Candidate SQL is evaluated in
disposable in-memory SQLite databases copied from immutable templates.  A
read-only authorizer, progress-handler deadline, and VM-step ceiling prevent
candidate actions from mutating fixtures or escaping into host-side effects.

Only development, calibration, and exploratory-pilot fixtures are materialized
here.  The future-confirmatory partition is a reservation, not a task set: no
confirmatory task instance or seed exists in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import Random
from typing import Any, cast

from ..adaptive import (
    AdaptiveComputePlanner,
    AdaptiveFrontierEvaluator,
    CalibratedLinearDiscrepancyModel,
    FixedQueryStopPolicy,
    LearnedEVCRouter,
    LinearEVCModel,
    MatchedRandomEscalationRouter,
    NeverStopPolicy,
    RunningDiscrepancyModel,
)
from ..adaptive.budget import AdaptiveResourceLedger
from ..models import ModelPortfolio
from ..planner import Planner, PlanResult
from ..replay import VerifiedReplayStore, VerifiedTransition
from ..routing import (
    AccurateOnlyRouter,
    CheapOnlyRouter,
    ComputeRouter,
    FixedCascadeRouter,
    ThresholdRouter,
)
from ..types import (
    Action,
    EvidenceProvenance,
    Fidelity,
    ModelObservation,
    ModelQuote,
    ResourceUsage,
    SearchBudget,
)

BENCHMARK_ID = "sqlite-query-repair-l2-v1"
CHEAP_MODEL_ID = "sqlite-lexical-predictor-v1"
EXECUTABLE_MODEL_ID = "sqlite-memory-verifier-v1"


class SQLitePartition(str, Enum):
    DEVELOPMENT = "development_training"
    CALIBRATION = "calibration"
    EXPLORATORY = "exploratory_pilot"
    FUTURE_CONFIRMATORY = "future_confirmatory"


@dataclass(frozen=True, slots=True)
class PartitionReservation:
    """A partition declaration that deliberately contains no task material."""

    partition: SQLitePartition
    task_count: int
    status: str
    seed_policy: str


FUTURE_CONFIRMATORY_RESERVATION = PartitionReservation(
    partition=SQLitePartition.FUTURE_CONFIRMATORY,
    task_count=120,
    status="reserved_unmaterialized",
    seed_policy=(
        "Generate and freeze task IDs and paired seeds only after user approval "
        "and external timestamping of the final protocol."
    ),
)


@dataclass(frozen=True, slots=True)
class SQLCandidate:
    candidate_id: str
    sql: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.sql.strip():
            raise ValueError("SQL candidates require non-empty IDs and text")


SQLiteScalar = str | int | float | None
SQLiteRow = tuple[SQLiteScalar, ...]


@dataclass(frozen=True, slots=True)
class SQLiteTask:
    """One immutable query-repair task and its objective expected result."""

    task_id: str
    partition: SQLitePartition
    template_family: str
    prompt: str
    setup_sql: str
    candidates: tuple[SQLCandidate, ...]
    expected_columns: tuple[str, ...]
    expected_rows: tuple[SQLiteRow, ...]
    semantic_tokens: tuple[str, ...]
    ordered: bool = True

    def __post_init__(self) -> None:
        if self.partition is SQLitePartition.FUTURE_CONFIRMATORY:
            raise ValueError("confirmatory fixtures must not be materialized")
        if not self.task_id or not self.template_family or not self.prompt:
            raise ValueError("SQLite task identity and prompt cannot be empty")
        if len(self.candidates) < 2:
            raise ValueError("SQLite tasks require at least two candidate queries")
        identifiers = [candidate.candidate_id for candidate in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("SQLite candidate IDs must be unique within a task")
        if not self.expected_columns:
            raise ValueError("SQLite tasks require expected result columns")

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    def candidate(self, action: Action) -> SQLCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == action:
                return candidate
        raise ValueError(f"unknown SQL candidate action: {action!r}")

    @property
    def fixture_sha256(self) -> str:
        payload = {
            "task_id": self.task_id,
            "partition": self.partition.value,
            "template_family": self.template_family,
            "prompt": self.prompt,
            "setup_sql": self.setup_sql,
            "candidates": [
                {"candidate_id": item.candidate_id, "sql": item.sql}
                for item in self.candidates
            ],
            "expected_columns": self.expected_columns,
            "expected_rows": self.expected_rows,
            "semantic_tokens": self.semantic_tokens,
            "ordered": self.ordered,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


_COMMERCE_SETUP = """
CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE orders(
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(id),
  status TEXT NOT NULL,
  total REAL NOT NULL
);
INSERT INTO customers VALUES (1, 'Ada'), (2, 'Ben'), (3, 'Cy'), (4, 'Dee');
INSERT INTO orders VALUES
  (1, 1, 'paid', 70.0), (2, 1, 'paid', 100.0),
  (3, 1, 'cancelled', 500.0), (4, 2, 'paid', 45.0),
  (5, 3, 'paid', 120.0), (6, 4, 'pending', 300.0);
"""

_INVENTORY_SETUP = """
CREATE TABLE products(
  sku TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  stock INTEGER NOT NULL,
  reorder_level INTEGER NOT NULL,
  unit_price REAL NOT NULL
);
INSERT INTO products VALUES
  ('P1', 'hardware', 10, 10, 3.0),
  ('P2', 'hardware', 2, 8, 7.5),
  ('P3', 'office', 20, 5, 1.5),
  ('P4', 'office', 0, 3, 4.0),
  ('P5', 'hardware', 6, 6, 2.0);
"""

_LIBRARY_SETUP = """
CREATE TABLE authors(id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE books(id INTEGER PRIMARY KEY, author_id INTEGER NOT NULL, title TEXT);
CREATE TABLE loans(
  id INTEGER PRIMARY KEY,
  book_id INTEGER NOT NULL,
  due_date TEXT NOT NULL,
  returned_date TEXT
);
INSERT INTO authors VALUES
  (1, 'Ishiguro'), (2, 'Le Guin'), (3, 'Morrison'), (4, 'Ng');
INSERT INTO books VALUES
  (1, 1, 'Remains'), (2, 2, 'Earthsea'), (3, 2, 'Dispossessed'),
  (4, 3, 'Beloved'), (5, 4, 'Everything'), (6, 4, 'Little Fires');
INSERT INTO loans VALUES
  (1, 1, '2026-01-10', '2026-01-08'),
  (2, 2, '2026-02-10', NULL),
  (3, 3, '2026-01-01', '2026-01-20'),
  (4, 5, '2026-03-01', '2026-02-20');
"""

_SUPPORT_SETUP = """
CREATE TABLE agents(id INTEGER PRIMARY KEY, name TEXT NOT NULL, team TEXT NOT NULL);
CREATE TABLE tickets(
  id INTEGER PRIMARY KEY,
  agent_id INTEGER,
  status TEXT NOT NULL,
  priority TEXT NOT NULL,
  opened_at TEXT NOT NULL,
  resolution_hours REAL
);
INSERT INTO agents VALUES
  (1, 'Ari', 'alpha'), (2, 'Bo', 'alpha'), (3, 'Cam', 'beta');
INSERT INTO tickets VALUES
  (1, 1, 'closed', 'high', '2026-01-01', 2.0),
  (2, 1, 'closed', 'low', '2026-01-02', 6.0),
  (3, 2, 'open', 'high', '2026-01-03', NULL),
  (4, 2, 'closed', 'high', '2026-01-04', 8.0),
  (5, 3, 'closed', 'high', '2026-01-05', 3.0),
  (6, 3, 'open', 'low', '2026-01-06', NULL);
"""


_TASKS: tuple[SQLiteTask, ...] = (
    SQLiteTask(
        task_id="sqlite-dev-paid-totals",
        partition=SQLitePartition.DEVELOPMENT,
        template_family="grouped_aggregation",
        prompt=(
            "Return each customer whose paid-order total is at least 120, with "
            "columns name and paid_total, ordered by paid_total descending then name."
        ),
        setup_sql=_COMMERCE_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT c.name, SUM(o.total) AS paid_total FROM customers c "
                "JOIN orders o ON o.customer_id=c.id WHERE o.status='paid' "
                "GROUP BY c.id, c.name HAVING SUM(o.total)>=120 "
                "ORDER BY paid_total DESC, c.name",
            ),
            SQLCandidate(
                "q1",
                "SELECT c.name, SUM(o.total) AS paid_total FROM customers c "
                "JOIN orders o ON o.customer_id=c.id GROUP BY c.id, c.name "
                "HAVING SUM(o.total)>=120 ORDER BY paid_total DESC, c.name",
            ),
            SQLCandidate(
                "q2",
                "SELECT c.name, SUM(o.total) AS paid_total FROM customers c "
                "JOIN orders o ON o.customer_id=c.id WHERE o.status='paid' "
                "AND o.total>=120 GROUP BY c.id, c.name ORDER BY paid_total DESC",
            ),
            SQLCandidate(
                "q3",
                "SELECT c.name, SUM(o.total) AS paid_total FROM customers c "
                "JOIN orders o ON o.customer_id=c.id WHERE o.status='paid' "
                "GROUP BY c.id, c.name HAVING SUM(o.total)>120 "
                "ORDER BY paid_total DESC, c.name",
            ),
        ),
        expected_columns=("name", "paid_total"),
        expected_rows=(("Ada", 170.0), ("Cy", 120.0)),
        semantic_tokens=("join", "status='paid'", "sum(", "having", ">=120", "desc"),
    ),
    SQLiteTask(
        task_id="sqlite-dev-low-stock",
        partition=SQLitePartition.DEVELOPMENT,
        template_family="row_filter",
        prompt=(
            "List product sku values whose stock is strictly below reorder_level, "
            "ordered alphabetically."
        ),
        setup_sql=_INVENTORY_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT sku FROM products WHERE stock < reorder_level ORDER BY sku",
            ),
            SQLCandidate(
                "q1",
                "SELECT sku FROM products WHERE stock <= reorder_level ORDER BY sku",
            ),
            SQLCandidate(
                "q2",
                "SELECT sku FROM products WHERE stock > reorder_level ORDER BY sku",
            ),
            SQLCandidate(
                "q3",
                "SELECT sku FROM products WHERE stock < reorder_level "
                "AND category='hardware' ORDER BY sku",
            ),
        ),
        expected_columns=("sku",),
        expected_rows=(("P2",), ("P4",)),
        semantic_tokens=("stock", "<", "reorder_level", "order by sku"),
    ),
    SQLiteTask(
        task_id="sqlite-dev-category-value",
        partition=SQLitePartition.DEVELOPMENT,
        template_family="grouped_arithmetic",
        prompt=(
            "Compute total inventory value stock*unit_price per category as "
            "inventory_value, ordered by category."
        ),
        setup_sql=_INVENTORY_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT category, SUM(stock*unit_price) AS inventory_value "
                "FROM products GROUP BY category ORDER BY category",
            ),
            SQLCandidate(
                "q1",
                "SELECT category, SUM(stock)+SUM(unit_price) AS inventory_value "
                "FROM products GROUP BY category ORDER BY category",
            ),
            SQLCandidate(
                "q2",
                "SELECT category, stock*unit_price AS inventory_value "
                "FROM products GROUP BY category ORDER BY category",
            ),
            SQLCandidate(
                "q3",
                "SELECT category, SUM(stock*unit_price) AS inventory_value "
                "FROM products ORDER BY category",
            ),
        ),
        expected_columns=("category", "inventory_value"),
        expected_rows=(("hardware", 57.0), ("office", 30.0)),
        semantic_tokens=("sum(", "stock*unit_price", "group by", "order by category"),
    ),
    SQLiteTask(
        task_id="sqlite-cal-authors-no-loans",
        partition=SQLitePartition.CALIBRATION,
        template_family="anti_join",
        prompt=(
            "Return authors for whom none of their books has ever appeared in loans, "
            "ordered by author name."
        ),
        setup_sql=_LIBRARY_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT a.name FROM authors a WHERE NOT EXISTS (SELECT 1 FROM books b "
                "JOIN loans l ON l.book_id=b.id WHERE b.author_id=a.id) "
                "ORDER BY a.name",
            ),
            SQLCandidate(
                "q1",
                "SELECT DISTINCT a.name FROM authors a JOIN books b "
                "ON b.author_id=a.id LEFT JOIN loans l ON l.book_id=b.id "
                "WHERE l.id IS NULL ORDER BY a.name",
            ),
            SQLCandidate(
                "q2",
                "SELECT DISTINCT a.name FROM authors a JOIN books b "
                "ON b.author_id=a.id JOIN loans l ON l.book_id=b.id "
                "ORDER BY a.name",
            ),
            SQLCandidate(
                "q3",
                "SELECT a.name FROM authors a LEFT JOIN books b ON b.author_id=a.id "
                "WHERE b.id IS NULL ORDER BY a.name",
            ),
        ),
        expected_columns=("name",),
        expected_rows=(("Morrison",),),
        semantic_tokens=("not exists", "books", "join loans", "order by a.name"),
    ),
    SQLiteTask(
        task_id="sqlite-cal-open-high-priority",
        partition=SQLitePartition.CALIBRATION,
        template_family="conditional_count",
        prompt=(
            "For every agent, return name and the count of open high-priority "
            "tickets as open_high, including agents with zero, ordered by name."
        ),
        setup_sql=_SUPPORT_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT a.name, SUM(CASE WHEN t.status='open' AND t.priority='high' "
                "THEN 1 ELSE 0 END) AS open_high FROM agents a LEFT JOIN tickets t "
                "ON t.agent_id=a.id GROUP BY a.id, a.name ORDER BY a.name",
            ),
            SQLCandidate(
                "q1",
                "SELECT a.name, COUNT(*) AS open_high FROM agents a JOIN tickets t "
                "ON t.agent_id=a.id WHERE t.status='open' AND t.priority='high' "
                "GROUP BY a.id, a.name ORDER BY a.name",
            ),
            SQLCandidate(
                "q2",
                "SELECT a.name, COUNT(t.id) AS open_high FROM agents a "
                "LEFT JOIN tickets t "
                "ON t.agent_id=a.id GROUP BY a.id, a.name ORDER BY a.name",
            ),
            SQLCandidate(
                "q3",
                "SELECT a.name, SUM(t.status='open') AS open_high FROM agents a "
                "LEFT JOIN tickets t ON t.agent_id=a.id GROUP BY a.id ORDER BY a.name",
            ),
        ),
        expected_columns=("name", "open_high"),
        expected_rows=(("Ari", 0), ("Bo", 1), ("Cam", 0)),
        semantic_tokens=(
            "left join",
            "case when",
            "status='open'",
            "priority='high'",
            "group by",
        ),
    ),
    SQLiteTask(
        task_id="sqlite-pilot-repeat-paid",
        partition=SQLitePartition.EXPLORATORY,
        template_family="join_group_having",
        prompt=(
            "Return customer names with at least two paid orders, plus paid_count, "
            "ordered by name."
        ),
        setup_sql=_COMMERCE_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT c.name, COUNT(*) AS paid_count FROM customers c JOIN orders o "
                "ON o.customer_id=c.id WHERE o.status='paid' GROUP BY c.id, c.name "
                "HAVING COUNT(*)>=2 ORDER BY c.name",
            ),
            SQLCandidate(
                "q1",
                "SELECT c.name, COUNT(*) AS paid_count FROM customers c JOIN orders o "
                "ON o.customer_id=c.id GROUP BY c.id, c.name HAVING COUNT(*)>=2 "
                "ORDER BY c.name",
            ),
            SQLCandidate(
                "q2",
                "SELECT c.name, COUNT(*) AS paid_count FROM customers c JOIN orders o "
                "ON o.customer_id=c.id WHERE o.status='paid' GROUP BY c.id, c.name "
                "HAVING COUNT(*)>2 ORDER BY c.name",
            ),
            SQLCandidate(
                "q3",
                "SELEC c.name, COUNT(*) AS paid_count FROM customers c JOIN orders o "
                "ON o.customer_id=c.id WHERE o.status='paid' GROUP BY c.id, c.name "
                "ORDER BY c.name",
            ),
        ),
        expected_columns=("name", "paid_count"),
        expected_rows=(("Ada", 2),),
        semantic_tokens=("join", "status='paid'", "count(", "having", ">=2"),
    ),
    SQLiteTask(
        task_id="sqlite-pilot-overdue-loans",
        partition=SQLitePartition.EXPLORATORY,
        template_family="nullable_date_filter",
        prompt=(
            "As of 2026-02-15, list overdue, not-yet-returned book titles "
            "and due_date, "
            "ordered by due_date then title."
        ),
        setup_sql=_LIBRARY_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT b.title, l.due_date FROM books b JOIN loans l "
                "ON l.book_id=b.id "
                "WHERE l.returned_date IS NULL AND l.due_date<'2026-02-15' "
                "ORDER BY l.due_date, b.title",
            ),
            SQLCandidate(
                "q1",
                "SELECT b.title, l.due_date FROM books b JOIN loans l "
                "ON l.book_id=b.id "
                "WHERE l.due_date<'2026-02-15' ORDER BY l.due_date, b.title",
            ),
            SQLCandidate(
                "q2",
                "SELECT b.title, l.due_date FROM books b JOIN loans l "
                "ON l.book_id=b.id "
                "WHERE l.returned_date=NULL AND l.due_date<'2026-02-15' "
                "ORDER BY l.due_date, b.title",
            ),
            SQLCandidate(
                "q3",
                "SELECT b.title, l.due_date FROM books b JOIN loans l "
                "ON l.book_id=b.id "
                "WHERE l.returned_date IS NULL AND l.due_date>'2026-02-15' "
                "ORDER BY l.due_date, b.title",
            ),
        ),
        expected_columns=("title", "due_date"),
        expected_rows=(("Earthsea", "2026-02-10"),),
        semantic_tokens=("join loans", "is null", "due_date<", "order by"),
    ),
    SQLiteTask(
        task_id="sqlite-pilot-agent-resolution",
        partition=SQLitePartition.EXPLORATORY,
        template_family="filtered_average",
        prompt=(
            "For agents with closed tickets, return name and average resolution_hours "
            "as avg_hours, rounded to one decimal, ordered fastest first then name."
        ),
        setup_sql=_SUPPORT_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT a.name, ROUND(AVG(t.resolution_hours),1) AS avg_hours "
                "FROM agents a "
                "JOIN tickets t ON t.agent_id=a.id WHERE t.status='closed' "
                "GROUP BY a.id, a.name ORDER BY avg_hours, a.name",
            ),
            SQLCandidate(
                "q1",
                "SELECT a.name, ROUND(AVG(COALESCE(t.resolution_hours,0)),1) "
                "AS avg_hours FROM agents a "
                "JOIN tickets t ON t.agent_id=a.id GROUP BY a.id, a.name "
                "ORDER BY avg_hours, a.name",
            ),
            SQLCandidate(
                "q2",
                "SELECT a.name, ROUND(SUM(t.resolution_hours),1) AS avg_hours "
                "FROM agents a "
                "JOIN tickets t ON t.agent_id=a.id WHERE t.status='closed' "
                "GROUP BY a.id, a.name ORDER BY avg_hours, a.name",
            ),
            SQLCandidate(
                "q3",
                "SELECT a.name, ROUND(AVG(t.resolution_hours),1) AS avg_hours "
                "FROM agents a "
                "JOIN tickets t ON t.agent_id=a.id WHERE t.status='closed' "
                "GROUP BY a.id, a.name ORDER BY avg_hours DESC, a.name",
            ),
        ),
        expected_columns=("name", "avg_hours"),
        expected_rows=(("Cam", 3.0), ("Ari", 4.0), ("Bo", 8.0)),
        semantic_tokens=("avg(", "status='closed'", "group by", "order by avg_hours"),
    ),
    SQLiteTask(
        task_id="sqlite-pilot-reorder-cost",
        partition=SQLitePartition.EXPLORATORY,
        template_family="conditional_arithmetic",
        prompt=(
            "For products below reorder level, return sku and the cost to replenish "
            "to reorder_level as replenish_cost, ordered highest cost first then sku."
        ),
        setup_sql=_INVENTORY_SETUP,
        candidates=(
            SQLCandidate(
                "q0",
                "SELECT sku, (reorder_level-stock)*unit_price AS replenish_cost "
                "FROM products WHERE stock<reorder_level "
                "ORDER BY replenish_cost DESC, sku",
            ),
            SQLCandidate(
                "q1",
                "SELECT sku, reorder_level*unit_price AS replenish_cost FROM products "
                "WHERE stock<reorder_level ORDER BY replenish_cost DESC, sku",
            ),
            SQLCandidate(
                "q2",
                "SELECT sku, (reorder_level-stock)*unit_price AS replenish_cost "
                "FROM products WHERE stock<=reorder_level "
                "ORDER BY replenish_cost DESC, sku",
            ),
            SQLCandidate(
                "q3",
                "SELECT sku, (stock-reorder_level)*unit_price AS replenish_cost "
                "FROM products WHERE stock<reorder_level "
                "ORDER BY replenish_cost DESC, sku",
            ),
        ),
        expected_columns=("sku", "replenish_cost"),
        expected_rows=(("P2", 45.0), ("P4", 12.0)),
        semantic_tokens=(
            "reorder_level-stock",
            "unit_price",
            "stock<reorder_level",
            "desc",
        ),
    ),
)


def partition_manifest() -> dict[str, Any]:
    """Return public split metadata without materializing confirmatory data."""

    manifest: dict[str, Any] = {}
    for partition in (
        SQLitePartition.DEVELOPMENT,
        SQLitePartition.CALIBRATION,
        SQLitePartition.EXPLORATORY,
    ):
        tasks = tuple(task for task in _TASKS if task.partition is partition)
        manifest[partition.value] = {
            "status": "materialized_immutable_fixture",
            "task_ids": [task.task_id for task in tasks],
            "fixture_sha256": {task.task_id: task.fixture_sha256 for task in tasks},
            "template_families": sorted({task.template_family for task in tasks}),
        }
    manifest[SQLitePartition.FUTURE_CONFIRMATORY.value] = {
        "status": FUTURE_CONFIRMATORY_RESERVATION.status,
        "task_count": FUTURE_CONFIRMATORY_RESERVATION.task_count,
        "task_ids": None,
        "seeds": None,
        "seed_policy": FUTURE_CONFIRMATORY_RESERVATION.seed_policy,
    }
    return manifest


def load_sqlite_partition(partition: SQLitePartition) -> tuple[SQLiteTask, ...]:
    """Load an allowed partition; confirmatory access always fails closed."""

    if partition is SQLitePartition.FUTURE_CONFIRMATORY:
        raise PermissionError(
            "future-confirmatory SQLite tasks are reserved and unmaterialized"
        )
    return tuple(task for task in _TASKS if task.partition is partition)


def get_sqlite_task(task_id: str) -> SQLiteTask:
    for task in _TASKS:
        if task.task_id == task_id:
            return task
    raise KeyError(f"unknown materialized SQLite task ID: {task_id}")


def paired_sqlite_task(task: SQLiteTask, paired_seed: int) -> SQLiteTask:
    """Return a deterministic candidate-order permutation for paired reruns."""

    candidates = list(task.candidates)
    seed_material = hashlib.sha256(f"{task.task_id}:{paired_seed}".encode()).digest()
    Random(int.from_bytes(seed_material[:8], "big")).shuffle(candidates)
    return SQLiteTask(
        task_id=task.task_id,
        partition=task.partition,
        template_family=task.template_family,
        prompt=task.prompt,
        setup_sql=task.setup_sql,
        candidates=tuple(candidates),
        expected_columns=task.expected_columns,
        expected_rows=task.expected_rows,
        semantic_tokens=task.semantic_tokens,
        ordered=task.ordered,
    )


class SQLiteExecutionStatus(str, Enum):
    COMPLETED = "completed"
    SQL_ERROR = "sql_error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SQLiteExecutionResult:
    status: SQLiteExecutionStatus
    columns: tuple[str, ...]
    rows: tuple[SQLiteRow, ...]
    verifier_passed: bool
    terminated: bool
    truncated: bool
    latency_s: float
    vm_steps: int
    error_type: str | None = None
    error_message: str | None = None


_DENIED_SQLITE_ACTIONS = frozenset(
    action
    for action in (
        getattr(sqlite3, "SQLITE_ATTACH", None),
        getattr(sqlite3, "SQLITE_DETACH", None),
        getattr(sqlite3, "SQLITE_ALTER_TABLE", None),
        getattr(sqlite3, "SQLITE_ANALYZE", None),
        getattr(sqlite3, "SQLITE_CREATE_INDEX", None),
        getattr(sqlite3, "SQLITE_CREATE_TABLE", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_INDEX", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TABLE", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_VIEW", None),
        getattr(sqlite3, "SQLITE_CREATE_TRIGGER", None),
        getattr(sqlite3, "SQLITE_CREATE_VIEW", None),
        getattr(sqlite3, "SQLITE_DELETE", None),
        getattr(sqlite3, "SQLITE_DROP_INDEX", None),
        getattr(sqlite3, "SQLITE_DROP_TABLE", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_INDEX", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TABLE", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_VIEW", None),
        getattr(sqlite3, "SQLITE_DROP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_DROP_VIEW", None),
        getattr(sqlite3, "SQLITE_INSERT", None),
        getattr(sqlite3, "SQLITE_PRAGMA", None),
        getattr(sqlite3, "SQLITE_REINDEX", None),
        getattr(sqlite3, "SQLITE_TRANSACTION", None),
        getattr(sqlite3, "SQLITE_UPDATE", None),
    )
    if action is not None
)


def _normalize_scalar(value: Any) -> SQLiteScalar:
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("SQLite verifier rows must contain finite numbers")
        return round(value, 10)
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"unsupported SQLite result value: {type(value).__name__}")


def _normalize_rows(rows: Sequence[Sequence[Any]]) -> tuple[SQLiteRow, ...]:
    return tuple(tuple(_normalize_scalar(value) for value in row) for row in rows)


class SQLiteSandbox:
    """Immutable in-memory template copied for every candidate execution."""

    def __init__(
        self,
        task: SQLiteTask,
        *,
        maximum_vm_steps: int = 100_000,
        timeout_s: float = 0.25,
        progress_granularity: int = 100,
    ) -> None:
        if maximum_vm_steps < 1 or progress_granularity < 1:
            raise ValueError("SQLite VM limits must be positive")
        if timeout_s <= 0:
            raise ValueError("SQLite timeout_s must be positive")
        self.task = task
        self.maximum_vm_steps = maximum_vm_steps
        self.timeout_s = timeout_s
        self.progress_granularity = progress_granularity
        self._template = sqlite3.connect(":memory:")
        self._template.execute("PRAGMA foreign_keys=ON")
        self._template.executescript(task.setup_sql)
        self._template.commit()
        self._fixture_checksum = self.fixture_checksum()

    def close(self) -> None:
        self._template.close()

    def __enter__(self) -> SQLiteSandbox:
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.close()

    def fixture_checksum(self) -> str:
        dump = "\n".join(self._template.iterdump()).encode("utf-8")
        return hashlib.sha256(dump).hexdigest()

    @property
    def fixture_intact(self) -> bool:
        return self.fixture_checksum() == self._fixture_checksum

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield one disposable clone and always close it after rollback."""

        connection = sqlite3.connect(":memory:")
        self._template.backup(connection)
        try:
            connection.execute("PRAGMA query_only=ON")
            yield connection
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()
            if not self.fixture_intact:
                raise RuntimeError("SQLite benchmark template was mutated")

    def execute(self, sql: str) -> SQLiteExecutionResult:
        started_at = time.perf_counter()
        vm_steps = 0
        timed_out = False

        def authorizer(
            action: int,
            argument1: str | None,
            argument2: str | None,
            database: str | None,
            source: str | None,
        ) -> int:
            del argument1, argument2, database, source
            return (
                sqlite3.SQLITE_DENY
                if action in _DENIED_SQLITE_ACTIONS
                else sqlite3.SQLITE_OK
            )

        deadline = started_at + self.timeout_s

        def progress() -> int:
            nonlocal vm_steps, timed_out
            vm_steps += self.progress_granularity
            timed_out = (
                vm_steps > self.maximum_vm_steps or time.perf_counter() >= deadline
            )
            return int(timed_out)

        try:
            with self.transaction() as connection:
                connection.set_authorizer(authorizer)
                connection.set_progress_handler(progress, self.progress_granularity)
                cursor = connection.execute(sql)
                if cursor.description is None:
                    raise sqlite3.OperationalError(
                        "candidate did not produce a read-only result set"
                    )
                columns = tuple(str(item[0]) for item in cursor.description)
                rows = _normalize_rows(cursor.fetchall())
                if not self.task.ordered:
                    rows = tuple(sorted(rows, key=repr))
                expected = self.task.expected_rows
                if not self.task.ordered:
                    expected = tuple(sorted(expected, key=repr))
                passed = columns == self.task.expected_columns and rows == expected
                return SQLiteExecutionResult(
                    status=SQLiteExecutionStatus.COMPLETED,
                    columns=columns,
                    rows=rows,
                    verifier_passed=passed,
                    terminated=True,
                    truncated=False,
                    latency_s=max(0.0, time.perf_counter() - started_at),
                    vm_steps=vm_steps,
                )
        except sqlite3.DatabaseError as exc:
            status = (
                SQLiteExecutionStatus.TIMEOUT
                if timed_out
                else SQLiteExecutionStatus.REJECTED
                if "not authorized" in str(exc).lower()
                else SQLiteExecutionStatus.SQL_ERROR
            )
            return SQLiteExecutionResult(
                status=status,
                columns=(),
                rows=(),
                verifier_passed=False,
                terminated=not timed_out,
                truncated=timed_out,
                latency_s=max(0.0, time.perf_counter() - started_at),
                vm_steps=vm_steps,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )


@dataclass(frozen=True, slots=True)
class SQLiteActionProvider:
    def legal_actions(self, state: SQLiteTask) -> tuple[str, ...]:
        return state.actions


@dataclass(frozen=True, slots=True)
class SQLiteTaskCodec:
    def key(self, state: SQLiteTask) -> tuple[str, str]:
        return state.task_id, state.fixture_sha256


@dataclass(frozen=True, slots=True)
class SQLiteCheapModel:
    """Deterministic local proxy using prompt-derived SQL lexical features."""

    cost: float = 0.25
    tokens_per_query: int = 8

    @property
    def model_id(self) -> str:
        return CHEAP_MODEL_ID

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.CHEAP

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del rollout_depth
        return ModelQuote(
            cost=self.cost,
            tokens=min(token_budget, self.tokens_per_query),
            expected_latency_s=0.0001,
        )

    def evaluate(
        self,
        state: SQLiteTask,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del rollout_depth, rng
        candidate = state.candidate(action)
        normalized = " ".join(candidate.sql.lower().split())
        matches = sum(token in normalized for token in state.semantic_tokens)
        coverage = matches / max(1, len(state.semantic_tokens))
        digest = hashlib.sha256(
            f"{state.task_id}:{candidate.candidate_id}".encode()
        ).digest()
        stable_bias = (digest[0] / 255.0 - 0.5) * 0.18
        suspicious = sum(
            token in normalized
            for token in ("=null", "select *", " delete ", " update ")
        )
        probability = min(
            0.98,
            max(0.02, 0.08 + 0.84 * coverage + stable_bias - 0.2 * suspicious),
        )
        variance = max(0.01, probability * (1.0 - probability))
        tokens = min(token_budget, self.tokens_per_query)
        return ModelObservation(
            value=probability,
            variance=variance,
            cost=self.cost,
            tokens=tokens,
            latency_s=0.0001,
            provenance=EvidenceProvenance.MODEL_PREDICTED,
            metadata={
                "execution_status": "model_prediction",
                "verifier_passed": None,
                "synthetic": False,
                "model_predicted": True,
                "executable": False,
                "verified": False,
                "semantic_coverage": coverage,
                "model_family": "deterministic_lexical_proxy",
            },
        )


@dataclass(frozen=True, slots=True)
class SQLiteExecutableModel:
    """Accurate fidelity backed by an isolated objective SQLite verifier."""

    cost: float = 4.0
    maximum_vm_steps: int = 100_000
    timeout_s: float = 0.25

    @property
    def model_id(self) -> str:
        return EXECUTABLE_MODEL_ID

    @property
    def fidelity(self) -> Fidelity:
        return Fidelity.ACCURATE

    def quote(self, *, token_budget: int, rollout_depth: int) -> ModelQuote:
        del token_budget, rollout_depth
        return ModelQuote(
            cost=self.cost,
            accurate_calls=1,
            expected_latency_s=min(self.timeout_s, 0.005),
            environment_calls=1,
        )

    def evaluate(
        self,
        state: SQLiteTask,
        action: Action,
        *,
        token_budget: int,
        rollout_depth: int,
        rng: Random,
    ) -> ModelObservation:
        del token_budget, rng
        candidate = state.candidate(action)
        step_limit = min(
            self.maximum_vm_steps,
            max(1, rollout_depth) * self.maximum_vm_steps,
        )
        with SQLiteSandbox(
            state,
            maximum_vm_steps=step_limit,
            timeout_s=self.timeout_s,
        ) as sandbox:
            result = sandbox.execute(candidate.sql)
        value = 1.0 if result.verifier_passed else -1.0
        return ModelObservation(
            value=value,
            variance=0.0,
            cost=self.cost,
            latency_s=result.latency_s,
            terminated=result.terminated,
            truncated=result.truncated,
            environment_calls=1,
            provenance=EvidenceProvenance.EXECUTABLE,
            # A failed query is still an objectively verified failure outcome.
            verified=True,
            metadata={
                "execution_status": result.status.value,
                "failure_type": result.error_type,
                "failure_message": result.error_message,
                "verifier_passed": result.verifier_passed,
                "synthetic": False,
                "model_predicted": False,
                "executable": True,
                "verified": True,
                "vm_steps": result.vm_steps,
                "fixture_sha256": state.fixture_sha256,
            },
        )


def make_sqlite_portfolio(
    *,
    cheap_cost: float = 0.25,
    executable_cost: float = 4.0,
    timeout_s: float = 0.25,
    maximum_vm_steps: int = 100_000,
) -> ModelPortfolio:
    return ModelPortfolio.from_models(
        (
            SQLiteCheapModel(cost=cheap_cost),
            SQLiteExecutableModel(
                cost=executable_cost,
                timeout_s=timeout_s,
                maximum_vm_steps=maximum_vm_steps,
            ),
        )
    )


def score_sqlite_action(task: SQLiteTask, action: Action) -> SQLiteExecutionResult:
    """Execute the selected task action once, outside the compute budget."""

    with SQLiteSandbox(task) as sandbox:
        return sandbox.execute(task.candidate(action).sql)


def _router_features_for_training(
    task: SQLiteTask,
    action: str,
    observation: ModelObservation,
    budget: SearchBudget,
) -> dict[str, float]:
    values = [
        SQLiteCheapModel()
        .evaluate(
            task,
            candidate,
            token_budget=8,
            rollout_depth=1,
            rng=Random(0),
        )
        .value
        for candidate in task.actions
    ]
    best = max(values)
    return {
        "action_mean": observation.value,
        "action_uncertainty": observation.variance**0.5,
        "action_risk": 0.0,
        "gap_to_best": max(0.0, best - observation.value),
        "evidence_count": 1.0,
        "search_depth": 0.0,
        "remaining_cost": budget.max_cost,
        "remaining_accurate_calls": float(budget.max_accurate_calls),
    }


def build_sqlite_training_replay(
    partition: SQLitePartition,
    *,
    budget: SearchBudget,
) -> tuple[VerifiedTransition, ...]:
    """Build exhaustive verified pairs from development/calibration only."""

    if partition not in {
        SQLitePartition.DEVELOPMENT,
        SQLitePartition.CALIBRATION,
    }:
        raise PermissionError(
            "router training replay may use development or calibration only"
        )
    cheap = SQLiteCheapModel()
    accurate = SQLiteExecutableModel()
    records: list[VerifiedTransition] = []
    for task in load_sqlite_partition(partition):
        for action in task.actions:
            cheap_observation = cheap.evaluate(
                task,
                action,
                token_budget=cheap.tokens_per_query,
                rollout_depth=1,
                rng=Random(0),
            )
            accurate_observation = accurate.evaluate(
                task,
                action,
                token_budget=0,
                rollout_depth=1,
                rng=Random(0),
            )
            features = _router_features_for_training(
                task,
                action,
                cheap_observation,
                budget,
            )
            records.append(
                VerifiedTransition(
                    state_id=task.task_id,
                    action=action,
                    cheap_model_id=cheap.model_id,
                    cheap_prediction=cheap_observation.value,
                    accurate_model_id=accurate.model_id,
                    verified_outcome=accurate_observation.value,
                    router_propensity=1.0,
                    cheap_provenance=cheap_observation.provenance,
                    accurate_provenance=accurate_observation.provenance,
                    context_features=features,
                    randomized_audit=False,
                    metadata={
                        "partition": partition.value,
                        "fixture_sha256": task.fixture_sha256,
                        "verifier_passed": accurate_observation.metadata[
                            "verifier_passed"
                        ],
                    },
                )
            )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class SQLiteLearnedComponents:
    evc_model: LinearEVCModel
    discrepancy: CalibratedLinearDiscrepancyModel
    training_records: tuple[VerifiedTransition, ...]
    calibration_records: tuple[VerifiedTransition, ...]


def train_sqlite_components(
    *,
    budget: SearchBudget,
    coverage: float = 0.9,
    ridge: float = 1.0,
) -> SQLiteLearnedComponents:
    training = build_sqlite_training_replay(
        SQLitePartition.DEVELOPMENT,
        budget=budget,
    )
    calibration = build_sqlite_training_replay(
        SQLitePartition.CALIBRATION,
        budget=budget,
    )
    evc_model = LinearEVCModel(ridge=ridge)
    evc_model.fit(
        training,
        calibration_records=calibration,
        coverage=coverage,
    )
    discrepancy = CalibratedLinearDiscrepancyModel(ridge=ridge)
    discrepancy.fit(
        training,
        calibration_records=calibration,
        coverage=coverage,
    )
    return SQLiteLearnedComponents(
        evc_model,
        discrepancy,
        training,
        calibration,
    )


@dataclass(frozen=True, slots=True)
class DirectSQLPlanner:
    """No-search baseline using a deterministic fixture-blind direct policy."""

    def plan(
        self,
        state: SQLiteTask,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        del models, budget, seed
        digest = hashlib.sha256(state.task_id.encode("utf-8")).digest()
        action = state.actions[digest[0] % len(state.actions)]
        return PlanResult(
            action=action,
            predicted_value=0.0,
            usage=ResourceUsage(),
            trace=(
                {
                    "decision_type": "task_action",
                    "task_action": action,
                    "model_id": None,
                    "provenance": EvidenceProvenance.SYNTHETIC.value,
                    "synthetic": True,
                    "model_predicted": False,
                    "executable": False,
                    "verified": False,
                    "route_propensity": 1.0,
                    "execution_status": "not_executed_during_planning",
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class LearnedDirectSQLPlanner:
    """Query-level learned baseline: score all actions, then act without routing."""

    token_budget: int = 8

    def plan(
        self,
        state: SQLiteTask,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        model = models.get(CHEAP_MODEL_ID)
        ledger = AdaptiveResourceLedger(budget)
        rng = Random(seed)
        predictions: dict[str, float] = {}
        trace: list[Mapping[str, Any]] = []
        for action in state.actions:
            quote = model.quote(token_budget=self.token_budget, rollout_depth=1)
            reservation = ledger.reserve(quote)
            if reservation is None:
                break
            try:
                observation = model.evaluate(
                    state,
                    action,
                    token_budget=self.token_budget,
                    rollout_depth=1,
                    rng=rng,
                )
            except Exception:
                ledger.fail(reservation)
                raise
            ledger.commit(reservation, observation)
            predictions[action] = observation.value
            trace.append(
                {
                    "decision_type": "compute_action",
                    "task_action": action,
                    "model_id": model.model_id,
                    "fidelity": model.fidelity.value,
                    "provenance": observation.provenance.value,
                    "synthetic": False,
                    "model_predicted": True,
                    "executable": False,
                    "verified": False,
                    "route_propensity": 1.0,
                    "execution_status": "model_prediction",
                    "value": observation.value,
                    "variance": observation.variance,
                    "cost": observation.cost,
                    "tokens": observation.tokens,
                    "latency_s": observation.latency_s,
                }
            )
        if not predictions:
            raise RuntimeError("budget cannot afford one learned-direct query")
        action = max(
            state.actions, key=lambda item: (predictions.get(item, -math.inf), item)
        )
        return PlanResult(action, predictions[action], ledger.usage(), tuple(trace))


@dataclass(frozen=True, slots=True)
class OracleSQLPlanner:
    """Executable counterfactual upper bound for exploratory fixtures only."""

    def plan(
        self,
        state: SQLiteTask,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult:
        if state.partition is not SQLitePartition.EXPLORATORY:
            raise PermissionError("oracle routing is restricted to exploratory tasks")
        model = models.get(EXECUTABLE_MODEL_ID)
        ledger = AdaptiveResourceLedger(budget)
        rng = Random(seed)
        outcomes: dict[str, float] = {}
        trace: list[Mapping[str, Any]] = []
        for action in state.actions:
            quote = model.quote(token_budget=0, rollout_depth=1)
            reservation = ledger.reserve(quote)
            if reservation is None:
                break
            try:
                observation = model.evaluate(
                    state,
                    action,
                    token_budget=0,
                    rollout_depth=1,
                    rng=rng,
                )
            except Exception:
                ledger.fail(reservation)
                raise
            ledger.commit(reservation, observation)
            outcomes[action] = observation.value
            trace.append(
                {
                    "decision_type": "compute_action",
                    "task_action": action,
                    "model_id": model.model_id,
                    "fidelity": model.fidelity.value,
                    "provenance": observation.provenance.value,
                    "synthetic": False,
                    "model_predicted": False,
                    "executable": True,
                    "verified": True,
                    "verifier_passed": observation.metadata["verifier_passed"],
                    "route_propensity": 1.0,
                    "execution_status": observation.metadata["execution_status"],
                    "failure_type": observation.metadata["failure_type"],
                    "value": observation.value,
                    "variance": observation.variance,
                    "cost": observation.cost,
                    "tokens": observation.tokens,
                    "latency_s": observation.latency_s,
                }
            )
        if not outcomes:
            raise RuntimeError("budget cannot afford one oracle executable query")
        action = max(
            state.actions, key=lambda item: (outcomes.get(item, -math.inf), item)
        )
        return PlanResult(action, outcomes[action], ledger.usage(), tuple(trace))


PRIMARY_SQLITE_METHODS = (
    "direct",
    "cheap_only",
    "accurate_only",
    "fixed_cascade",
    "random_matched",
    "uncertainty_threshold",
    "learned_direct",
    "learned_evc_proxy",
    "fidelity_mcts",
    "oracle_counterfactual",
)

SQLITE_ABLATIONS = (
    "fidelity_no_discrepancy",
    "fidelity_no_audit",
    "fidelity_fixed_stopping",
    "fidelity_no_verification",
    "fidelity_no_tree_reuse",
    "fidelity_cheap_fidelity_only",
)


def make_sqlite_planner(
    method: str,
    *,
    components: SQLiteLearnedComponents,
    matched_accurate_calls: int,
    seed: int,
    method_settings: Mapping[str, Any] | None = None,
) -> Planner:
    """Construct a declared baseline or ablation from public interfaces."""

    settings = {} if method_settings is None else dict(method_settings)
    if method == "direct":
        return DirectSQLPlanner()
    if method == "learned_direct":
        return LearnedDirectSQLPlanner(
            token_budget=int(settings.get("cheap_token_budget", 8))
        )
    if method == "oracle_counterfactual":
        return OracleSQLPlanner()

    router: ComputeRouter
    stop_policy: Any = NeverStopPolicy()
    discrepancy: Any = components.discrepancy
    replay: VerifiedReplayStore = VerifiedReplayStore()
    cheap_tokens = int(settings.get("cheap_token_budget", 8))
    if method in {"cheap_only", "fidelity_cheap_fidelity_only"}:
        router = CheapOnlyRouter(CHEAP_MODEL_ID, token_budget=cheap_tokens)
    elif method == "accurate_only":
        router = AccurateOnlyRouter(EXECUTABLE_MODEL_ID)
    elif method == "fixed_cascade":
        router = FixedCascadeRouter(
            CHEAP_MODEL_ID,
            EXECUTABLE_MODEL_ID,
            top_k=int(settings.get("top_k", 2)),
            cheap_token_budget=cheap_tokens,
        )
    elif method == "random_matched":
        router = MatchedRandomEscalationRouter(
            CHEAP_MODEL_ID,
            EXECUTABLE_MODEL_ID,
            target_accurate_calls=matched_accurate_calls,
            seed=seed,
            cheap_token_budget=cheap_tokens,
        )
    elif method == "uncertainty_threshold":
        router = ThresholdRouter(
            CHEAP_MODEL_ID,
            EXECUTABLE_MODEL_ID,
            z_score=float(settings.get("z_score", 1.0)),
            cheap_token_budget=cheap_tokens,
        )
    elif method in {
        "learned_evc_proxy",
        "fidelity_mcts",
        "fidelity_no_discrepancy",
        "fidelity_no_audit",
        "fidelity_fixed_stopping",
        "fidelity_no_tree_reuse",
    }:
        audit_probability = float(settings.get("audit_probability", 0.1))
        if method == "fidelity_no_discrepancy":
            discrepancy = RunningDiscrepancyModel()
        if method == "fidelity_no_audit":
            audit_probability = 0.0
        if method == "learned_evc_proxy":
            stop_policy = FixedQueryStopPolicy(int(settings.get("evc_query_limit", 5)))
        elif method == "fidelity_fixed_stopping":
            stop_policy = FixedQueryStopPolicy(
                int(settings.get("fixed_query_limit", 6))
            )
        router = LearnedEVCRouter(
            CHEAP_MODEL_ID,
            EXECUTABLE_MODEL_ID,
            components.evc_model,
            accurate_cost=float(settings.get("accurate_cost", 4.0)),
            cost_weight=float(settings.get("cost_weight", 0.1)),
            minimum_net_evc=float(settings.get("minimum_net_evc", 0.0)),
            audit_probability=audit_probability,
            seed=seed,
            cheap_token_budget=cheap_tokens,
        )
    elif method == "fidelity_no_verification":
        router = CheapOnlyRouter(CHEAP_MODEL_ID, token_budget=cheap_tokens)
    else:
        raise ValueError(f"unknown SQLite L2 method: {method}")

    return AdaptiveComputePlanner(
        action_provider=SQLiteActionProvider(),
        router=router,
        stop_policy=stop_policy,
        codec=SQLiteTaskCodec(),
        discrepancy=discrepancy,
        replay=replay,
    )


def make_sqlite_frontier_evaluator(
    *,
    components: SQLiteLearnedComponents,
    budget: SearchBudget,
    seed: int,
    method_settings: Mapping[str, Any] | None = None,
    portfolio: ModelPortfolio | None = None,
) -> AdaptiveFrontierEvaluator:
    """Bind the L2 portfolio to the classical adaptive frontier interface."""

    planner = make_sqlite_planner(
        "fidelity_mcts",
        components=components,
        matched_accurate_calls=0,
        seed=seed,
        method_settings=method_settings,
    )
    if not isinstance(planner, AdaptiveComputePlanner):
        raise TypeError("FidelityMCTS SQLite planner must be adaptive")
    resolved_portfolio = make_sqlite_portfolio() if portfolio is None else portfolio
    return AdaptiveFrontierEvaluator(planner, resolved_portfolio, budget)


def calibrate_matched_random_calls(
    budgets: Sequence[SearchBudget],
    *,
    components: SQLiteLearnedComponents,
    method_settings: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Choose random-baseline quotas using calibration tasks only."""

    tasks = load_sqlite_partition(SQLitePartition.CALIBRATION)
    targets: list[int] = []
    for budget_index, budget in enumerate(budgets):
        calls: list[int] = []
        for task_index, task in enumerate(tasks):
            planner = make_sqlite_planner(
                "fidelity_mcts",
                components=components,
                matched_accurate_calls=0,
                seed=10_000 + budget_index * 100 + task_index,
                method_settings={
                    **({} if method_settings is None else method_settings),
                    "audit_probability": 0.0,
                },
            )
            result = planner.plan(
                task,
                models=make_sqlite_portfolio(),
                budget=budget,
                seed=20_000 + budget_index * 100 + task_index,
            )
            calls.append(result.usage.accurate_calls)
        target = round(sum(calls) / len(calls))
        targets.append(max(0, min(target, budget.max_accurate_calls)))
    return tuple(targets)


def sqlite_fixture_artifact_hash() -> str:
    """Hash every materialized fixture and the unmaterialized split declaration."""

    payload = partition_manifest()
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def save_partition_manifest(path: Path) -> None:
    """Persist public split metadata; this never creates confirmatory tasks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(partition_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def planner_report_fields(result: PlanResult) -> tuple[str, Mapping[str, Any]]:
    """Extract adaptive stop/report metadata without coupling direct baselines."""

    report = getattr(result, "report", None)
    if report is None:
        return "direct_action", {}
    return str(report.stop_reason), cast(
        Mapping[str, Any],
        {
            "state_id": report.state_id,
            "replay_records_added": report.replay_records_added,
            "randomized_audit_queries": report.randomized_audit_queries,
        },
    )
