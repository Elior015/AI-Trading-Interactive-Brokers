"""SQLite (`StateStore`) and DuckDB (`BarStore`) persistence.

Reconstructed from the test suite and every call site that uses it: the
original module was never actually committed to git (an unanchored `data/`
line in `.gitignore` silently swallowed this whole package alongside the
intended top-level runtime `./data` directory), so no version history exists
to restore it from.

Two properties matter here more than anywhere else in the system:

* **Write-ahead order persistence.** `OrderManager` commits an order row
  *before* the wire call, so a crash between "we sent it" and "IBKR
  acknowledged it" leaves a row reconciliation can find and resolve, rather
  than a duplicate position after restart.
* **The rate limiters live elsewhere, but the durable bar store here is what
  makes them affordable**: `BarStore` persists every historical/streaming
  bar exactly once (deduplicated on symbol+ts+period), so a symbol already
  backfilled is never re-fetched just because the process restarted.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from ..domain.models import Bar
from ..logging_setup import get_logger

log = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _enum_value(x: Any) -> Any:
    """Unwrap a str-backed Enum (`OrderState`, ...) to its plain string value."""
    return x.value if hasattr(x, "value") else x


class StateStore:
    """SQLite-backed KV + orders + fills + risk events + narrative + review store.

    Every method here is synchronous by design: every call site in this
    codebase — `OrderManager`, `RiskEngine`, `KillSwitch`, `LLMGateway`,
    `SessionNarrative`, the dashboard — calls them without `await`. SQLite on
    local disk is fast enough that this has never needed to be async, and a
    single `threading.RLock` around the connection is enough to make it safe
    if it is ever touched from more than one thread.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_ref TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    limit_price REAL,
                    stop_price REAL,
                    target_price REAL,
                    state TEXT NOT NULL,
                    ib_order_id INTEGER,
                    ib_perm_id INTEGER,
                    parent_ref TEXT,
                    filled_quantity REAL DEFAULT 0,
                    avg_fill_price REAL,
                    cycle_id TEXT,
                    rationale TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS order_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_ref TEXT NOT NULL,
                    previous_state TEXT,
                    new_state TEXT,
                    raw_status TEXT,
                    ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exec_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    order_ref TEXT,
                    ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT,
                    symbol TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    reason TEXT,
                    detail TEXT,
                    ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS narrative (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_capabilities (
                    host TEXT NOT NULL,
                    model TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    capabilities TEXT,
                    checked_at TEXT NOT NULL,
                    PRIMARY KEY (host, model)
                );

                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT,
                    request TEXT,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    # -- generic kv ---------------------------------------------------------- #

    def get_state(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, payload),
            )

    # -- orders ---------------------------------------------------------------- #

    def save_order(self, order: Any, perm_id: int | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO orders (
                    order_ref, symbol, action, quantity, limit_price, stop_price,
                    target_price, state, ib_order_id, ib_perm_id, parent_ref,
                    filled_quantity, avg_fill_price, cycle_id, rationale,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_ref) DO UPDATE SET
                    symbol = excluded.symbol,
                    action = excluded.action,
                    quantity = excluded.quantity,
                    limit_price = excluded.limit_price,
                    stop_price = excluded.stop_price,
                    target_price = excluded.target_price,
                    state = excluded.state,
                    ib_order_id = excluded.ib_order_id,
                    ib_perm_id = COALESCE(excluded.ib_perm_id, orders.ib_perm_id),
                    parent_ref = excluded.parent_ref,
                    filled_quantity = excluded.filled_quantity,
                    avg_fill_price = excluded.avg_fill_price,
                    cycle_id = excluded.cycle_id,
                    rationale = excluded.rationale,
                    updated_at = excluded.updated_at
                """,
                (
                    order.order_ref,
                    order.symbol,
                    order.action,
                    order.quantity,
                    order.limit_price,
                    order.stop_price,
                    order.target_price,
                    _enum_value(order.state),
                    order.ib_order_id,
                    perm_id,
                    order.parent_ref,
                    order.filled_quantity,
                    order.avg_fill_price,
                    order.cycle_id,
                    order.rationale,
                    order.created_at.isoformat(),
                    order.updated_at.isoformat(),
                ),
            )

    def find_order_by_ref(self, order_ref: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE order_ref = ?", (order_ref,)
            ).fetchone()
        return dict(row) if row is not None else None

    def load_orders(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM orders ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def record_transition(self, order_ref: str, previous: Any, new: Any, raw_status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO order_transitions "
                "(order_ref, previous_state, new_state, raw_status, ts) VALUES (?, ?, ?, ?, ?)",
                (order_ref, _enum_value(previous), _enum_value(new), raw_status, _now_iso()),
            )

    # -- fills ------------------------------------------------------------------ #

    def save_fill(self, exec_id: str, fill: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO fills (exec_id, symbol, action, quantity, price, commission, "
                "order_ref, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(exec_id) DO NOTHING",
                (
                    exec_id,
                    fill.symbol,
                    fill.action,
                    fill.quantity,
                    fill.price,
                    fill.commission,
                    fill.order_ref,
                    fill.ts.isoformat(),
                ),
            )

    def load_fills(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fills ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- risk events / rejections ------------------------------------------------ #

    def save_risk_event(
        self, cycle_id: str, symbol: str, approved: bool, reason: str | None, detail: str
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO risk_events (cycle_id, symbol, approved, reason, detail, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cycle_id, symbol, 1 if approved else 0, reason, detail, _now_iso()),
            )

    def load_rejections(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, symbol, reason, detail FROM risk_events "
                "WHERE approved = 0 ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def rejection_counts(self) -> dict[str, int]:
        """Rejections by reason, for today (UTC)."""
        today = datetime.now(UTC).date().isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT reason, COUNT(*) AS n FROM risk_events "
                "WHERE approved = 0 AND reason IS NOT NULL AND ts >= ? "
                "GROUP BY reason ORDER BY n DESC",
                (today,),
            ).fetchall()
        return {r["reason"]: r["n"] for r in rows}

    # -- narrative ---------------------------------------------------------------- #

    def append_narrative(self, role: str, content: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO narrative (role, content, ts) VALUES (?, ?, ?)",
                (role, content, _now_iso()),
            )

    def replace_narrative(self, entries: list[tuple[str, str]]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM narrative")
            self._conn.executemany(
                "INSERT INTO narrative (role, content, ts) VALUES (?, ?, ?)",
                [(role, content, _now_iso()) for role, content in entries],
            )

    def load_narrative(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, ts FROM narrative ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- end-of-day review ---------------------------------------------------------- #

    def save_review(self, review: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reviews (data, ts) VALUES (?, ?)",
                (json.dumps(review, default=str), _now_iso()),
            )

    def load_last_review(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM reviews ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["data"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    # -- llm structured-output capability cache -------------------------------------- #

    def load_capability(self, host: str, model: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT strategy, capabilities, checked_at FROM llm_capabilities "
                "WHERE host = ? AND model = ?",
                (host, model),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        try:
            out["capabilities"] = json.loads(out["capabilities"]) if out["capabilities"] else []
        except (TypeError, ValueError, json.JSONDecodeError):
            out["capabilities"] = []
        return out

    def save_capability(
        self, host: str, model: str, strategy: str, capabilities: list[str] | None = None
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO llm_capabilities (host, model, strategy, capabilities, checked_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(host, model) DO UPDATE SET "
                "strategy = excluded.strategy, capabilities = excluded.capabilities, "
                "checked_at = excluded.checked_at",
                (host, model, strategy, json.dumps(capabilities or []), _now_iso()),
            )

    # -- llm response cache ------------------------------------------------------------ #

    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["response"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def cache_put(
        self, cache_key: str, model: str, request: dict[str, Any], response: dict[str, Any]
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO llm_cache (cache_key, model, request, response, created_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET "
                "response = excluded.response, created_at = excluded.created_at",
                (
                    cache_key,
                    model,
                    json.dumps(request, default=str),
                    json.dumps(response, default=str),
                    _now_iso(),
                ),
            )

    # -- lifecycle ----------------------------------------------------------------------- #

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class BarStore:
    """Durable OHLCV bar storage, deduplicated on `(symbol, ts, period)`.

    DuckDB's connection object is not safe for concurrent use from multiple
    threads at once, so every blocking call is funneled through a single
    `asyncio.Lock` before being dispatched (via `asyncio.to_thread`) to a
    worker thread. That serializes access rather than parallelizing it, but
    this is nowhere near a hot path — correctness matters far more here than
    squeezing out throughput.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._lock = asyncio.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                symbol VARCHAR NOT NULL,
                ts TIMESTAMP NOT NULL,
                period INTEGER NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, ts, period)
            )
            """
        )

    # ------------------------------------------------------------------ #

    def _save_bars_sync(self, bars: list[Bar]) -> int:
        rows = [
            (b.symbol, b.ts.replace(tzinfo=None), b.period, b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ]
        self._conn.executemany(
            """
            INSERT INTO bars (symbol, ts, period, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, ts, period) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """,
            rows,
        )
        return len(rows)

    async def save_bars(self, bars: list[Bar]) -> int:
        """Upsert `bars`, deduplicated on (symbol, ts, period). Returns the
        number of bars processed."""
        if not bars:
            return 0
        async with self._lock:
            return await asyncio.to_thread(self._save_bars_sync, bars)

    def _load_bars_sync(self, symbol: str, period_seconds: int, limit: int) -> list[Bar]:
        rows = self._conn.execute(
            """
            SELECT symbol, ts, period, open, high, low, close, volume
            FROM bars
            WHERE symbol = ? AND period = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (symbol, period_seconds, limit),
        ).fetchall()
        out = [
            Bar(
                symbol=r[0],
                ts=_as_utc(r[1]),
                open=r[3],
                high=r[4],
                low=r[5],
                close=r[6],
                volume=r[7],
                period=r[2],
            )
            for r in rows
        ]
        out.reverse()  # ascending by ts
        return out

    async def load_bars(self, symbol: str, period_seconds: int, limit: int) -> list[Bar]:
        """The most recent `limit` bars for `symbol`/`period_seconds`, ascending by ts."""
        async with self._lock:
            return await asyncio.to_thread(self._load_bars_sync, symbol, period_seconds, limit)

    def _row_count_sync(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM bars").fetchone()
        return int(row[0]) if row else 0

    async def row_count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._row_count_sync)

    def close(self) -> None:
        self._conn.close()


def _as_utc(ts: datetime) -> datetime:
    """DuckDB's TIMESTAMP column is naive; every Bar.ts in this system is UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts
