"""Shared runtime state, published to the dashboard.

Deliberately a plain object rather than a database read: the dashboard runs in
the same process and reads the same objects the trading loop writes, so there is
no serialization layer and no staleness between what the system believes and
what you see on screen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..domain.enums import ExecutionMode, SessionPhase
from ..domain.models import AccountSnapshot, FeaturePack
from ..domain.proposals import CycleDecision, DailyPlan, PendingApproval


@dataclass
class CycleRecord:
    """One decision cycle, for the dashboard's cycle history."""

    cycle_id: str
    started_at: datetime
    focus: list[str] = field(default_factory=list)
    market_read: str = ""
    proposals: int = 0
    approved: int = 0
    rejected: int = 0
    rejections: list[dict[str, str]] = field(default_factory=list)
    pending: int = 0
    llm_latency_ms: int = 0
    duration_ms: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "focus": self.focus,
            "market_read": self.market_read,
            "proposals": self.proposals,
            "approved": self.approved,
            "rejected": self.rejected,
            "pending": self.pending,
            "rejections": self.rejections,
            "llm_latency_ms": self.llm_latency_ms,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class AppState:
    """Everything the dashboard needs, and the loop's own bookkeeping."""

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    mode: str = "paper"
    phase: SessionPhase = SessionPhase.CLOSED
    session_description: str = "starting up"

    account: AccountSnapshot | None = None
    starting_equity: float = 0.0
    #: Highest equity seen today, for intraday drawdown.
    peak_equity: float = 0.0

    features: dict[str, FeaturePack] = field(default_factory=dict)
    focus: list[str] = field(default_factory=list)
    plan: DailyPlan | None = None
    last_decision: CycleDecision | None = None
    cycles: list[CycleRecord] = field(default_factory=list)

    #: Who pulls the trigger — the AI (auto) or a person via the dashboard
    #: (manual). Persisted across restarts by TradingEngine, not this class.
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    #: Manual-mode trade ideas waiting for a person to approve or skip.
    pending_approvals: list[PendingApproval] = field(default_factory=list)

    trades_today: int = 0
    entries_this_cycle: int = 0
    last_entry: dict[str, float] = field(default_factory=dict)
    last_close: dict[str, float] = field(default_factory=dict)

    connection: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    market_data: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)

    premarket_done: bool = False
    eod_done: bool = False
    halted_reason: str = ""
    last_error: str = ""

    # ------------------------------------------------------------------ #

    def record_cycle(self, record: CycleRecord, keep: int = 60) -> None:
        self.cycles.append(record)
        if len(self.cycles) > keep:
            del self.cycles[:-keep]

    def note_entry(self, symbol: str) -> None:
        self.last_entry[symbol] = time.monotonic()
        self.trades_today += 1
        self.entries_this_cycle += 1

    def note_close(self, symbol: str) -> None:
        self.last_close[symbol] = time.monotonic()

    def reset_for_new_session(self, equity: float) -> None:
        self.starting_equity = equity
        self.peak_equity = equity
        self.trades_today = 0
        self.entries_this_cycle = 0
        self.last_entry.clear()
        self.last_close.clear()
        self.cycles.clear()
        self.plan = None
        self.last_decision = None
        self.premarket_done = False
        self.eod_done = False
        self.halted_reason = ""
        # Yesterday's un-answered trade ideas are meaningless today. Note:
        # execution_mode itself is a standing preference and is NOT reset here.
        self.pending_approvals.clear()

    # ------------------------------------------------------------------ #

    @property
    def day_pnl(self) -> float:
        if self.account is None or self.starting_equity <= 0:
            return 0.0
        return self.account.equity - self.starting_equity

    @property
    def day_pnl_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return self.day_pnl / self.starting_equity * 100.0

    @property
    def drawdown_pct(self) -> float:
        if self.account is None or self.peak_equity <= 0:
            return 0.0
        return (self.account.equity - self.peak_equity) / self.peak_equity * 100.0

    @property
    def risk_used_pct(self) -> float:
        """Rough share of the day's risk budget consumed, for the prompt.

        Expressed as a percentage of the daily loss allowance so the model has a
        sense of how much room is left, without being handed a dollar figure.
        """
        if self.starting_equity <= 0:
            return 0.0
        loss = max(0.0, -self.day_pnl)
        allowance = self.starting_equity * 0.02
        return min(100.0, loss / allowance * 100.0) if allowance > 0 else 0.0

    def update_peak(self) -> None:
        if self.account and self.account.equity > self.peak_equity:
            self.peak_equity = self.account.equity

    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """The payload pushed to the dashboard over the WebSocket."""
        acct = self.account
        positions = []
        if acct:
            for sym, pos in acct.open_positions.items():
                positions.append(
                    {
                        "symbol": sym,
                        "quantity": pos.quantity,
                        "avg_cost": round(pos.avg_cost, 2),
                        "market_price": round(pos.market_price or pos.avg_cost, 2),
                        "unrealized_pnl": round(pos.unrealized_pnl, 2),
                        "market_value": round(pos.market_value, 2),
                    }
                )
        return {
            "ts": datetime.now(UTC).isoformat(),
            "mode": self.mode,
            "phase": self.phase.value,
            "session": self.session_description,
            "uptime_seconds": int(
                (datetime.now(UTC) - self.started_at).total_seconds()
            ),
            "account": {
                "id": acct.account_id if acct else "",
                "equity": round(acct.equity, 2) if acct else 0.0,
                "cash": round(acct.cash, 2) if acct else 0.0,
                "buying_power": round(acct.buying_power, 2) if acct else 0.0,
                "is_paper": acct.is_paper if acct else True,
                "age_seconds": round(acct.age_seconds(), 1) if acct else None,
            },
            "pnl": {
                "day": round(self.day_pnl, 2),
                "day_pct": round(self.day_pnl_pct, 3),
                "drawdown_pct": round(self.drawdown_pct, 3),
                "starting_equity": round(self.starting_equity, 2),
            },
            "positions": positions,
            "focus": self.focus,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "last_decision": (
                self.last_decision.model_dump(mode="json") if self.last_decision else None
            ),
            "cycles": [c.as_dict() for c in self.cycles[-12:]],
            "execution_mode": self.execution_mode.value,
            "pending_approvals": [p.model_dump(mode="json") for p in self.pending_approvals],
            "connection": self.connection,
            "llm": self.llm,
            "market_data": self.market_data,
            "risk": self.risk,
            "reconciliation": self.reconciliation,
            "trades_today": self.trades_today,
            "halted_reason": self.halted_reason,
            "last_error": self.last_error,
        }
