"""Core value objects.

Everything the LLM ever sees is built from these. They are deliberately plain and
fully validated: a number that reaches a trade decision should never have an
unexamined provenance.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import ConnectionState, OrderState


def utcnow() -> datetime:
    return datetime.now(UTC)


class Bar(BaseModel):
    """A single OHLCV bar."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    #: Bar width in seconds (5, 60, 300 ...).
    period: int


class Quote(BaseModel):
    """Latest top-of-book snapshot for a symbol."""

    symbol: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: float | None = None
    halted: bool = False
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float | None:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> float | None:
        s, m = self.spread, self.mid
        if s is None or not m:
            return None
        return s / m

    def age_seconds(self, now: datetime | None = None) -> float:
        return ((now or utcnow()) - self.updated_at).total_seconds()

    @property
    def is_usable(self) -> bool:
        """A quote we would be willing to price an order from."""
        return not self.halted and self.mid is not None and self.mid > 0


class Position(BaseModel):
    """A position as reported by IBKR. Never inferred locally."""

    symbol: str
    quantity: float
    avg_cost: float
    market_price: float | None = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def market_value(self) -> float:
        return self.quantity * (self.market_price or self.avg_cost)


class AccountSnapshot(BaseModel):
    """Ground truth from IBKR, refreshed every cycle.

    `as_of` is checked by the risk gate: acting on a stale snapshot is the
    failure mode that TradeTrap showed to be catastrophic, so a snapshot older
    than the configured tolerance blocks all trading.
    """

    account_id: str
    equity: float
    cash: float
    buying_power: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    as_of: datetime = Field(default_factory=utcnow)

    def age_seconds(self, now: datetime | None = None) -> float:
        return ((now or utcnow()) - self.as_of).total_seconds()

    @property
    def is_paper(self) -> bool:
        """IBKR paper accounts are prefixed 'DU' (and 'DF' for advisor paper)."""
        return self.account_id.upper().startswith(("DU", "DF"))

    @property
    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if not p.is_flat}

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class FeaturePack(BaseModel):
    """The complete numeric view of one symbol handed to the model.

    Everything here is computed deterministically from bars. The model is never
    asked to do arithmetic on prices; it only interprets these values.
    """

    symbol: str
    price: float
    #: Percent change from the previous session close.
    change_pct: float = 0.0
    gap_pct: float = 0.0
    #: Relative volume vs the 20-day average at this time of day.
    rvol: float = 1.0
    atr: float = 0.0
    atr_pct: float = 0.0
    vwap: float | None = None
    #: Signed distance from VWAP, in ATRs.
    vwap_distance_atr: float = 0.0
    rsi: float = 50.0
    ema_fast: float | None = None
    ema_slow: float | None = None
    #: +1 fast above slow, -1 below, 0 unknown.
    ema_trend: int = 0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_upper: float | None = None
    bb_lower: float | None = None
    #: Position within the opening range: 0 = low, 1 = high, >1 breakout.
    opening_range_position: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    avg_volume: float = 0.0
    spread_pct: float | None = None
    #: Deterministic pre-LLM rank score; higher is more interesting.
    rank_score: float = 0.0
    computed_at: datetime = Field(default_factory=utcnow)

    def to_row(self) -> dict[str, object]:
        """Compact representation for the prompt table.

        Rounded aggressively: extra decimal places cost tokens and add no signal.
        """

        def r(v: float | None, n: int = 2) -> float | None:
            return None if v is None else round(v, n)

        return {
            "sym": self.symbol,
            "px": r(self.price),
            "chg%": r(self.change_pct),
            "gap%": r(self.gap_pct),
            "rvol": r(self.rvol),
            "atr%": r(self.atr_pct),
            "vwap_d_atr": r(self.vwap_distance_atr),
            "rsi": r(self.rsi, 1),
            "ema_trend": self.ema_trend,
            "macd_hist": r(self.macd - self.macd_signal, 3),
            "or_pos": r(self.opening_range_position),
            "spread%": r((self.spread_pct or 0) * 100, 3),
        }


class ConnectionHealth(BaseModel):
    """Broker connection status, surfaced prominently on the dashboard."""

    state: ConnectionState = ConnectionState.DISCONNECTED
    last_connected_at: datetime | None = None
    last_error: str | None = None
    reconnect_attempts: int = 0
    #: Set when IBKR requires the weekly interactive 2FA re-login.
    needs_manual_2fa: bool = False

    @property
    def is_tradeable(self) -> bool:
        return self.state == ConnectionState.CONNECTED and not self.needs_manual_2fa


class ManagedOrder(BaseModel):
    """Our record of an order. Reconciled against IBKR on every connect."""

    order_ref: str
    symbol: str
    action: str
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    state: OrderState = OrderState.PENDING
    ib_order_id: int | None = None
    parent_ref: str | None = None
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    cycle_id: str | None = None
    rationale: str | None = None

    @property
    def is_working(self) -> bool:
        return self.state.is_working


class Fill(BaseModel):
    symbol: str
    action: str
    quantity: float
    price: float
    commission: float = 0.0
    ts: datetime = Field(default_factory=utcnow)
    order_ref: str | None = None
