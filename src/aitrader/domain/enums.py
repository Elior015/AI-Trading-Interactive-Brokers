"""Enumerations shared across the system."""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    """What the model proposes doing with a symbol."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class SessionPhase(str, Enum):
    """Where we are in the US equity trading day."""

    CLOSED = "CLOSED"
    PREMARKET = "PREMARKET"
    OPENING_RANGE = "OPENING_RANGE"
    RTH = "RTH"
    CLOSING = "CLOSING"
    AFTER_HOURS = "AFTER_HOURS"

    @property
    def is_tradeable(self) -> bool:
        return self in (SessionPhase.OPENING_RANGE, SessionPhase.RTH, SessionPhase.CLOSING)


class OrderState(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED)

    @property
    def is_working(self) -> bool:
        return self in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL)


class RejectReason(str, Enum):
    """Every way the risk gate can say no.

    These are surfaced verbatim on the dashboard's /rejections page, so if the
    system is not trading you can see exactly which rule stopped it.
    """

    KILL_SWITCH = "KILL_SWITCH"
    TRADING_MODE_INTERLOCK = "TRADING_MODE_INTERLOCK"
    MARKET_CLOSED = "MARKET_CLOSED"
    TOO_LATE_IN_SESSION = "TOO_LATE_IN_SESSION"
    STALE_ACCOUNT_STATE = "STALE_ACCOUNT_STATE"
    INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"
    PER_TRADE_RISK_EXCEEDED = "PER_TRADE_RISK_EXCEEDED"
    POSITION_NOTIONAL_EXCEEDED = "POSITION_NOTIONAL_EXCEEDED"
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    MAX_ENTRIES_PER_CYCLE = "MAX_ENTRIES_PER_CYCLE"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DAILY_TRADE_CAP = "DAILY_TRADE_CAP"
    PRICE_COLLAR = "PRICE_COLLAR"
    ILLIQUID = "ILLIQUID"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    HALTED = "HALTED"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    COOLDOWN = "COOLDOWN"
    FLIP_FLOP_GUARD = "FLIP_FLOP_GUARD"
    ZERO_QUANTITY = "ZERO_QUANTITY"
    NO_MARKET_DATA = "NO_MARKET_DATA"
    RISK_OFFICER_VETO = "RISK_OFFICER_VETO"
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class KillSwitchAction(str, Enum):
    HALT_NEW_ENTRIES = "halt_new_entries"
    FLATTEN_ALL = "flatten_all"


class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    #: Socket is up but IBKR has invalidated the session (the weekly 2FA event).
    UNAUTHENTICATED = "UNAUTHENTICATED"
