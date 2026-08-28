"""Absolute ceilings that configuration cannot loosen.

`RiskConfig` clamps itself to these at load time. The point is that a
fat-fingered YAML edit — `max_position_notional_pct: 10.0` instead of `0.10` —
gets clamped with a loud warning rather than becoming a ten-times-equity order.

Config can always make a limit *tighter*. It can never make one looser.
"""

from __future__ import annotations

#: Never risk more than this fraction of equity on one trade, whatever config says.
ABS_MAX_RISK_PER_TRADE_PCT = 0.02

#: Never let one position exceed this fraction of equity.
ABS_MAX_POSITION_NOTIONAL_PCT = 0.25

#: Never hold more than this many positions at once.
ABS_MAX_CONCURRENT_POSITIONS = 15

#: Never open more than this many new positions in a single decision cycle.
ABS_MAX_NEW_ENTRIES_PER_CYCLE = 5

#: Never let the daily loss limit be set looser than this.
ABS_MAX_DAILY_LOSS_PCT = 0.10

#: Never place more than this many orders in a day.
ABS_MAX_TRADES_PER_DAY = 200

#: The runaway-loop circuit breaker. This is the single most important limit in
#: the file: a bug that loops placing orders is stopped here regardless of every
#: other check passing.
ABS_MAX_ORDERS_PER_MINUTE = 10

#: Never accept a limit price further than this from the last trade.
ABS_MAX_PRICE_COLLAR_PCT = 0.05

#: Never trade something with a spread wider than this.
ABS_MAX_SPREAD_PCT = 0.02

#: Never act on a quote older than this.
ABS_MAX_QUOTE_AGE_SECONDS = 60.0

#: Never act on an account snapshot older than this.
ABS_MAX_ACCOUNT_AGE_SECONDS = 300.0

#: Never act on an LLM decision older than this.
ABS_MAX_DECISION_AGE_SECONDS = 180.0

#: IBKR closes the socket above ~50 messages/sec. Stay meaningfully under it.
ABS_MAX_BROKER_MESSAGES_PER_SECOND = 45

#: IBKR allows 60 historical requests per 10 minutes. Leave a safety margin.
ABS_MAX_HISTORICAL_REQUESTS_PER_10MIN = 50

#: IBKR's default market-data line allowance is 100.
ABS_MAX_MARKET_DATA_LINES = 95

#: IBKR caps scanner results at 50 rows per scan code and 10 active API scans.
ABS_MAX_SCANNER_ROWS = 50
ABS_MAX_ACTIVE_SCANNERS = 10


def clamp(value: float, ceiling: float, name: str, warnings: list[str]) -> float:
    """Return `value` capped at `ceiling`, recording a warning if it was capped."""
    if value > ceiling:
        warnings.append(
            f"{name}={value} exceeds the hard limit {ceiling}; clamped to {ceiling}. "
            "Check your config — this is usually a decimal-point mistake."
        )
        return ceiling
    return value


def clamp_int(value: int, ceiling: int, name: str, warnings: list[str]) -> int:
    if value > ceiling:
        warnings.append(
            f"{name}={value} exceeds the hard limit {ceiling}; clamped to {ceiling}."
        )
        return ceiling
    return value
