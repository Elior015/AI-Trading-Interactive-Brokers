"""The broker boundary.

Everything above this layer depends only on `BrokerPort`. Two payoffs:

1. `ib_async` is maintained by one person and its predecessor was archived
   after the original author died. This protocol is what lets us replace it.
2. The test suite drives a `FakeBroker` implementing this protocol, so tests
   need neither a running Gateway nor a network — which is the only way tests
   actually get run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ..domain.models import AccountSnapshot, Bar, Quote


@dataclass
class BrokerOrderSpec:
    """A single order leg, expressed without reference to any broker library."""

    symbol: str
    action: str  # "BUY" | "SELL"
    quantity: float
    order_type: str  # "LMT" | "MKT" | "STP" | "STP LMT"
    limit_price: float | None = None
    stop_price: float | None = None
    order_ref: str = ""
    oca_group: str | None = None
    parent_ref: str | None = None
    transmit: bool = True
    tif: str = "DAY"
    outside_rth: bool = False


@dataclass
class BrokerOrderStatus:
    """What the broker says about an order."""

    order_ref: str
    order_id: int
    #: IBKR-assigned and stable across restarts and clientIds. `order_id` is not.
    perm_id: int | None
    status: str
    filled: float
    remaining: float
    avg_fill_price: float | None
    symbol: str = ""
    action: str = ""


@dataclass
class BrokerExecution:
    """A fill as reported by the broker."""

    exec_id: str
    order_ref: str
    symbol: str
    action: str
    quantity: float
    price: float
    ts: datetime
    commission: float = 0.0
    perm_id: int | None = None


@dataclass
class ScannerHit:
    symbol: str
    rank: int
    scan_code: str


@dataclass
class BrokerEvents:
    """Callbacks the broker layer invokes. Set by whoever owns the connection."""

    on_order_status: Any = None
    on_execution: Any = None
    on_disconnect: Any = None
    on_error: Any = None
    on_bar: Any = None


@runtime_checkable
class BrokerPort(Protocol):
    """The only broker surface the rest of the system may use."""

    events: BrokerEvents

    # -- lifecycle -------------------------------------------------------- #

    async def connect(self, host: str, port: int, client_id: int, timeout: float) -> str:
        """Connect and return the managed account id. Raises on failure."""
        ...

    async def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    async def is_authenticated(self) -> bool:
        """True when the session is genuinely usable.

        A connected socket is not enough: after IBKR's weekly token reset the
        socket connects but no account data flows. This must actually exercise
        the session, not just check the socket.
        """
        ...

    # -- reference data ---------------------------------------------------- #

    async def qualify(self, symbols: list[str]) -> dict[str, int]:
        """Resolve symbols to IBKR conIds. Unresolvable symbols are omitted."""
        ...

    # -- market data ------------------------------------------------------- #

    async def subscribe_quote(self, symbol: str) -> None: ...
    async def unsubscribe_quote(self, symbol: str) -> None: ...
    def get_quote(self, symbol: str) -> Quote | None: ...
    async def subscribe_bars(self, symbol: str) -> None: ...
    async def unsubscribe_bars(self, symbol: str) -> None: ...

    def clear_subscription_cache(self) -> None:
        """Drop any cached "already subscribed" state after a reconnect.

        IBKR does not carry market-data subscriptions across a new session.
        Without this, subscribe_quote()/subscribe_bars() would keep skipping
        every symbol they believe is still subscribed, leaving feeds silently
        dead after every reconnect.
        """
        ...

    async def historical_bars(
        self, symbol: str, duration: str, bar_size: str, what_to_show: str = "TRADES"
    ) -> list[Bar]:
        """Fetch history. Callers must hold a pacer token before calling."""
        ...

    async def scan(self, scan_code: str, rows: int, min_price: float, min_volume: int) -> list[ScannerHit]:
        """Server-side scanner. Costs no market-data lines."""
        ...

    # -- account ----------------------------------------------------------- #

    async def account_snapshot(self) -> AccountSnapshot: ...

    # -- orders ------------------------------------------------------------ #

    async def place_bracket(
        self, entry: BrokerOrderSpec, stop: BrokerOrderSpec, target: BrokerOrderSpec
    ) -> list[BrokerOrderStatus]:
        """Place parent + stop + target atomically as one native IBKR bracket.

        The protective legs must live at the broker, not in this process: there
        are guaranteed windows (daily Gateway restart, the weekly 2FA outage, a
        crash) where we cannot reach IBKR, and a position must stay protected
        through all of them.
        """
        ...

    async def place_single(self, spec: BrokerOrderSpec) -> BrokerOrderStatus: ...
    async def cancel_order(self, order_id: int) -> None: ...
    async def cancel_all(self) -> None: ...
    async def open_orders(self) -> list[BrokerOrderStatus]: ...
    async def completed_orders(self) -> list[BrokerOrderStatus]: ...
    async def executions(self) -> list[BrokerExecution]: ...
    async def modify_stop(self, order_id: int, new_stop: float) -> None: ...


@dataclass
class BrokerStats:
    """Counters surfaced on the dashboard."""

    messages_sent: int = 0
    historical_requests: int = 0
    pacing_violations: int = 0
    reconnects: int = 0
    errors: dict[str, int] = field(default_factory=dict)


# IBKR error-code classification. Reacting correctly to these is most of what
# separates a bot that survives a session from one that gets disconnected.
PACING_ERRORS = {100, 162, 165, 366, 420}
NO_SUBSCRIPTION_ERRORS = {354, 10089, 10090, 10091, 10167, 10197}
CONNECTIVITY_ERRORS = {502, 504, 1100, 1101, 1102, 2103, 2105, 2157}
ORDER_REJECT_ERRORS = {201, 202, 203, 399, 10147, 10148}
#: Informational codes that are not failures and must not raise alarms.
BENIGN_ERRORS = {2104, 2106, 2107, 2108, 2119, 2158, 399}


def classify_error(code: int) -> str:
    if code in PACING_ERRORS:
        return "PACING"
    if code in NO_SUBSCRIPTION_ERRORS:
        return "NO_SUBSCRIPTION"
    if code in CONNECTIVITY_ERRORS:
        return "CONNECTIVITY"
    if code in ORDER_REJECT_ERRORS:
        return "ORDER_REJECT"
    if code in BENIGN_ERRORS:
        return "BENIGN"
    return "OTHER"
