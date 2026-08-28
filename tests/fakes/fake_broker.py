"""An in-process broker double implementing `BrokerPort`.

This is why the port/adapter split earns its keep: the whole test suite runs
with no IB Gateway and no network, which is the only way tests actually get run.

It simulates the failure modes that matter: partial fills, rejections,
disconnects mid-order, duplicate executions, and — importantly — the
connected-but-unauthenticated state that IBKR's weekly token reset produces.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime

from aitrader.broker.port import (
    BrokerEvents,
    BrokerExecution,
    BrokerOrderSpec,
    BrokerOrderStatus,
    ScannerHit,
)
from aitrader.domain.models import AccountSnapshot, Bar, Position, Quote


@dataclass
class FakeBroker:
    """Configurable broker double."""

    account_id: str = "DU1234567"
    equity: float = 100_000.0
    cash: float = 100_000.0
    buying_power: float = 200_000.0

    connected: bool = False
    authenticated: bool = True
    #: Set True to simulate the weekly IBKR token reset: socket up, session dead.
    simulate_unauthenticated: bool = False
    #: Raise on the next order placement.
    fail_next_order: bool = False
    #: Fill entries immediately rather than leaving them working.
    auto_fill: bool = True

    events: BrokerEvents = field(default_factory=BrokerEvents)
    quotes: dict[str, Quote] = field(default_factory=dict)
    bars: dict[str, list[Bar]] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    scanner_results: dict[str, list[str]] = field(default_factory=dict)

    placed: list[BrokerOrderSpec] = field(default_factory=list)
    statuses: dict[str, BrokerOrderStatus] = field(default_factory=dict)
    cancelled: list[int] = field(default_factory=list)
    executions_list: list[BrokerExecution] = field(default_factory=list)
    global_cancels: int = 0
    historical_calls: int = 0

    _ids: itertools.count = field(default_factory=lambda: itertools.count(1000))
    _exec_ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    # -- lifecycle -------------------------------------------------------- #

    async def connect(self, host: str, port: int, client_id: int, timeout: float = 20.0) -> str:
        self.connected = True
        return self.account_id

    async def disconnect(self) -> None:
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def is_authenticated(self) -> bool:
        if self.simulate_unauthenticated:
            return False
        return self.connected and self.authenticated

    def simulate_disconnect(self) -> None:
        self.connected = False
        if self.events.on_disconnect:
            self.events.on_disconnect()

    # -- reference data ---------------------------------------------------- #

    async def qualify(self, symbols: list[str]) -> dict[str, int]:
        return {s: 1000 + i for i, s in enumerate(symbols)}

    # -- market data ------------------------------------------------------- #

    async def subscribe_quote(self, symbol: str) -> None:
        self.quotes.setdefault(
            symbol, Quote(symbol=symbol, bid=99.95, ask=100.05, last=100.0, volume=1_000_000)
        )

    async def unsubscribe_quote(self, symbol: str) -> None:
        self.quotes.pop(symbol, None)

    def get_quote(self, symbol: str) -> Quote | None:
        return self.quotes.get(symbol)

    def set_quote(self, symbol: str, **kwargs) -> Quote:
        q = Quote(symbol=symbol, **kwargs)
        self.quotes[symbol] = q
        return q

    async def subscribe_bars(self, symbol: str) -> None:
        self.bars.setdefault(symbol, [])

    async def unsubscribe_bars(self, symbol: str) -> None:
        self.bars.pop(symbol, None)

    async def historical_bars(
        self, symbol: str, duration: str = "2 D", bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
    ) -> list[Bar]:
        self.historical_calls += 1
        return self.bars.get(symbol, [])

    async def scan(
        self, scan_code: str, rows: int = 50, min_price: float = 3.0, min_volume: int = 500_000
    ) -> list[ScannerHit]:
        symbols = self.scanner_results.get(scan_code, [])
        return [
            ScannerHit(symbol=s, rank=i, scan_code=scan_code)
            for i, s in enumerate(symbols[:rows])
        ]

    # -- account ------------------------------------------------------------ #

    async def account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id=self.account_id,
            equity=self.equity,
            cash=self.cash,
            buying_power=self.buying_power,
            positions=dict(self.positions),
            unrealized_pnl=sum(p.unrealized_pnl for p in self.positions.values()),
            as_of=datetime.now(UTC),
        )

    # -- orders -------------------------------------------------------------- #

    def _status(self, spec: BrokerOrderSpec, status: str, filled: float = 0.0) -> BrokerOrderStatus:
        oid = next(self._ids)
        st = BrokerOrderStatus(
            order_ref=spec.order_ref,
            order_id=oid,
            perm_id=oid * 10,
            status=status,
            filled=filled,
            remaining=spec.quantity - filled,
            avg_fill_price=spec.limit_price if filled else None,
            symbol=spec.symbol,
            action=spec.action,
        )
        self.statuses[spec.order_ref] = st
        return st

    async def place_bracket(
        self, entry: BrokerOrderSpec, stop: BrokerOrderSpec, target: BrokerOrderSpec
    ) -> list[BrokerOrderStatus]:
        if self.fail_next_order:
            self.fail_next_order = False
            raise RuntimeError("simulated order rejection")

        self.placed.extend([entry, stop, target])

        if self.auto_fill:
            entry_status = self._status(entry, "Filled", entry.quantity)
            qty = entry.quantity if entry.action == "BUY" else -entry.quantity
            existing = self.positions.get(entry.symbol)
            new_qty = (existing.quantity if existing else 0) + qty
            if new_qty == 0:
                self.positions.pop(entry.symbol, None)
            else:
                self.positions[entry.symbol] = Position(
                    symbol=entry.symbol,
                    quantity=new_qty,
                    avg_cost=entry.limit_price or 100.0,
                    market_price=entry.limit_price or 100.0,
                )
            execution = BrokerExecution(
                exec_id=f"exec-{next(self._exec_ids)}",
                order_ref=entry.order_ref,
                symbol=entry.symbol,
                action=entry.action,
                quantity=entry.quantity,
                price=entry.limit_price or 100.0,
                ts=datetime.now(UTC),
            )
            self.executions_list.append(execution)
            if self.events.on_execution:
                self.events.on_execution(execution)
        else:
            entry_status = self._status(entry, "Submitted")

        return [entry_status, self._status(target, "PreSubmitted"), self._status(stop, "PreSubmitted")]

    async def place_single(self, spec: BrokerOrderSpec) -> BrokerOrderStatus:
        if self.fail_next_order:
            self.fail_next_order = False
            raise RuntimeError("simulated order rejection")
        self.placed.append(spec)
        return self._status(spec, "Submitted")

    async def cancel_order(self, order_id: int) -> None:
        self.cancelled.append(order_id)
        for ref, st in self.statuses.items():
            if st.order_id == order_id:
                self.statuses[ref] = BrokerOrderStatus(
                    order_ref=ref, order_id=order_id, perm_id=st.perm_id,
                    status="Cancelled", filled=st.filled, remaining=0,
                    avg_fill_price=st.avg_fill_price, symbol=st.symbol, action=st.action,
                )
                break

    async def cancel_all(self) -> None:
        self.global_cancels += 1

    async def open_orders(self) -> list[BrokerOrderStatus]:
        return [
            s for s in self.statuses.values()
            if s.status in ("Submitted", "PreSubmitted", "PendingSubmit")
        ]

    async def completed_orders(self) -> list[BrokerOrderStatus]:
        return [s for s in self.statuses.values() if s.status in ("Filled", "Cancelled")]

    async def executions(self) -> list[BrokerExecution]:
        return list(self.executions_list)

    async def modify_stop(self, order_id: int, new_stop: float) -> None:
        pass


def make_bars(
    symbol: str = "TEST",
    count: int = 60,
    start_price: float = 100.0,
    trend: float = 0.0,
    period: int = 300,
) -> list[Bar]:
    """Generate a deterministic bar series for tests."""
    bars: list[Bar] = []
    price = start_price
    base = datetime(2026, 8, 24, 14, 0, tzinfo=UTC)
    for i in range(count):
        price += trend
        wiggle = 0.5 if i % 2 == 0 else -0.3
        close = price + wiggle
        bars.append(
            Bar(
                symbol=symbol,
                ts=base.replace(minute=(base.minute + i * 5) % 60, hour=base.hour + (i * 5) // 60),
                open=price,
                high=max(price, close) + 0.4,
                low=min(price, close) - 0.4,
                close=close,
                volume=1_000_000 + i * 1000,
                period=period,
            )
        )
    return bars
