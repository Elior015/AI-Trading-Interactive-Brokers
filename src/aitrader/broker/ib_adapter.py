"""The one and only module that talks to `ib_async`.

An architecture test (`tests/test_architecture.py`) fails CI if `placeOrder` or
`ib_async` appear anywhere else in `src/`. That test, not convention, is what
keeps this boundary intact over time.

Every outbound call passes through a global message pacer first: IBKR closes the
socket above roughly 50 messages per second, and a closed socket mid-session is
far worse than a slow one.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ib_async import (
    IB,
    Contract,
    ExecutionFilter,
    LimitOrder,
    Order,
    ScannerSubscription,
    Stock,
    StopOrder,
)

from ..data.ratelimit import TokenBucket
from ..domain.models import AccountSnapshot, Bar, Position, Quote, utcnow
from ..logging_setup import get_logger
from .port import (
    BrokerEvents,
    BrokerExecution,
    BrokerOrderSpec,
    BrokerOrderStatus,
    BrokerStats,
    ScannerHit,
    classify_error,
)

log = get_logger(__name__)

# Mirrors ib_async's own default tag list (ib.py: reqAccountSummaryAsync). Kept
# local because we bypass that helper below — see _account_summary_once.
_ACCOUNT_SUMMARY_TAGS = (
    "AccountType,NetLiquidation,TotalCashValue,SettledCash,"
    "AccruedCash,BuyingPower,EquityWithLoanValue,"
    "PreviousDayEquityWithLoanValue,GrossPositionValue,RegTEquity,"
    "RegTMargin,SMA,InitMarginReq,MaintMarginReq,AvailableFunds,"
    "ExcessLiquidity,Cushion,FullInitMarginReq,FullMaintMarginReq,"
    "FullAvailableFunds,FullExcessLiquidity,LookAheadNextChange,"
    "LookAheadInitMarginReq,LookAheadMaintMarginReq,"
    "LookAheadAvailableFunds,LookAheadExcessLiquidity,"
    "HighestSeverity,DayTradesRemaining,DayTradesRemainingT+1,"
    "DayTradesRemainingT+2,DayTradesRemainingT+3,"
    "DayTradesRemainingT+4,Leverage,$LEDGER:ALL"
)


def _f(value: Any, default: float | None = None) -> float | None:
    """IBKR sends NaN and -1 as 'no value'. Normalize both to None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v < 0:  # NaN or the -1 sentinel
        return default
    return v


class IbAsyncBroker:
    """`BrokerPort` implementation over `ib_async`."""

    def __init__(self, messages_per_second: int = 45) -> None:
        self.ib = IB()
        self.events = BrokerEvents()
        self.stats = BrokerStats()
        self._pacer = TokenBucket(rate=messages_per_second, capacity=messages_per_second)
        self._con_ids: dict[str, int] = {}
        # Values are usually Stock, but a qualified contract from IBKR is
        # typed as the base Contract, so the dict is typed to match.
        self._contracts: dict[str, Contract] = {}
        self._tickers: dict[str, Any] = {}
        self._bar_subs: dict[str, Any] = {}
        self._account: str = ""
        self._ref_by_order_id: dict[int, str] = {}
        # Serializes account-summary requests: the watchdog (every 10s) and
        # account-sync (every 30s) loops both call into
        # `_account_summary_once` independently, and IBKR allows only one
        # open account-summary subscription per client — two in flight at
        # once fails every later request with error 322 until the next
        # Gateway restart. See `_account_summary_once`.
        self._account_summary_lock = asyncio.Lock()
        # IBKR reuses error 162 for genuine historical-data pacing violations
        # *and* for unrelated scanner rejections (disabled filter, entitlement
        # cancellation). `_on_error` treats every 162 as PACING and throttles
        # the shared historical-data pacer, so a permanently broken scanner
        # would otherwise punish real historical/quote requests forever. This
        # tracks in-flight scanner reqIds so their errors can be excluded.
        self._scanner_req_ids: set[int] = set()
        self._wire_events()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def _wire_events(self) -> None:
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        self.ib.barUpdateEvent += self._on_bar_update

    async def connect(self, host: str, port: int, client_id: int, timeout: float = 20.0) -> str:
        log.info("broker_connecting", host=host, port=port, client_id=client_id)
        await self.ib.connectAsync(host, port, clientId=client_id, timeout=timeout)

        accounts = self.ib.managedAccounts()
        if not accounts:
            raise ConnectionError(
                "Connected to IB Gateway but it reported no managed accounts. "
                "This usually means the session is not authenticated."
            )
        self._account = accounts[0]

        # Real-time data only. Delayed US equity data is not offered to IB LLC
        # clients, so silently degrading to it is not an option — we would
        # rather fail loudly than trade on data we cannot get.
        self.ib.reqMarketDataType(1)

        log.info("broker_connected", account=self._account, accounts=accounts)
        return self._account

    async def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception as exc:  # noqa: BLE001 - shutdown must never raise
            log.warning("broker_disconnect_error", error=str(exc))

    @property
    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def clear_subscription_cache(self) -> None:
        """Drop cached ticker/bar handles after a reconnect.

        `self.ib` (and its wrapper) persist across a reconnect, but IBKR's
        subscriptions do not — a fresh session starts with none. Without
        this, subscribe_quote()/subscribe_bars() below see the symbol
        already in _tickers/_bar_subs and return without re-issuing the
        request, leaving every feed silently dead.
        """
        self._tickers.clear()
        self._bar_subs.clear()

    async def is_authenticated(self) -> bool:
        """Prove the session actually works, rather than trusting the socket.

        After IBKR's weekly token reset the socket connects happily but no
        account data flows. Requiring a real account summary within a short
        timeout is how we detect that state instead of trading blind.
        """
        if not self.ib.isConnected():
            return False
        try:
            if not self.ib.managedAccounts():
                return False
            summary = await self._account_summary_once()
            return bool(summary)
        except TimeoutError:
            log.warning("broker_auth_probe_timeout")
            return False
        except Exception as exc:  # noqa: BLE001
            log.warning("broker_auth_probe_failed", error=str(exc))
            return False

    async def _account_summary_once(self, timeout: float = 15.0) -> list[Any]:
        """Fetch account summary via a fresh, self-cancelling subscription.

        `ib.accountSummaryAsync()` subscribes once and caches forever, with no
        way to clean up if the caller times out mid-flight (a real risk here:
        this runs on every watchdog tick during a connectivity blip). IBKR
        allows only one active account-summary subscription per client, so an
        orphaned one then makes every later request fail with error 322
        ("Maximum number of account summary requests exceeded") until the
        next full Gateway restart. Using our own reqId and cancelling it in
        `finally`, on every path including timeout, means a blip can never
        leak a subscription.
        """
        async with self._account_summary_lock:
            req_id = self.ib.client.getReqId()
            future = self.ib.wrapper.startReq(req_id)
            self.ib.client.reqAccountSummary(req_id, "All", _ACCOUNT_SUMMARY_TAGS)
            try:
                async with asyncio.timeout(timeout):
                    await future
            finally:
                self.ib.client.cancelAccountSummary(req_id)
                self.ib.wrapper._futures.pop(req_id, None)
                self.ib.wrapper._results.pop(req_id, None)
            if self._account:
                return [v for v in self.ib.wrapper.acctSummary.values() if v.account == self._account]
            return list(self.ib.wrapper.acctSummary.values())

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def _on_order_status(self, trade: Any) -> None:
        try:
            status = self._to_status(trade)
            if status.order_ref:
                self._ref_by_order_id[status.order_id] = status.order_ref
            if self.events.on_order_status:
                self.events.on_order_status(status)
        except Exception as exc:
            log.exception("order_status_handler_failed", error=str(exc))

    def _on_exec_details(self, trade: Any, fill: Any) -> None:
        try:
            ex = fill.execution
            ref = getattr(trade.order, "orderRef", "") or self._ref_by_order_id.get(
                getattr(ex, "orderId", -1), ""
            )
            commission = 0.0
            report = getattr(fill, "commissionReport", None)
            if report is not None:
                commission = _f(getattr(report, "commission", 0.0), 0.0) or 0.0
            ts = getattr(ex, "time", None) or utcnow()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            execution = BrokerExecution(
                exec_id=ex.execId,
                order_ref=ref,
                symbol=fill.contract.symbol,
                action="BUY" if ex.side.upper().startswith("B") else "SELL",
                quantity=float(ex.shares),
                price=float(ex.price),
                ts=ts,
                commission=commission,
                perm_id=getattr(ex, "permId", None),
            )
            if self.events.on_execution:
                self.events.on_execution(execution)
        except Exception as exc:
            log.exception("exec_details_handler_failed", error=str(exc))

    def _on_disconnected(self) -> None:
        log.warning("broker_disconnected")
        if self.events.on_disconnect:
            self.events.on_disconnect()

    def _on_error(self, req_id: int, code: int, message: str, contract: Any = None) -> None:
        kind = classify_error(code)
        if kind == "PACING" and (
            req_id in self._scanner_req_ids or "scanner subscription cancelled" in message.lower()
        ):
            # IBKR overloads this code for scanner rejections unrelated to
            # pacing, including its own async echo of the cancelScannerSubscription
            # call `scan()` makes in its `finally` on every successful pass —
            # by the time that echo arrives, the reqId has usually already been
            # discarded from `_scanner_req_ids`, hence the message match too.
            # Letting either through as PACING would throttle the shared
            # historical-data pacer over routine, expected scanner noise.
            kind = "BENIGN"
        if kind == "BENIGN":
            return
        self.stats.errors[str(code)] = self.stats.errors.get(str(code), 0) + 1
        if kind == "PACING":
            self.stats.pacing_violations += 1
            log.error("ibkr_pacing_violation", code=code, message=message)
        elif kind == "NO_SUBSCRIPTION":
            # This is fatal for trading the affected symbol: there is no delayed
            # fallback for US equities on an IB LLC account.
            log.error("ibkr_market_data_not_subscribed", code=code, message=message)
        elif kind == "CONNECTIVITY":
            log.warning("ibkr_connectivity", code=code, message=message)
        else:
            log.warning("ibkr_error", code=code, message=message, kind=kind)
        if self.events.on_error:
            self.events.on_error(code, message, kind)

    def _on_bar_update(self, bars: Any, has_new_bar: bool) -> None:
        if not has_new_bar or not bars:
            return
        try:
            last = bars[-1]
            symbol = bars.contract.symbol
            ts = last.time
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            bar = Bar(
                symbol=symbol,
                ts=ts,
                open=float(last.open),
                high=float(last.high),
                low=float(last.low),
                close=float(last.close),
                volume=float(last.volume or 0),
                period=5,
            )
            if self.events.on_bar:
                self.events.on_bar(bar)
        except Exception as exc:
            log.exception("bar_update_handler_failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # reference data
    # ------------------------------------------------------------------ #

    def _stock(self, symbol: str) -> Contract:
        """Return a usable contract for `symbol`, creating an unqualified Stock
        the first time it is seen. Once `qualify()` has run for the symbol, this
        returns the qualified contract instead."""
        if symbol not in self._contracts:
            self._contracts[symbol] = Stock(symbol, "SMART", "USD")
        return self._contracts[symbol]

    async def qualify(self, symbols: list[str]) -> dict[str, int]:
        """Resolve symbols to conIds, caching results.

        Qualification is itself a paced request, so the cache matters: a 300
        symbol universe re-qualified every cycle would eat the message budget.
        """
        pending = [s for s in symbols if s not in self._con_ids]
        if not pending:
            return {s: self._con_ids[s] for s in symbols if s in self._con_ids}

        for chunk_start in range(0, len(pending), 50):
            chunk = pending[chunk_start : chunk_start + 50]
            await self._pacer.acquire(len(chunk))
            contracts = [self._stock(s) for s in chunk]
            try:
                qualified = await self.ib.qualifyContractsAsync(*contracts)
            except Exception as exc:  # noqa: BLE001
                log.warning("qualify_failed", symbols=chunk, error=str(exc))
                continue
            for c in qualified:
                # ib_async's return type covers ambiguous-match (nested list)
                # and failed-qualification (None) entries; only a resolved
                # Contract carries a usable conId.
                if isinstance(c, Contract) and c.conId:
                    self._con_ids[c.symbol] = c.conId
                    self._contracts[c.symbol] = c

        missing = [s for s in symbols if s not in self._con_ids]
        if missing:
            log.warning("symbols_unqualified", symbols=missing[:20], count=len(missing))
        return {s: self._con_ids[s] for s in symbols if s in self._con_ids}

    # ------------------------------------------------------------------ #
    # market data
    # ------------------------------------------------------------------ #

    async def subscribe_quote(self, symbol: str) -> None:
        if symbol in self._tickers:
            return
        await self._pacer.acquire()
        contract = self._contracts.get(symbol) or self._stock(symbol)
        # Generic tick 233 gives RTVolume, which carries the last trade size.
        self._tickers[symbol] = self.ib.reqMktData(contract, "233", False, False)

    async def unsubscribe_quote(self, symbol: str) -> None:
        if symbol not in self._tickers:
            return
        await self._pacer.acquire()
        try:
            self.ib.cancelMktData(self._contracts.get(symbol) or self._stock(symbol))
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel_mkt_data_failed", symbol=symbol, error=str(exc))
        self._tickers.pop(symbol, None)

    def get_quote(self, symbol: str) -> Quote | None:
        t = self._tickers.get(symbol)
        if t is None:
            return None
        ts = getattr(t, "time", None) or utcnow()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return Quote(
            symbol=symbol,
            bid=_f(getattr(t, "bid", None)),
            ask=_f(getattr(t, "ask", None)),
            last=_f(getattr(t, "last", None)) or _f(getattr(t, "close", None)),
            volume=_f(getattr(t, "volume", None), 0.0),
            halted=bool(getattr(t, "halted", 0)),
            updated_at=ts,
        )

    async def subscribe_bars(self, symbol: str) -> None:
        if symbol in self._bar_subs:
            return
        await self._pacer.acquire()
        contract = self._contracts.get(symbol) or self._stock(symbol)
        self._bar_subs[symbol] = self.ib.reqRealTimeBars(contract, 5, "TRADES", False)

    async def unsubscribe_bars(self, symbol: str) -> None:
        sub = self._bar_subs.pop(symbol, None)
        if sub is None:
            return
        await self._pacer.acquire()
        try:
            self.ib.cancelRealTimeBars(sub)
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel_realtime_bars_failed", symbol=symbol, error=str(exc))

    async def historical_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
    ) -> list[Bar]:
        await self._pacer.acquire()
        self.stats.historical_requests += 1
        contract = self._contracts.get(symbol) or self._stock(symbol)
        raw = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=2,  # UTC epoch-style, avoids local-timezone ambiguity
        )
        period = _bar_size_seconds(bar_size)
        out: list[Bar] = []
        for b in raw or []:
            ts = b.date
            if isinstance(ts, str):
                continue
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
            else:  # a `date` for daily bars
                ts = datetime(ts.year, ts.month, ts.day, tzinfo=UTC)
            out.append(
                Bar(
                    symbol=symbol,
                    ts=ts,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume or 0),
                    period=period,
                )
            )
        return out

    async def scan(
        self,
        scan_code: str,
        rows: int = 50,
        min_price: float = 3.0,
        min_volume: int = 500_000,
        timeout: float = 15.0,
    ) -> list[ScannerHit]:
        """Server-side scanner.

        Scanner results carry no bid/ask/last fields, which is exactly why they
        cost no market-data lines — the reason a 100+ symbol universe is
        affordable at all.

        Bypasses `ib.reqScannerDataAsync`: it only cancels the subscription
        on the happy path (`await future` then `cancelScannerSubscription`),
        so any rejection or slow response leaks a subscription with no
        timeout on the wait. Leaked subscriptions stack up until IBKR starts
        auto-cancelling new ones with "API scanner subscription cancelled"
        (error 162) — this mirrors the self-cancelling pattern in
        `_account_summary_once` to guarantee cleanup on every path.
        """
        await self._pacer.acquire()
        sub = ScannerSubscription(
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode=scan_code,
            numberOfRows=min(rows, 50),
            abovePrice=min_price,
            aboveVolume=min_volume,
        )
        # No TagValue filters: `usdMarketCapAbove` requires a fundamentals
        # data entitlement this account doesn't have and IBKR rejects it
        # with error 162 on every call, which kills scanning entirely.
        # abovePrice/aboveVolume on the subscription already filter junk.
        data_list = self.ib.reqScannerSubscription(sub, [], [])
        self._scanner_req_ids.add(data_list.reqId)
        future = self.ib.wrapper.startReq(data_list.reqId, container=data_list)
        try:
            async with asyncio.timeout(timeout):
                await future
            data = future.result()
        except Exception as exc:  # noqa: BLE001
            log.warning("scanner_failed", scan_code=scan_code, error=str(exc))
            return []
        finally:
            self.ib.client.cancelScannerSubscription(data_list.reqId)
            self.ib.wrapper.endSubscription(data_list)
            self.ib.wrapper._futures.pop(data_list.reqId, None)
            self.ib.wrapper._results.pop(data_list.reqId, None)
            self._scanner_req_ids.discard(data_list.reqId)
        hits: list[ScannerHit] = []
        for item in data or []:
            contract = getattr(item.contractDetails, "contract", None) if item else None
            if contract is None or not contract.symbol:
                continue
            hits.append(
                ScannerHit(symbol=contract.symbol, rank=int(getattr(item, "rank", 0)), scan_code=scan_code)
            )
        return hits

    # ------------------------------------------------------------------ #
    # account
    # ------------------------------------------------------------------ #

    async def account_snapshot(self) -> AccountSnapshot:
        """Ground truth. Never inferred, always fetched."""
        await self._pacer.acquire(3)
        summary = await self._account_summary_once()
        values: dict[str, float] = {}
        for row in summary:
            if row.currency in ("USD", "", "BASE"):
                try:
                    values[row.tag] = float(row.value)
                except (TypeError, ValueError):
                    continue

        portfolio = self.ib.portfolio(self._account) if self._account else self.ib.portfolio()
        positions: dict[str, Position] = {}
        for item in portfolio:
            sym = item.contract.symbol
            positions[sym] = Position(
                symbol=sym,
                quantity=float(item.position),
                avg_cost=float(item.averageCost or 0),
                market_price=_f(item.marketPrice),
                unrealized_pnl=float(item.unrealizedPNL or 0),
                realized_pnl=float(item.realizedPNL or 0),
            )

        equity = values.get("NetLiquidation", 0.0)
        return AccountSnapshot(
            account_id=self._account,
            equity=equity,
            cash=values.get("TotalCashValue", 0.0),
            buying_power=values.get("BuyingPower", values.get("AvailableFunds", 0.0)),
            realized_pnl=values.get("RealizedPnL", 0.0),
            unrealized_pnl=sum(p.unrealized_pnl for p in positions.values()),
            positions=positions,
            as_of=utcnow(),
        )

    # ------------------------------------------------------------------ #
    # orders
    # ------------------------------------------------------------------ #

    def _apply_common(self, order: Order, spec: BrokerOrderSpec) -> Order:
        order.orderRef = spec.order_ref
        order.tif = spec.tif
        order.outsideRth = spec.outside_rth
        order.transmit = spec.transmit
        if spec.oca_group:
            order.ocaGroup = spec.oca_group
            order.ocaType = 1  # cancel remaining legs on fill
        if self._account:
            order.account = self._account
        return order

    def _build(self, spec: BrokerOrderSpec) -> Order:
        t = spec.order_type.upper()
        if t == "LMT":
            if spec.limit_price is None:
                raise ValueError("LMT order requires a limit price")
            o: Order = LimitOrder(spec.action, spec.quantity, spec.limit_price)
        elif t == "STP":
            if spec.stop_price is None:
                raise ValueError("STP order requires a stop price")
            o = StopOrder(spec.action, spec.quantity, spec.stop_price)
        elif t == "MKT":
            o = Order(action=spec.action, totalQuantity=spec.quantity, orderType="MKT")
        else:
            raise ValueError(f"unsupported order type {spec.order_type!r}")
        return self._apply_common(o, spec)

    async def place_bracket(
        self, entry: BrokerOrderSpec, stop: BrokerOrderSpec, target: BrokerOrderSpec
    ) -> list[BrokerOrderStatus]:
        """Place a native IBKR bracket so the protective legs live at the broker.

        `transmit` is False on the parent and target and True only on the last
        leg, so IBKR receives the whole group atomically. Sending the parent
        alone would leave a naked position if the children failed to arrive.
        """
        await self._pacer.acquire(3)
        contract = self._contracts.get(entry.symbol) or self._stock(entry.symbol)
        if entry.limit_price is None:
            raise ValueError("bracket entry must be a limit order")
        if target.limit_price is None:
            raise ValueError("bracket target must be a limit order")
        if stop.stop_price is None:
            raise ValueError("bracket stop must carry a stop price")

        parent = LimitOrder(entry.action, entry.quantity, entry.limit_price)
        parent.transmit = False
        self._apply_common(parent, entry)
        parent.ocaGroup = ""  # the parent is not part of the exit OCA group
        parent.orderId = self.ib.client.getReqId()

        exit_action = "SELL" if entry.action.upper() == "BUY" else "BUY"
        oca = entry.oca_group or f"oca-{parent.orderId}"

        take = LimitOrder(exit_action, entry.quantity, target.limit_price)
        take.parentId = parent.orderId
        take.transmit = False
        self._apply_common(take, target)
        take.ocaGroup = oca
        take.ocaType = 1
        take.orderId = self.ib.client.getReqId()

        stop_order = StopOrder(exit_action, entry.quantity, stop.stop_price)
        stop_order.parentId = parent.orderId
        stop_order.transmit = True  # the last leg transmits the whole group
        self._apply_common(stop_order, stop)
        stop_order.ocaGroup = oca
        stop_order.ocaType = 1
        stop_order.orderId = self.ib.client.getReqId()

        statuses: list[BrokerOrderStatus] = []
        for order in (parent, take, stop_order):
            trade = self.ib.placeOrder(contract, order)
            self.stats.messages_sent += 1
            self._ref_by_order_id[order.orderId] = order.orderRef
            statuses.append(self._to_status(trade))

        log.info(
            "bracket_placed",
            symbol=entry.symbol,
            action=entry.action,
            quantity=entry.quantity,
            entry=entry.limit_price,
            stop=stop.stop_price,
            target=target.limit_price,
            oca_group=oca,
        )
        return statuses

    async def place_single(self, spec: BrokerOrderSpec) -> BrokerOrderStatus:
        await self._pacer.acquire()
        contract = self._contracts.get(spec.symbol) or self._stock(spec.symbol)
        order = self._build(spec)
        trade = self.ib.placeOrder(contract, order)
        self.stats.messages_sent += 1
        if getattr(order, "orderId", None):
            self._ref_by_order_id[order.orderId] = spec.order_ref
        return self._to_status(trade)

    async def cancel_order(self, order_id: int) -> None:
        await self._pacer.acquire()
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                self.stats.messages_sent += 1
                return
        log.warning("cancel_order_not_found", order_id=order_id)

    async def cancel_all(self) -> None:
        """Account-wide cancel. Reserved for the kill switch.

        This cancels manually-placed orders too, which is why it is not used for
        routine cleanup.
        """
        await self._pacer.acquire()
        self.ib.reqGlobalCancel()
        self.stats.messages_sent += 1
        log.warning("global_cancel_issued")

    def _to_status(self, trade: Any) -> BrokerOrderStatus:
        st = trade.orderStatus
        return BrokerOrderStatus(
            order_ref=getattr(trade.order, "orderRef", "") or "",
            order_id=trade.order.orderId,
            perm_id=getattr(trade.order, "permId", None) or getattr(st, "permId", None),
            status=st.status,
            filled=float(st.filled or 0),
            remaining=float(st.remaining or 0),
            avg_fill_price=_f(st.avgFillPrice),
            symbol=getattr(trade.contract, "symbol", ""),
            action=getattr(trade.order, "action", ""),
        )

    async def open_orders(self) -> list[BrokerOrderStatus]:
        await self._pacer.acquire()
        trades = await self.ib.reqAllOpenOrdersAsync()
        return [self._to_status(t) for t in trades]

    async def completed_orders(self) -> list[BrokerOrderStatus]:
        """Needed to resolve 'maybe-sent' orders after a crash."""
        await self._pacer.acquire()
        try:
            trades = await self.ib.reqCompletedOrdersAsync(apiOnly=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("completed_orders_failed", error=str(exc))
            return []
        return [self._to_status(t) for t in trades]

    async def executions(self) -> list[BrokerExecution]:
        await self._pacer.acquire()
        try:
            fills = await self.ib.reqExecutionsAsync(ExecutionFilter())
        except Exception as exc:  # noqa: BLE001
            log.warning("executions_failed", error=str(exc))
            return []
        out: list[BrokerExecution] = []
        for fill in fills:
            ex = fill.execution
            ts = getattr(ex, "time", None) or utcnow()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            report = getattr(fill, "commissionReport", None)
            out.append(
                BrokerExecution(
                    exec_id=ex.execId,
                    order_ref=self._ref_by_order_id.get(ex.orderId, ""),
                    symbol=fill.contract.symbol,
                    action="BUY" if ex.side.upper().startswith("B") else "SELL",
                    quantity=float(ex.shares),
                    price=float(ex.price),
                    ts=ts,
                    commission=_f(getattr(report, "commission", 0.0), 0.0) or 0.0,
                    perm_id=getattr(ex, "permId", None),
                )
            )
        return out

    async def modify_stop(self, order_id: int, new_stop: float) -> None:
        """Move an existing stop by re-placing it with the same orderId.

        Modifying in place matters: cancelling and re-creating would leave a
        window in which the position is unprotected.
        """
        await self._pacer.acquire()
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                trade.order.auxPrice = new_stop
                self.ib.placeOrder(trade.contract, trade.order)
                self.stats.messages_sent += 1
                log.info("stop_modified", order_id=order_id, new_stop=new_stop)
                return
        log.warning("modify_stop_not_found", order_id=order_id)


def _bar_size_seconds(bar_size: str) -> int:
    table = {
        "1 secs": 1, "5 secs": 5, "10 secs": 10, "15 secs": 15, "30 secs": 30,
        "1 min": 60, "2 mins": 120, "3 mins": 180, "5 mins": 300, "10 mins": 600,
        "15 mins": 900, "20 mins": 1200, "30 mins": 1800,
        "1 hour": 3600, "2 hours": 7200, "4 hours": 14400,
        "1 day": 86400, "1 week": 604800,
    }
    return table.get(bar_size.strip().lower(), 300)
