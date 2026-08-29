"""Order lifecycle: placement, state, reconciliation.

Two ideas do most of the work here.

**Brackets live at the broker.** Every entry is a native IBKR bracket, so the
stop and target are real resting orders at IBKR rather than intentions held in
this process. There are guaranteed windows when we cannot reach the broker — the
daily Gateway restart, the weekly 2FA outage, a crash — and a position must stay
protected through all of them.

**Write-ahead, then place.** The order row is committed to SQLite *before* the
wire call. On restart, a row we believe is working but that IBKR has never heard
of is a "maybe-sent" that must be resolved against the broker before we are
allowed to trade that symbol again. Without this ordering, a crash between the
call and the acknowledgement produces a duplicate position.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..domain.enums import Action, OrderState
from ..domain.models import AccountSnapshot, Fill, ManagedOrder, utcnow
from ..domain.proposals import SizedOrder
from ..logging_setup import get_logger
from .port import BrokerExecution, BrokerOrderSpec, BrokerOrderStatus, BrokerPort

log = get_logger(__name__)

REF_PREFIX = "ait"

#: How long a symbol stays blocked after an unresolved submit error before it
#: is auto-unblocked. `reconcile()` only runs on connect/reconnect, not on a
#: timer, so a transient error on an otherwise-stable connection could
#: otherwise leave a symbol frozen for the rest of the process's life.
_BLOCK_TTL_SECONDS = 600.0

#: IBKR status strings mapped to our state machine.
_STATUS_MAP = {
    "PendingSubmit": OrderState.PENDING,
    "PendingCancel": OrderState.SUBMITTED,
    "PreSubmitted": OrderState.SUBMITTED,
    "Submitted": OrderState.SUBMITTED,
    "ApiPending": OrderState.PENDING,
    "ApiCancelled": OrderState.CANCELLED,
    "Cancelled": OrderState.CANCELLED,
    "Filled": OrderState.FILLED,
    "Inactive": OrderState.REJECTED,
}


def make_ref(cycle_id: str, symbol: str, kind: str, day: str | None = None) -> str:
    """Deterministic order reference, used for idempotency and adoption.

    Because it is derived rather than random, the same logical order produces
    the same ref after a restart — which is what lets reconciliation recognise
    an order we already placed instead of placing it again.
    """
    d = day or datetime.now(UTC).date().isoformat()
    return f"{REF_PREFIX}:{d}:{cycle_id or 'manual'}:{symbol}:{kind}"


def parse_ref(ref: str) -> dict[str, str] | None:
    parts = (ref or "").split(":")
    if len(parts) != 5 or parts[0] != REF_PREFIX:
        return None
    return {"day": parts[1], "cycle_id": parts[2], "symbol": parts[3], "kind": parts[4]}


def is_ours(ref: str) -> bool:
    return (ref or "").startswith(f"{REF_PREFIX}:")


@dataclass
class ReconciliationReport:
    """What reconciliation found. Surfaced on the dashboard after every reconnect."""

    adopted: list[str] = field(default_factory=list)
    orphaned_local: list[str] = field(default_factory=list)
    orphaned_broker: list[str] = field(default_factory=list)
    unprotected_positions: list[str] = field(default_factory=list)
    protective_stops_placed: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    blocked_symbols: set[str] = field(default_factory=set)
    ran_at: datetime = field(default_factory=utcnow)

    @property
    def is_clean(self) -> bool:
        return not self.unresolved

    def as_dict(self) -> dict[str, Any]:
        return {
            "adopted": self.adopted,
            "orphaned_local": self.orphaned_local,
            "orphaned_broker": self.orphaned_broker,
            "unprotected_positions": self.unprotected_positions,
            "protective_stops_placed": self.protective_stops_placed,
            "unresolved": self.unresolved,
            "blocked_symbols": sorted(self.blocked_symbols),
            "ran_at": self.ran_at.isoformat(),
            "clean": self.is_clean,
        }


class OrderManager:
    """Owns every order this system has placed.

    Only `RiskEngine.submit` may call `place_bracket`. An architecture test
    enforces that.
    """

    def __init__(self, broker: BrokerPort, store: Any = None, limit_offset_pct: float = 0.001):
        self.broker = broker
        self.store = store
        self.limit_offset_pct = limit_offset_pct
        self.orders: dict[str, ManagedOrder] = {}
        self.fills: list[Fill] = []
        #: Symbols we must not trade until reconciliation resolves them.
        #: Value is the monotonic time the block was set, so it can expire —
        #: see _BLOCK_TTL_SECONDS.
        self.blocked_symbols: dict[str, float] = {}
        self._seen_exec_ids: set[str] = set()
        self._perm_ids: dict[str, int] = {}

    def _unblock_expired(self) -> None:
        now = time.monotonic()
        expired = [s for s, blocked_at in self.blocked_symbols.items() if now - blocked_at > _BLOCK_TTL_SECONDS]
        for symbol in expired:
            del self.blocked_symbols[symbol]
            log.info("symbol_block_expired", symbol=symbol, ttl_seconds=_BLOCK_TTL_SECONDS)

    # ------------------------------------------------------------------ #
    # placement
    # ------------------------------------------------------------------ #

    async def place_bracket(self, order: SizedOrder, cycle_id: str = "") -> list[ManagedOrder]:
        """Place a native bracket, persisting our intent first."""
        self._unblock_expired()
        if order.symbol in self.blocked_symbols:
            raise RuntimeError(
                f"{order.symbol} is blocked pending reconciliation of an unresolved order"
            )

        entry_ref = make_ref(cycle_id, order.symbol, "ENTRY")
        stop_ref = make_ref(cycle_id, order.symbol, "STOP")
        target_ref = make_ref(cycle_id, order.symbol, "TARGET")

        # Idempotency: if we already have this ref, the order exists.
        if entry_ref in self.orders:
            log.warning("duplicate_order_ref_suppressed", order_ref=entry_ref)
            return [self.orders[entry_ref]]
        if self.store is not None and self.store.find_order_by_ref(entry_ref):
            log.warning("duplicate_order_ref_in_store", order_ref=entry_ref)
            return []

        action = "BUY" if order.action == Action.BUY else "SELL"
        exit_action = "SELL" if action == "BUY" else "BUY"

        managed = [
            ManagedOrder(
                order_ref=entry_ref, symbol=order.symbol, action=action,
                quantity=order.quantity, limit_price=order.entry_price,
                stop_price=order.stop_price, target_price=order.target_price,
                state=OrderState.PENDING, cycle_id=cycle_id,
                rationale=order.proposal.rationale[:500],
            ),
            ManagedOrder(
                order_ref=stop_ref, symbol=order.symbol, action=exit_action,
                quantity=order.quantity, stop_price=order.stop_price,
                state=OrderState.PENDING, parent_ref=entry_ref, cycle_id=cycle_id,
            ),
            ManagedOrder(
                order_ref=target_ref, symbol=order.symbol, action=exit_action,
                quantity=order.quantity, limit_price=order.target_price,
                state=OrderState.PENDING, parent_ref=entry_ref, cycle_id=cycle_id,
            ),
        ]

        # Write-ahead. If we die immediately after this, reconciliation knows to
        # go looking for these orders at the broker.
        for mo in managed:
            self.orders[mo.order_ref] = mo
            self._persist(mo)

        oca_group = f"{REF_PREFIX}-{cycle_id or 'm'}-{order.symbol}"
        try:
            statuses = await self.broker.place_bracket(
                BrokerOrderSpec(
                    symbol=order.symbol, action=action, quantity=order.quantity,
                    order_type="LMT", limit_price=order.entry_price, order_ref=entry_ref,
                ),
                BrokerOrderSpec(
                    symbol=order.symbol, action=exit_action, quantity=order.quantity,
                    order_type="STP", stop_price=order.stop_price, order_ref=stop_ref,
                    oca_group=oca_group,
                ),
                BrokerOrderSpec(
                    symbol=order.symbol, action=exit_action, quantity=order.quantity,
                    order_type="LMT", limit_price=order.target_price, order_ref=target_ref,
                    oca_group=oca_group,
                ),
            )
        except Exception:
            # Leave the rows in place: they are the record that we *may* have
            # sent something, and reconciliation must check rather than assume.
            for mo in managed:
                self.blocked_symbols[mo.symbol] = time.monotonic()
            raise

        for status in statuses:
            self._apply_status(status)

        log.info(
            "bracket_submitted",
            symbol=order.symbol, action=action, quantity=order.quantity,
            entry=order.entry_price, stop=order.stop_price, target=order.target_price,
            risk=round(order.risk_amount, 2), cycle_id=cycle_id,
        )
        return managed

    async def close_position(self, symbol: str, quantity: float, is_long: bool, reason: str) -> None:
        """Exit a position with a marketable limit.

        Any resting bracket legs are cancelled first: relying on OCA to fire
        correctly while we send a competing exit is a race.
        """
        await self.cancel_symbol_orders(symbol)

        quote = self.broker.get_quote(symbol)
        action = "SELL" if is_long else "BUY"
        ref = make_ref(f"exit-{int(utcnow().timestamp())}", symbol, "EXIT")

        price = None
        if quote and quote.is_usable and quote.mid:
            # Cross the spread to get out; an unfilled exit is worse than a
            # slightly worse price.
            offset = quote.mid * self.limit_offset_pct * 3
            price = round(quote.mid - offset if is_long else quote.mid + offset, 2)

        spec = BrokerOrderSpec(
            symbol=symbol, action=action, quantity=abs(quantity),
            order_type="LMT" if price else "MKT", limit_price=price, order_ref=ref,
        )
        mo = ManagedOrder(
            order_ref=ref, symbol=symbol, action=action, quantity=abs(quantity),
            limit_price=price, state=OrderState.PENDING, rationale=reason,
        )
        self.orders[ref] = mo
        self._persist(mo)

        status = await self.broker.place_single(spec)
        self._apply_status(status)
        log.info("position_close_submitted", symbol=symbol, quantity=quantity, reason=reason)

    async def cancel_symbol_orders(self, symbol: str) -> int:
        cancelled = 0
        for mo in list(self.orders.values()):
            if mo.symbol == symbol and mo.is_working and mo.ib_order_id:
                try:
                    await self.broker.cancel_order(mo.ib_order_id)
                    cancelled += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("cancel_failed", order_ref=mo.order_ref, error=str(exc))
        return cancelled

    async def cancel_all_working(self) -> int:
        """Cancel our own working orders individually.

        Deliberately not `reqGlobalCancel`, which is account-wide and would also
        cancel manually-placed orders. Global cancel is reserved for the kill
        switch.
        """
        cancelled = 0
        for mo in list(self.orders.values()):
            if mo.is_working and mo.ib_order_id:
                try:
                    await self.broker.cancel_order(mo.ib_order_id)
                    cancelled += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("cancel_failed", order_ref=mo.order_ref, error=str(exc))
        log.warning("cancelled_all_working_orders", count=cancelled)
        return cancelled

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def _apply_status(self, status: BrokerOrderStatus) -> None:
        ref = status.order_ref
        mo = self.orders.get(ref)
        if mo is None:
            if not is_ours(ref):
                return
            mo = ManagedOrder(
                order_ref=ref, symbol=status.symbol, action=status.action,
                quantity=status.filled + status.remaining,
            )
            self.orders[ref] = mo

        previous = mo.state
        new_state = _STATUS_MAP.get(status.status, mo.state)
        if status.filled > 0 and new_state not in (OrderState.FILLED, OrderState.CANCELLED):
            new_state = OrderState.PARTIAL

        mo.state = new_state
        mo.ib_order_id = status.order_id or mo.ib_order_id
        mo.filled_quantity = status.filled
        mo.avg_fill_price = status.avg_fill_price or mo.avg_fill_price
        mo.updated_at = utcnow()
        if status.perm_id:
            self._perm_ids[ref] = status.perm_id

        self._persist(mo, status.perm_id)
        if previous != new_state and self.store is not None:
            try:
                self.store.record_transition(ref, previous, new_state, status.status)
            except Exception as exc:  # noqa: BLE001
                log.warning("transition_persist_failed", error=str(exc))

    def on_order_status(self, status: BrokerOrderStatus) -> None:
        try:
            self._apply_status(status)
        except Exception as exc:
            log.exception("order_status_apply_failed", error=str(exc))

    def on_execution(self, execution: BrokerExecution) -> None:
        """Record a fill, deduped on execId.

        IBKR re-sends executions after a reconnect, so without this dedupe a
        reconnect would double-count realized P&L.
        """
        if execution.exec_id in self._seen_exec_ids:
            return
        self._seen_exec_ids.add(execution.exec_id)

        fill = Fill(
            symbol=execution.symbol, action=execution.action, quantity=execution.quantity,
            price=execution.price, commission=execution.commission, ts=execution.ts,
            order_ref=execution.order_ref,
        )
        self.fills.append(fill)
        if self.store is not None:
            try:
                self.store.save_fill(execution.exec_id, fill)
            except Exception as exc:  # noqa: BLE001
                log.warning("fill_persist_failed", error=str(exc))
        log.info(
            "fill",
            symbol=execution.symbol, action=execution.action,
            quantity=execution.quantity, price=execution.price,
        )

    def _persist(self, mo: ManagedOrder, perm_id: int | None = None) -> None:
        if self.store is None:
            return
        try:
            self.store.save_order(mo, perm_id or self._perm_ids.get(mo.order_ref))
        except Exception as exc:  # noqa: BLE001
            log.warning("order_persist_failed", order_ref=mo.order_ref, error=str(exc))

    # ------------------------------------------------------------------ #
    # reconciliation
    # ------------------------------------------------------------------ #

    async def reconcile(self, account: AccountSnapshot) -> ReconciliationReport:
        """Rebuild our view of the world from the broker's.

        Runs on every connect and reconnect. The decision loop is gated on this
        completing, because trading on a stale picture of our own positions is
        the failure that compounds worst.
        """
        report = ReconciliationReport()
        self._unblock_expired()

        broker_open = await self.broker.open_orders()
        broker_completed = await self.broker.completed_orders()

        by_ref: dict[str, BrokerOrderStatus] = {}
        for status in [*broker_open, *broker_completed]:
            if status.order_ref:
                by_ref[status.order_ref] = status

        # 1. Absorb everything the broker knows about.
        for status in broker_open:
            if is_ours(status.order_ref):
                if status.order_ref not in self.orders:
                    report.adopted.append(status.order_ref)
                self._apply_status(status)
            else:
                report.orphaned_broker.append(
                    status.order_ref or f"orderId={status.order_id}"
                )

        for status in broker_completed:
            if is_ours(status.order_ref) and status.order_ref in self.orders:
                self._apply_status(status)

        # 2. Resolve local orders the broker has never heard of. These are the
        #    "maybe-sent" rows: we committed the intent, then possibly died.
        for mo in list(self.orders.values()):
            if not mo.is_working:
                continue
            if mo.order_ref in by_ref:
                continue
            report.orphaned_local.append(mo.order_ref)
            # The broker has no record in either open or completed orders, so
            # the order never reached it. Marking it cancelled is safe.
            previous = mo.state
            mo.state = OrderState.CANCELLED
            mo.updated_at = utcnow()
            self._persist(mo)
            if self.store is not None:
                self.store.record_transition(
                    mo.order_ref, previous, OrderState.CANCELLED,
                    "not found at broker during reconciliation",
                )

        # 3. Re-absorb executions, deduped, so realized P&L is correct.
        for execution in await self.broker.executions():
            self.on_execution(execution)

        # 4. The case that actually saves money: a position with no protective
        #    stop resting at the broker. This happens when the process dies
        #    between the parent filling and the children being acknowledged.
        protected: set[str] = {
            mo.symbol
            for mo in self.orders.values()
            if mo.is_working and mo.stop_price is not None
        }
        for symbol, position in account.open_positions.items():
            if symbol not in protected:
                report.unprotected_positions.append(symbol)
                placed = await self._emit_protective_stop(symbol, position)
                if placed:
                    report.protective_stops_placed.append(symbol)
                else:
                    report.unresolved.append(f"{symbol}: unprotected, stop placement failed")
                    self.blocked_symbols[symbol] = time.monotonic()

        # 5. Cancel protective orders for positions we no longer hold.
        for mo in list(self.orders.values()):
            if (
                mo.is_working
                and mo.parent_ref is not None
                and mo.symbol not in account.open_positions
                and mo.ib_order_id
            ):
                try:
                    await self.broker.cancel_order(mo.ib_order_id)
                    log.info("cancelled_orphan_protective_order", order_ref=mo.order_ref)
                except Exception as exc:  # noqa: BLE001
                    log.warning("orphan_cancel_failed", order_ref=mo.order_ref, error=str(exc))

        report.blocked_symbols = set(self.blocked_symbols)

        log.info("reconciliation_complete", **report.as_dict())
        if not report.is_clean:
            log.error("reconciliation_unresolved", unresolved=report.unresolved)
        return report

    async def _emit_protective_stop(self, symbol: str, position: Any) -> bool:
        """Place a stop for a position that has none.

        The stop is placed at a fixed percentage rather than an ATR multiple,
        because at this point we may have no indicator history — getting *a*
        stop in place immediately matters more than getting the ideal one.
        """
        try:
            quote = self.broker.get_quote(symbol)
            reference = None
            if quote and quote.is_usable and quote.mid:
                reference = float(quote.mid)
            elif position.market_price:
                reference = float(position.market_price)
            elif position.avg_cost:
                reference = float(position.avg_cost)
            if not reference or reference <= 0:
                log.error("protective_stop_no_reference_price", symbol=symbol)
                return False

            is_long = position.quantity > 0
            stop_price = round(reference * (0.97 if is_long else 1.03), 2)
            ref = make_ref(f"protect-{int(utcnow().timestamp())}", symbol, "STOP")

            spec = BrokerOrderSpec(
                symbol=symbol,
                action="SELL" if is_long else "BUY",
                quantity=abs(position.quantity),
                order_type="STP",
                stop_price=stop_price,
                order_ref=ref,
            )
            mo = ManagedOrder(
                order_ref=ref, symbol=symbol, action=spec.action,
                quantity=abs(position.quantity), stop_price=stop_price,
                state=OrderState.PENDING, parent_ref="recovered",
                rationale="protective stop added during reconciliation",
            )
            self.orders[ref] = mo
            self._persist(mo)

            status = await self.broker.place_single(spec)
            self._apply_status(status)
            log.warning(
                "protective_stop_placed_for_unprotected_position",
                symbol=symbol, quantity=position.quantity, stop=stop_price,
            )
            return True
        except Exception as exc:
            log.exception("protective_stop_failed", symbol=symbol, error=str(exc))
            return False

    # ------------------------------------------------------------------ #

    def load_from_store(self) -> None:
        """Restore today's orders after a restart."""
        if self.store is None:
            return
        try:
            rows = self.store.load_orders()
        except Exception as exc:  # noqa: BLE001
            log.warning("order_restore_failed", error=str(exc))
            return
        for row in rows:
            try:
                mo = ManagedOrder(
                    order_ref=row["order_ref"], symbol=row["symbol"], action=row["action"],
                    quantity=row["quantity"], limit_price=row["limit_price"],
                    stop_price=row["stop_price"], target_price=row["target_price"],
                    state=OrderState(row["state"]), ib_order_id=row["ib_order_id"],
                    parent_ref=row["parent_ref"], filled_quantity=row["filled_quantity"] or 0,
                    avg_fill_price=row["avg_fill_price"], cycle_id=row["cycle_id"],
                    rationale=row["rationale"],
                )
                self.orders[mo.order_ref] = mo
                if row.get("ib_perm_id"):
                    self._perm_ids[mo.order_ref] = row["ib_perm_id"]
            except Exception as exc:  # noqa: BLE001
                log.warning("order_row_restore_failed", row=row.get("order_ref"), error=str(exc))
        if self.orders:
            log.info("orders_restored", count=len(self.orders))

    @property
    def working_symbols(self) -> set[str]:
        return {mo.symbol for mo in self.orders.values() if mo.is_working}

    def working_orders(self) -> list[ManagedOrder]:
        return [mo for mo in self.orders.values() if mo.is_working]

    def stop_order_for(self, symbol: str) -> ManagedOrder | None:
        for mo in self.orders.values():
            if mo.symbol == symbol and mo.is_working and mo.stop_price is not None:
                return mo
        return None
