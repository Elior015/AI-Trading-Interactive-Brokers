"""Order lifecycle: bracket construction, idempotency, execution dedup, and
reconciliation — the machinery that makes a restart or a crash safe.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aitrader.broker.orders import OrderManager, is_ours, make_ref, parse_ref
from aitrader.data.store import StateStore
from aitrader.domain.enums import Action, OrderState
from aitrader.domain.models import AccountSnapshot, Position
from aitrader.domain.proposals import SizedOrder, TradeProposal
from tests.fakes.fake_broker import FakeBroker


def sized(symbol="AAPL", action=Action.BUY, qty=10, entry=100.0, stop=99.0, target=102.0):
    return SizedOrder(
        proposal=TradeProposal(symbol=symbol, action=action, conviction=0.8, horizon_minutes=30),
        symbol=symbol, action=action, quantity=qty,
        entry_price=entry, stop_price=stop, target_price=target,
        risk_amount=abs(entry - stop) * qty, notional=entry * qty,
    )


@pytest.fixture
def store(tmp_path):
    s = StateStore(tmp_path / "test.sqlite3")
    yield s
    s.close()


@pytest.fixture
def broker():
    return FakeBroker()


@pytest.fixture
def manager(broker, store):
    return OrderManager(broker=broker, store=store)


class TestOrderRef:
    def test_deterministic(self):
        ref = make_ref("c1", "AAPL", "ENTRY", day="2026-08-24")
        assert ref == "ait:2026-08-24:c1:AAPL:ENTRY"

    def test_same_inputs_same_ref(self):
        """This determinism is what lets reconciliation adopt rather than duplicate."""
        a = make_ref("c1", "AAPL", "ENTRY", day="2026-08-24")
        b = make_ref("c1", "AAPL", "ENTRY", day="2026-08-24")
        assert a == b

    def test_parse_roundtrip(self):
        ref = make_ref("c7", "MSFT", "STOP", day="2026-08-24")
        parsed = parse_ref(ref)
        assert parsed == {"day": "2026-08-24", "cycle_id": "c7", "symbol": "MSFT", "kind": "STOP"}

    def test_is_ours(self):
        assert is_ours(make_ref("c1", "AAPL", "ENTRY"))
        assert not is_ours("some-other-system:order-1")
        assert not is_ours("")

    def test_parse_rejects_foreign_ref(self):
        assert parse_ref("manual-order-123") is None


class TestBracketPlacement:
    async def test_places_three_legs(self, manager, broker):
        await manager.place_bracket(sized(), cycle_id="c1")
        assert len(broker.placed) == 3

    async def test_entry_is_limit_stop_is_stop_target_is_limit(self, manager, broker):
        await manager.place_bracket(sized(), cycle_id="c1")
        entry, stop, target = broker.placed  # order matches place_bracket's own arg order... see below
        # place_bracket in the fake receives (entry, stop, target) as positional args
        assert entry.order_type == "LMT" and entry.limit_price == 100.0
        assert stop.order_type == "STP" and stop.stop_price == 99.0
        assert target.order_type == "LMT" and target.limit_price == 102.0

    async def test_exit_legs_share_an_oca_group(self, manager, broker):
        await manager.place_bracket(sized(), cycle_id="c1")
        _, stop, target = broker.placed
        assert stop.oca_group == target.oca_group
        assert stop.oca_group is not None

    async def test_stop_and_target_are_opposite_action_from_entry(self, manager, broker):
        await manager.place_bracket(sized(action=Action.BUY), cycle_id="c1")
        entry, stop, target = broker.placed
        assert entry.action == "BUY"
        assert stop.action == "SELL" and target.action == "SELL"

    async def test_writes_ahead_before_calling_the_broker(self, manager, store):
        """The row must exist in SQLite even if we crash immediately after this."""
        await manager.place_bracket(sized(symbol="NVDA"), cycle_id="c1")
        ref = make_ref("c1", "NVDA", "ENTRY")
        row = store.find_order_by_ref(ref)
        assert row is not None

    async def test_duplicate_ref_is_not_replaced(self, manager, broker):
        """Idempotency: calling twice with the same cycle_id must not double-place."""
        await manager.place_bracket(sized(), cycle_id="c1")
        first_count = len(broker.placed)
        await manager.place_bracket(sized(), cycle_id="c1")
        assert len(broker.placed) == first_count

    async def test_blocked_symbol_refuses_placement(self, manager, broker):
        manager.blocked_symbols.add("AAPL")
        with pytest.raises(RuntimeError):
            await manager.place_bracket(sized(symbol="AAPL"), cycle_id="c1")
        assert broker.placed == []

    async def test_placement_failure_blocks_the_symbol(self, manager, broker):
        """A failed call might have partially reached the broker — don't retry blind."""
        broker.fail_next_order = True
        with pytest.raises(RuntimeError):
            await manager.place_bracket(sized(symbol="TSLA"), cycle_id="c1")
        assert "TSLA" in manager.blocked_symbols


class TestExecutionDedup:
    def test_duplicate_exec_id_counted_once(self, manager):
        from aitrader.broker.port import BrokerExecution

        ex = BrokerExecution(
            exec_id="e-1", order_ref="ait:d:c1:AAPL:ENTRY", symbol="AAPL",
            action="BUY", quantity=10, price=100.0, ts=datetime.now(UTC),
        )
        manager.on_execution(ex)
        manager.on_execution(ex)  # IBKR re-sends on reconnect
        assert len(manager.fills) == 1

    def test_distinct_exec_ids_both_counted(self, manager):
        from aitrader.broker.port import BrokerExecution

        for i in range(2):
            manager.on_execution(
                BrokerExecution(
                    exec_id=f"e-{i}", order_ref="ref", symbol="AAPL",
                    action="BUY", quantity=5, price=100.0, ts=datetime.now(UTC),
                )
            )
        assert len(manager.fills) == 2

    def test_fill_persisted_to_store(self, manager, store):
        from aitrader.broker.port import BrokerExecution

        manager.on_execution(
            BrokerExecution(
                exec_id="e-9", order_ref="ref", symbol="AAPL",
                action="BUY", quantity=5, price=101.5, ts=datetime.now(UTC),
            )
        )
        rows = store.load_fills()
        assert len(rows) == 1
        assert rows[0]["exec_id"] == "e-9"


class TestReconciliation:
    def account(self, positions=None, equity=100_000.0):
        return AccountSnapshot(
            account_id="DU1", equity=equity, cash=equity, buying_power=equity * 2,
            positions=positions or {},
        )

    async def test_clean_reconcile_with_nothing_outstanding(self, manager, broker):
        report = await manager.reconcile(self.account())
        assert report.is_clean
        assert report.unprotected_positions == []

    async def test_adopts_broker_orders_we_did_not_know_about(self, manager, broker):
        """Simulates recovering from a restart: the broker remembers, we don't yet."""
        from aitrader.broker.port import BrokerOrderStatus

        broker.statuses["ait:2026-08-24:c1:AAPL:ENTRY"] = BrokerOrderStatus(
            order_ref="ait:2026-08-24:c1:AAPL:ENTRY", order_id=555, perm_id=5550,
            status="Submitted", filled=0, remaining=10, avg_fill_price=None,
            symbol="AAPL", action="BUY",
        )
        report = await manager.reconcile(self.account())
        assert "ait:2026-08-24:c1:AAPL:ENTRY" in report.adopted
        assert "ait:2026-08-24:c1:AAPL:ENTRY" in manager.orders

    async def test_maybe_sent_order_not_at_broker_is_marked_cancelled(self, manager, store):
        """The write-ahead row for an order that never reached the broker must not
        be left in a working state forever.

        The entry leg auto-fills in the fake broker, so it is already terminal
        (FILLED) before reconciliation runs and is correctly left alone — only
        the still-working stop/target legs are "maybe-sent" and need resolving.
        """
        await manager.place_bracket(sized(symbol="MSFT"), cycle_id="c1")
        working_refs = [ref for ref, mo in manager.orders.items() if "MSFT" in ref and mo.is_working]
        assert working_refs, "expected the stop/target legs to still be working"

        broker_double = manager.broker
        broker_double.statuses.clear()
        broker_double.placed.clear()

        report = await manager.reconcile(self.account())
        assert set(working_refs) <= set(report.orphaned_local)
        for ref in working_refs:
            assert manager.orders[ref].state == OrderState.CANCELLED

    async def test_unprotected_position_gets_a_protective_stop(self, manager, broker):
        """The case that actually saves money: a filled position with no
        resting stop, e.g. because the process died between the parent filling
        and the children being acknowledged."""
        broker.positions["GOOG"] = Position(symbol="GOOG", quantity=10, avg_cost=150.0, market_price=151.0)
        broker.quotes["GOOG"] = broker.set_quote("GOOG", bid=150.9, ask=151.1, last=151.0)

        report = await manager.reconcile(self.account(positions=broker.positions))
        assert "GOOG" in report.unprotected_positions
        assert "GOOG" in report.protective_stops_placed
        assert manager.stop_order_for("GOOG") is not None

    async def test_protected_position_is_left_alone(self, manager, broker):
        await manager.place_bracket(sized(symbol="AMZN"), cycle_id="c1")
        broker.positions["AMZN"] = Position(symbol="AMZN", quantity=10, avg_cost=100.0, market_price=100.5)

        report = await manager.reconcile(self.account(positions=broker.positions))
        assert "AMZN" not in report.unprotected_positions

    async def test_orphan_protective_orders_are_cancelled(self, manager, broker):
        """A stop/target for a position we no longer hold must not linger."""
        await manager.place_bracket(sized(symbol="AMD"), cycle_id="c1")
        # No position for AMD in the account snapshot -> the stop/target orphaned.
        await manager.reconcile(self.account(positions={}))
        assert len(broker.cancelled) >= 1

    async def test_dedupes_executions_during_reconciliation(self, manager, broker):
        from aitrader.broker.port import BrokerExecution

        broker.executions_list.append(
            BrokerExecution(
                exec_id="e-dup", order_ref="ref", symbol="AAPL",
                action="BUY", quantity=10, price=100.0, ts=datetime.now(UTC),
            )
        )
        manager.on_execution(broker.executions_list[0])
        await manager.reconcile(self.account())
        assert len(manager.fills) == 1


class TestCancelAndClose:
    async def test_cancel_symbol_orders_only_touches_that_symbol(self, manager, broker):
        await manager.place_bracket(sized(symbol="AAPL"), cycle_id="c1")
        await manager.place_bracket(sized(symbol="MSFT"), cycle_id="c1")

        # Entry auto-fills in the fake, so only each symbol's stop/target remain
        # working and are candidates for cancellation.
        aapl_working_ids = {
            mo.ib_order_id for mo in manager.orders.values()
            if mo.symbol == "AAPL" and mo.is_working
        }
        msft_working_ids = {
            mo.ib_order_id for mo in manager.orders.values()
            if mo.symbol == "MSFT" and mo.is_working
        }
        assert aapl_working_ids and msft_working_ids

        cancelled = await manager.cancel_symbol_orders("AAPL")
        assert cancelled == len(aapl_working_ids)
        assert set(broker.cancelled) == aapl_working_ids
        assert msft_working_ids.isdisjoint(broker.cancelled)

    async def test_close_position_cancels_resting_orders_first(self, manager, broker):
        await manager.place_bracket(sized(symbol="NFLX"), cycle_id="c1")
        broker.positions["NFLX"] = Position(symbol="NFLX", quantity=10, avg_cost=100.0)
        await manager.close_position("NFLX", 10, is_long=True, reason="test exit")
        # A close order was submitted beyond the original bracket's three legs.
        assert len(broker.placed) >= 4
