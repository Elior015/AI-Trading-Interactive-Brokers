"""The auto/manual execution-mode branch inside the decision cycle.

Proves two things: auto mode's behavior is byte-for-byte unchanged from
before this feature existed (a trade still reaches the broker directly), and
manual mode holds every trade -- entries and closes -- for a human instead of
ever calling the broker.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from aitrader.broker.orders import OrderManager
from aitrader.config import RiskConfig
from aitrader.data.store import StateStore
from aitrader.domain.enums import Action, ExecutionMode, SessionPhase
from aitrader.domain.models import AccountSnapshot, FeaturePack, Position, Quote
from aitrader.domain.proposals import CycleDecision, TradeProposal
from aitrader.engine.calendar import SessionInfo
from aitrader.engine.cycle import DecisionCycle
from aitrader.engine.state import AppState, CycleRecord
from aitrader.llm.narrative import SessionNarrative
from aitrader.risk.engine import RiskEngine
from aitrader.risk.killswitch import KillSwitch
from tests.fakes.fake_broker import FakeBroker


def _account(equity=100_000.0, positions=None):
    return AccountSnapshot(
        account_id="DU1234567", equity=equity, cash=equity,
        buying_power=equity * 2, positions=positions or {}, as_of=datetime.now(UTC),
    )


def _features(symbol="AAPL", price=100.0):
    return FeaturePack(symbol=symbol, price=price, atr=1.0)


def _session():
    return SessionInfo(
        phase=SessionPhase.RTH, is_trading_day=True, open_at=None, close_at=None,
        minutes_to_open=0.0, minutes_to_close=180.0,
    )


class FakeMarketData:
    def __init__(self, quotes: dict[str, Quote] | None = None):
        self._quotes = quotes or {}

    def quotes(self, symbols):
        return {s: self._quotes[s] for s in symbols if s in self._quotes}


@pytest.fixture
def store(tmp_path):
    s = StateStore(tmp_path / "test.sqlite3")
    yield s
    s.close()


@pytest.fixture
def broker():
    return FakeBroker(connected=True)


@pytest.fixture
def orders(broker, store):
    return OrderManager(broker=broker, store=store)


@pytest.fixture
def risk(tmp_path, store, orders):
    return RiskEngine(
        cfg=RiskConfig(require_risk_officer=False),
        kill_switch=KillSwitch(sentinel=tmp_path / "KILL", store=store),
        store=store,
        order_manager=orders,
    )


@pytest.fixture
def cycle(risk, orders, store):
    settings = SimpleNamespace(
        strategy=SimpleNamespace(
            risk=risk.cfg,
            cadence=SimpleNamespace(opening_range_minutes=15),
            broker=SimpleNamespace(limit_offset_pct=0.001),
        ),
        is_live=False,
    )
    state = AppState()
    state.features = {"AAPL": _features()}
    state.account = _account()
    state.starting_equity = 100_000.0
    quote = Quote(symbol="AAPL", bid=99.95, ask=100.05, last=100.0, volume=1_000_000)
    agents = SimpleNamespace(narrative=SessionNarrative(store=store))
    return DecisionCycle(
        settings=settings,
        state=state,
        market_data=FakeMarketData({"AAPL": quote}),
        agents=agents,
        risk=risk,
        orders=orders,
        calendar=SimpleNamespace(),
        focus_manager=SimpleNamespace(),
    )


def _decision(action=Action.BUY):
    proposal = TradeProposal(
        symbol="AAPL", action=action, conviction=0.8, horizon_minutes=60,
        rationale="strong momentum",
    )
    return CycleDecision(proposals=[proposal], cycle_id="c1")


class TestAutoModeUnchanged:
    """Locks in today's behavior: auto mode must keep reaching the broker
    directly, exactly as it did before manual mode existed."""

    async def test_entry_reaches_the_broker(self, cycle, broker):
        cycle.state.execution_mode = ExecutionMode.AUTO
        record = CycleRecord(cycle_id="c1", started_at=datetime.now(UTC))
        await cycle._execute(_decision(), cycle.state.account, _session(), record, "c1")

        assert broker.placed, "auto mode must place the order with the broker"
        assert record.approved == 1
        assert record.pending == 0
        assert cycle.state.pending_approvals == []

    async def test_close_reaches_the_broker(self, cycle, broker):
        cycle.state.execution_mode = ExecutionMode.AUTO
        position = Position(symbol="AAPL", quantity=10, avg_cost=100.0, market_price=101.0)
        account = _account(positions={"AAPL": position})
        cycle.state.account = account
        record = CycleRecord(cycle_id="c1", started_at=datetime.now(UTC))

        await cycle._handle_close("AAPL", account, _session(), record, "c1")

        assert broker.placed, "auto mode must close the position at the broker"
        assert record.approved == 1
        assert record.pending == 0
        assert cycle.state.pending_approvals == []


class TestManualModeHoldsForApproval:
    """The whole point of the feature: nothing reaches the broker until a
    person says yes."""

    async def test_entry_is_queued_not_placed(self, cycle, broker):
        cycle.state.execution_mode = ExecutionMode.MANUAL
        record = CycleRecord(cycle_id="c1", started_at=datetime.now(UTC))

        await cycle._execute(_decision(), cycle.state.account, _session(), record, "c1")

        assert broker.placed == [], "manual mode must never reach the broker on its own"
        assert record.approved == 0
        assert record.pending == 1
        assert len(cycle.state.pending_approvals) == 1
        approval = cycle.state.pending_approvals[0]
        assert approval.kind == "entry"
        assert approval.symbol == "AAPL"
        assert approval.sized is not None
        assert approval.sized.proposal.rationale == "strong momentum"

    async def test_close_is_queued_not_placed(self, cycle, broker):
        cycle.state.execution_mode = ExecutionMode.MANUAL
        position = Position(symbol="AAPL", quantity=10, avg_cost=100.0, market_price=101.0)
        account = _account(positions={"AAPL": position})
        cycle.state.account = account
        record = CycleRecord(cycle_id="c1", started_at=datetime.now(UTC))

        await cycle._handle_close("AAPL", account, _session(), record, "c1")

        assert broker.placed == [], "manual mode must never close a position on its own"
        assert record.approved == 0
        assert record.pending == 1
        assert len(cycle.state.pending_approvals) == 1
        approval = cycle.state.pending_approvals[0]
        assert approval.kind == "close"
        assert approval.symbol == "AAPL"
        assert approval.close_quantity == 10

    async def test_low_conviction_is_still_rejected_before_queuing(self, cycle, broker):
        """Manual mode holds trades that already passed every other check --
        it does not relax the checks that come before it."""
        cycle.state.execution_mode = ExecutionMode.MANUAL
        decision = CycleDecision(
            proposals=[
                TradeProposal(symbol="AAPL", action=Action.BUY, conviction=0.1, horizon_minutes=60)
            ],
            cycle_id="c1",
        )
        record = CycleRecord(cycle_id="c1", started_at=datetime.now(UTC))

        await cycle._execute(decision, cycle.state.account, _session(), record, "c1")

        assert broker.placed == []
        assert cycle.state.pending_approvals == []
        assert record.rejected == 1
        assert record.pending == 0
