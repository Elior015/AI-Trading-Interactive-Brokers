"""A full decision cycle, end to end, over test doubles.

No IB Gateway, no network, no real LLM. This is what proves the pieces fit
together: a proposal from a scripted model becomes a real bracket order through
sizing and the risk gate, a restart cannot duplicate that order, and a tripped
kill switch stops new ones without touching what is already resting at the
broker.
"""

from __future__ import annotations

from aitrader.agents.roles import AgentRunner
from aitrader.analytics.ranking import FocusListManager
from aitrader.broker.market_data import MarketDataService
from aitrader.broker.orders import OrderManager
from aitrader.config import Secrets, Settings, StrategyConfig
from aitrader.data.store import BarStore, StateStore
from aitrader.domain.enums import SessionPhase
from aitrader.domain.models import Quote
from aitrader.engine.calendar import MarketCalendar, SessionInfo
from aitrader.engine.cycle import DecisionCycle
from aitrader.engine.state import AppState
from aitrader.llm.audit import AuditLog
from aitrader.llm.base import LLMResponse, ModelCapabilities
from aitrader.llm.gateway import LLMGateway
from aitrader.llm.narrative import SessionNarrative
from aitrader.risk.engine import RiskEngine
from aitrader.risk.killswitch import KillSwitch
from tests.fakes.fake_broker import FakeBroker, make_bars

BUY_AAPL = """{
  "market_read": "AAPL showing unusual relative volume with a clean break above VWAP.",
  "proposals": [
    {
      "symbol": "AAPL",
      "action": "BUY",
      "conviction": 0.9,
      "horizon_minutes": 30,
      "stop_atr_multiple": 1.5,
      "target_r_multiple": 2.0,
      "rationale": "Breaking the opening range on strong relative volume.",
      "evidence": ["rvol 3.2x average", "vwap_distance_atr +1.1"]
    }
  ],
  "watching": [],
  "notes_for_next_cycle": ""
}"""

NO_TRADE = '{"market_read": "nothing actionable", "proposals": [], "watching": [], "notes_for_next_cycle": ""}'


class ScriptedProvider:
    """Routes canned JSON replies by which agent role's system prompt asked."""

    name = "scripted"
    host = "fake://scripted"

    def __init__(self, trader_reply: str = NO_TRADE, risk_reply: str = '{"approve": true}'):
        self.trader_reply = trader_reply
        self.risk_reply = risk_reply
        self.calls: list[str] = []

    async def chat(self, req):
        system = req.messages[0]["content"]
        self.calls.append(system[:40])
        if "trading desk" in system:
            content = self.trader_reply
        elif "risk officer" in system:
            content = self.risk_reply
        else:
            content = "{}"
        return LLMResponse(content=content, model=req.model)

    async def show(self, model):
        return ModelCapabilities(model=model, capabilities=["tools"])

    async def healthy(self):
        return True


def build_settings() -> Settings:
    settings = Settings(secrets=Secrets(), strategy=StrategyConfig())
    settings.strategy.risk.require_risk_officer = False
    return settings


def build_rig(tmp_path, trader_reply: str = NO_TRADE, risk_reply: str = '{"approve": true}'):
    """Wire up one full stack of test doubles, mirroring TradingEngine.__post_init__
    but with a FakeBroker and a scripted LLM instead of the real adapter."""
    settings = build_settings()

    store = StateStore(tmp_path / "state.sqlite3")
    bar_store = BarStore(tmp_path / "bars.duckdb")
    broker = FakeBroker(equity=100_000.0, cash=100_000.0, buying_power=200_000.0)

    kill_switch = KillSwitch(sentinel=tmp_path / "KILL", store=store)
    order_manager = OrderManager(broker=broker, store=store, limit_offset_pct=0.001)
    risk = RiskEngine(
        cfg=settings.strategy.risk, kill_switch=kill_switch, store=store,
        order_manager=order_manager,
    )

    provider = ScriptedProvider(trader_reply=trader_reply, risk_reply=risk_reply)
    llm = LLMGateway(
        provider=provider, audit=AuditLog(tmp_path / "audit", enabled=False),
        store=store, max_concurrent=1,
    )
    narrative = SessionNarrative(store=store)
    agents = AgentRunner(gateway=llm, cfg=settings.strategy.llm, narrative=narrative)

    market_data = MarketDataService(
        broker=broker, bar_store=bar_store,
        universe_cfg=settings.strategy.universe, broker_cfg=settings.strategy.broker,
        data_cfg=settings.strategy.data,
    )

    focus_manager = FocusListManager(size=5, max_promotions_per_cycle=10, persistence_cycles=1)
    calendar = MarketCalendar()

    state = AppState()
    state.starting_equity = broker.equity
    state.peak_equity = broker.equity
    state.session_description = "RTH test session"

    cycle = DecisionCycle(
        settings=settings, state=state, market_data=market_data, agents=agents,
        risk=risk, orders=order_manager, calendar=calendar, focus_manager=focus_manager,
    )

    return {
        "settings": settings, "store": store, "bar_store": bar_store, "broker": broker,
        "kill_switch": kill_switch, "order_manager": order_manager, "risk": risk,
        "provider": provider, "market_data": market_data, "cycle": cycle, "state": state,
    }


def seed_symbol(rig, symbol: str = "AAPL", start_price: float = 100.0) -> None:
    """Pre-populate a symbol as if it had already been backfilled and streamed."""
    bars = make_bars(symbol, count=60, start_price=start_price, trend=0.05)
    rig["market_data"].intraday[symbol] = bars
    rig["market_data"].daily[symbol] = make_bars(symbol, count=20, start_price=start_price * 0.98)
    rig["market_data"].universe = [symbol]
    rig["broker"].quotes[symbol] = Quote(
        symbol=symbol, bid=start_price + 2.9, ask=start_price + 3.1, last=start_price + 3.0,
        volume=2_000_000,
    )


async def account_provider(rig):
    return await rig["broker"].account_snapshot()


def rth_session(minutes_to_close: float = 180.0) -> SessionInfo:
    return SessionInfo(
        phase=SessionPhase.RTH, is_trading_day=True, open_at=None, close_at=None,
        minutes_to_open=-90.0, minutes_to_close=minutes_to_close,
    )


class TestFullCycleProducesAnOrder:
    async def test_buy_proposal_becomes_a_real_bracket(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=BUY_AAPL)
        seed_symbol(rig)

        record = await rig["cycle"].run(rth_session(), lambda: account_provider(rig))

        assert record.error == ""
        assert record.proposals == 1
        assert record.approved == 1, record.rejections
        assert record.rejected == 0

        broker = rig["broker"]
        assert len(broker.placed) == 3  # entry + stop + target
        entry, stop, target = broker.placed
        assert entry.symbol == "AAPL" and entry.action == "BUY"
        assert stop.order_type == "STP"
        assert target.order_type == "LMT"

    async def test_order_is_write_ahead_persisted(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=BUY_AAPL)
        seed_symbol(rig)
        await rig["cycle"].run(rth_session(), lambda: account_provider(rig))

        rows = rig["store"].load_orders()
        assert any(r["symbol"] == "AAPL" for r in rows)

    async def test_no_proposals_places_nothing(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=NO_TRADE)
        seed_symbol(rig)
        record = await rig["cycle"].run(rth_session(), lambda: account_provider(rig))
        assert record.proposals == 0
        assert rig["broker"].placed == []

    async def test_narrative_records_the_placed_trade(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=BUY_AAPL)
        seed_symbol(rig)
        await rig["cycle"].run(rth_session(), lambda: account_provider(rig))
        combined = " ".join(e.content for e in rig["cycle"].agents.narrative.entries)
        assert "AAPL" in combined


class TestKillSwitchStopsNewEntriesWithoutTouchingExisting:
    async def test_tripped_kill_switch_blocks_the_next_cycle(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=NO_TRADE)
        seed_symbol(rig)
        # First cycle, clean: establishes a baseline with nothing placed.
        await rig["cycle"].run(rth_session(), lambda: account_provider(rig))
        assert rig["broker"].placed == []

        rig["kill_switch"].trip("integration test halt")
        rig["provider"].trader_reply = BUY_AAPL  # now the model wants to trade

        record = await rig["cycle"].run(rth_session(), lambda: account_provider(rig))
        assert record.approved == 0
        assert record.rejected == 1
        assert rig["broker"].placed == []  # nothing reached the broker

    async def test_kill_switch_does_not_cancel_a_position_already_placed(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=BUY_AAPL)
        seed_symbol(rig)
        await rig["cycle"].run(rth_session(), lambda: account_provider(rig))
        assert len(rig["broker"].placed) == 3

        rig["kill_switch"].trip("halt after position opened")
        rig["provider"].trader_reply = NO_TRADE
        await rig["cycle"].run(rth_session(), lambda: account_provider(rig))

        # The halt-mode kill switch must not have cancelled the resting bracket.
        assert len(rig["broker"].placed) == 3
        assert len(rig["broker"].cancelled) == 0


class TestRestartCannotDuplicateAnOrder:
    async def test_reload_and_reconcile_then_replace_the_same_cycle_id_is_a_noop(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=BUY_AAPL)
        seed_symbol(rig)
        record = await rig["cycle"].run(rth_session(), lambda: account_provider(rig))
        assert record.approved == 1
        placed_before = len(rig["broker"].placed)

        # Simulate a process restart: a brand new OrderManager, same durable
        # store, same broker (which represents IBKR's memory surviving the
        # restart even though ours did not).
        restarted = OrderManager(broker=rig["broker"], store=rig["store"], limit_offset_pct=0.001)
        restarted.load_from_store()
        assert any("AAPL" in ref for ref in restarted.orders)

        report = await restarted.reconcile(await account_provider(rig))
        assert report.orphaned_local == []
        assert report.unprotected_positions == []

        # Attempting to place the identical trade again (same cycle_id,
        # symbol, kind) must be recognized as the order we already have.
        from aitrader.domain.enums import Action
        from aitrader.domain.proposals import TradeProposal
        from aitrader.risk.sizing import size_proposal

        account = await account_provider(rig)
        features = rig["state"].features["AAPL"]
        proposal = TradeProposal(
            symbol="AAPL", action=Action.BUY, conviction=0.9, horizon_minutes=30,
        )
        sized = size_proposal(proposal, features, account, rig["broker"].get_quote("AAPL"), rig["settings"].strategy.risk)
        assert sized is not None

        # Reuse the cycle_id from the very first placement.
        first_cycle_id = rig["state"].cycles[0].cycle_id
        result = await restarted.place_bracket(sized, cycle_id=first_cycle_id)
        assert result == [] or len(result) <= 3
        assert len(rig["broker"].placed) == placed_before  # nothing new was sent


class TestUniverseVsPacing:
    async def test_backfill_respects_the_historical_pacer_and_persists_bars(self, tmp_path):
        """The mechanism that makes a 100+ symbol universe survive IBKR's
        60-requests-per-10-minutes limit: every symbol goes through the pacer,
        and results land in the durable bar store rather than being re-fetched."""
        rig = build_rig(tmp_path)
        symbols = [f"SYM{i}" for i in range(12)]
        for s in symbols:
            rig["broker"].bars[s] = make_bars(s, count=25, start_price=50.0)

        rig["market_data"].pacer.min_spacing = 0.001  # keep the test fast
        loaded = await rig["market_data"].backfill_universe(symbols)

        assert loaded == len(symbols)
        assert rig["broker"].historical_calls == len(symbols)
        row_count = await rig["bar_store"].row_count()
        assert row_count > 0

    async def test_focus_list_never_exceeds_the_configured_size(self, tmp_path):
        rig = build_rig(tmp_path, trader_reply=NO_TRADE)
        symbols = [f"SYM{i}" for i in range(30)]
        for i, s in enumerate(symbols):
            seed_symbol(rig, symbol=s, start_price=50.0 + i)
        rig["market_data"].universe = symbols

        await rig["cycle"].run(rth_session(), lambda: account_provider(rig))
        assert len(rig["state"].focus) <= rig["cycle"].focus_manager.size
