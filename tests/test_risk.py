"""The risk gate.

One table-driven test per check, covering both the accept and reject path, plus
the structural property that the gate is the only route to the broker.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aitrader.config import RiskConfig
from aitrader.domain.enums import Action, KillSwitchAction, RejectReason, SessionPhase
from aitrader.domain.models import AccountSnapshot, Position, Quote
from aitrader.domain.proposals import SizedOrder, TradeProposal
from aitrader.risk.engine import RiskContext, RiskEngine
from aitrader.risk.killswitch import KillSwitch


@pytest.fixture
def kill(tmp_path):
    return KillSwitch(sentinel=tmp_path / "KILL")


@pytest.fixture
def cfg():
    return RiskConfig()


@pytest.fixture
def engine(cfg, kill):
    return RiskEngine(cfg=cfg, kill_switch=kill)


def account(equity=100_000.0, positions=None, buying_power=None, account_id="DU123", age=0.0):
    return AccountSnapshot(
        account_id=account_id,
        equity=equity,
        cash=equity,
        buying_power=buying_power if buying_power is not None else equity * 2,
        positions=positions or {},
        as_of=datetime.now(UTC) - timedelta(seconds=age),
    )


def order(symbol="TEST", action=Action.BUY, qty=100, entry=100.0, stop=99.0, target=102.0):
    return SizedOrder(
        proposal=TradeProposal(
            symbol=symbol, action=action, conviction=0.8, horizon_minutes=60
        ),
        symbol=symbol,
        action=action,
        quantity=qty,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        risk_amount=abs(entry - stop) * qty,
        notional=entry * qty,
    )


def ctx(**overrides):
    base = {
        "account": account(),
        "quote": Quote(symbol="TEST", bid=99.95, ask=100.05, last=100.0),
        "phase": SessionPhase.RTH,
        "minutes_to_close": 180.0,
        "is_live": False,
        "starting_equity": 100_000.0,
    }
    base.update(overrides)
    return RiskContext(**base)


class TestHappyPath:
    def test_clean_order_is_approved(self, engine):
        assert engine.evaluate(order(), ctx()).approved


class TestKillSwitch:
    def test_tripped_switch_blocks(self, engine, kill):
        kill.trip("manual")
        v = engine.evaluate(order(), ctx())
        assert not v.approved and v.reason == RejectReason.KILL_SWITCH

    def test_sentinel_file_blocks_without_api_call(self, tmp_path, cfg):
        """`touch data/KILL` from any shell must halt the system."""
        sentinel = tmp_path / "KILL"
        ks = KillSwitch(sentinel=sentinel)
        eng = RiskEngine(cfg=cfg, kill_switch=ks)
        assert eng.evaluate(order(), ctx()).approved
        sentinel.write_text("stop now")
        assert not eng.evaluate(order(), ctx()).approved

    def test_survives_restart(self, tmp_path):
        sentinel = tmp_path / "KILL"
        KillSwitch(sentinel=sentinel).trip("before restart")
        revived = KillSwitch(sentinel=sentinel)
        assert revived.is_tripped
        assert "before restart" in revived.reason

    def test_reset_clears_it(self, tmp_path):
        ks = KillSwitch(sentinel=tmp_path / "KILL")
        ks.trip("x")
        ks.reset()
        assert not ks.is_tripped

    def test_flatten_mode(self, tmp_path):
        ks = KillSwitch(sentinel=tmp_path / "KILL", action=KillSwitchAction.FLATTEN_ALL)
        ks.trip("flatten")
        assert ks.should_flatten


class TestTradingModeInterlock:
    def test_live_mode_against_paper_account_blocked(self, engine):
        v = engine.evaluate(order(), ctx(is_live=True, account=account(account_id="DU999")))
        assert not v.approved and v.reason == RejectReason.TRADING_MODE_INTERLOCK

    def test_paper_mode_against_live_account_blocked(self, engine):
        """The dangerous direction: believing we are on paper while wired to live."""
        v = engine.evaluate(order(), ctx(is_live=False, account=account(account_id="U7654321")))
        assert not v.approved and v.reason == RejectReason.TRADING_MODE_INTERLOCK

    def test_live_mode_against_live_account_allowed(self, engine):
        v = engine.evaluate(order(), ctx(is_live=True, account=account(account_id="U7654321")))
        assert v.approved


class TestSessionChecks:
    @pytest.mark.parametrize(
        "phase", [SessionPhase.CLOSED, SessionPhase.PREMARKET, SessionPhase.AFTER_HOURS]
    )
    def test_untradeable_phases_blocked(self, engine, phase):
        v = engine.evaluate(order(), ctx(phase=phase))
        assert not v.approved and v.reason == RejectReason.MARKET_CLOSED

    def test_too_close_to_the_close(self, engine):
        v = engine.evaluate(order(), ctx(minutes_to_close=5.0))
        assert not v.approved and v.reason == RejectReason.TOO_LATE_IN_SESSION


class TestFreshness:
    def test_stale_account_snapshot_blocked(self, engine):
        """Acting on a stale view of our own state is the worst failure mode."""
        v = engine.evaluate(order(), ctx(account=account(age=600)))
        assert not v.approved and v.reason == RejectReason.STALE_ACCOUNT_STATE

    def test_stale_decision_blocked(self, engine):
        v = engine.evaluate(order(), ctx(decision_age_seconds=999))
        assert not v.approved

    def test_stale_quote_blocked(self, engine):
        stale = Quote(
            symbol="TEST", bid=99.9, ask=100.1,
            updated_at=datetime.now(UTC) - timedelta(seconds=300),
        )
        v = engine.evaluate(order(), ctx(quote=stale))
        assert not v.approved


class TestMarketData:
    def test_missing_quote_blocked(self, engine):
        """No delayed fallback exists for US equities, so this fails loudly."""
        v = engine.evaluate(order(), ctx(quote=None))
        assert not v.approved and v.reason == RejectReason.NO_MARKET_DATA

    def test_halted_contract_blocked(self, engine):
        q = Quote(symbol="TEST", bid=99.9, ask=100.1, halted=True)
        v = engine.evaluate(order(), ctx(quote=q))
        assert not v.approved and v.reason == RejectReason.HALTED

    def test_wide_spread_blocked(self, engine):
        q = Quote(symbol="TEST", bid=95.0, ask=105.0)
        v = engine.evaluate(order(), ctx(quote=q))
        assert not v.approved


class TestPriceAndSize:
    def test_fat_finger_limit_blocked(self, engine):
        v = engine.evaluate(order(entry=150.0, stop=148.0, target=155.0), ctx())
        assert not v.approved and v.reason == RejectReason.PRICE_COLLAR

    def test_zero_quantity_blocked(self, engine):
        v = engine.evaluate(order(qty=0), ctx())
        assert not v.approved and v.reason == RejectReason.ZERO_QUANTITY

    def test_penny_stock_blocked(self, engine):
        # Tight spread so the price-band check is the one that fires.
        q = Quote(symbol="TEST", bid=0.999, ask=1.001)
        v = engine.evaluate(order(entry=1.0, stop=0.99, target=1.02, qty=10), ctx(quote=q))
        assert not v.approved and v.reason == RejectReason.ILLIQUID

    def test_illiquid_symbol_blocked(self, engine):
        v = engine.evaluate(order(), ctx(avg_volume=1000))
        assert not v.approved and v.reason == RejectReason.ILLIQUID

    def test_excess_per_trade_risk_blocked(self, engine):
        """The gate recomputes risk independently, so a sizer bug cannot leak."""
        v = engine.evaluate(order(qty=5000, entry=100.0, stop=90.0), ctx())
        assert not v.approved
        assert v.reason in (
            RejectReason.PER_TRADE_RISK_EXCEEDED,
            RejectReason.POSITION_NOTIONAL_EXCEEDED,
            RejectReason.INSUFFICIENT_BUYING_POWER,
        )

    def test_excess_notional_blocked(self, engine):
        v = engine.evaluate(order(qty=2000, entry=100.0, stop=99.9), ctx())
        assert not v.approved

    def test_insufficient_buying_power_blocked(self, engine):
        v = engine.evaluate(order(qty=100, entry=100.0), ctx(account=account(buying_power=1000)))
        assert not v.approved and v.reason == RejectReason.INSUFFICIENT_BUYING_POWER


class TestPortfolioLimits:
    def test_max_concurrent_positions(self, engine, cfg):
        positions = {
            f"S{i}": Position(symbol=f"S{i}", quantity=10, avg_cost=100.0)
            for i in range(cfg.max_concurrent_positions)
        }
        v = engine.evaluate(order(), ctx(account=account(positions=positions)))
        assert not v.approved and v.reason == RejectReason.MAX_POSITIONS_REACHED

    def test_max_entries_per_cycle(self, engine, cfg):
        v = engine.evaluate(order(), ctx(entries_this_cycle=cfg.max_new_entries_per_cycle))
        assert not v.approved and v.reason == RejectReason.MAX_ENTRIES_PER_CYCLE

    def test_daily_trade_cap(self, engine, cfg):
        v = engine.evaluate(order(), ctx(trades_today=cfg.max_trades_per_day))
        assert not v.approved

    def test_daily_loss_limit(self, engine):
        # Small enough that no notional or buying-power check fires first.
        v = engine.evaluate(
            order(qty=20), ctx(account=account(equity=95_000), starting_equity=100_000)
        )
        assert not v.approved and v.reason == RejectReason.DAILY_LOSS_LIMIT

    def test_daily_loss_check_trips_the_kill_switch(self, engine, kill):
        tripped = engine.check_daily_loss(account(equity=90_000), starting_equity=100_000)
        assert tripped and kill.is_tripped

    def test_daily_loss_check_passes_when_within_limit(self, engine, kill):
        assert not engine.check_daily_loss(account(equity=99_500), starting_equity=100_000)
        assert not kill.is_tripped


class TestDuplicatesAndTiming:
    def test_working_order_blocks_a_second(self, engine):
        v = engine.evaluate(order(), ctx(working_symbols={"TEST"}))
        assert not v.approved and v.reason == RejectReason.DUPLICATE_ORDER

    def test_adding_to_an_existing_long_blocked(self, engine):
        positions = {"TEST": Position(symbol="TEST", quantity=100, avg_cost=99.0)}
        v = engine.evaluate(order(), ctx(account=account(positions=positions)))
        assert not v.approved and v.reason == RejectReason.DUPLICATE_ORDER

    def test_cooldown_blocks_rapid_reentry(self, engine):
        v = engine.evaluate(order(), ctx(last_entry={"TEST": time.monotonic()}))
        assert not v.approved and v.reason == RejectReason.COOLDOWN

    def test_flip_flop_guard(self, engine):
        v = engine.evaluate(order(), ctx(last_close={"TEST": time.monotonic()}))
        assert not v.approved and v.reason == RejectReason.FLIP_FLOP_GUARD

    def test_order_rate_circuit_breaker(self, engine, cfg):
        """The runaway-bug breaker: stops a loop regardless of other checks."""
        for _ in range(cfg.max_orders_per_minute):
            engine._record_order()
        v = engine.evaluate(order(), ctx())
        assert not v.approved


class TestShortsAndRiskOfficer:
    def test_shorts_disabled_by_default(self, engine):
        v = engine.evaluate(order(action=Action.SELL, stop=101.0, target=98.0), ctx())
        assert not v.approved

    def test_shorts_allowed_when_configured(self, kill):
        eng = RiskEngine(cfg=RiskConfig(allow_shorts=True), kill_switch=kill)
        v = eng.evaluate(order(action=Action.SELL, stop=101.0, target=98.0), ctx())
        assert v.approved

    def test_risk_officer_can_veto(self, engine):
        v = engine.evaluate(
            order(), ctx(risk_officer_approved=False, risk_officer_comment="too extended")
        )
        assert not v.approved and v.reason == RejectReason.RISK_OFFICER_VETO

    def test_risk_officer_cannot_authorize_what_the_gate_rejects(self, engine, kill):
        """Advisory means veto-only. Approval never overrides a deterministic check."""
        kill.trip("halt")
        v = engine.evaluate(order(), ctx(risk_officer_approved=True))
        assert not v.approved


class TestHardLimitClamping:
    def test_config_cannot_loosen_a_hard_limit(self):
        """A decimal-point mistake gets clamped, not honoured."""
        cfg = RiskConfig(max_risk_per_trade_pct=0.05, max_position_notional_pct=0.9)
        from aitrader import hard_limits

        assert cfg.max_risk_per_trade_pct <= hard_limits.ABS_MAX_RISK_PER_TRADE_PCT
        assert cfg.max_position_notional_pct <= hard_limits.ABS_MAX_POSITION_NOTIONAL_PCT
        assert cfg.clamp_warnings

    def test_config_may_tighten(self):
        cfg = RiskConfig(max_risk_per_trade_pct=0.001, min_risk_per_trade_pct=0.0005)
        assert cfg.max_risk_per_trade_pct == pytest.approx(0.001)
        assert not cfg.clamp_warnings

    def test_min_risk_above_max_is_rejected_outright(self):
        """A contradictory pair is a config error, not something to clamp."""
        with pytest.raises(ValidationError):
            RiskConfig(max_risk_per_trade_pct=0.001, min_risk_per_trade_pct=0.01)


class TestSubmitPath:
    async def test_rejected_order_never_reaches_the_broker(self, cfg, kill):
        class Recorder:
            def __init__(self):
                self.calls = 0

            async def place_bracket(self, order, cycle_id=""):
                self.calls += 1

        rec = Recorder()
        eng = RiskEngine(cfg=cfg, kill_switch=kill, order_manager=rec)
        kill.trip("blocked")
        verdict = await eng.submit(order(), ctx(), cycle_id="c1")
        assert not verdict.approved
        assert rec.calls == 0

    async def test_approved_order_reaches_the_broker(self, cfg, kill):
        class Recorder:
            def __init__(self):
                self.calls = 0

            async def place_bracket(self, order, cycle_id=""):
                self.calls += 1

        rec = Recorder()
        eng = RiskEngine(cfg=cfg, kill_switch=kill, order_manager=rec)
        verdict = await eng.submit(order(), ctx(), cycle_id="c1")
        assert verdict.approved
        assert rec.calls == 1

    async def test_kill_switch_tripped_after_evaluation_still_blocks(self, cfg, kill):
        """The last-look check, for an order that sat in a queue."""

        class TrippingRecorder:
            def __init__(self):
                self.calls = 0

            async def place_bracket(self, order, cycle_id=""):
                self.calls += 1

        rec = TrippingRecorder()
        eng = RiskEngine(cfg=cfg, kill_switch=kill, order_manager=rec)

        original = eng.evaluate

        def evaluate_then_trip(o, c):
            verdict = original(o, c)
            kill.trip("tripped mid-flight")
            return verdict

        eng.evaluate = evaluate_then_trip  # type: ignore[method-assign]
        verdict = await eng.submit(order(), ctx(), cycle_id="c1")
        assert not verdict.approved
        assert rec.calls == 0
