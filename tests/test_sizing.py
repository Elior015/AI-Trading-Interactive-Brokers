"""Position sizing.

The invariant that matters: no combination of inputs may produce a position
whose stop-loss risk exceeds the configured fraction of equity.
"""

from __future__ import annotations

import pytest

from aitrader.config import RiskConfig
from aitrader.domain.enums import Action
from aitrader.domain.models import AccountSnapshot, FeaturePack, Quote
from aitrader.domain.proposals import TradeProposal
from aitrader.risk.sizing import compute_stop_distance, conviction_scalar, size_proposal


def account(equity: float = 100_000.0, buying_power: float | None = None) -> AccountSnapshot:
    return AccountSnapshot(
        account_id="DU123",
        equity=equity,
        cash=equity,
        buying_power=buying_power if buying_power is not None else equity * 2,
    )


def features(price: float = 100.0, atr: float = 1.0, avg_volume: float = 5_000_000) -> FeaturePack:
    return FeaturePack(
        symbol="TEST", price=price, atr=atr, atr_pct=atr / price * 100, avg_volume=avg_volume
    )


def proposal(
    action: Action = Action.BUY, conviction: float = 1.0,
    stop_mult: float = 1.5, target_r: float = 2.0,
) -> TradeProposal:
    return TradeProposal(
        symbol="TEST", action=action, conviction=conviction,
        horizon_minutes=60, stop_atr_multiple=stop_mult, target_r_multiple=target_r,
    )


class TestConvictionScalar:
    def test_bounded_to_half_and_one(self):
        """Conviction modulates size within a band; it can never uncap it."""
        assert conviction_scalar(0.0) == pytest.approx(0.5)
        assert conviction_scalar(1.0) == pytest.approx(1.0)
        assert conviction_scalar(0.5) == pytest.approx(0.75)

    @pytest.mark.parametrize("bad", [-5.0, 99.0, float("inf")])
    def test_out_of_range_still_bounded(self, bad):
        assert 0.5 <= conviction_scalar(bad) <= 1.0


class TestStopDistance:
    def test_uses_atr_multiple(self):
        d = compute_stop_distance(proposal(stop_mult=2.0), features(price=100, atr=1.5))
        assert d == pytest.approx(3.0)

    def test_floored_so_it_cannot_collapse(self):
        """A near-zero stop would make the sizer produce an enormous position."""
        d = compute_stop_distance(proposal(stop_mult=0.5), features(price=100, atr=0.0001))
        assert d >= 100 * 0.0015

    def test_never_zero(self):
        d = compute_stop_distance(proposal(stop_mult=0.5), features(price=0.5, atr=0.0))
        assert d > 0


class TestSizeProposal:
    def test_basic_long(self):
        cfg = RiskConfig(max_risk_per_trade_pct=0.01)
        order = size_proposal(
            proposal(), features(price=100, atr=1.0), account(100_000),
            Quote(symbol="TEST", bid=99.95, ask=100.05), cfg,
        )
        assert order is not None
        assert order.action == Action.BUY
        assert order.stop_price < order.entry_price < order.target_price
        assert order.quantity > 0

    def test_short_bracket_is_inverted(self):
        cfg = RiskConfig(max_risk_per_trade_pct=0.01, allow_shorts=True)
        order = size_proposal(
            proposal(action=Action.SELL), features(price=100, atr=1.0), account(),
            Quote(symbol="TEST", bid=99.95, ask=100.05), cfg,
        )
        assert order is not None
        assert order.target_price < order.entry_price < order.stop_price

    def test_target_respects_r_multiple(self):
        cfg = RiskConfig(max_risk_per_trade_pct=0.01)
        order = size_proposal(
            proposal(stop_mult=1.0, target_r=3.0), features(price=100, atr=2.0),
            account(), Quote(symbol="TEST", bid=99.95, ask=100.05), cfg,
        )
        assert order is not None
        risk = order.entry_price - order.stop_price
        reward = order.target_price - order.entry_price
        assert reward / risk == pytest.approx(3.0, rel=0.05)

    def test_hold_is_not_sized(self):
        cfg = RiskConfig()
        assert size_proposal(
            proposal(action=Action.HOLD), features(), account(), None, cfg
        ) is None

    def test_zero_equity_is_not_sized(self):
        cfg = RiskConfig()
        assert size_proposal(proposal(), features(), account(equity=0), None, cfg) is None

    def test_tiny_account_produces_no_order_rather_than_a_bad_one(self):
        cfg = RiskConfig(max_risk_per_trade_pct=0.005)
        order = size_proposal(
            proposal(), features(price=500.0, atr=10.0), account(equity=100.0), None, cfg
        )
        assert order is None

    def test_notional_cap_binds(self):
        # Large risk allowance, small notional cap: the cap must win.
        cfg = RiskConfig(max_risk_per_trade_pct=0.02, max_position_notional_pct=0.05)
        acct = account(1_000_000)
        order = size_proposal(proposal(), features(price=100, atr=0.10), acct, None, cfg)
        assert order is not None
        assert order.notional <= acct.equity * 0.05 * 1.01

    def test_buying_power_cap_binds(self):
        cfg = RiskConfig(max_risk_per_trade_pct=0.02, buying_power_reserve_pct=0.5)
        acct = account(equity=100_000, buying_power=10_000)
        order = size_proposal(proposal(), features(price=100, atr=0.5), acct, None, cfg)
        if order is not None:
            assert order.notional <= 10_000 * 0.5 * 1.01

    def test_liquidity_cap_binds(self):
        """Never take a meaningful share of the symbol's daily volume."""
        cfg = RiskConfig(max_risk_per_trade_pct=0.02)
        order = size_proposal(
            proposal(), features(price=10, atr=0.05, avg_volume=10_000),
            account(1_000_000), None, cfg,
        )
        if order is not None:
            assert order.quantity <= 10_000 * 0.005 + 1

    def test_uses_live_quote_over_stale_feature_price(self):
        cfg = RiskConfig()
        order = size_proposal(
            proposal(), features(price=100.0, atr=1.0), account(),
            Quote(symbol="TEST", bid=109.9, ask=110.1), cfg,
        )
        assert order is not None
        assert order.entry_price > 105  # priced from the quote, not the stale feature


class TestRiskInvariant:
    """The property that must hold for every input combination."""

    @pytest.mark.parametrize("equity", [5_000, 50_000, 250_000, 2_000_000])
    @pytest.mark.parametrize("price", [3.0, 42.5, 180.0, 900.0])
    @pytest.mark.parametrize("atr", [0.01, 0.5, 5.0, 40.0])
    @pytest.mark.parametrize("conviction", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("stop_mult", [0.5, 1.5, 5.0])
    def test_risk_never_exceeds_configured_fraction(
        self, equity, price, atr, conviction, stop_mult
    ):
        cfg = RiskConfig(max_risk_per_trade_pct=0.01, min_risk_per_trade_pct=0.001)
        order = size_proposal(
            proposal(conviction=conviction, stop_mult=stop_mult),
            features(price=price, atr=atr),
            account(equity),
            None,
            cfg,
        )
        if order is None:
            return
        risk = abs(order.entry_price - order.stop_price) * order.quantity
        # One share of tick-rounding tolerance.
        tolerance = abs(order.entry_price - order.stop_price)
        assert risk <= equity * cfg.max_risk_per_trade_pct + tolerance, (
            f"risk {risk:.2f} exceeded cap {equity * cfg.max_risk_per_trade_pct:.2f}"
        )

    @pytest.mark.parametrize("conviction", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_higher_conviction_never_shrinks_size(self, conviction):
        cfg = RiskConfig()
        base = size_proposal(
            proposal(conviction=0.0), features(price=100, atr=1.0), account(), None, cfg
        )
        test = size_proposal(
            proposal(conviction=conviction), features(price=100, atr=1.0), account(), None, cfg
        )
        assert base is not None and test is not None
        assert test.quantity >= base.quantity

    def test_quantity_is_always_a_whole_number(self):
        cfg = RiskConfig()
        order = size_proposal(
            proposal(), features(price=137.77, atr=2.31), account(88_888), None, cfg
        )
        assert order is not None
        assert isinstance(order.quantity, int)
        assert order.quantity == int(order.quantity)
