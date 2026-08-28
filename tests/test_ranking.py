"""Deterministic candidate scoring and focus-list hysteresis.

The hysteresis is what stops the focus list thrashing against IBKR's limit of
60 new real-time-bar subscriptions per 10 minutes: a challenger must clearly
beat the weakest incumbent, and hold that advantage for more than one cycle,
before it displaces anyone.
"""

from __future__ import annotations

import pytest

from aitrader.analytics.ranking import (
    FocusListManager,
    rank_candidates,
    scanner_rank_scores,
    score_symbol,
)
from aitrader.domain.models import FeaturePack


def fp(symbol="X", **kw) -> FeaturePack:
    base = {
        "symbol": symbol, "price": 100.0, "rvol": 1.0, "atr_pct": 1.5, "vwap_distance_atr": 0.0,
        "rsi": 50.0, "ema_trend": 0, "macd": 0.0, "macd_signal": 0.0, "avg_volume": 2_000_000,
        "spread_pct": 0.001, "gap_pct": 0.0,
    }
    base.update(kw)
    return FeaturePack(**base)


class TestScoreSymbol:
    def test_high_relative_volume_scores_higher(self):
        quiet = score_symbol(fp(rvol=1.0))
        busy = score_symbol(fp(rvol=4.0))
        assert busy > quiet

    def test_extreme_volatility_is_penalized(self):
        normal = score_symbol(fp(atr_pct=2.0))
        wild = score_symbol(fp(atr_pct=15.0))
        assert wild < normal

    def test_wide_spread_is_heavily_penalized(self):
        tight = score_symbol(fp(spread_pct=0.001))
        wide = score_symbol(fp(spread_pct=0.02))
        assert wide < tight - 2.0

    def test_illiquid_is_penalized(self):
        liquid = score_symbol(fp(avg_volume=5_000_000))
        illiquid = score_symbol(fp(avg_volume=100_000))
        assert illiquid < liquid

    def test_scanner_presence_adds_score(self):
        base = score_symbol(fp(), scanner_rank=0.0)
        boosted = score_symbol(fp(), scanner_rank=2.0)
        assert boosted > base

    def test_direction_agnostic(self):
        """A big move down should score similarly to a big move up — direction
        is the model's call, not the ranker's."""
        up = score_symbol(fp(vwap_distance_atr=2.5, rsi=80))
        down = score_symbol(fp(vwap_distance_atr=-2.5, rsi=20))
        assert up == pytest.approx(down)


class TestRankCandidates:
    def test_orders_by_score_descending(self):
        features = {
            "QUIET": fp("QUIET", rvol=1.0),
            "HOT": fp("HOT", rvol=5.0),
            "MID": fp("MID", rvol=2.0),
        }
        ranked = rank_candidates(features, limit=10)
        assert ranked == ["HOT", "MID", "QUIET"]

    def test_respects_limit(self):
        features = {f"S{i}": fp(f"S{i}", rvol=float(i)) for i in range(30)}
        ranked = rank_candidates(features, limit=5)
        assert len(ranked) == 5


class TestScannerRankScores:
    def test_top_of_a_scan_scores_higher_than_bottom(self):
        scores = scanner_rank_scores({"MOST_ACTIVE": ["AAPL", "MSFT", "GOOG", "AMZN"]})
        assert scores["AAPL"] > scores["AMZN"]

    def test_appearing_in_multiple_scans_compounds(self):
        scores = scanner_rank_scores({
            "MOST_ACTIVE": ["AAPL", "MSFT"],
            "TOP_PERC_GAIN": ["AAPL", "TSLA"],
        })
        assert scores["AAPL"] > scores["MSFT"]
        assert scores["AAPL"] > scores["TSLA"]

    def test_empty_scan_does_not_crash(self):
        assert scanner_rank_scores({"EMPTY": []}) == {}


class TestFocusListHysteresis:
    def test_fills_empty_slots_immediately(self):
        mgr = FocusListManager(size=3, max_promotions_per_cycle=10)
        features = {f"S{i}": fp(f"S{i}", rvol=float(i + 1)) for i in range(5)}
        focus, added, _removed = mgr.update(features)
        assert len(focus) == 3
        assert set(added) == set(focus)

    def test_weak_challenger_does_not_displace_a_strong_incumbent(self):
        mgr = FocusListManager(size=1, churn_margin=0.5, persistence_cycles=1)
        strong = {"STRONG": fp("STRONG", rvol=5.0)}
        mgr.update(strong)
        assert mgr.current == ["STRONG"]

        weak_challenger = {"STRONG": fp("STRONG", rvol=5.0), "WEAK": fp("WEAK", rvol=5.1)}
        focus, added, _removed = mgr.update(weak_challenger)
        # 5.1 does not beat 5.0 by the required 50% margin.
        assert focus == ["STRONG"]
        assert added == []

    def test_persistent_strong_challenger_eventually_displaces(self):
        mgr = FocusListManager(size=1, churn_margin=0.1, persistence_cycles=2)
        mgr.update({"INCUMBENT": fp("INCUMBENT", rvol=1.0)})

        challenge = {"INCUMBENT": fp("INCUMBENT", rvol=1.0), "CHALLENGER": fp("CHALLENGER", rvol=5.0)}
        focus1, _added1, _removed1 = mgr.update(challenge)
        assert focus1 == ["INCUMBENT"]  # first cycle: not persistent yet

        focus2, added2, removed2 = mgr.update(challenge)
        assert focus2 == ["CHALLENGER"]
        assert "CHALLENGER" in added2
        assert "INCUMBENT" in removed2

    def test_protected_symbols_are_never_displaced(self):
        """A symbol with an open position must never lose its data feed."""
        mgr = FocusListManager(size=1, churn_margin=0.0, persistence_cycles=1)
        mgr.update({"HELD": fp("HELD", rvol=0.1)})

        strong_challenger = {"HELD": fp("HELD", rvol=0.1), "HOT": fp("HOT", rvol=10.0)}
        focus, _added, _removed = mgr.update(strong_challenger, protected={"HELD"})
        assert "HELD" in focus
        assert "HOT" not in focus  # no free slots, HELD is protected

    def test_max_promotions_per_cycle_caps_churn(self):
        mgr = FocusListManager(size=10, max_promotions_per_cycle=2)
        features = {f"S{i}": fp(f"S{i}", rvol=float(i)) for i in range(20)}
        _focus, added, _removed = mgr.update(features)
        assert len(added) <= 2

    def test_protected_symbol_with_no_features_is_still_kept(self):
        """A held position must not vanish from the focus list just because its
        features could not be computed this cycle (e.g. a data gap)."""
        mgr = FocusListManager(size=2)
        mgr.update({"HELD": fp("HELD"), "OTHER": fp("OTHER")})
        focus, _added, _removed = mgr.update({"OTHER": fp("OTHER")}, protected={"HELD"})
        assert "HELD" in focus
