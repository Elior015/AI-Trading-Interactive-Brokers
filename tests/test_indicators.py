"""Indicator correctness.

Every value here feeds a trade decision, so each indicator is checked against
hand-computed expectations rather than against another library's output.
"""

from __future__ import annotations

import numpy as np
import pytest

from aitrader.analytics import indicators as ind


class TestSMA:
    def test_simple_average(self):
        out = ind.sma([1, 2, 3, 4, 5], 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert out[2] == pytest.approx(2.0)
        assert out[3] == pytest.approx(3.0)
        assert out[4] == pytest.approx(4.0)

    def test_warmup_is_nan_not_zero(self):
        """A zero-seeded warm-up looks like a real value and would mislead."""
        out = ind.sma([10, 20, 30], 3)
        assert np.isnan(out[:2]).all()

    def test_too_short(self):
        assert np.isnan(ind.sma([1, 2], 5)).all()


class TestEMA:
    def test_seeded_with_sma(self):
        out = ind.ema([1, 2, 3, 4, 5], 3)
        assert out[2] == pytest.approx(2.0)  # SMA of 1,2,3
        # alpha = 2/(3+1) = 0.5
        assert out[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)  # 3.0
        assert out[4] == pytest.approx(0.5 * 5 + 0.5 * 3.0)  # 4.0

    def test_constant_series_converges_to_constant(self):
        out = ind.ema([7.0] * 30, 10)
        assert out[-1] == pytest.approx(7.0)


class TestWilderSmooth:
    def test_uses_one_over_period_not_two_over_period_plus_one(self):
        """Wilder's alpha is 1/n. Using EMA's 2/(n+1) breaks RSI and ATR."""
        out = ind.wilder_smooth([1, 2, 3, 4], 3)
        assert out[2] == pytest.approx(2.0)  # seed = mean(1,2,3)
        assert out[3] == pytest.approx((2.0 * 2 + 4) / 3)


class TestRSI:
    def test_all_gains_is_100(self):
        out = ind.rsi(list(range(1, 40)), 14)
        assert out[-1] == pytest.approx(100.0)

    def test_all_losses_is_zero(self):
        out = ind.rsi(list(range(40, 1, -1)), 14)
        assert out[-1] == pytest.approx(0.0)

    def test_bounded_zero_to_hundred(self):
        rng = np.random.default_rng(7)
        series = 100 + np.cumsum(rng.normal(0, 1, 300))
        out = ind.rsi(series, 14)
        finite = out[np.isfinite(out)]
        assert finite.size > 0
        assert finite.min() >= 0.0 and finite.max() <= 100.0

    def test_flat_series_is_neutral(self):
        out = ind.rsi([50.0] * 40, 14)
        assert out[-1] == pytest.approx(50.0)


class TestTrueRangeAndATR:
    def test_true_range_takes_the_largest_of_three(self):
        highs = [10, 12, 11]
        lows = [8, 9, 7]
        closes = [9, 11, 8]
        tr = ind.true_range(highs, lows, closes)
        assert tr[0] == pytest.approx(2.0)          # first bar: high - low
        assert tr[1] == pytest.approx(3.0)          # max(3, |12-9|, |9-9|)
        assert tr[2] == pytest.approx(4.0)          # max(4, |11-11|, |7-11|)

    def test_atr_is_positive_for_real_data(self):
        rng = np.random.default_rng(3)
        closes = 100 + np.cumsum(rng.normal(0, 0.5, 100))
        highs, lows = closes + 0.5, closes - 0.5
        out = ind.atr(highs, lows, closes, 14)
        assert ind.last_valid(out) > 0


class TestMACD:
    def test_histogram_is_line_minus_signal(self):
        rng = np.random.default_rng(11)
        closes = 100 + np.cumsum(rng.normal(0, 1, 200))
        line, sig, hist = ind.macd(closes)
        valid = np.isfinite(line) & np.isfinite(sig)
        assert valid.any()
        assert np.allclose(hist[valid], line[valid] - sig[valid])

    def test_rising_series_has_positive_macd(self):
        line, _, _ = ind.macd(list(range(1, 200)))
        assert ind.last_valid(line) > 0


class TestBollinger:
    def test_bands_straddle_the_middle(self):
        rng = np.random.default_rng(5)
        closes = 100 + rng.normal(0, 2, 100)
        up, mid, low = ind.bollinger(closes, 20, 2.0)
        i = -1
        assert up[i] > mid[i] > low[i]

    def test_zero_volatility_collapses_the_bands(self):
        up, _mid, low = ind.bollinger([42.0] * 40, 20, 2.0)
        assert up[-1] == pytest.approx(42.0)
        assert low[-1] == pytest.approx(42.0)


class TestVWAP:
    def test_single_bar_equals_typical_price(self):
        out = ind.vwap([10], [8], [9], [1000])
        assert out[0] == pytest.approx(9.0)

    def test_weights_by_volume(self):
        # Two bars: typical prices 10 and 20, volumes 1 and 3.
        out = ind.vwap([10, 20], [10, 20], [10, 20], [1, 3])
        assert out[-1] == pytest.approx((10 * 1 + 20 * 3) / 4)

    def test_zero_volume_does_not_divide_by_zero(self):
        out = ind.vwap([10, 11], [10, 11], [10, 11], [0, 0])
        assert np.isnan(out).all()


class TestRelativeVolume:
    def test_double_the_average_is_two(self):
        assert ind.relative_volume([100] * 20 + [200]) == pytest.approx(2.0)

    def test_insufficient_history_is_neutral(self):
        """Must return 1.0, never something that reads as a signal."""
        assert ind.relative_volume([500]) == pytest.approx(1.0)
        assert ind.relative_volume([]) == pytest.approx(1.0)


class TestOpeningRange:
    def test_uses_only_the_first_n_bars(self):
        hi, lo = ind.opening_range([10, 12, 20], [8, 9, 5], 2)
        assert hi == pytest.approx(12.0)
        assert lo == pytest.approx(8.0)

    def test_position_within_range(self):
        assert ind.opening_range_position(10.0, 12.0, 8.0) == pytest.approx(0.5)
        assert ind.opening_range_position(14.0, 12.0, 8.0) == pytest.approx(1.5)  # breakout
        assert ind.opening_range_position(6.0, 12.0, 8.0) == pytest.approx(-0.5)  # breakdown

    def test_degenerate_range_returns_none(self):
        assert ind.opening_range_position(10.0, 10.0, 10.0) is None
        assert ind.opening_range_position(10.0, None, None) is None


class TestLastValid:
    def test_skips_trailing_nans(self):
        assert ind.last_valid(np.array([1.0, 2.0, np.nan])) == pytest.approx(2.0)

    def test_all_nan_returns_default(self):
        assert ind.last_valid(np.array([np.nan, np.nan]), default=5.0) == pytest.approx(5.0)

    def test_empty_returns_default(self):
        assert ind.last_valid(np.array([]), default=3.0) == pytest.approx(3.0)
