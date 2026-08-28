"""Technical indicators, implemented directly rather than pulled from a library.

Three reasons this is worth the ~200 lines: `pandas-ta` has publicly flagged
maintenance funding problems, TA-Lib needs a C toolchain that complicates the
Docker build, and — most importantly — every number here feeds a trade decision,
so each one is unit-tested against hand-computed values in
`tests/test_indicators.py`.

All functions take and return numpy arrays, are pure, and return `nan` for
positions inside the warm-up window rather than silently seeding with zeros.
"""

from __future__ import annotations

import numpy as np


def _as_array(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.ravel()


def sma(values: object, period: int) -> np.ndarray:
    """Simple moving average."""
    x = _as_array(values)
    out = np.full(x.size, np.nan)
    if period <= 0 or x.size < period:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def ema(values: object, period: int) -> np.ndarray:
    """Exponential moving average, seeded with the SMA of the first `period` values.

    Seeding from the SMA (rather than the first observation) is the convention
    charting packages use; matching it means our EMA agrees with what you see in
    TWS.
    """
    x = _as_array(values)
    out = np.full(x.size, np.nan)
    if period <= 0 or x.size < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = x[:period].mean()
    for i in range(period, x.size):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def wilder_smooth(values: object, period: int) -> np.ndarray:
    """Wilder's smoothing, used by RSI and ATR.

    This is an EMA with alpha = 1/period, not 2/(period+1). Using the wrong one
    is the classic source of RSI values that disagree with every chart.
    """
    x = _as_array(values)
    out = np.full(x.size, np.nan)
    if period <= 0 or x.size < period:
        return out
    out[period - 1] = x[:period].mean()
    for i in range(period, x.size):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def rsi(closes: object, period: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder)."""
    c = _as_array(closes)
    out = np.full(c.size, np.nan)
    if c.size < period + 1:
        return out
    delta = np.diff(c)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.divide(avg_gain, avg_loss)
        vals = 100.0 - (100.0 / (1.0 + rs))
    # A window with no losses is RSI 100 by definition, not a division error.
    vals = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, vals)
    vals = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, vals)
    out[1:] = vals
    return out


def true_range(highs: object, lows: object, closes: object) -> np.ndarray:
    h, low, c = _as_array(highs), _as_array(lows), _as_array(closes)
    n = min(h.size, low.size, c.size)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    out[0] = h[0] - low[0]
    if n > 1:
        prev = c[:-1]
        out[1:] = np.maximum.reduce(
            [h[1:] - low[1:], np.abs(h[1:] - prev), np.abs(low[1:] - prev)]
        )
    return out


def atr(highs: object, lows: object, closes: object, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder). The basis for every stop distance we set."""
    tr = true_range(highs, lows, closes)
    return wilder_smooth(tr, period)


def macd(
    closes: object, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (macd_line, signal_line, histogram)."""
    c = _as_array(closes)
    fast_e, slow_e = ema(c, fast), ema(c, slow)
    line = fast_e - slow_e
    valid = ~np.isnan(line)
    sig = np.full(c.size, np.nan)
    if valid.any():
        first = int(np.argmax(valid))
        sig_vals = ema(line[first:], signal)
        sig[first:] = sig_vals
    return line, sig, line - sig


def bollinger(
    closes: object, period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (upper, middle, lower)."""
    c = _as_array(closes)
    mid = sma(c, period)
    std = np.full(c.size, np.nan)
    for i in range(period - 1, c.size):
        std[i] = c[i - period + 1 : i + 1].std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def vwap(highs: object, lows: object, closes: object, volumes: object) -> np.ndarray:
    """Session-anchored VWAP using the typical price.

    Callers must pass only the current session's bars — VWAP that runs across a
    session boundary is meaningless, and this function cannot detect that for you.
    """
    h, low, c, v = _as_array(highs), _as_array(lows), _as_array(closes), _as_array(volumes)
    n = min(h.size, low.size, c.size, v.size)
    if n == 0:
        return np.array([])
    typical = (h[:n] + low[:n] + c[:n]) / 3.0
    vol = np.where(v[:n] > 0, v[:n], 0.0)
    cum_pv = np.cumsum(typical * vol)
    cum_v = np.cumsum(vol)
    out = np.full(n, np.nan)
    nz = cum_v > 0
    out[nz] = cum_pv[nz] / cum_v[nz]
    return out


def relative_volume(volumes: object, lookback: int = 20) -> float:
    """Today's volume so far versus the average of the previous `lookback` values.

    Returns 1.0 when there is not enough history, which is the neutral value —
    it must never look like a signal.
    """
    v = _as_array(volumes)
    if v.size < 2:
        return 1.0
    current = float(v[-1])
    prior = v[max(0, v.size - 1 - lookback) : -1]
    prior = prior[prior > 0]
    if prior.size == 0:
        return 1.0
    avg = float(prior.mean())
    return current / avg if avg > 0 else 1.0


def opening_range(
    highs: object, lows: object, bars_in_range: int
) -> tuple[float, float] | tuple[None, None]:
    """High and low of the first `bars_in_range` bars of the session."""
    h, low = _as_array(highs), _as_array(lows)
    if h.size < bars_in_range or bars_in_range <= 0:
        return None, None
    return float(h[:bars_in_range].max()), float(low[:bars_in_range].min())


def opening_range_position(price: float, or_high: float | None, or_low: float | None) -> float | None:
    """Where price sits in the opening range: 0 = low, 1 = high, >1 = breakout."""
    if or_high is None or or_low is None:
        return None
    span = or_high - or_low
    if span <= 0:
        return None
    return (price - or_low) / span


def last_valid(arr: np.ndarray, default: float = 0.0) -> float:
    """Most recent non-nan value, or `default`."""
    if arr is None or len(arr) == 0:
        return default
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if finite.size else default
