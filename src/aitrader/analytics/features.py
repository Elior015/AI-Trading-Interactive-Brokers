"""Turning bars into the numeric view the model reasons over.

The model is never asked to do arithmetic on prices. It receives these values
already computed, and `FEATURE_REGISTRY` is the allow-list of keys it is
permitted to cite as evidence — a cheap and effective check on invented
justifications, and the hook where untrusted inputs would be constrained if news
ingestion is added later.
"""

from __future__ import annotations

import numpy as np

from ..domain.models import Bar, FeaturePack, Quote, utcnow
from ..logging_setup import get_logger
from . import indicators as ind

log = get_logger(__name__)

#: Every feature key the model may reference in `evidence`. Anything else is
#: rejected at validation time.
FEATURE_REGISTRY: dict[str, str] = {
    "price": "Last traded price",
    "change_pct": "Percent change from the previous session close",
    "gap_pct": "Percent gap between today's open and yesterday's close",
    "rvol": "Relative volume versus the recent average",
    "atr": "Average True Range (14)",
    "atr_pct": "ATR as a percentage of price",
    "vwap": "Session volume-weighted average price",
    "vwap_distance_atr": "Signed distance from VWAP measured in ATRs",
    "rsi": "Relative Strength Index (14)",
    "ema_fast": "9-period EMA",
    "ema_slow": "21-period EMA",
    "ema_trend": "+1 when the fast EMA is above the slow EMA, -1 below",
    "macd": "MACD line (12/26)",
    "macd_signal": "MACD signal line (9)",
    "macd_hist": "MACD histogram",
    "bb_upper": "Upper Bollinger band (20, 2)",
    "bb_lower": "Lower Bollinger band (20, 2)",
    "opening_range_position": "Position within the opening range (0=low, 1=high)",
    "day_high": "Session high",
    "day_low": "Session low",
    "avg_volume": "Average daily volume",
    "spread_pct": "Bid/ask spread as a fraction of mid",
    "rank_score": "Deterministic pre-model ranking score",
}

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14


def _arrays(bars: list[Bar]) -> dict[str, np.ndarray]:
    return {
        "open": np.array([b.open for b in bars], dtype=float),
        "high": np.array([b.high for b in bars], dtype=float),
        "low": np.array([b.low for b in bars], dtype=float),
        "close": np.array([b.close for b in bars], dtype=float),
        "volume": np.array([b.volume for b in bars], dtype=float),
    }


def build_feature_pack(
    symbol: str,
    intraday: list[Bar],
    daily: list[Bar] | None = None,
    quote: Quote | None = None,
    opening_range_bars: int = 3,
) -> FeaturePack | None:
    """Compute every feature for one symbol.

    Returns None when there is not enough history to compute anything
    trustworthy — a half-warm indicator is worse than no indicator, because it
    looks like a signal.
    """
    if len(intraday) < EMA_SLOW:
        return None

    a = _arrays(intraday)
    closes, highs, lows, volumes = a["close"], a["high"], a["low"], a["volume"]

    price = float(quote.mid) if quote and quote.is_usable and quote.mid else float(closes[-1])
    if price <= 0:
        return None

    atr_arr = ind.atr(highs, lows, closes, ATR_PERIOD)
    atr_val = ind.last_valid(atr_arr, 0.0)
    # A zero ATR would make every stop distance zero, so fall back to a small
    # fraction of price rather than emitting a degenerate value.
    if atr_val <= 0:
        atr_val = max(price * 0.005, 0.01)

    vwap_arr = ind.vwap(highs, lows, closes, volumes)
    vwap_val = ind.last_valid(vwap_arr, price)
    ema_f = ind.last_valid(ind.ema(closes, EMA_FAST), price)
    ema_s = ind.last_valid(ind.ema(closes, EMA_SLOW), price)
    macd_line, macd_sig, _ = ind.macd(closes)
    bb_u, _, bb_l = ind.bollinger(closes)

    prev_close = float(closes[0])
    day_open = float(a["open"][0])
    avg_volume = 0.0
    if daily and len(daily) >= 2:
        prev_close = float(daily[-1].close)
        dv = np.array([b.volume for b in daily], dtype=float)
        dv = dv[dv > 0]
        avg_volume = float(dv.mean()) if dv.size else 0.0

    or_high, or_low = ind.opening_range(highs, lows, opening_range_bars)

    return FeaturePack(
        symbol=symbol,
        price=price,
        change_pct=((price - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0,
        gap_pct=((day_open - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0,
        rvol=ind.relative_volume(volumes),
        atr=atr_val,
        atr_pct=atr_val / price * 100.0,
        vwap=vwap_val,
        vwap_distance_atr=(price - vwap_val) / atr_val if atr_val > 0 else 0.0,
        rsi=ind.last_valid(ind.rsi(closes, RSI_PERIOD), 50.0),
        ema_fast=ema_f,
        ema_slow=ema_s,
        ema_trend=1 if ema_f > ema_s else (-1 if ema_f < ema_s else 0),
        macd=ind.last_valid(macd_line, 0.0),
        macd_signal=ind.last_valid(macd_sig, 0.0),
        bb_upper=ind.last_valid(bb_u, price),
        bb_lower=ind.last_valid(bb_l, price),
        opening_range_position=ind.opening_range_position(price, or_high, or_low),
        day_high=float(highs.max()),
        day_low=float(lows.min()),
        avg_volume=avg_volume,
        spread_pct=quote.spread_pct if quote else None,
        computed_at=utcnow(),
    )


def build_all(
    intraday: dict[str, list[Bar]],
    daily: dict[str, list[Bar]] | None = None,
    quotes: dict[str, Quote] | None = None,
    opening_range_bars: int = 3,
) -> dict[str, FeaturePack]:
    """Compute feature packs for every symbol that has enough data."""
    out: dict[str, FeaturePack] = {}
    daily = daily or {}
    quotes = quotes or {}
    for symbol, bars in intraday.items():
        try:
            pack = build_feature_pack(
                symbol, bars, daily.get(symbol), quotes.get(symbol), opening_range_bars
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the cycle
            log.warning("feature_build_failed", symbol=symbol, error=str(exc))
            continue
        if pack is not None:
            out[symbol] = pack
    return out


def validate_evidence(evidence: list[str]) -> list[str]:
    """Return the evidence entries that reference no known feature key.

    Used to flag (not reject) proposals whose justification cites nothing we
    actually supplied.
    """
    unknown: list[str] = []
    for item in evidence:
        lowered = item.lower()
        if not any(key in lowered for key in FEATURE_REGISTRY):
            unknown.append(item)
    return unknown
