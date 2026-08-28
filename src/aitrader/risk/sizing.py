"""Turning a proposal into a concrete order size.

This is where the model's abstract risk preference (an ATR multiple, an R
multiple) becomes actual prices and a share count. Doing it here rather than in
the model is the boundary that stops a hallucinated number from becoming a
hallucinated trade.

Conviction may only *modulate* size within a configured band. It can never
uncap it: a model that is confidently wrong should not be able to bet more than
a model that is tentatively wrong.
"""

from __future__ import annotations

import math

from ..config import RiskConfig
from ..domain.enums import Action
from ..domain.models import AccountSnapshot, FeaturePack, Quote
from ..domain.proposals import SizedOrder, TradeProposal
from ..logging_setup import get_logger

log = get_logger(__name__)

#: Below this the trade is not worth the commission.
MIN_SHARES = 1


def conviction_scalar(conviction: float) -> float:
    """Map conviction onto [0.5, 1.0] of the configured risk budget.

    Deliberately narrow. The difference between a 0.6 and a 0.95 conviction is
    not reliable enough to justify a 60% swing in position size.
    """
    return max(0.5, min(1.0, 0.5 + 0.5 * conviction))


def compute_stop_distance(proposal: TradeProposal, features: FeaturePack) -> float:
    """Stop distance in dollars, floored so it can never collapse to zero.

    A degenerate stop would make the sizer divide by something near zero and
    produce an enormous position, so the floors here are load-bearing.
    """
    atr_stop = proposal.stop_atr_multiple * features.atr
    #: Never risk less than 0.15% of price — tighter than that and normal noise
    #: stops you out immediately.
    min_stop = features.price * 0.0015
    return max(atr_stop, min_stop, 0.01)


def round_to_tick(price: float, tick: float = 0.01) -> float:
    return round(round(price / tick) * tick, 2)


def size_proposal(
    proposal: TradeProposal,
    features: FeaturePack,
    account: AccountSnapshot,
    quote: Quote | None,
    cfg: RiskConfig,
    limit_offset_pct: float = 0.001,
) -> SizedOrder | None:
    """Produce a fully-specified order, or None if it cannot be sized sensibly.

    Returning None is a normal outcome — an equity too small to take the trade,
    a stop too wide, a price we cannot determine.
    """
    if proposal.action not in (Action.BUY, Action.SELL):
        return None
    if account.equity <= 0:
        log.warning("sizing_skipped_no_equity", symbol=proposal.symbol)
        return None

    # Prefer the live quote; fall back to the last computed price.
    price = features.price
    if quote is not None and quote.is_usable and quote.mid:
        price = float(quote.mid)
    if price <= 0:
        return None

    is_long = proposal.action == Action.BUY

    # Marketable limit entry. Never a bare market order: a market order against
    # a stale or thin book is the fastest way to lose money in this design.
    offset = price * limit_offset_pct
    entry = round_to_tick(price + offset if is_long else price - offset)

    stop_distance = compute_stop_distance(proposal, features)
    stop = round_to_tick(entry - stop_distance if is_long else entry + stop_distance)
    target_distance = stop_distance * proposal.target_r_multiple
    target = round_to_tick(entry + target_distance if is_long else entry - target_distance)

    # A bracket whose stop is on the wrong side of entry would be rejected by
    # IBKR anyway, but catching it here gives a clearer diagnostic.
    if is_long and not (stop < entry < target):
        log.warning("sizing_invalid_bracket", symbol=proposal.symbol, entry=entry, stop=stop, target=target)
        return None
    if not is_long and not (target < entry < stop):
        log.warning("sizing_invalid_bracket", symbol=proposal.symbol, entry=entry, stop=stop, target=target)
        return None

    risk_budget = account.equity * cfg.max_risk_per_trade_pct * conviction_scalar(proposal.conviction)
    risk_budget = max(risk_budget, account.equity * cfg.min_risk_per_trade_pct)

    actual_stop_distance = abs(entry - stop)
    if actual_stop_distance <= 0:
        return None

    shares = math.floor(risk_budget / actual_stop_distance)

    # Clamp by notional exposure.
    max_notional = account.equity * cfg.max_position_notional_pct
    shares = min(shares, math.floor(max_notional / entry) if entry > 0 else 0)

    # Clamp by buying power, keeping a reserve.
    usable_bp = account.buying_power * (1.0 - cfg.buying_power_reserve_pct)
    shares = min(shares, math.floor(usable_bp / entry) if entry > 0 else 0)

    # Clamp so we never take a meaningful share of the symbol's daily volume.
    if features.avg_volume > 0:
        shares = min(shares, math.floor(features.avg_volume * 0.005))

    if shares < MIN_SHARES:
        log.info(
            "sizing_below_minimum",
            symbol=proposal.symbol,
            shares=shares,
            equity=round(account.equity, 2),
            stop_distance=round(actual_stop_distance, 4),
        )
        return None

    return SizedOrder(
        proposal=proposal,
        symbol=proposal.symbol,
        action=proposal.action,
        quantity=int(shares),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        risk_amount=shares * actual_stop_distance,
        notional=shares * entry,
    )
