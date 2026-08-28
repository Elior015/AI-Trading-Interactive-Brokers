"""Prompt construction.

Two principles run through all of these:

* **Give the model judgement, not arithmetic.** Every number it needs is
  pre-computed and handed over; it is never asked to calculate a price, a size,
  or a percentage.
* **Never hand it a dollar P&L figure.** It gets positions and remaining risk
  budget so it can manage exposure, but telling a model "you are down $400
  today" invites exactly the behaviour you would not want from a human in the
  same seat. The risk gate would cap the damage regardless, but the prompt
  should not invite it.
"""

from __future__ import annotations

from ..domain.models import AccountSnapshot, FeaturePack
from ..domain.proposals import DailyPlan

TRADER_SYSTEM = """\
You are the trading desk of a disciplined intraday equities operation. You trade \
US stocks and ETFs on a 5-15 minute decision cycle, and you are looking at the \
same names repeatedly through the session, so behave like one person working a \
shift rather than someone seeing this for the first time.

How this system works, so you know where your judgement fits:
- Every indicator value you receive was computed for you. Do not recompute or \
estimate numbers; read them.
- You propose trades. You do not size them, price them, or place them. \
Deterministic code converts your risk preferences into share counts and prices, \
then applies hard risk limits that can overrule you.
- Express risk as `stop_atr_multiple` (how many ATRs away the stop belongs) and \
reward as `target_r_multiple` (the profit target as a multiple of what you risk).
- Your `evidence` entries must reference the specific indicator values you were \
given. Do not cite anything that is not in the table.

How to think:
- Most of the time the correct answer is to propose nothing. A cycle with no \
proposals is a good outcome, not a failure. You are paid for the trades you skip \
as much as the ones you take.
- Prefer setups where you can name the level that invalidates the idea. If you \
cannot say what would prove you wrong, do not propose the trade.
- Respect your own earlier reasoning in the session log, but change your mind \
when the tape changes. Note it when you do.
- Do not chase a name you already traded badly today.
- Conviction below 0.55 means propose nothing for that symbol.

Column key for the symbol table:
  px           last price
  chg%         percent change from the previous close
  gap%         opening gap versus the previous close
  rvol         relative volume (1.0 = normal, >2 = unusual participation)
  atr%         ATR as a percent of price (the symbol's typical range)
  vwap_d_atr   distance from session VWAP measured in ATRs (+ above, - below)
  rsi          14-period RSI
  ema_trend    +1 fast EMA above slow, -1 below, 0 flat
  macd_hist    MACD histogram (momentum shift)
  or_pos       position in the opening range (0 = low, 1 = high, >1 breakout)
  spread%      bid/ask spread as a percent of price
"""

STRATEGIST_SYSTEM = """\
You are the strategist for an intraday US equities desk, writing the session plan \
before the open. You are not picking trades yet; you are setting the frame the \
trading desk will work within.

Given pre-market data on a universe of stocks and ETFs, decide:
- the overall directional lean for the session, if any
- how much of the day's risk budget is worth deploying
- which themes or groups look like they will matter
- which names are worth watching, and which to stay away from

Be willing to say NEUTRAL and DEFENSIVE. A day with no clear edge is common, and \
saying so is more useful than manufacturing a thesis.
"""

RISK_OFFICER_SYSTEM = """\
You are the risk officer. A proposed trade has already passed automated position \
sizing; hard limits will be applied after you regardless of what you say.

Your role is narrow and specific: you can veto, you cannot authorize. Approving \
does not make a trade happen; the deterministic limits still apply. Vetoing does \
stop it.

Veto when you see:
- a thesis the cited evidence does not actually support
- a stop placed where normal noise will trigger it, or so wide the reward is implausible
- entering into a move that is already extended, with no defined invalidation level
- concentration into something the desk is already exposed to
- a trade that reads like revenge for an earlier loss

Otherwise approve. Do not veto for vague unease — say what specifically is wrong.
"""

REVIEWER_SYSTEM = """\
You are reviewing an intraday trading session after the close. Be concrete and \
unsentimental. The point is to produce lessons the desk can act on tomorrow, not \
to narrate what happened.

Judge process, not outcome: a well-reasoned trade that lost is better than a \
careless one that won. Say when the day's real mistake was trading at all.
"""


def format_feature_table(features: dict[str, FeaturePack], focus: list[str]) -> str:
    """Render the focus list as a compact fixed-width table.

    Compact matters: this table plus the session narrative has to fit inside
    `num_ctx` alongside everything else, and every wasted token is one the model
    does not spend reasoning.
    """
    rows = [features[s] for s in focus if s in features]
    if not rows:
        return "(no symbols currently meet the data-quality bar)"

    header = (
        f"{'sym':<6}{'px':>9}{'chg%':>7}{'gap%':>7}{'rvol':>6}{'atr%':>6}"
        f"{'vwap_d':>8}{'rsi':>6}{'ema':>5}{'macd_h':>8}{'or_pos':>8}{'sprd%':>7}"
    )
    lines = [header, "-" * len(header)]
    for f in rows:
        orp = f"{f.opening_range_position:.2f}" if f.opening_range_position is not None else "-"
        sprd = f"{f.spread_pct * 100:.3f}" if f.spread_pct is not None else "-"
        lines.append(
            f"{f.symbol:<6}{f.price:>9.2f}{f.change_pct:>7.2f}{f.gap_pct:>7.2f}"
            f"{f.rvol:>6.2f}{f.atr_pct:>6.2f}{f.vwap_distance_atr:>8.2f}"
            f"{f.rsi:>6.1f}{f.ema_trend:>5d}{f.macd - f.macd_signal:>8.3f}"
            f"{orp:>8}{sprd:>7}"
        )
    return "\n".join(lines)


def build_cycle_prompt(
    features: dict[str, FeaturePack],
    focus: list[str],
    narrative: str,
    position_context: str,
    session_description: str,
    minutes_to_close: float,
    plan: DailyPlan | None = None,
    max_new_entries: int = 2,
) -> str:
    """The per-cycle user message."""
    parts: list[str] = [
        f"Session status: {session_description}",
        f"Time remaining in the session: {minutes_to_close:.0f} minutes.",
    ]

    if plan:
        parts.append(
            f"\nYour pre-market plan: bias {plan.bias}, posture {plan.risk_posture}."
            + (f" Themes: {', '.join(plan.themes[:5])}." if plan.themes else "")
            + (f" Avoid today: {', '.join(plan.avoid[:8])}." if plan.avoid else "")
        )

    parts.append(f"\n{position_context}")
    parts.append(f"\n{narrative}")
    parts.append("\nCurrent focus list:\n" + format_feature_table(features, focus))

    if minutes_to_close < 60:
        parts.append(
            "\nThe close is approaching. Favour managing or exiting what you hold "
            "over opening anything new."
        )

    parts.append(
        f"\nPropose at most {max_new_entries} new entries. Use CLOSE to exit a position "
        "you hold. Proposing nothing is a valid and often correct answer."
    )
    return "\n".join(parts)


def build_premarket_prompt(
    features: dict[str, FeaturePack],
    candidates: list[str],
    session_description: str,
    yesterday_lessons: list[str],
    equity: float,
) -> str:
    parts: list[str] = [
        f"Pre-market planning. {session_description}",
        f"Account size: roughly ${equity:,.0f}.",
    ]
    if yesterday_lessons:
        parts.append(
            "\nLessons you wrote at the end of the last session:\n"
            + "\n".join(f"  - {x}" for x in yesterday_lessons[:6])
        )
    parts.append(
        "\nPre-market scan of the universe (ranked by a deterministic interest score):\n"
        + format_feature_table(features, candidates[:40])
    )
    parts.append(
        "\nSet the frame for today: directional lean, risk posture, themes, a watchlist "
        "of up to 20 names, and anything to avoid."
    )
    return "\n".join(parts)


def build_risk_review_prompt(
    symbol: str,
    action: str,
    quantity: int,
    entry: float,
    stop: float,
    target: float,
    risk_amount: float,
    equity: float,
    rationale: str,
    evidence: list[str],
    features: FeaturePack | None,
    account: AccountSnapshot,
) -> str:
    risk_pct = (risk_amount / equity * 100) if equity > 0 else 0.0
    reward = abs(target - entry)
    risk_per_share = abs(entry - stop)
    rr = reward / risk_per_share if risk_per_share > 0 else 0.0

    parts = [
        f"Proposed trade: {action} {quantity} {symbol}",
        f"  entry {entry:.2f}, stop {stop:.2f}, target {target:.2f}",
        f"  risking {risk_pct:.2f}% of equity for a {rr:.1f}:1 reward-to-risk",
        f"\nDesk rationale: {rationale}",
    ]
    if evidence:
        parts.append("Cited evidence: " + "; ".join(evidence[:6]))
    if features:
        parts.append(
            f"\nCurrent readings: price {features.price:.2f}, RSI {features.rsi:.1f}, "
            f"ATR {features.atr_pct:.2f}% of price, {features.vwap_distance_atr:+.2f} ATR "
            f"from VWAP, relative volume {features.rvol:.2f}"
        )
    held = account.open_positions
    parts.append(
        f"\nDesk currently holds {len(held)} position(s)"
        + (f": {', '.join(held)}" if held else ".")
    )
    parts.append("\nApprove or veto. If you veto, say specifically why.")
    return "\n".join(parts)


def build_eod_prompt(
    narrative: str,
    trades: list[dict],
    starting_equity: float,
    ending_equity: float,
    rejection_counts: dict[str, int],
) -> str:
    pnl_pct = (
        (ending_equity - starting_equity) / starting_equity * 100 if starting_equity else 0.0
    )
    parts = [
        "The session has closed. Review it.",
        f"\nResult: {pnl_pct:+.2f}% on the day across {len(trades)} fills.",
        f"\n{narrative}",
    ]
    if trades:
        lines = [
            f"  {t.get('action', '')} {t.get('quantity', 0):.0f} {t.get('symbol', '')} "
            f"at {t.get('price', 0):.2f}"
            for t in trades[:40]
        ]
        parts.append("\nFills:\n" + "\n".join(lines))
    if rejection_counts:
        parts.append(
            "\nOrders blocked by risk limits today:\n"
            + "\n".join(f"  {reason}: {n}" for reason, n in rejection_counts.items())
        )
        parts.append(
            "If one limit blocked a lot of trades, say whether that reflects a bad "
            "setting or the desk repeatedly trying something it should not."
        )
    parts.append(
        "\nGive a grade, what worked, what failed, and specific lessons for tomorrow."
    )
    return "\n".join(parts)


COMPACTION_SYSTEM = """\
You are compressing a trading session log so it fits in limited context. Preserve \
decisions, their reasoning, outcomes, and anything the desk said it would watch for. \
Drop routine chatter. Write it as a continuous account in past tense, no more than \
250 words.
"""
