"""Deterministic narrowing of the broad universe down to a focus list.

This is what makes a 100+ symbol universe affordable. The model never sees the
whole universe; it sees the top ~20 by a score computed here in plain Python.
That keeps prompts small, latency low, LLM quota usage sane, and — because
scoring is deterministic — makes the selection reproducible and reviewable in a
way an LLM-chosen shortlist would not be.

Focus-list membership is damped by hysteresis so subscriptions do not thrash
against IBKR's limit of 60 new real-time-bar subscriptions per 10 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.models import FeaturePack
from ..logging_setup import get_logger

log = get_logger(__name__)


def score_symbol(f: FeaturePack, scanner_rank: float = 0.0) -> float:
    """Score a candidate on how much a day trader should care about it right now.

    The components deliberately reward *movement and participation*, not
    direction: picking a side is the model's job, and baking a directional bias
    in here would double-count it.
    """
    score = 0.0

    # Participation. Unusual volume is the single best filter for "something is
    # happening here", so it carries the most weight.
    score += min(f.rvol, 5.0) * 2.0

    # Range. A symbol needs enough daily range to pay for the spread, but
    # extreme volatility is a risk, not an opportunity.
    if 0.5 <= f.atr_pct <= 6.0:
        score += min(f.atr_pct, 4.0)
    elif f.atr_pct > 6.0:
        score -= 1.0

    # Displacement from the session's fair value, either direction.
    score += min(abs(f.vwap_distance_atr), 3.0)

    # Momentum extremes, either direction.
    score += abs(f.rsi - 50.0) / 25.0

    # Trend agreement between EMA and MACD is worth a small bonus.
    macd_hist = f.macd - f.macd_signal
    if f.ema_trend != 0 and (macd_hist > 0) == (f.ema_trend > 0):
        score += 0.5

    # Opening-range breakouts in either direction.
    orp = f.opening_range_position
    if orp is not None and (orp > 1.0 or orp < 0.0):
        score += 1.0

    # Gaps draw attention early in the session.
    score += min(abs(f.gap_pct) / 2.0, 2.0)

    # Being on a server-side scanner list is independent evidence.
    score += scanner_rank

    # Penalties: an untradeable spread makes everything above irrelevant.
    if f.spread_pct is not None and f.spread_pct > 0.003:
        score -= 3.0
    if f.avg_volume and f.avg_volume < 300_000:
        score -= 2.0

    return round(score, 4)


@dataclass
class FocusListManager:
    """Chooses the focus list, with hysteresis so membership does not thrash.

    A challenger must beat the weakest incumbent by `churn_margin` and hold that
    advantage for `persistence_cycles` consecutive cycles before it displaces
    anyone. Without this, symbols oscillate in and out and every swap costs a
    subscription against IBKR's rate limits.
    """

    size: int = 20
    churn_margin: float = 0.15
    persistence_cycles: int = 2
    max_promotions_per_cycle: int = 4

    current: list[str] = field(default_factory=list)
    _challenger_streak: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        features: dict[str, FeaturePack],
        scanner_ranks: dict[str, float] | None = None,
        protected: set[str] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Recompute the focus list.

        `protected` symbols (anything with an open position or a working order)
        are always members: dropping the data feed for a position we hold would
        blind the exit logic.

        Returns (focus_list, added, removed).
        """
        scanner_ranks = scanner_ranks or {}
        protected = protected or set()

        scored: dict[str, float] = {}
        for symbol, pack in features.items():
            s = score_symbol(pack, scanner_ranks.get(symbol, 0.0))
            pack.rank_score = s
            scored[symbol] = s

        # Protected names occupy slots before anything else is considered.
        keep = [s for s in protected if s in scored or s in self.current]
        slots = max(self.size - len(keep), 0)

        incumbents = [s for s in self.current if s not in keep and s in scored]
        challengers = sorted(
            (s for s in scored if s not in keep and s not in incumbents),
            key=lambda s: scored[s],
            reverse=True,
        )

        incumbents.sort(key=lambda s: scored[s], reverse=True)
        retained = incumbents[:slots]
        free_slots = slots - len(retained)

        added: list[str] = []

        # Free slots are filled immediately; nobody is being displaced.
        for symbol in challengers:
            if free_slots <= 0 or len(added) >= self.max_promotions_per_cycle:
                break
            retained.append(symbol)
            added.append(symbol)
            free_slots -= 1

        # Contested slots require beating the weakest incumbent persistently.
        for symbol in challengers:
            if symbol in added or len(added) >= self.max_promotions_per_cycle:
                continue
            if not retained:
                break
            weakest = min(retained, key=lambda s: scored.get(s, 0.0))
            if weakest in protected:
                break
            if scored[symbol] > scored.get(weakest, 0.0) * (1.0 + self.churn_margin):
                streak = self._challenger_streak.get(symbol, 0) + 1
                self._challenger_streak[symbol] = streak
                if streak >= self.persistence_cycles:
                    retained.remove(weakest)
                    retained.append(symbol)
                    added.append(symbol)
                    self._challenger_streak.pop(symbol, None)
            else:
                self._challenger_streak.pop(symbol, None)

        new_focus = keep + [s for s in retained if s not in keep]
        removed = [s for s in self.current if s not in new_focus]

        # Forget streaks for anything that made it in or fell out of contention.
        for symbol in list(self._challenger_streak):
            if symbol in new_focus or symbol not in scored:
                self._challenger_streak.pop(symbol, None)

        self.current = new_focus
        if added or removed:
            log.info("focus_list_updated", added=added, removed=removed, size=len(new_focus))
        return new_focus, added, removed


def rank_candidates(
    features: dict[str, FeaturePack],
    scanner_ranks: dict[str, float] | None = None,
    limit: int = 25,
) -> list[str]:
    """One-shot ranking, used by the pre-market pass before hysteresis applies."""
    scanner_ranks = scanner_ranks or {}
    scored = [
        (symbol, score_symbol(pack, scanner_ranks.get(symbol, 0.0)))
        for symbol, pack in features.items()
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in scored[:limit]]


def scanner_rank_scores(
    hits_by_code: dict[str, list[str]], weight: float = 1.5
) -> dict[str, float]:
    """Convert scanner positions into a bonus score.

    A symbol appearing near the top of several scans is more interesting than
    one appearing near the bottom of a single scan, so both position and breadth
    of appearance contribute.
    """
    out: dict[str, float] = {}
    for symbols in hits_by_code.values():
        n = len(symbols)
        if n == 0:
            continue
        for i, symbol in enumerate(symbols):
            out[symbol] = out.get(symbol, 0.0) + weight * (1.0 - i / n)
    return out
