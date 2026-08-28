"""The session narrative — the system's working memory.

This is what makes the AI behave like *one trader working a session* rather than
forty independent, amnesiac invocations that contradict each other every five
minutes. Each cycle reads the notebook and appends to it: the pre-market thesis,
what it did and why, what it is waiting for, what it got wrong earlier.

Two deliberate constraints:

* It is compacted when it grows past a character budget, so the prompt cannot
  silently overflow `num_ctx` and lose the beginning of the day.
* It records positions and remaining risk budget, but **not** running P&L in
  money terms. Telling a model "you are down $400 today" invites exactly the
  behaviour you would not want from a human in the same seat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..domain.models import AccountSnapshot
from ..domain.proposals import CycleDecision, DailyPlan
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class NarrativeEntry:
    role: str
    content: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def render(self) -> str:
        return f"[{self.ts.strftime('%H:%M')} UTC] {self.role}: {self.content}"


@dataclass
class SessionNarrative:
    """The rolling notebook for one trading day."""

    max_chars: int = 6000
    entries: list[NarrativeEntry] = field(default_factory=list)
    plan: DailyPlan | None = None
    #: Carried over from yesterday's end-of-day review.
    yesterday_lessons: list[str] = field(default_factory=list)
    store: Any = None

    # ------------------------------------------------------------------ #

    def append(self, role: str, content: str) -> None:
        content = content.strip()
        if not content:
            return
        self.entries.append(NarrativeEntry(role=role, content=content))
        if self.store is not None:
            try:
                self.store.append_narrative(role, content)
            except Exception as exc:  # noqa: BLE001
                log.warning("narrative_persist_failed", error=str(exc))

    def record_plan(self, plan: DailyPlan) -> None:
        self.plan = plan
        self.append(
            "plan",
            f"Session bias {plan.bias}, risk posture {plan.risk_posture}. "
            f"{plan.reasoning[:400]}"
            + (f" Themes: {', '.join(plan.themes[:5])}." if plan.themes else "")
            + (f" Avoiding: {', '.join(plan.avoid[:8])}." if plan.avoid else ""),
        )

    def record_decision(self, decision: CycleDecision) -> None:
        if decision.market_read:
            self.append("read", decision.market_read)
        for p in decision.proposals:
            self.append(
                "intent",
                f"{p.action.value} {p.symbol} (conviction {p.conviction:.2f}) — {p.rationale[:200]}",
            )
        if decision.notes_for_next_cycle:
            self.append("note", decision.notes_for_next_cycle)

    def record_execution(self, symbol: str, action: str, quantity: int, price: float) -> None:
        self.append("filled", f"{action} {quantity} {symbol} at {price:.2f}")

    def record_rejection(self, symbol: str, reason: str, detail: str = "") -> None:
        self.append("blocked", f"{symbol}: {reason}{(' — ' + detail) if detail else ''}")

    def record_exit(self, symbol: str, reason: str, pnl_r: float | None = None) -> None:
        """Record an exit in R multiples rather than dollars.

        R is the unit the model reasons in when it sets a stop, and it keeps the
        feedback about trade quality rather than about money.
        """
        suffix = f" ({pnl_r:+.2f}R)" if pnl_r is not None else ""
        self.append("exit", f"Closed {symbol}: {reason}{suffix}")

    # ------------------------------------------------------------------ #

    def render(self, limit_chars: int | None = None) -> str:
        """The notebook as it appears in the prompt."""
        limit = limit_chars or self.max_chars
        parts: list[str] = []

        if self.yesterday_lessons:
            parts.append(
                "From yesterday's review:\n"
                + "\n".join(f"  - {x}" for x in self.yesterday_lessons[:5])
            )
        if self.plan:
            parts.append(
                f"Today's plan: bias {self.plan.bias}, posture {self.plan.risk_posture}."
                + (f" Themes: {', '.join(self.plan.themes[:5])}." if self.plan.themes else "")
            )

        rendered = [e.render() for e in self.entries]
        # Keep the most recent entries; the tail is what matters for the next
        # decision, and the head has already been compacted into a summary.
        body: list[str] = []
        used = sum(len(p) for p in parts)
        for line in reversed(rendered):
            if used + len(line) > limit:
                break
            body.append(line)
            used += len(line) + 1
        body.reverse()

        if body:
            parts.append("Session log so far:\n" + "\n".join(body))
        else:
            parts.append("Session log so far: (nothing yet — this is the first look today)")
        return "\n\n".join(parts)

    @property
    def total_chars(self) -> int:
        return sum(len(e.content) for e in self.entries)

    def needs_compaction(self) -> bool:
        return self.total_chars > self.max_chars

    def compact(self, summary: str, keep_recent: int = 10) -> None:
        """Replace older entries with a summary, keeping the most recent verbatim."""
        recent = self.entries[-keep_recent:] if keep_recent else []
        self.entries = [
            NarrativeEntry(role="summary", content=summary),
            *recent,
        ]
        if self.store is not None:
            try:
                self.store.replace_narrative([(e.role, e.content) for e in self.entries])
            except Exception as exc:  # noqa: BLE001
                log.warning("narrative_compaction_persist_failed", error=str(exc))
        log.info("narrative_compacted", kept=len(recent), summary_chars=len(summary))

    # ------------------------------------------------------------------ #

    def position_context(self, account: AccountSnapshot, risk_used_pct: float) -> str:
        """Current holdings and remaining risk budget, stated without dollar P&L.

        The model gets what it needs to manage exposure without being handed a
        loss figure to react to emotionally.
        """
        lines: list[str] = []
        open_positions = account.open_positions
        if not open_positions:
            lines.append("You currently hold no positions.")
        else:
            lines.append("Current positions:")
            for sym, pos in open_positions.items():
                direction = "long" if pos.is_long else "short"
                move = ""
                if pos.market_price and pos.avg_cost:
                    pct = (pos.market_price - pos.avg_cost) / pos.avg_cost * 100.0
                    if not pos.is_long:
                        pct = -pct
                    move = f", currently {pct:+.1f}% from entry"
                lines.append(f"  - {sym}: {direction} {abs(pos.quantity):.0f} shares{move}")
        remaining = max(0.0, 100.0 - risk_used_pct)
        lines.append(f"Risk budget used today: {risk_used_pct:.0f}%. Remaining: {remaining:.0f}%.")
        return "\n".join(lines)

    def load_from_store(self, store: Any) -> None:
        """Restore today's notebook after a restart, so continuity survives one."""
        try:
            rows = store.load_narrative()
        except Exception as exc:  # noqa: BLE001
            log.warning("narrative_load_failed", error=str(exc))
            return
        self.entries = [
            NarrativeEntry(
                role=r["role"],
                content=r["content"],
                ts=datetime.fromisoformat(r["ts"]),
            )
            for r in rows
        ]
        if self.entries:
            log.info("narrative_restored", entries=len(self.entries))
