"""Schemas at the LLM boundary.

The critical design property here: `TradeProposal` contains no share count, no
dollar amount, no limit price and no order id. The model expresses risk as an
ATR multiple and reward as an R multiple; deterministic code turns those into
prices and quantities. A hallucinated number therefore cannot become a
hallucinated trade.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .enums import Action, RejectReason
from .models import utcnow


class TradeProposal(BaseModel):
    """A single trade idea from the model."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(description="Ticker symbol, exactly as given in the input table.")
    action: Action = Field(description="BUY to open long, SELL to open short, CLOSE to exit an existing position, HOLD to do nothing.")
    conviction: float = Field(ge=0.0, le=1.0, description="Confidence 0-1. Scales position size. Use <0.5 only for HOLD.")
    horizon_minutes: int = Field(ge=5, le=390, description="How long this trade is expected to take to work out.")
    stop_atr_multiple: float = Field(default=1.5, ge=0.5, le=5.0, description="Stop distance in ATRs from entry.")
    target_r_multiple: float = Field(default=2.0, ge=0.5, le=10.0, description="Profit target as a multiple of the risked amount.")
    rationale: str = Field(default="", max_length=600, description="One short paragraph: why this trade, now.")
    evidence: list[str] = Field(default_factory=list, description="Up to 6 short factual references to specific indicator values.")

    @field_validator("symbol")
    @classmethod
    def _known_symbol(cls, v: str, info: ValidationInfo) -> str:
        """Reject tickers we did not put in front of the model.

        Without this the model can name a symbol we hold no data for, and the
        whole downstream pipeline would be pricing a contract it never saw.
        """
        sym = v.strip().upper()
        allowed = (info.context or {}).get("allowed_symbols")
        if allowed is not None and sym not in allowed:
            raise ValueError(f"{sym!r} was not in the focus list supplied to the model")
        return sym

    @field_validator("evidence")
    @classmethod
    def _cap_evidence(cls, v: list[str]) -> list[str]:
        return [str(e)[:200] for e in v[:6]]

    @property
    def is_actionable(self) -> bool:
        return self.action != Action.HOLD

    @property
    def is_entry(self) -> bool:
        return self.action in (Action.BUY, Action.SELL)


class CycleDecision(BaseModel):
    """The Trader agent's full output for one decision cycle."""

    model_config = ConfigDict(extra="ignore")

    market_read: str = Field(default="", max_length=800, description="Two or three sentences on what the tape is doing right now.")
    proposals: list[TradeProposal] = Field(default_factory=list, description="One entry per symbol you want to act on. Omit symbols you do not care about.")
    watching: list[str] = Field(default_factory=list, description="Symbols to keep an eye on but not trade yet.")
    notes_for_next_cycle: str = Field(default="", max_length=400, description="What you want to remember when you look again in a few minutes.")

    #: Stamped by us, not the model. Execution rejects decisions older than the
    #: configured tolerance, because a stale decision is a wrong decision.
    decided_at: datetime = Field(default_factory=utcnow)
    cycle_id: str = ""

    @field_validator("proposals")
    @classmethod
    def _drop_holds(cls, v: list[TradeProposal]) -> list[TradeProposal]:
        # HOLD proposals carry no instruction; keeping them only adds noise downstream.
        return [p for p in v if p.action != Action.HOLD]

    @field_validator("proposals")
    @classmethod
    def _dedupe_symbols(cls, v: list[TradeProposal]) -> list[TradeProposal]:
        """Keep only the first proposal per symbol.

        A model that emits two conflicting instructions for one ticker has told
        us nothing; taking both would open and close in the same cycle.
        """
        seen: set[str] = set()
        out: list[TradeProposal] = []
        for p in v:
            if p.symbol not in seen:
                seen.add(p.symbol)
                out.append(p)
        return out

    def age_seconds(self, now: datetime | None = None) -> float:
        return ((now or utcnow()) - self.decided_at).total_seconds()

    @classmethod
    def safe_default(cls, reason: str = "") -> CycleDecision:
        """The fail-safe returned when the model cannot produce valid output."""
        return cls(
            market_read=f"No decision produced ({reason})." if reason else "No decision produced.",
            proposals=[],
            watching=[],
            notes_for_next_cycle="",
        )


class DailyPlan(BaseModel):
    """The Strategist agent's pre-market output."""

    model_config = ConfigDict(extra="ignore")

    bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(default="NEUTRAL", description="Overall directional lean for the session.")
    reasoning: str = Field(default="", max_length=1200, description="Why you hold that bias.")
    themes: list[str] = Field(default_factory=list, description="Up to 5 things you expect to matter today.")
    watchlist: list[str] = Field(default_factory=list, description="Up to 20 symbols worth watching from the pre-market data.")
    risk_posture: Literal["AGGRESSIVE", "NORMAL", "DEFENSIVE"] = Field(default="NORMAL", description="How much of the day's risk budget to deploy.")
    avoid: list[str] = Field(default_factory=list, description="Symbols to stay away from today, with no explanation needed.")

    @classmethod
    def safe_default(cls) -> DailyPlan:
        return cls(bias="NEUTRAL", risk_posture="DEFENSIVE", reasoning="No plan produced; defaulting to defensive.")


class RiskCritique(BaseModel):
    """The Risk Officer's advisory second opinion.

    Advisory means exactly one thing: it can veto, never approve. The
    deterministic gate in `risk/engine.py` always has the final word.
    """

    model_config = ConfigDict(extra="ignore")

    approve: bool = Field(default=False, description="False to veto this trade.")
    concerns: list[str] = Field(default_factory=list, description="Specific risks you see, if any.")
    comment: str = Field(default="", max_length=400)

    @classmethod
    def safe_default(cls) -> RiskCritique:
        # Fail closed: if the Risk Officer could not answer, do not take the trade.
        return cls(approve=False, comment="Risk review unavailable; failing closed.")


class SessionReview(BaseModel):
    """The Reviewer agent's end-of-day output, carried into tomorrow."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(default="", max_length=1500)
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    lessons_for_tomorrow: list[str] = Field(default_factory=list)
    grade: Literal["A", "B", "C", "D", "F"] = Field(default="C")

    @classmethod
    def safe_default(cls) -> SessionReview:
        return cls(summary="No review produced.")


class SizedOrder(BaseModel):
    """A proposal after deterministic sizing, before the risk gate."""

    proposal: TradeProposal
    symbol: str
    action: Action
    quantity: int
    entry_price: float
    stop_price: float
    target_price: float
    risk_amount: float
    notional: float

    @property
    def is_long(self) -> bool:
        return self.action == Action.BUY


class RiskVerdict(BaseModel):
    """The gate's answer. Every rejection is logged and shown on the dashboard."""

    approved: bool
    reason: RejectReason | None = None
    detail: str = ""
    symbol: str = ""
    checked_at: datetime = Field(default_factory=utcnow)

    @classmethod
    def ok(cls, symbol: str = "") -> RiskVerdict:
        return cls(approved=True, symbol=symbol)

    @classmethod
    def reject(cls, reason: RejectReason, detail: str = "", symbol: str = "") -> RiskVerdict:
        return cls(approved=False, reason=reason, detail=detail, symbol=symbol)

    def __bool__(self) -> bool:
        return self.approved


class PendingApproval(BaseModel):
    """A trade idea that cleared sizing and the advisory risk review and is
    now waiting for a person to say yes, in manual mode.

    Carries the already-computed `SizedOrder` (whose `.proposal` is the
    original `TradeProposal`, needed to re-size on the current price at
    approval time) so the number a person sees is the number that would be
    sent to the broker, not a re-derived one.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    cycle_id: str = ""
    kind: Literal["entry", "close"]
    symbol: str
    action: Action
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)

    #: Entry-only. None for a close.
    sized: SizedOrder | None = None

    #: Close-only. Unused for an entry.
    close_quantity: float = 0.0
    close_is_long: bool = True

    created_at: datetime = Field(default_factory=utcnow)

    def age_seconds(self, now: datetime | None = None) -> float:
        return ((now or utcnow()) - self.created_at).total_seconds()
