"""The four agent roles.

Each one is a thin wrapper: build a prompt, call the gateway, validate, and fall
back to a safe default if the model could not deliver. The safe default is
always "do nothing" — never "proceed anyway".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import LLMConfig, ModelSpec
from ..domain.models import AccountSnapshot, FeaturePack
from ..domain.proposals import (
    CycleDecision,
    DailyPlan,
    RiskCritique,
    SessionReview,
    SizedOrder,
)
from ..llm.gateway import LLMGateway
from ..llm.narrative import SessionNarrative
from ..logging_setup import get_logger
from . import prompts

log = get_logger(__name__)


@dataclass
class AgentRunner:
    """Owns the four roles and the shared session narrative."""

    gateway: LLMGateway
    cfg: LLMConfig
    narrative: SessionNarrative

    async def _call(
        self,
        role: str,
        spec: ModelSpec,
        system: str,
        user: str,
        response_model: type,
        cycle_id: str,
        context: dict[str, Any] | None = None,
    ):
        return await self.gateway.complete(
            role=role,
            cycle_id=cycle_id,
            model=spec.model,
            system=system,
            user=user,
            response_model=response_model,
            context=context,
            temperature=spec.temperature,
            num_ctx=spec.num_ctx,
            num_predict=spec.num_predict,
            seed=spec.seed,
            keep_alive=self.cfg.keep_alive,
            timeout=spec.timeout_seconds,
            think=spec.think,
        )

    # ------------------------------------------------------------------ #

    async def plan_session(
        self,
        features: dict[str, FeaturePack],
        candidates: list[str],
        session_description: str,
        equity: float,
        cycle_id: str = "premarket",
    ) -> DailyPlan:
        """Pre-market: set the frame for the day."""
        user = prompts.build_premarket_prompt(
            features=features,
            candidates=candidates,
            session_description=session_description,
            yesterday_lessons=self.narrative.yesterday_lessons,
            equity=equity,
        )
        plan = await self._call(
            "strategist", self.cfg.strategist, prompts.STRATEGIST_SYSTEM,
            user, DailyPlan, cycle_id,
        )
        if plan is None:
            log.warning("premarket_plan_unavailable_using_defensive_default")
            plan = DailyPlan.safe_default()
        self.narrative.record_plan(plan)
        return plan

    async def decide(
        self,
        features: dict[str, FeaturePack],
        focus: list[str],
        account: AccountSnapshot,
        risk_used_pct: float,
        session_description: str,
        minutes_to_close: float,
        max_new_entries: int,
        cycle_id: str,
    ) -> CycleDecision:
        """One decision cycle, as a single batched call for the whole focus list.

        Batching is not just an optimization: on a Free-tier Ollama Cloud account
        with one concurrent model, per-symbol calls would blow both the latency
        budget and the weekly quota. It also lets the model compare names against
        each other, which per-symbol calls destroy.
        """
        user = prompts.build_cycle_prompt(
            features=features,
            focus=focus,
            narrative=self.narrative.render(),
            position_context=self.narrative.position_context(account, risk_used_pct),
            session_description=session_description,
            minutes_to_close=minutes_to_close,
            plan=self.narrative.plan,
            max_new_entries=max_new_entries,
        )
        decision = await self._call(
            "trader", self.cfg.trader, prompts.TRADER_SYSTEM, user,
            CycleDecision, cycle_id,
            # The model may only name symbols we actually put in front of it,
            # plus anything we already hold.
            context={"allowed_symbols": set(focus) | set(account.open_positions)},
        )
        if decision is None:
            log.warning("cycle_decision_unavailable_holding", cycle_id=cycle_id)
            return CycleDecision.safe_default("model unavailable or invalid output")

        decision.cycle_id = cycle_id
        self.narrative.record_decision(decision)
        return decision

    async def review_risk(
        self,
        order: SizedOrder,
        features: FeaturePack | None,
        account: AccountSnapshot,
        cycle_id: str,
    ) -> RiskCritique:
        """Advisory second opinion. Can veto; cannot authorize.

        Fails closed: if the reviewer cannot answer, the trade does not happen.
        """
        user = prompts.build_risk_review_prompt(
            symbol=order.symbol,
            action=order.action.value,
            quantity=order.quantity,
            entry=order.entry_price,
            stop=order.stop_price,
            target=order.target_price,
            risk_amount=order.risk_amount,
            equity=account.equity,
            rationale=order.proposal.rationale,
            evidence=order.proposal.evidence,
            features=features,
            account=account,
        )
        critique = await self._call(
            "risk_officer", self.cfg.risk_officer, prompts.RISK_OFFICER_SYSTEM,
            user, RiskCritique, cycle_id,
        )
        if critique is None:
            log.warning("risk_review_unavailable_failing_closed", symbol=order.symbol)
            return RiskCritique.safe_default()
        if not critique.approve:
            self.narrative.append(
                "risk-veto",
                f"{order.symbol}: {critique.comment or '; '.join(critique.concerns[:2])}",
            )
        return critique

    async def review_session(
        self,
        trades: list[dict],
        starting_equity: float,
        ending_equity: float,
        rejection_counts: dict[str, int],
        cycle_id: str = "eod",
    ) -> SessionReview:
        """End of day: write the lessons that carry into tomorrow."""
        user = prompts.build_eod_prompt(
            narrative=self.narrative.render(limit_chars=8000),
            trades=trades,
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            rejection_counts=rejection_counts,
        )
        review = await self._call(
            "reviewer", self.cfg.reviewer, prompts.REVIEWER_SYSTEM,
            user, SessionReview, cycle_id,
        )
        return review or SessionReview.safe_default()

    async def compact_narrative(self, cycle_id: str = "compaction") -> None:
        """Summarize the notebook so the prompt cannot silently overflow `num_ctx`."""
        if not self.narrative.needs_compaction():
            return

        class _Summary(SessionReview):
            pass

        result = await self.gateway.complete(
            role="compaction",
            cycle_id=cycle_id,
            model=self.cfg.trader.model,
            system=prompts.COMPACTION_SYSTEM,
            user=self.narrative.render(limit_chars=12000),
            response_model=SessionReview,
            temperature=0.0,
            num_ctx=self.cfg.trader.num_ctx,
            num_predict=700,
            keep_alive=self.cfg.keep_alive,
            timeout=self.cfg.trader.timeout_seconds,
        )
        if result and result.summary:
            self.narrative.compact(result.summary)
        else:
            # Even if the model cannot summarize, the notebook must not grow
            # without bound — drop the oldest entries.
            self.narrative.compact("(earlier session activity omitted)", keep_recent=15)
