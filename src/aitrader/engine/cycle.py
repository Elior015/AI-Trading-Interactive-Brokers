"""One decision cycle, start to finish.

The flow, and the reason for its order:

  reconcile account (ground truth)
    -> compute features deterministically
    -> rank and update the focus list
    -> one batched LLM call for proposals
    -> deterministic sizing
    -> optional advisory risk review
    -> deterministic risk gate
    -> execution

The model appears exactly once, in the middle, and everything downstream of it
is plain Python. That is what makes the system auditable: for any order placed
you can point at the numbers that produced it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..agents.roles import AgentRunner
from ..analytics import features as feat
from ..analytics.ranking import FocusListManager, scanner_rank_scores
from ..broker.market_data import MarketDataService
from ..broker.orders import OrderManager
from ..config import Settings
from ..domain.enums import Action, ExecutionMode
from ..domain.proposals import CycleDecision, PendingApproval
from ..logging_setup import get_logger
from ..risk.engine import RiskContext, RiskEngine
from ..risk.sizing import size_proposal
from .calendar import MarketCalendar, SessionInfo
from .state import AppState, CycleRecord

log = get_logger(__name__)


@dataclass
class DecisionCycle:
    settings: Settings
    state: AppState
    market_data: MarketDataService
    agents: AgentRunner
    risk: RiskEngine
    orders: OrderManager
    calendar: MarketCalendar
    focus_manager: FocusListManager

    # ------------------------------------------------------------------ #

    async def refresh_features(self, symbols: list[str]) -> None:
        """Recompute the numeric view for the symbols we care about."""
        quotes = self.market_data.quotes(symbols)
        intraday = {s: self.market_data.intraday.get(s, []) for s in symbols}
        daily = {s: self.market_data.daily.get(s, []) for s in symbols}
        self.state.features = feat.build_all(
            intraday=intraday,
            daily=daily,
            quotes=quotes,
            opening_range_bars=max(
                1, self.settings.strategy.cadence.opening_range_minutes // 5
            ),
        )

    async def update_focus(self, account_symbols: set[str]) -> tuple[list[str], list[str]]:
        """Re-rank candidates and adjust streaming subscriptions."""
        ranks = scanner_rank_scores(self.market_data.scanner_hits)
        protected = account_symbols | self.orders.working_symbols

        focus, added, removed = self.focus_manager.update(
            self.state.features, ranks, protected
        )

        if added:
            await self.market_data.subscribe(added)
            # Promotion is the only intraday historical request we make, and it
            # is capped per cycle so the 60-per-10-minutes budget stays healthy.
            await self.market_data.load_intraday(added)
        if removed:
            await self.market_data.unsubscribe(removed)

        self.state.focus = focus
        return added, removed

    # ------------------------------------------------------------------ #

    async def run(self, session: SessionInfo, account_provider: Any) -> CycleRecord:
        """Execute one full cycle."""
        cycle_id = f"c-{datetime.now(UTC).strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
        started = time.monotonic()
        record = CycleRecord(
            cycle_id=cycle_id, started_at=datetime.now(UTC)
        )
        self.state.entries_this_cycle = 0

        try:
            # 1. Ground truth first. Everything downstream depends on this being
            #    the broker's view rather than ours.
            account = await account_provider()
            self.state.account = account
            self.state.update_peak()

            # 2. Deterministic features over the candidate set.
            candidates = self._candidate_symbols(account)
            await self.refresh_features(candidates)

            # 3. Narrow to the focus list.
            await self.update_focus(set(account.open_positions))
            record.focus = list(self.state.focus)

            if not self.state.focus:
                record.error = "no symbols passed the data-quality bar"
                log.info("cycle_skipped_no_focus", cycle_id=cycle_id)
                return record

            # 4. The single LLM call.
            llm_started = time.monotonic()
            decision = await self.agents.decide(
                features=self.state.features,
                focus=self.state.focus,
                account=account,
                risk_used_pct=self.state.risk_used_pct,
                session_description=self.state.session_description,
                minutes_to_close=session.minutes_to_close,
                max_new_entries=self.settings.strategy.risk.max_new_entries_per_cycle,
                cycle_id=cycle_id,
            )
            record.llm_latency_ms = int((time.monotonic() - llm_started) * 1000)
            self.state.last_decision = decision
            record.market_read = decision.market_read
            record.proposals = len(decision.proposals)

            # 5. Everything from here is deterministic.
            await self._execute(decision, account, session, record, cycle_id)

        except Exception as exc:
            record.error = str(exc)
            self.state.last_error = str(exc)
            log.exception("cycle_failed", cycle_id=cycle_id, error=str(exc))

        record.duration_ms = int((time.monotonic() - started) * 1000)
        self.state.record_cycle(record)
        return record

    def _candidate_symbols(self, account: Any) -> list[str]:
        """Which symbols to compute features for this cycle.

        Bounded deliberately: computing indicators for 400 names every cycle
        would be wasted work, since only the focus list reaches the model.
        """
        pool = set(self.market_data.subscribed)
        pool |= set(account.open_positions)
        pool |= self.orders.working_symbols
        for symbols in self.market_data.scanner_hits.values():
            pool.update(symbols[:30])
        pool |= set(self.market_data.universe[:150])
        pool -= self.market_data.no_subscription
        return sorted(pool)

    # ------------------------------------------------------------------ #

    async def _execute(
        self,
        decision: CycleDecision,
        account: Any,
        session: SessionInfo,
        record: CycleRecord,
        cycle_id: str,
    ) -> None:
        cfg = self.settings.strategy.risk

        for proposal in decision.proposals:
            # Exits are handled separately: they are not new risk, so they do
            # not pass through sizing or the entry limits.
            if proposal.action == Action.CLOSE:
                await self._handle_close(proposal.symbol, account, session, record, cycle_id)
                continue

            features = self.state.features.get(proposal.symbol)
            if features is None:
                record.rejected += 1
                record.rejections.append(
                    {"symbol": proposal.symbol, "reason": "NO_FEATURES", "detail": "no data"}
                )
                continue

            if proposal.conviction < 0.55:
                record.rejected += 1
                record.rejections.append(
                    {
                        "symbol": proposal.symbol,
                        "reason": "LOW_CONVICTION",
                        "detail": f"conviction {proposal.conviction:.2f} below 0.55",
                    }
                )
                continue

            quote = self.market_data.quotes([proposal.symbol]).get(proposal.symbol)

            sized = size_proposal(
                proposal=proposal,
                features=features,
                account=account,
                quote=quote,
                cfg=cfg,
                limit_offset_pct=self.settings.strategy.broker.limit_offset_pct,
            )
            if sized is None:
                record.rejected += 1
                record.rejections.append(
                    {
                        "symbol": proposal.symbol,
                        "reason": "UNSIZEABLE",
                        "detail": "position size rounds to zero under current limits",
                    }
                )
                continue

            # Advisory review. It can veto; it cannot authorize.
            approved, comment = True, ""
            if cfg.require_risk_officer:
                critique = await self.agents.review_risk(
                    sized, features, account, cycle_id
                )
                approved = critique.approve
                comment = critique.comment or "; ".join(critique.concerns[:2])

            ctx = RiskContext(
                account=account,
                quote=quote,
                phase=session.phase,
                minutes_to_close=session.minutes_to_close,
                is_live=self.settings.is_live,
                starting_equity=self.state.starting_equity,
                entries_this_cycle=self.state.entries_this_cycle,
                trades_today=self.state.trades_today,
                working_symbols=self.orders.working_symbols,
                last_entry=self.state.last_entry,
                last_close=self.state.last_close,
                decision_age_seconds=decision.age_seconds(),
                risk_officer_approved=approved,
                risk_officer_comment=comment,
                avg_volume=features.avg_volume,
            )

            if self.state.execution_mode == ExecutionMode.MANUAL:
                self.state.pending_approvals.append(
                    PendingApproval(
                        cycle_id=cycle_id,
                        kind="entry",
                        symbol=sized.symbol,
                        action=sized.action,
                        rationale=proposal.rationale,
                        evidence=proposal.evidence,
                        sized=sized,
                    )
                )
                record.pending += 1
                self.agents.narrative.append(
                    "awaiting_approval",
                    f"{sized.action.value} {sized.quantity} {sized.symbol} at "
                    f"{sized.entry_price:.2f} — waiting for your OK",
                )
                continue

            verdict = await self.risk.submit(sized, ctx, cycle_id=cycle_id)
            if verdict.approved:
                record.approved += 1
                self.state.note_entry(sized.symbol)
                self.agents.narrative.append(
                    "placed",
                    f"{sized.action.value} {sized.quantity} {sized.symbol} at "
                    f"{sized.entry_price:.2f}, stop {sized.stop_price:.2f}, "
                    f"target {sized.target_price:.2f}",
                )
            else:
                record.rejected += 1
                reason = verdict.reason.value if verdict.reason else "UNKNOWN"
                record.rejections.append(
                    {"symbol": sized.symbol, "reason": reason, "detail": verdict.detail}
                )
                self.agents.narrative.record_rejection(sized.symbol, reason, verdict.detail)

    async def _handle_close(
        self,
        symbol: str,
        account: Any,
        session: SessionInfo,
        record: CycleRecord,
        cycle_id: str = "",
    ) -> None:
        position = account.open_positions.get(symbol)
        if position is None:
            record.rejections.append(
                {"symbol": symbol, "reason": "NO_POSITION", "detail": "nothing to close"}
            )
            return

        quote = self.market_data.quotes([symbol]).get(symbol)
        verdict = self.risk.evaluate_close(symbol, quote, session.phase, self.orders.working_symbols)
        if not verdict.approved:
            record.rejected += 1
            reason = verdict.reason.value if verdict.reason else "UNKNOWN"
            record.rejections.append({"symbol": symbol, "reason": reason, "detail": verdict.detail})
            self.agents.narrative.record_rejection(symbol, reason, verdict.detail)
            return

        if self.state.execution_mode == ExecutionMode.MANUAL:
            self.state.pending_approvals.append(
                PendingApproval(
                    cycle_id=cycle_id,
                    kind="close",
                    symbol=symbol,
                    action=Action.SELL if position.is_long else Action.BUY,
                    rationale="model requested exit",
                    close_quantity=position.quantity,
                    close_is_long=position.is_long,
                )
            )
            record.pending += 1
            self.agents.narrative.append("awaiting_approval", f"Close {symbol} — waiting for your OK")
            return

        try:
            await self.orders.close_position(
                symbol, position.quantity, position.is_long, reason="model requested exit"
            )
            self.state.note_close(symbol)
            self.agents.narrative.record_exit(symbol, "model requested exit")
            record.approved += 1
        except Exception as exc:
            log.exception("close_failed", symbol=symbol, error=str(exc))
            record.rejections.append(
                {"symbol": symbol, "reason": "CLOSE_FAILED", "detail": str(exc)}
            )

    # ------------------------------------------------------------------ #

    async def run_premarket(self, account_provider: Any) -> None:
        """Pre-market: backfill, scan, and set the day's frame."""
        log.info("premarket_starting")
        account = await account_provider()
        self.state.account = account
        self.state.reset_for_new_session(account.equity)
        self.risk.reset_for_new_session()

        await self.market_data.refresh_scanners()
        await self.market_data.backfill_universe()

        candidates = self.market_data.scanner_universe()[:200]
        await self.market_data.load_cached(candidates)

        from ..analytics.ranking import rank_candidates

        # Real intraday bars don't exist yet this early in the session, so
        # `refresh_features` (which reads `market_data.intraday`) would return
        # nothing for every symbol and leave the focus list permanently empty
        # -- no symbol ever gets subscribed, so intraday bars never start
        # accumulating either. Rank premarket candidates on daily bars instead
        # (already loaded by backfill_universe); once a seed focus list is
        # subscribed below, the first live cycle's refresh_features() takes
        # over with real intraday data.
        daily = {s: self.market_data.daily.get(s, []) for s in candidates}
        quotes = self.market_data.quotes(candidates)
        self.state.features = feat.build_all(
            intraday=daily,
            daily=daily,
            quotes=quotes,
            opening_range_bars=max(
                1, self.settings.strategy.cadence.opening_range_minutes // 5
            ),
        )

        ranked = rank_candidates(
            self.state.features,
            scanner_rank_scores(self.market_data.scanner_hits),
            limit=40,
        )

        plan = await self.agents.plan_session(
            features=self.state.features,
            candidates=ranked,
            session_description=self.state.session_description,
            equity=account.equity,
        )
        self.state.plan = plan

        # Seed the focus list from the plan's watchlist, then fill from the
        # deterministic ranking. The model gets a say in what to watch, but not
        # the final word — a name it likes with no data is still excluded.
        seed = [s for s in plan.watchlist if s in self.state.features]
        seed += [s for s in ranked if s not in seed and s not in plan.avoid]
        seed = seed[: self.settings.strategy.universe.focus_list_size]

        self.focus_manager.current = seed
        self.state.focus = seed
        await self.market_data.subscribe(seed)
        await self.market_data.load_intraday(seed)

        self.state.premarket_done = True
        log.info("premarket_complete", focus=seed, bias=plan.bias, posture=plan.risk_posture)

    async def run_eod(self, account_provider: Any) -> None:
        """End of day: review the session and write tomorrow's lessons."""
        if self.state.eod_done:
            return
        log.info("eod_review_starting")
        account = await account_provider()
        self.state.account = account

        fills = [
            {
                "symbol": f.symbol, "action": f.action,
                "quantity": f.quantity, "price": f.price,
            }
            for f in self.orders.fills
        ]
        rejections: dict[str, int] = {}
        if self.risk.store is not None:
            try:
                rejections = self.risk.store.rejection_counts()
            except Exception:  # noqa: BLE001
                rejections = {}

        review = await self.agents.review_session(
            trades=fills,
            starting_equity=self.state.starting_equity,
            ending_equity=account.equity,
            rejection_counts=rejections,
        )

        if self.risk.store is not None:
            try:
                self.risk.store.save_review(review.model_dump(mode="json"))
            except Exception as exc:  # noqa: BLE001
                log.warning("review_persist_failed", error=str(exc))

        await self.market_data.persist_session()
        self.state.eod_done = True
        log.info(
            "eod_review_complete",
            grade=review.grade,
            pnl_pct=round(self.state.day_pnl_pct, 2),
            lessons=len(review.lessons_for_tomorrow),
        )
