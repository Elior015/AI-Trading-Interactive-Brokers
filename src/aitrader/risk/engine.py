"""The risk gate: the only path from a proposal to the broker.

Two properties make this genuinely load-bearing rather than decorative:

* **Every check is deterministic Python.** No LLM participates in deciding
  whether an order is allowed. A probabilistic filter cannot be a risk control.
* **It is the only caller of the order manager.** `tests/test_architecture.py`
  walks the AST of `src/` and fails if any other module reaches the broker's
  order methods. That test, not convention, is what holds the line over time.

All checks run even after one fails, so the dashboard can show *every* reason an
order was blocked rather than just the first. Any failure vetoes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..config import RiskConfig
from ..domain.enums import Action, RejectReason, SessionPhase
from ..domain.models import AccountSnapshot, Quote
from ..domain.proposals import RiskVerdict, SizedOrder
from ..logging_setup import get_logger
from .killswitch import KillSwitch

log = get_logger(__name__)


@dataclass
class RiskContext:
    """Everything the gate needs to judge one order."""

    account: AccountSnapshot
    quote: Quote | None
    phase: SessionPhase
    minutes_to_close: float
    is_live: bool
    #: Equity at the start of the session, for the daily loss limit.
    starting_equity: float
    entries_this_cycle: int = 0
    trades_today: int = 0
    working_symbols: set[str] = field(default_factory=set)
    #: symbol -> monotonic timestamp of last entry
    last_entry: dict[str, float] = field(default_factory=dict)
    #: symbol -> monotonic timestamp of last close
    last_close: dict[str, float] = field(default_factory=dict)
    decision_age_seconds: float = 0.0
    risk_officer_approved: bool = True
    risk_officer_comment: str = ""
    avg_volume: float = 0.0


class RiskEngine:
    """Deterministic pre-trade controls."""

    def __init__(
        self,
        cfg: RiskConfig,
        kill_switch: KillSwitch,
        store: Any = None,
        order_manager: Any = None,
    ) -> None:
        self.cfg = cfg
        self.kill_switch = kill_switch
        self.store = store
        #: Private on purpose. Reaching the broker means going through `submit`.
        self._order_manager = order_manager
        self._order_times: deque[float] = deque()
        self.daily_loss_tripped = False

    # ------------------------------------------------------------------ #
    # the checks
    # ------------------------------------------------------------------ #

    def evaluate(self, order: SizedOrder, ctx: RiskContext) -> RiskVerdict:
        """Run every check. Returns the first failure, having logged them all."""
        failures: list[RiskVerdict] = []

        def fail(reason: RejectReason, detail: str) -> None:
            failures.append(RiskVerdict.reject(reason, detail, order.symbol))

        cfg = self.cfg
        acct = ctx.account

        # 1. Kill switch.
        if self.kill_switch.is_tripped:
            fail(RejectReason.KILL_SWITCH, self.kill_switch.reason or "kill switch active")

        # 2. Live/paper interlock. Re-asserted here, not only at startup, so a
        #    mid-session state change cannot slip an order through.
        if ctx.is_live and acct.is_paper:
            fail(
                RejectReason.TRADING_MODE_INTERLOCK,
                f"live mode but account {acct.account_id} is a paper account",
            )
        if not ctx.is_live and not acct.is_paper and acct.account_id:
            fail(
                RejectReason.TRADING_MODE_INTERLOCK,
                f"paper mode but account {acct.account_id} is not a paper account",
            )

        # 3. Session phase.
        if not ctx.phase.is_tradeable:
            fail(RejectReason.MARKET_CLOSED, f"session phase is {ctx.phase.value}")

        # 4. Too close to the close for a new position.
        if ctx.minutes_to_close <= cfg.no_entry_minutes_before_close:
            fail(
                RejectReason.TOO_LATE_IN_SESSION,
                f"{ctx.minutes_to_close:.0f} min to close, "
                f"cutoff is {cfg.no_entry_minutes_before_close}",
            )

        # 5. Account state freshness. Acting on a stale snapshot is the failure
        #    mode that propagates worst, so it is a hard veto.
        age = acct.age_seconds()
        if age > cfg.max_account_age_seconds:
            fail(
                RejectReason.STALE_ACCOUNT_STATE,
                f"account snapshot is {age:.0f}s old (max {cfg.max_account_age_seconds:.0f}s)",
            )

        # 6. Decision freshness. A stale decision is a wrong decision.
        if ctx.decision_age_seconds > cfg.max_decision_age_seconds:
            fail(
                RejectReason.STALE_MARKET_DATA,
                f"decision is {ctx.decision_age_seconds:.0f}s old "
                f"(max {cfg.max_decision_age_seconds:.0f}s)",
            )

        # 7. Market data present and usable. There is no delayed fallback for US
        #    equities on an IB LLC account, so missing data fails loudly.
        q = ctx.quote
        if q is None:
            fail(RejectReason.NO_MARKET_DATA, "no quote available")
        else:
            if q.halted:
                fail(RejectReason.HALTED, "contract is halted")
            if not q.is_usable:
                fail(RejectReason.NO_MARKET_DATA, "quote has no usable price")
            qage = q.age_seconds()
            if qage > cfg.max_quote_age_seconds:
                fail(
                    RejectReason.STALE_MARKET_DATA,
                    f"quote is {qage:.0f}s old (max {cfg.max_quote_age_seconds:.0f}s)",
                )
            sp = q.spread_pct
            if sp is not None and sp > cfg.max_spread_pct:
                fail(
                    RejectReason.SPREAD_TOO_WIDE,
                    f"spread {sp * 100:.2f}% exceeds {cfg.max_spread_pct * 100:.2f}%",
                )

        # 8. Quantity sanity.
        if order.quantity < 1:
            fail(RejectReason.ZERO_QUANTITY, f"quantity is {order.quantity}")

        # 9. Price collar / fat finger.
        if q is not None and q.mid:
            deviation = abs(order.entry_price - q.mid) / q.mid
            if deviation > cfg.price_collar_pct:
                fail(
                    RejectReason.PRICE_COLLAR,
                    f"limit {order.entry_price:.2f} is {deviation * 100:.2f}% from "
                    f"mid {q.mid:.2f} (max {cfg.price_collar_pct * 100:.2f}%)",
                )

        # 10. Price band.
        if not (cfg.min_price <= order.entry_price <= cfg.max_price):
            fail(
                RejectReason.ILLIQUID,
                f"price {order.entry_price:.2f} outside "
                f"[{cfg.min_price}, {cfg.max_price}]",
            )

        # 11. Liquidity.
        if ctx.avg_volume and ctx.avg_volume < cfg.min_avg_volume:
            fail(
                RejectReason.ILLIQUID,
                f"average volume {ctx.avg_volume:,.0f} below {cfg.min_avg_volume:,.0f}",
            )

        # 12. Per-trade risk, recomputed here independently of the sizer.
        #     Verifying rather than trusting means a sizer bug cannot leak through.
        max_risk = acct.equity * cfg.max_risk_per_trade_pct
        actual_risk = abs(order.entry_price - order.stop_price) * order.quantity
        if actual_risk > max_risk * 1.02:  # 2% tolerance for tick rounding
            fail(
                RejectReason.PER_TRADE_RISK_EXCEEDED,
                f"risk ${actual_risk:,.2f} exceeds cap ${max_risk:,.2f}",
            )

        # 13. Position notional.
        max_notional = acct.equity * cfg.max_position_notional_pct
        if order.notional > max_notional * 1.02:
            fail(
                RejectReason.POSITION_NOTIONAL_EXCEEDED,
                f"notional ${order.notional:,.2f} exceeds cap ${max_notional:,.2f}",
            )

        # 14. Buying power.
        usable = acct.buying_power * (1.0 - cfg.buying_power_reserve_pct)
        if order.notional > usable:
            fail(
                RejectReason.INSUFFICIENT_BUYING_POWER,
                f"notional ${order.notional:,.2f} exceeds usable buying power ${usable:,.2f}",
            )

        # 15. Concurrent positions.
        open_count = len(acct.open_positions)
        if order.symbol not in acct.open_positions and open_count >= cfg.max_concurrent_positions:
            fail(
                RejectReason.MAX_POSITIONS_REACHED,
                f"{open_count} positions open, max {cfg.max_concurrent_positions}",
            )

        # 16. Entries per cycle.
        if ctx.entries_this_cycle >= cfg.max_new_entries_per_cycle:
            fail(
                RejectReason.MAX_ENTRIES_PER_CYCLE,
                f"{ctx.entries_this_cycle} entries already this cycle",
            )

        # 17. Daily loss limit.
        if ctx.starting_equity > 0:
            pnl_pct = (acct.equity - ctx.starting_equity) / ctx.starting_equity
            if pnl_pct <= -cfg.daily_loss_limit_pct:
                self.daily_loss_tripped = True
                fail(
                    RejectReason.DAILY_LOSS_LIMIT,
                    f"day P&L {pnl_pct * 100:.2f}% breached "
                    f"limit -{cfg.daily_loss_limit_pct * 100:.2f}%",
                )

        # 18. Daily trade cap.
        if ctx.trades_today >= cfg.max_trades_per_day:
            fail(
                RejectReason.DAILY_TRADE_CAP,
                f"{ctx.trades_today} trades today, max {cfg.max_trades_per_day}",
            )

        # 19. Order rate. The runaway-bug circuit breaker: a loop placing orders
        #     is stopped here regardless of every other check passing.
        if not self._check_order_rate():
            fail(
                RejectReason.DAILY_TRADE_CAP,
                f"order rate exceeded {cfg.max_orders_per_minute}/minute",
            )

        # 20. Duplicate / already working.
        if order.symbol in ctx.working_symbols:
            fail(RejectReason.DUPLICATE_ORDER, "an order for this symbol is already working")
        if order.symbol in acct.open_positions and order.action in (Action.BUY, Action.SELL):
            existing = acct.open_positions[order.symbol]
            same_side = (existing.is_long and order.action == Action.BUY) or (
                not existing.is_long and order.action == Action.SELL
            )
            if same_side:
                fail(RejectReason.DUPLICATE_ORDER, "already holding this position")

        # 21. Cooldown.
        now = time.monotonic()
        last = ctx.last_entry.get(order.symbol)
        if last is not None and (now - last) < cfg.symbol_cooldown_seconds:
            fail(
                RejectReason.COOLDOWN,
                f"entered {now - last:.0f}s ago, cooldown is {cfg.symbol_cooldown_seconds:.0f}s",
            )

        # 22. Flip-flop guard.
        closed = ctx.last_close.get(order.symbol)
        if closed is not None and (now - closed) < cfg.flip_flop_guard_seconds:
            fail(
                RejectReason.FLIP_FLOP_GUARD,
                f"closed {now - closed:.0f}s ago; not reversing within "
                f"{cfg.flip_flop_guard_seconds:.0f}s",
            )

        # 23. Shorting.
        if (
            order.action == Action.SELL
            and order.symbol not in acct.open_positions
            and not cfg.allow_shorts
        ):
            fail(RejectReason.TRADING_MODE_INTERLOCK, "short selling is disabled")

        # 24. Risk Officer veto. Advisory only: it can block, never authorize.
        if cfg.require_risk_officer and not ctx.risk_officer_approved:
            fail(
                RejectReason.RISK_OFFICER_VETO,
                ctx.risk_officer_comment[:200] or "risk officer declined",
            )

        verdict = failures[0] if failures else RiskVerdict.ok(order.symbol)

        if failures:
            log.info(
                "risk_rejected",
                symbol=order.symbol,
                action=order.action.value,
                quantity=order.quantity,
                reasons=[f.reason.value for f in failures if f.reason],
                detail=verdict.detail,
            )
        return verdict

    def _check_order_rate(self) -> bool:
        now = time.monotonic()
        while self._order_times and self._order_times[0] <= now - 60.0:
            self._order_times.popleft()
        return len(self._order_times) < self.cfg.max_orders_per_minute

    def _record_order(self) -> None:
        self._order_times.append(time.monotonic())

    # ------------------------------------------------------------------ #
    # the only path to the broker
    # ------------------------------------------------------------------ #

    async def submit(self, order: SizedOrder, ctx: RiskContext, cycle_id: str = "") -> RiskVerdict:
        """Evaluate and, if approved, place the order.

        Nothing else in the system may call the order manager. The check is
        re-run immediately before the wire call in case the order sat in a queue
        while conditions changed.
        """
        verdict = self.evaluate(order, ctx)
        self._persist(cycle_id, order, verdict)
        if not verdict.approved:
            return verdict

        # Last look. The kill switch in particular may have been tripped
        # between evaluation and here.
        if self.kill_switch.is_tripped:
            late = RiskVerdict.reject(
                RejectReason.KILL_SWITCH, "tripped after evaluation", order.symbol
            )
            self._persist(cycle_id, order, late)
            return late

        if self._order_manager is None:
            return RiskVerdict.reject(
                RejectReason.TRADING_MODE_INTERLOCK, "no order manager wired", order.symbol
            )

        try:
            await self._order_manager.place_bracket(order, cycle_id=cycle_id)
            self._record_order()
        except Exception as exc:
            log.exception("order_placement_failed", symbol=order.symbol, error=str(exc))
            return RiskVerdict.reject(
                RejectReason.DUPLICATE_ORDER, f"placement failed: {exc}", order.symbol
            )
        return verdict

    def _persist(self, cycle_id: str, order: SizedOrder, verdict: RiskVerdict) -> None:
        if self.store is None:
            return
        try:
            self.store.save_risk_event(
                cycle_id,
                order.symbol,
                verdict.approved,
                verdict.reason.value if verdict.reason else None,
                verdict.detail,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("risk_event_persist_failed", error=str(exc))

    # ------------------------------------------------------------------ #

    def check_daily_loss(self, account: AccountSnapshot, starting_equity: float) -> bool:
        """Out-of-band daily loss check, run by the fast loop.

        Returns True when the limit is breached, having tripped the kill switch.
        This runs independently of any proposal so a losing position can halt
        trading even when the model is proposing nothing.
        """
        if starting_equity <= 0 or self.daily_loss_tripped:
            return self.daily_loss_tripped
        pnl_pct = (account.equity - starting_equity) / starting_equity
        if pnl_pct <= -self.cfg.daily_loss_limit_pct:
            self.daily_loss_tripped = True
            self.kill_switch.trip(
                f"daily loss limit breached: {pnl_pct * 100:.2f}% "
                f"(limit -{self.cfg.daily_loss_limit_pct * 100:.2f}%)"
            )
            return True
        return False

    def reset_for_new_session(self) -> None:
        self.daily_loss_tripped = False
        self._order_times.clear()

    def status(self) -> dict[str, Any]:
        return {
            "daily_loss_tripped": self.daily_loss_tripped,
            "orders_last_minute": len(self._order_times),
            "max_orders_per_minute": self.cfg.max_orders_per_minute,
            "kill_switch": self.kill_switch.status(),
            "checked_at": datetime.now(UTC).isoformat(),
        }
