"""The trading engine: builds every service and runs the concurrent loops.

The loops are deliberately separated by how fast they must react:

* `fast_loop` (seconds) is entirely deterministic — kill switch, daily loss,
  end-of-day flatten, stale order cleanup. It never calls the LLM, so it keeps
  working when the model is slow, unavailable, or out of quota.
* `decision_loop` (minutes) is where the model runs.

That split is what lets the system degrade rather than fail: with the LLM gone
it stops opening new positions but keeps managing existing ones, and the stops
protecting those positions are resting at IBKR regardless.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..agents.roles import AgentRunner
from ..analytics.ranking import FocusListManager
from ..broker.gateway import GatewayService
from ..broker.ib_adapter import IbAsyncBroker
from ..broker.market_data import MarketDataService
from ..broker.orders import OrderManager
from ..config import LiveTradingNotPermitted, Settings
from ..data.store import BarStore, StateStore
from ..domain.enums import KillSwitchAction, SessionPhase
from ..llm.audit import AuditLog
from ..llm.gateway import LLMGateway
from ..llm.narrative import SessionNarrative
from ..llm.providers import build_provider
from ..logging_setup import get_logger
from ..risk.engine import RiskEngine
from ..risk.killswitch import KillSwitch
from .calendar import MarketCalendar
from .cycle import DecisionCycle
from .state import AppState

log = get_logger(__name__)


@dataclass
class TradingEngine:
    settings: Settings

    state: AppState = field(default_factory=AppState)
    broker: Any = None
    _tasks: list[asyncio.Task] = field(default_factory=list)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _last_account_fetch: float = 0.0

    def __post_init__(self) -> None:
        s = self.settings
        cfg = s.strategy

        self.state.mode = "live" if s.is_live else "paper"

        # Persistence
        self.store = StateStore(s.data_dir / "aitrader.sqlite3")
        self.bar_store = BarStore(s.duckdb_path)

        # Broker
        self.broker = self.broker or IbAsyncBroker(
            messages_per_second=cfg.broker.max_messages_per_second
        )
        self.gateway = GatewayService(
            broker=self.broker,
            host=cfg.broker.host,
            port=s.port,
            client_id=cfg.broker.client_id,
            connect_timeout=cfg.broker.connect_timeout,
            base_delay=cfg.broker.reconnect_base_delay,
            max_delay=cfg.broker.reconnect_max_delay,
            on_reconnect=self._on_reconnect,
        )

        self.market_data = MarketDataService(
            broker=self.broker,
            bar_store=self.bar_store,
            universe_cfg=cfg.universe,
            broker_cfg=cfg.broker,
            data_cfg=cfg.data,
        )
        self.market_data.universe = s.load_universe()

        self.orders = OrderManager(
            broker=self.broker,
            store=self.store,
            limit_offset_pct=cfg.broker.limit_offset_pct,
        )
        self.broker.events.on_order_status = self.orders.on_order_status
        self.broker.events.on_execution = self.orders.on_execution
        self.broker.events.on_error = self._on_broker_error

        # Risk
        self.kill_switch = KillSwitch(
            sentinel=s.kill_file,
            store=self.store,
            action=cfg.risk.kill_switch_action,
        )
        self.risk = RiskEngine(
            cfg=cfg.risk,
            kill_switch=self.kill_switch,
            store=self.store,
            order_manager=self.orders,
        )

        # LLM
        provider = build_provider(
            cfg.llm.provider,
            api_key=s.secrets.ollama_api_key,
            cloud_host=cfg.llm.cloud_host,
            local_host=cfg.llm.local_host,
        )
        self.llm = LLMGateway(
            provider=provider,
            audit=AuditLog(s.audit_dir, enabled=cfg.llm.audit_enabled),
            store=self.store,
            max_concurrent=cfg.llm.max_concurrent_requests,
            max_retries=cfg.llm.max_retries,
            retry_base_delay=cfg.llm.retry_base_delay,
            cache_enabled=cfg.llm.cache_enabled,
            # Never serve a cached decision to a live cycle: the market moved
            # even if the prompt hashed identically.
            cache_read_enabled=False,
        )

        self.narrative = SessionNarrative(
            max_chars=cfg.llm.narrative_max_chars, store=self.store
        )
        self.agents = AgentRunner(gateway=self.llm, cfg=cfg.llm, narrative=self.narrative)

        # Scheduling
        self.calendar = MarketCalendar(
            premarket_minutes=cfg.cadence.premarket_start_minutes_before_open,
            opening_range_minutes=cfg.cadence.opening_range_minutes,
        )
        self.focus_manager = FocusListManager(
            size=cfg.universe.focus_list_size,
            churn_margin=cfg.universe.churn_margin,
            persistence_cycles=cfg.universe.churn_persistence_cycles,
            max_promotions_per_cycle=cfg.universe.max_promotions_per_cycle,
        )
        self.cycle = DecisionCycle(
            settings=s,
            state=self.state,
            market_data=self.market_data,
            agents=self.agents,
            risk=self.risk,
            orders=self.orders,
            calendar=self.calendar,
            focus_manager=self.focus_manager,
        )

    # ------------------------------------------------------------------ #
    # startup
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        cfg = self.settings.strategy

        for warning in cfg.risk.clamp_warnings:
            log.warning("config_clamped_to_hard_limit", detail=warning)

        account_id = await self.gateway.connect()

        # The live/paper interlock. Any disagreement aborts rather than
        # continuing in an ambiguous state.
        try:
            self.settings.verify_live_interlock(account_id)
        except LiveTradingNotPermitted as exc:
            log.error("LIVE_TRADING_INTERLOCK_FAILED", detail=str(exc))
            await self.gateway.disconnect()
            raise

        log.info(
            "trading_mode_confirmed",
            mode=self.state.mode,
            account=account_id,
            port=self.settings.port,
        )

        # Restore what we knew before any restart.
        self.orders.load_from_store()
        self.narrative.load_from_store(self.store)
        last_review = self.store.load_last_review()
        if last_review:
            self.narrative.yesterday_lessons = last_review.get("lessons_for_tomorrow", [])

        account = await self._account()
        self.state.account = account
        if self.state.starting_equity <= 0:
            saved = self.store.get_state("starting_equity")
            today = datetime.now(UTC).date().isoformat()
            if isinstance(saved, dict) and saved.get("day") == today:
                self.state.starting_equity = float(saved.get("equity", account.equity))
            else:
                self.state.starting_equity = account.equity
                self.store.set_state(
                    "starting_equity", {"day": today, "equity": account.equity}
                )
            self.state.peak_equity = max(self.state.starting_equity, account.equity)

        await self._reconcile()

        # Qualify contracts once; the conId cache makes this cheap afterwards.
        universe = self.market_data.universe
        if universe:
            await self.broker.qualify(universe[:200])

        await self.market_data.load_cached(universe[:100])

        # Negotiate the structured-output strategy before the first real call.
        try:
            await self.llm.ensure_strategy(cfg.llm.trader.model)
        except Exception as exc:  # noqa: BLE001
            log.warning("llm_strategy_negotiation_failed", error=str(exc))

        self._spawn_loops()
        log.info("engine_started", mode=self.state.mode, universe=len(universe))

    def _spawn_loops(self) -> None:
        cad = self.settings.strategy.cadence
        specs = [
            ("watchdog", self.gateway.watchdog(cad.connection_check_seconds)),
            ("account_sync", self._loop(self._sync_account, cad.account_sync_seconds)),
            ("fast", self._loop(self._fast_tick, cad.fast_loop_seconds)),
            ("scanner", self._loop(self._scanner_tick, cad.scanner_refresh_seconds)),
            ("schedule", self._loop(self._schedule_tick, 30.0)),
            ("decision", self._decision_loop()),
        ]
        for name, coro in specs:
            self._tasks.append(asyncio.create_task(coro, name=name))

    async def _loop(self, fn: Any, interval: float) -> None:
        """Run `fn` every `interval` seconds until stopped.

        A failing iteration is logged and the loop continues: one bad tick must
        never take down a loop that other safety behaviour depends on.
        """
        while not self._stop.is_set():
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("loop_iteration_failed", loop=fn.__name__, error=str(exc))
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval)

    # ------------------------------------------------------------------ #
    # loops
    # ------------------------------------------------------------------ #

    async def _account(self):
        account = await self.broker.account_snapshot()
        self.state.account = account
        self.state.update_peak()
        return account

    async def _sync_account(self) -> None:
        if not self.broker.is_connected:
            return
        await self._account()

    async def _fast_tick(self) -> None:
        """Deterministic safety loop. Never calls the LLM."""
        session = self.calendar.info()
        self.state.phase = session.phase
        self.state.session_description = self.calendar.describe()
        self.state.connection = self.gateway.status()
        self.state.llm = self.llm.health.as_dict()
        self.state.market_data = self.market_data.status()
        self.state.risk = self.risk.status()

        if not self.broker.is_connected or self.state.account is None:
            return

        account = self.state.account

        # Daily loss limit, checked out of band so a losing position can halt
        # trading even when the model is proposing nothing.
        if self.risk.check_daily_loss(account, self.state.starting_equity):
            self.state.halted_reason = "daily loss limit"

        # Kill switch in flatten mode.
        if self.kill_switch.should_flatten and account.open_positions:
            log.error("kill_switch_flattening_all_positions")
            await self._flatten_all("kill switch")

        # End-of-day flatten. Being flat overnight is the default: this is a
        # day-trading system, and it removes both gap risk and any exposure to
        # the weekend re-authentication window.
        cfg = self.settings.strategy.risk
        if (
            session.is_trading_day
            and 0 < session.minutes_to_close <= cfg.flatten_minutes_before_close
            and account.open_positions
        ):
            log.info("eod_flatten_triggered", minutes_to_close=session.minutes_to_close)
            await self._flatten_all("end of day")

        await self._expire_stale_orders()

    async def _expire_stale_orders(self) -> None:
        """Cancel entry limits that never filled.

        Leaving them working into the next cycle means acting on a decision the
        model made against a market that has since moved.
        """
        timeout = self.settings.strategy.broker.order_timeout_seconds
        now = datetime.now(UTC)
        for order in self.orders.working_orders():
            if order.parent_ref is not None:
                continue  # protective legs stay until the position closes
            age = (now - order.created_at).total_seconds()
            if age > timeout and order.filled_quantity == 0 and order.ib_order_id:
                try:
                    await self.broker.cancel_order(order.ib_order_id)
                    log.info(
                        "stale_entry_cancelled",
                        symbol=order.symbol, order_ref=order.order_ref, age=round(age),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("stale_cancel_failed", order_ref=order.order_ref, error=str(exc))

    async def _scanner_tick(self) -> None:
        if not self.gateway.is_tradeable:
            return
        session = self.calendar.info()
        if session.phase in (SessionPhase.CLOSED,):
            return
        await self.market_data.refresh_scanners()

    async def _schedule_tick(self) -> None:
        """Fire the once-a-day passes at the right point in the session."""
        session = self.calendar.info()
        if not session.is_trading_day or not self.gateway.is_tradeable:
            return

        cad = self.settings.strategy.cadence

        if (
            not self.state.premarket_done
            and session.phase == SessionPhase.PREMARKET
            and session.minutes_to_open <= cad.premarket_start_minutes_before_open
        ):
            await self.cycle.run_premarket(self._account)

        if (
            not self.state.eod_done
            and session.phase == SessionPhase.AFTER_HOURS
            and session.minutes_to_close < -5
        ):
            await self.cycle.run_eod(self._account)

    async def _decision_loop(self) -> None:
        """The cycle driver, aligned to bar boundaries.

        The offset past the boundary matters: firing exactly on it means acting
        on a bar that has not finished forming yet.
        """
        interval = self.settings.strategy.cadence.decision_interval_seconds
        offset = 20

        while not self._stop.is_set():
            try:
                now = datetime.now(UTC)
                epoch = int(now.timestamp())
                next_boundary = epoch - (epoch % interval) + interval + offset
                wait = max(next_boundary - epoch, 1)
                with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=wait)
                if self._stop.is_set():
                    return

                await self._maybe_run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("decision_loop_failed", error=str(exc))
                await asyncio.sleep(10)

    async def _maybe_run_cycle(self) -> None:
        session = self.calendar.info()

        if not session.phase.is_tradeable:
            return
        if not self.gateway.is_tradeable:
            log.debug("cycle_skipped_not_connected")
            return
        if self.kill_switch.is_tripped:
            return
        if not self.state.premarket_done:
            # Started mid-session: do the planning pass now rather than trading blind.
            await self.cycle.run_premarket(self._account)
            return
        if self.llm.health.deterministic_only:
            log.info("cycle_skipped_llm_disabled_managing_existing_only")
            return

        record = await self.cycle.run(session, self._account)
        log.info(
            "cycle_complete",
            cycle_id=record.cycle_id,
            focus=len(record.focus),
            proposals=record.proposals,
            approved=record.approved,
            rejected=record.rejected,
            llm_ms=record.llm_latency_ms,
            total_ms=record.duration_ms,
        )

        if self.narrative.needs_compaction():
            await self.agents.compact_narrative()

    # ------------------------------------------------------------------ #

    async def _flatten_all(self, reason: str) -> int:
        account = self.state.account
        if account is None:
            return 0
        closed = 0
        await self.orders.cancel_all_working()
        for symbol, position in list(account.open_positions.items()):
            try:
                await self.orders.close_position(
                    symbol, position.quantity, position.is_long, reason=reason
                )
                self.state.note_close(symbol)
                self.narrative.record_exit(symbol, reason)
                closed += 1
            except Exception as exc:
                log.exception("flatten_failed", symbol=symbol, error=str(exc))
        if closed:
            log.warning("flattened_positions", count=closed, reason=reason)
        return closed

    async def _reconcile(self) -> None:
        account = await self._account()
        report = await self.orders.reconcile(account)
        self.state.reconciliation = report.as_dict()
        if report.protective_stops_placed:
            log.error(
                "PROTECTIVE_STOPS_ADDED_DURING_RECONCILIATION",
                symbols=report.protective_stops_placed,
                detail="positions were found without a resting stop at the broker",
            )

    async def _on_reconnect(self) -> None:
        log.info("reconnected_reconciling")
        await self._reconcile()

    def _on_broker_error(self, code: int, message: str, kind: str) -> None:
        if kind == "PACING":
            self.market_data.pacer.register_violation()
        elif kind == "NO_SUBSCRIPTION":
            # Extract the symbol if we can; otherwise this is still worth a loud
            # log, because there is no delayed fallback to fall back to.
            self.state.last_error = f"market data not subscribed: {message}"

    # ------------------------------------------------------------------ #

    async def stop(self) -> None:
        log.info("engine_stopping")
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        with contextlib.suppress(Exception):
            await self.market_data.persist_session()
        with contextlib.suppress(Exception):
            await self.gateway.disconnect()
        with contextlib.suppress(Exception):
            self.store.close()
        with contextlib.suppress(Exception):
            self.bar_store.close()
        log.info("engine_stopped")

    async def shutdown_gracefully(self) -> None:
        """On SIGTERM: cancel working entries, leave protective stops in place.

        Deliberately does *not* flatten. Brackets resting at IBKR are safer than
        a rushed market exit during a container restart, and this is exactly the
        situation they exist for.
        """
        log.info("graceful_shutdown_started")
        with contextlib.suppress(Exception):
            for order in self.orders.working_orders():
                if order.parent_ref is None and order.ib_order_id:
                    await self.broker.cancel_order(order.ib_order_id)
        await self.stop()

    # ------------------------------------------------------------------ #

    async def trip_kill_switch(self, reason: str, action: KillSwitchAction | None = None) -> None:
        self.kill_switch.trip(reason, action)
        if self.kill_switch.should_flatten:
            await self._flatten_all(f"kill switch: {reason}")

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()
