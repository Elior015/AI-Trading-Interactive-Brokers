"""Market data: the three-tier universe that makes 100+ symbols affordable.

IBKR's limits are the binding constraint on this whole system:

* 60 historical requests per rolling 10 minutes
* ~100 concurrent market-data lines
* 60 new real-time-bar subscriptions per 10 minutes

The design that fits inside them:

* **Broad tier** (100-500 names): daily bars backfilled once pre-market through
  a token bucket, then never re-fetched. Intraday interest comes from IBKR's
  server-side scanner, which returns no quote fields and therefore costs no
  market-data lines at all — this is the single fact that makes a large universe
  workable.
* **Focus tier** (~20 names): real streaming subscriptions. Once a symbol is
  streaming, its bars build themselves from 5-second real-time bars and we never
  issue another historical request for it.
* **Positions**: always subscribed regardless of rank. Dropping the feed for
  something we hold would blind the exit logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..config import BrokerConfig, DataConfig, UniverseConfig
from ..data.ratelimit import HistoricalDataPacer, MarketDataLineBudget
from ..data.store import BarStore
from ..domain.models import Bar, Quote
from ..logging_setup import get_logger
from .port import BrokerPort

log = get_logger(__name__)

DAILY_PERIOD = 86400
INTRADAY_PERIOD = 300


def aggregate(bars: list[Bar], period: int) -> list[Bar]:
    """Roll finer bars up into `period`-second bars.

    Used to build 5-minute bars from the 5-second real-time stream, so a
    streaming symbol never needs another historical request.
    """
    if not bars:
        return []
    buckets: dict[int, list[Bar]] = defaultdict(list)
    for b in bars:
        epoch = int(b.ts.timestamp())
        buckets[epoch - (epoch % period)].append(b)

    out: list[Bar] = []
    for start in sorted(buckets):
        group = sorted(buckets[start], key=lambda x: x.ts)
        out.append(
            Bar(
                symbol=group[0].symbol,
                ts=datetime.fromtimestamp(start, tz=UTC),
                open=group[0].open,
                high=max(g.high for g in group),
                low=min(g.low for g in group),
                close=group[-1].close,
                volume=sum(g.volume for g in group),
                period=period,
            )
        )
    return out


@dataclass
class MarketDataService:
    broker: BrokerPort
    bar_store: BarStore
    universe_cfg: UniverseConfig
    broker_cfg: BrokerConfig
    data_cfg: DataConfig

    pacer: HistoricalDataPacer = field(init=False)
    lines: MarketDataLineBudget = field(init=False)

    universe: list[str] = field(default_factory=list)
    subscribed: set[str] = field(default_factory=set)
    #: symbol -> 5-second bars accumulated this session
    raw_bars: dict[str, list[Bar]] = field(default_factory=lambda: defaultdict(list))
    intraday: dict[str, list[Bar]] = field(default_factory=dict)
    daily: dict[str, list[Bar]] = field(default_factory=dict)
    scanner_hits: dict[str, list[str]] = field(default_factory=dict)
    backfill_complete: bool = False
    backfill_progress: tuple[int, int] = (0, 0)
    #: Symbols IBKR told us we have no market-data subscription for.
    no_subscription: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.pacer = HistoricalDataPacer(
            capacity=50,
            window=600.0,
            min_spacing=self.broker_cfg.historical_request_period / 10.0 or 1.0,
        )
        self.lines = MarketDataLineBudget(self.broker_cfg.max_market_data_lines)
        self.broker.events.on_bar = self._on_bar

    # ------------------------------------------------------------------ #
    # streaming
    # ------------------------------------------------------------------ #

    def _on_bar(self, bar: Bar) -> None:
        """Accumulate 5-second bars and roll them into 5-minute bars."""
        series = self.raw_bars[bar.symbol]
        series.append(bar)
        # Keep a bounded window; the durable copy is in DuckDB.
        if len(series) > 5000:
            del series[:-4000]

        rolled = aggregate(series, INTRADAY_PERIOD)
        if rolled:
            existing = self.intraday.get(bar.symbol, [])
            # Replace only the tail that the streaming data covers, preserving
            # the backfilled history in front of it.
            cutoff = rolled[0].ts
            kept = [b for b in existing if b.ts < cutoff]
            self.intraday[bar.symbol] = (kept + rolled)[-self.data_cfg.memory_bars :]

    async def subscribe(self, symbols: list[str]) -> list[str]:
        """Start streaming for `symbols`, respecting the line budget."""
        added: list[str] = []
        for symbol in symbols:
            if symbol in self.subscribed:
                continue
            # Two lines per symbol: quote plus real-time bars.
            if not await self.lines.reserve(symbol, 2):
                log.warning("subscription_skipped_no_lines", symbol=symbol)
                continue
            try:
                await self.broker.subscribe_quote(symbol)
                await self.broker.subscribe_bars(symbol)
                self.subscribed.add(symbol)
                added.append(symbol)
            except Exception as exc:  # noqa: BLE001
                await self.lines.release(symbol)
                log.warning("subscribe_failed", symbol=symbol, error=str(exc))
        if added:
            log.info(
                "subscribed", symbols=added,
                lines_used=self.lines.used, lines_free=self.lines.free,
            )
        return added

    async def resubscribe(self, symbols: list[str]) -> list[str]:
        """Re-issue subscriptions for `symbols` after a broker reconnect.

        `subscribe()` alone would skip every symbol already in `subscribed`,
        but a reconnect gets a fresh IBKR session that carries none of the
        old subscriptions over. The line budget is left untouched — it still
        reflects lines we intend to hold, and `MarketDataLineBudget.reserve`
        is a no-op for a symbol it already counts.
        """
        self.subscribed.clear()
        return await self.subscribe(symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        for symbol in symbols:
            if symbol not in self.subscribed:
                continue
            try:
                await self.broker.unsubscribe_quote(symbol)
                await self.broker.unsubscribe_bars(symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("unsubscribe_failed", symbol=symbol, error=str(exc))
            self.subscribed.discard(symbol)
            await self.lines.release(symbol)

    def quotes(self, symbols: list[str] | None = None) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for symbol in symbols or list(self.subscribed):
            q = self.broker.get_quote(symbol)
            if q is not None:
                out[symbol] = q
        return out

    # ------------------------------------------------------------------ #
    # historical
    # ------------------------------------------------------------------ #

    async def backfill_universe(self, symbols: list[str] | None = None) -> int:
        """Pre-market bulk backfill of daily bars.

        Paced at roughly one request every second with a rolling 50-per-10-minutes
        cap, so a 300-name universe takes a few minutes of the pre-market window
        and uses none of the intraday budget. Results land in DuckDB and are
        never re-fetched.
        """
        targets = symbols or self.universe
        if not targets:
            return 0

        self.backfill_complete = False
        loaded = 0
        total = len(targets)
        log.info(
            "backfill_started",
            symbols=total,
            estimated_seconds=round(self.pacer.estimated_wait(total)),
        )

        for i, symbol in enumerate(targets, start=1):
            self.backfill_progress = (i, total)
            if symbol in self.no_subscription:
                continue
            try:
                await self.pacer.acquire(symbol, "1 day", "TRADES")
                bars = await self.broker.historical_bars(
                    symbol,
                    duration=f"{self.data_cfg.backfill_days} D",
                    bar_size="1 day",
                    what_to_show="TRADES",
                )
                if bars:
                    self.daily[symbol] = bars
                    await self.bar_store.save_bars(bars)
                    loaded += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("backfill_symbol_failed", symbol=symbol, error=str(exc))

        self.backfill_complete = True
        log.info("backfill_complete", loaded=loaded, requested=total)
        return loaded

    async def load_intraday(self, symbols: list[str]) -> int:
        """Fetch intraday history for newly-promoted symbols.

        Called only on promotion into the focus list, and capped per cycle, so
        intraday historical usage stays far under the 60-per-10-minutes limit.
        """
        loaded = 0
        for symbol in symbols[: self.universe_cfg.max_promotions_per_cycle]:
            if symbol in self.intraday and len(self.intraday[symbol]) > 30:
                continue
            try:
                await self.pacer.acquire(symbol, self.data_cfg.bar_size, "TRADES")
                bars = await self.broker.historical_bars(
                    symbol, duration="2 D", bar_size=self.data_cfg.bar_size,
                    what_to_show="TRADES",
                )
                if bars:
                    self.intraday[symbol] = bars[-self.data_cfg.memory_bars :]
                    await self.bar_store.save_bars(bars)
                    loaded += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("intraday_load_failed", symbol=symbol, error=str(exc))
        return loaded

    async def load_cached(self, symbols: list[str]) -> None:
        """Warm memory from DuckDB after a restart, costing no IBKR requests."""
        for symbol in symbols:
            if symbol not in self.daily:
                bars = await self.bar_store.load_bars(symbol, DAILY_PERIOD, 60)
                if bars:
                    self.daily[symbol] = bars
            if symbol not in self.intraday:
                bars = await self.bar_store.load_bars(
                    symbol, INTRADAY_PERIOD, self.data_cfg.memory_bars
                )
                if bars:
                    self.intraday[symbol] = bars

    # ------------------------------------------------------------------ #
    # scanner
    # ------------------------------------------------------------------ #

    async def refresh_scanners(self) -> dict[str, list[str]]:
        """Run the configured scan codes.

        This is the intraday discovery mechanism for the broad tier. It costs no
        market-data lines and no historical requests, which is what makes
        watching hundreds of names practical.
        """
        results: dict[str, list[str]] = {}
        for code in self.universe_cfg.scan_codes:
            try:
                hits = await self.broker.scan(
                    code,
                    rows=self.universe_cfg.scan_rows_per_code,
                    min_price=self.universe_cfg.scanner_min_price,
                    min_volume=self.universe_cfg.scanner_min_volume,
                )
                results[code] = [h.symbol for h in hits]
            except Exception as exc:  # noqa: BLE001
                log.warning("scanner_refresh_failed", scan_code=code, error=str(exc))
                results[code] = []
        self.scanner_hits = results
        total = len({s for v in results.values() for s in v})
        log.debug("scanners_refreshed", codes=len(results), unique_symbols=total)
        return results

    def scanner_universe(self) -> list[str]:
        """Every symbol any scanner surfaced, plus the static universe."""
        found = {s for symbols in self.scanner_hits.values() for s in symbols}
        return sorted(found | set(self.universe))

    # ------------------------------------------------------------------ #

    def mark_no_subscription(self, symbol: str) -> None:
        """Record that IBKR refuses data for a symbol.

        There is no delayed fallback for US equities on an IB LLC account, so a
        symbol without data is simply untradeable and must be excluded rather
        than traded on stale prices.
        """
        if symbol not in self.no_subscription:
            self.no_subscription.add(symbol)
            log.error("market_data_not_subscribed", symbol=symbol)

    async def persist_session(self) -> int:
        """Flush accumulated intraday bars to DuckDB."""
        saved = 0
        for bars in self.intraday.values():
            if bars:
                saved += await self.bar_store.save_bars(bars)
        log.info("session_bars_persisted", rows=saved)
        return saved

    def status(self) -> dict[str, Any]:
        return {
            "universe_size": len(self.universe),
            "subscribed": sorted(self.subscribed),
            "lines_used": self.lines.used,
            "lines_capacity": self.lines.capacity,
            "historical_utilization": round(self.pacer.utilization(), 3),
            "backfill_complete": self.backfill_complete,
            "backfill_progress": self.backfill_progress,
            "scanner_codes": list(self.scanner_hits),
            "no_subscription": sorted(self.no_subscription),
            "symbols_with_intraday": len(self.intraday),
        }
