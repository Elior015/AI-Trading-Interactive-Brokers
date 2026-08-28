"""US equity session state machine.

All scheduling is computed in exchange-local time from the XNYS calendar and
converted for display only. This matters more than it looks: the US session is
16:30-23:00 Israel time for most of the year, but Israel and the US change
daylight saving on *different dates*, so for a few weeks each year the offset
shifts by an hour. Hard-coding Israeli clock times would mean the bot starts
trading an hour late (or early) twice a year.

Half-days and holidays come free with `exchange_calendars`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from ..domain.enums import SessionPhase
from ..logging_setup import get_logger

log = get_logger(__name__)

ET = ZoneInfo("America/New_York")
ISRAEL = ZoneInfo("Asia/Jerusalem")


@dataclass
class SessionInfo:
    phase: SessionPhase
    is_trading_day: bool
    open_at: datetime | None
    close_at: datetime | None
    minutes_to_open: float
    minutes_to_close: float
    is_half_day: bool = False

    @property
    def is_tradeable(self) -> bool:
        return self.phase.is_tradeable


class MarketCalendar:
    """Thin wrapper over the XNYS calendar."""

    def __init__(
        self,
        premarket_minutes: int = 150,
        opening_range_minutes: int = 15,
        closing_minutes: int = 30,
    ) -> None:
        self.calendar = xcals.get_calendar("XNYS")
        self.premarket_minutes = premarket_minutes
        self.opening_range_minutes = opening_range_minutes
        self.closing_minutes = closing_minutes

    @lru_cache(maxsize=64)  # noqa: B019 - bounded by trading days, fine for a long-lived process
    def _session_bounds(self, day: date) -> tuple[datetime, datetime] | None:
        try:
            if not self.calendar.is_session(day):
                return None
            open_ts = self.calendar.session_open(day)
            close_ts = self.calendar.session_close(day)
        except Exception as exc:  # noqa: BLE001
            log.warning("calendar_lookup_failed", day=str(day), error=str(exc))
            return None
        return open_ts.to_pydatetime(), close_ts.to_pydatetime()

    def session_bounds(self, day: date | None = None) -> tuple[datetime, datetime] | None:
        """Open and close for `day`, in UTC. None when it is not a trading day."""
        d = day or datetime.now(ET).date()
        return self._session_bounds(d)

    def is_trading_day(self, day: date | None = None) -> bool:
        return self.session_bounds(day) is not None

    def next_trading_day(self, after: date | None = None) -> date:
        d = (after or datetime.now(ET).date()) + timedelta(days=1)
        for _ in range(10):
            if self.is_trading_day(d):
                return d
            d += timedelta(days=1)
        return d

    def info(self, now: datetime | None = None) -> SessionInfo:
        """Where we are in the trading day right now."""
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        et_date = now.astimezone(ET).date()
        bounds = self.session_bounds(et_date)

        if bounds is None:
            return SessionInfo(
                phase=SessionPhase.CLOSED,
                is_trading_day=False,
                open_at=None,
                close_at=None,
                minutes_to_open=float("inf"),
                minutes_to_close=float("inf"),
            )

        open_at, close_at = bounds
        to_open = (open_at - now).total_seconds() / 60.0
        to_close = (close_at - now).total_seconds() / 60.0

        # A half day closes early (typically 13:00 ET).
        full_length = (close_at - open_at).total_seconds() / 3600.0
        is_half_day = full_length < 6.0

        if now < open_at:
            phase = (
                SessionPhase.PREMARKET
                if to_open <= self.premarket_minutes
                else SessionPhase.CLOSED
            )
        elif now >= close_at:
            phase = SessionPhase.AFTER_HOURS
        else:
            elapsed = (now - open_at).total_seconds() / 60.0
            if elapsed <= self.opening_range_minutes:
                phase = SessionPhase.OPENING_RANGE
            elif to_close <= self.closing_minutes:
                phase = SessionPhase.CLOSING
            else:
                phase = SessionPhase.RTH

        return SessionInfo(
            phase=phase,
            is_trading_day=True,
            open_at=open_at,
            close_at=close_at,
            minutes_to_open=to_open,
            minutes_to_close=to_close,
            is_half_day=is_half_day,
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def to_israel(dt: datetime) -> datetime:
        """Convert for display. Never used for scheduling."""
        return dt.astimezone(ISRAEL)

    @staticmethod
    def to_et(dt: datetime) -> datetime:
        return dt.astimezone(ET)

    def describe(self, now: datetime | None = None) -> str:
        """A human-readable line for the dashboard, in the user's local time."""
        now = now or datetime.now(UTC)
        info = self.info(now)
        if not info.is_trading_day:
            # Anchor the lookup on the date being described, not wall-clock
            # "now" — otherwise describing a past or future date silently
            # reports the wrong next session.
            nxt = self.next_trading_day(now.astimezone(ET).date())
            return f"Market closed. Next trading day: {nxt.isoformat()}"
        if info.open_at is None or info.close_at is None:
            return "Market closed."
        open_il = self.to_israel(info.open_at).strftime("%H:%M")
        close_il = self.to_israel(info.close_at).strftime("%H:%M")
        half = " (half day)" if info.is_half_day else ""
        return (
            f"{info.phase.value} — session {open_il}-{close_il} Israel time{half}, "
            f"{info.minutes_to_close:.0f} min to close"
        )
