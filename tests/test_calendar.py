"""The session calendar.

The Israel/US daylight-saving gap is the case that actually matters here: for a
few weeks each year the two regions are offset by different amounts, so any
design that hard-codes "market opens at 16:30 Israel time" is wrong for part of
the year. Everything is computed in exchange-local time instead.
"""

from __future__ import annotations

from datetime import datetime

from aitrader.domain.enums import SessionPhase
from aitrader.engine.calendar import ET, ISRAEL, MarketCalendar


def utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


class TestSessionPhases:
    def test_before_premarket_window_is_closed(self):
        cal = MarketCalendar(premarket_minutes=150)
        # 2026-08-24 is a Monday; market opens 13:30 UTC.
        info = cal.info(utc("2026-08-24T09:00:00Z"))
        assert info.phase == SessionPhase.CLOSED
        assert info.is_trading_day

    def test_within_premarket_window(self):
        cal = MarketCalendar(premarket_minutes=150)
        info = cal.info(utc("2026-08-24T12:00:00Z"))
        assert info.phase == SessionPhase.PREMARKET

    def test_opening_range(self):
        cal = MarketCalendar(opening_range_minutes=15)
        info = cal.info(utc("2026-08-24T13:35:00Z"))
        assert info.phase == SessionPhase.OPENING_RANGE

    def test_regular_trading_hours(self):
        cal = MarketCalendar()
        info = cal.info(utc("2026-08-24T18:00:00Z"))
        assert info.phase == SessionPhase.RTH
        assert info.phase.is_tradeable

    def test_closing_window(self):
        cal = MarketCalendar(closing_minutes=30)
        info = cal.info(utc("2026-08-24T19:45:00Z"))
        assert info.phase == SessionPhase.CLOSING

    def test_after_hours(self):
        cal = MarketCalendar()
        info = cal.info(utc("2026-08-24T21:00:00Z"))
        assert info.phase == SessionPhase.AFTER_HOURS
        assert not info.phase.is_tradeable

    def test_weekend_is_not_a_trading_day(self):
        cal = MarketCalendar()
        info = cal.info(utc("2026-08-23T18:00:00Z"))  # a Sunday
        assert not info.is_trading_day
        assert info.phase == SessionPhase.CLOSED
        assert info.minutes_to_open == float("inf")


class TestHalfDays:
    def test_day_after_thanksgiving_closes_early(self):
        cal = MarketCalendar()
        bounds = cal.session_bounds(datetime(2026, 11, 27).date())
        assert bounds is not None
        _, close = bounds
        assert close.astimezone(ET).strftime("%H:%M") == "13:00"

    def test_half_day_flag_is_set(self):
        cal = MarketCalendar()
        info = cal.info(utc("2026-11-27T16:00:00Z"))
        assert info.is_half_day

    def test_regular_day_is_not_flagged_half(self):
        cal = MarketCalendar()
        info = cal.info(utc("2026-08-24T16:00:00Z"))
        assert not info.is_half_day


class TestIsraelDstGap:
    """The whole reason scheduling must never be done in Israel-local time."""

    def test_us_dst_ends_before_israel_in_late_october(self):
        cal = MarketCalendar()
        bounds = cal.session_bounds(datetime(2026, 10, 27).date())
        assert bounds is not None
        open_at, close_at = bounds
        open_il = open_at.astimezone(ISRAEL).strftime("%H:%M")
        close_il = close_at.astimezone(ISRAEL).strftime("%H:%M")
        # During the gap weeks the session is an hour earlier in Israel time
        # than the "normal" 16:30-23:00 window.
        assert open_il == "15:30"
        assert close_il == "22:00"

    def test_normal_summer_session_is_1630_2300_israel(self):
        cal = MarketCalendar()
        bounds = cal.session_bounds(datetime(2026, 8, 24).date())
        assert bounds is not None
        open_at, close_at = bounds
        assert open_at.astimezone(ISRAEL).strftime("%H:%M") == "16:30"
        assert close_at.astimezone(ISRAEL).strftime("%H:%M") == "23:00"

    def test_winter_session_is_1630_2300_israel_too(self):
        """Once both regions are on standard time the offset returns to normal."""
        cal = MarketCalendar()
        bounds = cal.session_bounds(datetime(2026, 12, 15).date())
        assert bounds is not None
        open_at, close_at = bounds
        assert open_at.astimezone(ISRAEL).strftime("%H:%M") == "16:30"
        assert close_at.astimezone(ISRAEL).strftime("%H:%M") == "23:00"


class TestNextTradingDay:
    def test_skips_the_weekend(self):
        cal = MarketCalendar()
        # Friday -> next trading day is Monday.
        nxt = cal.next_trading_day(datetime(2026, 8, 21).date())
        assert nxt.weekday() == 0
        assert nxt == datetime(2026, 8, 24).date()

    def test_skips_a_holiday(self):
        cal = MarketCalendar()
        # Around Thanksgiving 2026 (Nov 26) — the 27th is a half day, not
        # closed, so this just checks we land on an actual session.
        nxt = cal.next_trading_day(datetime(2026, 11, 25).date())
        assert cal.is_trading_day(nxt)


class TestDescribe:
    def test_produces_a_readable_line(self):
        cal = MarketCalendar()
        text = cal.describe(utc("2026-08-24T18:00:00Z"))
        assert "RTH" in text
        assert "Israel time" in text

    def test_closed_day_names_the_next_session(self):
        cal = MarketCalendar()
        text = cal.describe(utc("2026-08-23T12:00:00Z"))
        assert "closed" in text.lower()
        assert "2026-08-24" in text
