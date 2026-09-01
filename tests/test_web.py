"""The dashboard, exercised over real HTTP against a real ASGI app.

This is what actually proves the pages render — a type checker can catch a
signature mismatch, but only serving the templates for real catches a Jinja
error or a missing attribute on the engine object passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from aitrader.data.store import StateStore
from aitrader.domain.enums import ConnectionState, SessionPhase
from aitrader.llm.narrative import SessionNarrative
from aitrader.risk.killswitch import KillSwitch
from aitrader.web.app import create_app


@dataclass
class FakeOrders:
    orders: dict = field(default_factory=dict)
    fills: list = field(default_factory=list)


@dataclass
class FakeMarketData:
    scanner_hits: dict = field(default_factory=dict)

    def status(self) -> dict:
        return {
            "universe_size": 3, "subscribed": [], "lines_used": 0, "lines_capacity": 90,
            "historical_utilization": 0.0, "backfill_complete": True,
            "backfill_progress": (0, 0), "scanner_codes": [], "no_subscription": [],
            "symbols_with_intraday": 0,
        }


class FakeEngine:
    """Just enough of `TradingEngine`'s surface for the dashboard routes."""

    def __init__(self, tmp_path):
        self.store = StateStore(tmp_path / "web_test.sqlite3")
        self.narrative = SessionNarrative(store=self.store)
        self.narrative.append("plan", "Bias NEUTRAL today.")
        self.narrative.append("placed", "BUY 10 AAPL at 101.20")

        self.state = SimpleNamespace(
            plan=None, focus=["AAPL", "MSFT"], features={}, phase=SessionPhase.RTH,
            execution_mode="auto", pending_approvals=[],
        )
        self.orders = FakeOrders()
        self.market_data = FakeMarketData()
        self.kill_switch = KillSwitch(sentinel=tmp_path / "KILL", store=self.store)

        self.gateway = SimpleNamespace(
            is_tradeable=True,
            status=lambda: {
                "state": ConnectionState.CONNECTED.value, "account_id": "DU123",
                "needs_manual_2fa": False, "reconnect_attempts": 0, "last_error": None,
                "last_connected_at": None, "ready": True, "tradeable": True,
            },
        )
        self.broker = SimpleNamespace(is_connected=True)
        self.settings = SimpleNamespace(
            strategy=SimpleNamespace(cadence=SimpleNamespace(dashboard_push_seconds=1)),
            get_or_create_dashboard_token=lambda: "test-token",
        )
        self._killed_with: tuple[str, Any] | None = None
        self._approved_id: str | None = None
        self._rejected_id: str | None = None

    def snapshot(self) -> dict:
        return {
            "ts": "2026-08-24T18:00:00+00:00", "mode": "paper", "phase": "RTH",
            "session": "RTH — session 16:30-23:00 Israel time, 120 min to close",
            "uptime_seconds": 120,
            "account": {
                "id": "DU123", "equity": 100_000.0, "cash": 100_000.0,
                "buying_power": 200_000.0, "is_paper": True, "age_seconds": 1.2,
            },
            "pnl": {"day": 150.0, "day_pct": 0.15, "drawdown_pct": 0.0, "starting_equity": 100_000.0},
            "positions": [
                {
                    "symbol": "AAPL", "quantity": 10, "avg_cost": 101.2, "market_price": 102.0,
                    "unrealized_pnl": 8.0, "market_value": 1020.0,
                }
            ],
            "focus": ["AAPL", "MSFT"],
            "plan": None,
            "last_decision": None,
            "cycles": [],
            "execution_mode": self.state.execution_mode,
            "pending_approvals": self.state.pending_approvals,
            "connection": self.gateway.status(),
            "llm": {
                "available": True, "consecutive_failures": 0, "quota_exhausted": False,
                "deterministic_only": False, "last_error": "", "last_success": None,
                "total_calls": 3, "total_tokens": 500, "strategy": "json_mode",
            },
            "market_data": self.market_data.status(),
            "risk": {"kill_switch": self.kill_switch.status()},
            "reconciliation": {},
            "trades_today": 0,
            "halted_reason": "",
            "last_error": "",
        }

    async def trip_kill_switch(self, reason: str, action=None) -> None:
        self._killed_with = (reason, action)
        self.kill_switch.trip(reason, action)

    def set_execution_mode(self, mode) -> None:
        self.state.execution_mode = mode.value if hasattr(mode, "value") else mode

    async def approve_proposal(self, approval_id: str):
        if approval_id == "missing":
            return None
        self._approved_id = approval_id
        return SimpleNamespace(approved=True, detail="")

    def reject_proposal(self, approval_id: str, reason: str = "") -> bool:
        if approval_id == "missing":
            return False
        self._rejected_id = approval_id
        return True


@pytest.fixture
def client(tmp_path):
    engine = FakeEngine(tmp_path)
    app = create_app(engine)
    with TestClient(app) as c:
        c.engine = engine  # type: ignore[attr-defined]
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c


class TestPagesRender:
    """Every server-rendered page must return 200 with real content — this is
    the check that would have caught the TemplateResponse argument-order bug."""

    @pytest.mark.parametrize(
        "path", ["/", "/narrative", "/trades", "/rejections", "/universe"]
    )
    def test_page_returns_200(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 200, resp.text[:500]
        assert "text/html" in resp.headers["content-type"]

    def test_overview_shows_account_data(self, client):
        resp = client.get("/")
        assert "DU123" in resp.text
        assert "100000.00" in resp.text or "100,000.00" in resp.text

    def test_overview_shows_the_position(self, client):
        resp = client.get("/")
        assert "AAPL" in resp.text

    def test_narrative_shows_logged_entries(self, client):
        resp = client.get("/narrative")
        assert "Bias NEUTRAL today" in resp.text
        assert "BUY 10 AAPL" in resp.text

    def test_narrative_with_no_plan_does_not_crash(self, client):
        resp = client.get("/narrative")
        assert resp.status_code == 200

    def test_universe_shows_focus_list(self, client):
        resp = client.get("/universe")
        assert "AAPL" in resp.text and "MSFT" in resp.text

    def test_kill_button_present_on_every_page(self, client):
        for path in ["/", "/narrative", "/trades", "/rejections", "/universe"]:
            resp = client.get(path)
            assert "kill-btn" in resp.text
            assert "flatten-btn" in resp.text

    def test_mode_buttons_present_on_every_page(self, client):
        for path in ["/", "/narrative", "/trades", "/rejections", "/universe"]:
            resp = client.get(path)
            assert "mode-auto-btn" in resp.text
            assert "mode-manual-btn" in resp.text


class TestJsonEndpoints:
    def test_api_state_returns_the_snapshot(self, client):
        resp = client.get("/api/state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["account"]["id"] == "DU123"

    def test_health_ok_when_connected(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestControlEndpoints:
    def test_kill_routes_through_the_engine_not_the_broker_directly(self, client):
        resp = client.post("/control/kill", params={"mode": "halt_new_entries", "reason": "test"})
        assert resp.status_code == 200
        assert client.engine._killed_with is not None
        assert client.engine._killed_with[0] == "test"

    def test_reset_clears_the_kill_switch(self, client):
        client.post("/control/kill", params={"mode": "halt_new_entries", "reason": "t"})
        assert client.engine.kill_switch.is_tripped
        resp = client.post("/control/reset", json={"confirm": True})
        assert resp.status_code == 200
        assert not client.engine.kill_switch.is_tripped

    def test_flatten_mode_is_recorded(self, client):
        client.post("/control/kill", params={"mode": "flatten_all", "reason": "flatten test"})
        from aitrader.domain.enums import KillSwitchAction

        assert client.engine._killed_with[1] == KillSwitchAction.FLATTEN_ALL

    def test_switch_to_manual_needs_no_confirmation(self, client):
        resp = client.post("/control/mode", json={"mode": "manual"})
        assert resp.status_code == 200
        assert client.engine.state.execution_mode == "manual"

    def test_switch_to_auto_without_confirm_is_refused(self, client):
        resp = client.post("/control/mode", json={"mode": "auto"})
        assert resp.status_code == 400
        # Refused before it ever reached the engine.
        assert client.engine.state.execution_mode == "auto"

    def test_switch_to_auto_with_confirm_succeeds(self, client):
        client.post("/control/mode", json={"mode": "manual"})
        resp = client.post("/control/mode", json={"mode": "auto", "confirm": True})
        assert resp.status_code == 200
        assert client.engine.state.execution_mode == "auto"

    def test_unknown_mode_is_rejected(self, client):
        resp = client.post("/control/mode", json={"mode": "sometimes"})
        assert resp.status_code == 400

    def test_approve_routes_through_the_engine_not_the_broker_directly(self, client):
        resp = client.post("/control/approve", json={"id": "abc123"})
        assert resp.status_code == 200
        assert client.engine._approved_id == "abc123"
        assert resp.json()["approved"] is True

    def test_approving_a_trade_that_is_gone_returns_404(self, client):
        resp = client.post("/control/approve", json={"id": "missing"})
        assert resp.status_code == 404

    def test_reject_routes_through_the_engine(self, client):
        resp = client.post("/control/reject", json={"id": "abc123", "reason": "too risky"})
        assert resp.status_code == 200
        assert client.engine._rejected_id == "abc123"

    def test_rejecting_a_trade_that_is_gone_returns_404(self, client):
        resp = client.post("/control/reject", json={"id": "missing"})
        assert resp.status_code == 404
