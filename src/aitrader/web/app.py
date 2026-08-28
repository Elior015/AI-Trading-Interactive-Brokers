"""The dashboard: FastAPI + server-rendered HTML + one WebSocket.

Deliberately not a React SPA. This runs in the same process as the trading
engine and reads the same in-memory `AppState` the loop writes — no
serialization boundary, no second container, no build step. For a single-user
local control panel with a read-mostly surface and four buttons, that is a
better trade than a JS toolchain would be.

The dashboard is a *view* onto the engine, never a second path to the broker:
every mutating endpoint goes through `TradingEngine` (the kill switch, the
sentinel file) rather than touching orders or the broker directly.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..domain.enums import KillSwitchAction
from ..logging_setup import get_logger

log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def create_app(engine: Any) -> FastAPI:
    """Build the FastAPI app bound to a running `TradingEngine`."""
    app = FastAPI(title="AI Trader", docs_url=None, redoc_url=None)
    app.state.engine = engine

    static_dir = WEB_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def render(name: str, request: Request, **context: Any) -> HTMLResponse:
        # Current Starlette signature is (request, name, context) — the older
        # (name, {"request": request, ...}) form is deprecated and, on newer
        # versions, mismatches arguments outright rather than warning.
        return TEMPLATES.TemplateResponse(request, name, context)

    # ------------------------------------------------------------------ #
    # pages
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        return render("overview.html", request, snapshot=engine.snapshot())

    @app.get("/narrative", response_class=HTMLResponse)
    async def narrative_page(request: Request) -> HTMLResponse:
        entries = [
            {"role": e.role, "content": e.content, "ts": e.render()}
            for e in engine.narrative.entries[-200:]
        ]
        return render(
            "narrative.html",
            request,
            entries=entries,
            plan=engine.state.plan.model_dump(mode="json") if engine.state.plan else None,
        )

    @app.get("/trades", response_class=HTMLResponse)
    async def trades_page(request: Request) -> HTMLResponse:
        orders = sorted(
            engine.orders.orders.values(), key=lambda o: o.updated_at, reverse=True
        )[:200]
        fills = list(reversed(engine.orders.fills[-200:]))
        return render("trades.html", request, orders=orders, fills=fills)

    @app.get("/rejections", response_class=HTMLResponse)
    async def rejections_page(request: Request) -> HTMLResponse:
        rows = engine.store.load_rejections(limit=200) if engine.store else []
        counts = engine.store.rejection_counts() if engine.store else {}
        return render("rejections.html", request, rows=rows, counts=counts)

    @app.get("/universe", response_class=HTMLResponse)
    async def universe_page(request: Request) -> HTMLResponse:
        md = engine.market_data
        return render(
            "universe.html",
            request,
            status=md.status(),
            focus=engine.state.focus,
            features={
                s: f.to_row() for s, f in sorted(engine.state.features.items())
            },
            scanner_hits=md.scanner_hits,
        )

    # ------------------------------------------------------------------ #
    # control endpoints — every one routes through the engine, never the broker
    # ------------------------------------------------------------------ #

    @app.post("/control/kill")
    async def kill(mode: str = "halt_new_entries", reason: str = "manual dashboard trip") -> JSONResponse:
        action = (
            KillSwitchAction.FLATTEN_ALL
            if mode == "flatten_all"
            else KillSwitchAction.HALT_NEW_ENTRIES
        )
        await engine.trip_kill_switch(reason, action)
        log.warning("dashboard_kill_triggered", mode=mode, reason=reason)
        return JSONResponse({"ok": True, "status": engine.kill_switch.status()})

    @app.post("/control/reset")
    async def reset_kill() -> JSONResponse:
        engine.kill_switch.reset()
        log.warning("dashboard_kill_reset")
        return JSONResponse({"ok": True})

    # ------------------------------------------------------------------ #
    # json + health
    # ------------------------------------------------------------------ #

    @app.get("/health")
    async def health() -> JSONResponse:
        healthy = engine.gateway.is_tradeable or not engine.state.phase.is_tradeable
        return JSONResponse(
            {
                "ok": bool(engine.broker.is_connected),
                "connection": engine.gateway.status(),
                "phase": engine.state.phase.value,
            },
            status_code=200 if healthy or engine.broker.is_connected else 503,
        )

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(engine.snapshot())

    # ------------------------------------------------------------------ #
    # websocket
    # ------------------------------------------------------------------ #

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_json(engine.snapshot())
                await asyncio.sleep(engine.settings.strategy.cadence.dashboard_push_seconds)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("ws_stream_closed", error=str(exc))
        finally:
            with contextlib.suppress(Exception):
                await ws.close()

    return app
