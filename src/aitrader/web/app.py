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
import secrets as _secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..domain.enums import KillSwitchAction
from ..logging_setup import get_logger

log = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))


class ResetRequest(BaseModel):
    confirm: bool = False


def create_app(engine: Any) -> FastAPI:
    """Build the FastAPI app bound to a running `TradingEngine`."""
    app = FastAPI(title="AI Trader", docs_url=None, redoc_url=None)
    app.state.engine = engine

    # Bearer token gating everything except /health (the Docker healthcheck
    # hits it with no auth header) and /static. A page load can't set a
    # custom header, so the token is also accepted as ?token=... — visit
    # once at http://host:port/?token=<the value printed at startup>.
    dashboard_token = engine.settings.get_or_create_dashboard_token()
    log.warning(
        "dashboard_token_ready",
        detail="append ?token=<value> to the dashboard URL, or send it as "
        "'Authorization: Bearer <value>' — see secrets/dashboard_token.txt",
    )

    def _token_ok(supplied: str | None) -> bool:
        return bool(supplied) and _secrets.compare_digest(supplied, dashboard_token)

    def require_token(
        request: Request, authorization: str | None = Header(default=None)
    ) -> None:
        supplied = request.query_params.get("token")
        if not supplied and authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not _token_ok(supplied):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "missing or invalid dashboard token"
            )

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

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_token)])
    async def overview(request: Request) -> HTMLResponse:
        return render("overview.html", request, snapshot=engine.snapshot())

    @app.get(
        "/narrative", response_class=HTMLResponse, dependencies=[Depends(require_token)]
    )
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

    @app.get("/trades", response_class=HTMLResponse, dependencies=[Depends(require_token)])
    async def trades_page(request: Request) -> HTMLResponse:
        orders = sorted(
            engine.orders.orders.values(), key=lambda o: o.updated_at, reverse=True
        )[:200]
        fills = list(reversed(engine.orders.fills[-200:]))
        return render("trades.html", request, orders=orders, fills=fills)

    @app.get(
        "/rejections", response_class=HTMLResponse, dependencies=[Depends(require_token)]
    )
    async def rejections_page(request: Request) -> HTMLResponse:
        rows = engine.store.load_rejections(limit=200) if engine.store else []
        counts = engine.store.rejection_counts() if engine.store else {}
        return render("rejections.html", request, rows=rows, counts=counts)

    @app.get(
        "/universe", response_class=HTMLResponse, dependencies=[Depends(require_token)]
    )
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

    @app.post("/control/kill", dependencies=[Depends(require_token)])
    async def kill(mode: str = "halt_new_entries", reason: str = "manual dashboard trip") -> JSONResponse:
        action = (
            KillSwitchAction.FLATTEN_ALL
            if mode == "flatten_all"
            else KillSwitchAction.HALT_NEW_ENTRIES
        )
        await engine.trip_kill_switch(reason, action)
        log.warning("dashboard_kill_triggered", mode=mode, reason=reason)
        return JSONResponse({"ok": True, "status": engine.kill_switch.status()})

    @app.post("/control/reset", dependencies=[Depends(require_token)])
    async def reset_kill(body: ResetRequest) -> JSONResponse:
        # Un-tripping is the dangerous direction — it re-enables trading —
        # so it takes an explicit confirmation, not just the token.
        if not body.confirm:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                'POST {"confirm": true} to reset the kill switch',
            )
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

    @app.get("/api/state", dependencies=[Depends(require_token)])
    async def api_state() -> JSONResponse:
        return JSONResponse(engine.snapshot())

    # ------------------------------------------------------------------ #
    # websocket
    # ------------------------------------------------------------------ #

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:
        # A WebSocket handshake can't carry a Depends() dependency the same
        # way; check the token from the query string before accepting.
        if not _token_ok(ws.query_params.get("token")):
            await ws.close(code=4401)
            return
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
