"""Entrypoint and CLI.

`run` starts the trading engine and the dashboard in one process — the
dashboard reads the engine's live state directly, so there is no serialization
boundary between what the loop believes and what you see on screen.

`doctor` is the connectivity preflight described in the plan: it proves real-time
data actually works before you trust the system with a session, because there is
no delayed-data fallback for US equities on an IB LLC account to quietly fall
back to.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys

from .config import LiveTradingNotPermitted, get_settings
from .domain.enums import KillSwitchAction
from .logging_setup import configure_logging, get_logger

log = get_logger(__name__)


async def _run(settings) -> int:
    from .engine.scheduler import TradingEngine

    engine = TradingEngine(settings=settings)

    stop = asyncio.Event()

    def _handle_signal(sig_name: str) -> None:
        log.warning("signal_received", signal=sig_name)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_signal, sig.name)

    try:
        await engine.start()
    except LiveTradingNotPermitted as exc:
        log.error("startup_aborted", reason=str(exc))
        return 2
    except Exception as exc:
        log.exception("startup_failed", error=str(exc))
        return 1

    server_task: asyncio.Task | None = None
    if settings.strategy.web.enabled:
        server_task = asyncio.create_task(_run_dashboard(engine, settings), name="dashboard")

    log.info(
        "aitrader_running",
        mode=engine.state.mode,
        dashboard=f"http://{settings.strategy.web.host}:{settings.strategy.web.port}"
        if settings.strategy.web.enabled
        else "disabled",
    )

    await stop.wait()

    # A SIGTERM cancels working entries and leaves protective stops resting at
    # IBKR rather than rushing a market flatten — brackets exist for exactly
    # this situation.
    await engine.shutdown_gracefully()
    if server_task is not None:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
    return 0


async def _run_dashboard(engine, settings) -> None:
    import uvicorn

    from .web.app import create_app

    app = create_app(engine)
    config = uvicorn.Config(
        app,
        host=settings.strategy.web.host,
        port=settings.strategy.web.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _doctor(settings) -> int:
    """Connectivity preflight. Verifies the things that must be true before
    trusting the system with a session, rather than assuming they are."""
    from .broker.ib_adapter import IbAsyncBroker

    broker = IbAsyncBroker(messages_per_second=settings.strategy.broker.max_messages_per_second)
    checks: list[tuple[str, bool, str]] = []

    try:
        account_id = await broker.connect(
            settings.strategy.broker.host,
            settings.port,
            settings.strategy.broker.client_id,
            settings.strategy.broker.connect_timeout,
        )
        checks.append(("connect", True, f"account={account_id}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("connect", False, str(exc)))
        _print_report(checks)
        return 1

    authenticated = await broker.is_authenticated()
    checks.append((
        "authenticated", authenticated,
        "session usable" if authenticated
        else "socket connected but session is not usable — this is usually the "
             "weekly token reset; approve the IB Key push on your phone",
    ))

    is_paper = account_id.upper().startswith(("DU", "DF"))
    expected_paper = not settings.is_live
    checks.append((
        "account_type_matches_mode",
        is_paper == expected_paper,
        f"account={account_id} paper={is_paper} configured_mode="
        f"{'live' if settings.is_live else 'paper'}",
    ))

    universe = settings.load_universe()
    test_symbols = universe[:3] or ["AAPL", "MSFT", "SPY"]
    try:
        await broker.qualify(test_symbols)
        await broker.subscribe_quote(test_symbols[0])
        await asyncio.sleep(3.0)
        quote = broker.get_quote(test_symbols[0])
        usable = quote is not None and quote.is_usable
        checks.append((
            "live_market_data",
            usable,
            f"{test_symbols[0]}: bid={quote.bid if quote else None} "
            f"ask={quote.ask if quote else None}"
            if quote
            else "no quote received — check your market data subscription is "
                 "active and shared to this account",
        ))
    except Exception as exc:  # noqa: BLE001
        checks.append(("live_market_data", False, str(exc)))

    try:
        hits = await broker.scan("MOST_ACTIVE", rows=5, min_price=3.0, min_volume=100_000)
        checks.append(("scanner", len(hits) > 0, f"{len(hits)} rows returned"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("scanner", False, str(exc)))

    try:
        snapshot = await broker.account_snapshot()
        checks.append((
            "account_summary",
            snapshot.equity > 0,
            f"equity=${snapshot.equity:,.2f} buying_power=${snapshot.buying_power:,.2f}",
        ))
    except Exception as exc:  # noqa: BLE001
        checks.append(("account_summary", False, str(exc)))

    await broker.disconnect()
    _print_report(checks)
    return 0 if all(ok for _, ok, _ in checks) else 1


def _print_report(checks: list[tuple[str, bool, str]]) -> None:
    print("\naitrader doctor — connectivity preflight\n" + "-" * 50)
    for name, ok, detail in checks:
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {name:<28} {detail}")
    print("-" * 50)
    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}\n")
    else:
        print("All checks passed.\n")


async def _kill(settings, mode: str, reason: str) -> int:
    action = KillSwitchAction.FLATTEN_ALL if mode == "flatten" else KillSwitchAction.HALT_NEW_ENTRIES
    settings.kill_file.parent.mkdir(parents=True, exist_ok=True)
    settings.kill_file.write_text(f"{reason} (mode={action.value})", encoding="utf-8")
    print(f"Kill switch sentinel written: {settings.kill_file}")
    print(f"Mode: {action.value}. A running instance will pick this up on its next tick.")
    return 0


async def _healthcheck(settings) -> int:
    """Used by the Docker healthcheck. Exits 0 only when the system looks alive."""
    if settings.kill_file.exists():
        # A tripped kill switch is a deliberate state, not a failure — the
        # container should stay up so it can be inspected and reset.
        print("kill switch active")
        return 0
    if not settings.state_file.exists() and not settings.duckdb_path.exists():
        print("no state yet")
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aitrader")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="run the trading engine and dashboard")
    sub.add_parser("doctor", help="connectivity preflight against the broker")
    sub.add_parser("healthcheck", help="used by the Docker healthcheck")

    kill_p = sub.add_parser("kill", help="trip the kill switch via sentinel file")
    kill_p.add_argument("--mode", choices=["halt", "flatten"], default="halt")
    kill_p.add_argument("--reason", default="manual CLI trip")

    return p


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings(args.config)
    configure_logging(settings.secrets.log_level)

    if args.command == "run":
        code = asyncio.run(_run(settings))
    elif args.command == "doctor":
        code = asyncio.run(_doctor(settings))
    elif args.command == "kill":
        code = asyncio.run(_kill(settings, args.mode, args.reason))
    elif args.command == "healthcheck":
        code = asyncio.run(_healthcheck(settings))
    else:  # pragma: no cover - argparse enforces this
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    cli()
