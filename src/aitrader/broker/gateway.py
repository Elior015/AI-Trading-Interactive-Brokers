"""Broker connection lifecycle.

Two outages are certain and must be absorbed rather than treated as errors:

* **The daily Gateway restart.** IBKR requires it; IBC performs it at
  `AUTO_RESTART_TIME` without needing 2FA. We reconnect with the same clientId
  (order visibility is per-client) and re-reconcile before trading resumes.

* **The weekly re-authentication.** IBKR invalidates the security token on
  Sunday around 01:00 ET, and the next login needs an interactive IB Key tap.
  This cannot be automated — by design. We detect it, refuse to trade, and say
  so loudly rather than pretending otherwise.

Detecting the second case is the subtle part: the socket connects happily while
the session is dead, so we probe the session rather than trusting `isConnected`.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..domain.enums import ConnectionState
from ..domain.models import ConnectionHealth
from ..logging_setup import get_logger
from .port import BrokerPort

log = get_logger(__name__)


@dataclass
class GatewayService:
    """Maintains the broker connection and gates trading on its health."""

    broker: BrokerPort
    host: str
    port: int
    client_id: int
    connect_timeout: float = 20.0
    base_delay: float = 5.0
    max_delay: float = 300.0
    on_reconnect: Callable[[], Any] | None = None

    health: ConnectionHealth = field(default_factory=ConnectionHealth)
    account_id: str = ""
    #: Cleared on disconnect, set after reconciliation. The decision loop waits
    #: on this, so it can never trade on an unreconciled view of positions.
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _reconnecting: bool = False

    def __post_init__(self) -> None:
        self.broker.events.on_disconnect = self._handle_disconnect

    # ------------------------------------------------------------------ #

    async def connect(self) -> str:
        self.health.state = ConnectionState.CONNECTING
        try:
            self.account_id = await self.broker.connect(
                self.host, self.port, self.client_id, self.connect_timeout
            )
        except Exception as exc:
            self.health.state = ConnectionState.DISCONNECTED
            self.health.last_error = str(exc)
            raise

        if not await self.broker.is_authenticated():
            self.health.state = ConnectionState.UNAUTHENTICATED
            self.health.needs_manual_2fa = True
            raise ConnectionError(
                "Connected to IB Gateway but the session is not authenticated. "
                "This is usually IBKR's weekly token reset — approve the IB Key "
                "push on your phone, then the system will reconnect."
            )

        self.health.state = ConnectionState.CONNECTED
        self.health.needs_manual_2fa = False
        self.health.last_connected_at = datetime.now(UTC)
        self.health.last_error = None
        self.health.reconnect_attempts = 0
        log.info("gateway_connected", account=self.account_id, port=self.port)
        return self.account_id

    def _handle_disconnect(self) -> None:
        self.ready.clear()
        if self.health.state == ConnectionState.CONNECTED:
            self.health.state = ConnectionState.DISCONNECTED
            log.warning("gateway_connection_lost")

    async def disconnect(self) -> None:
        self._stop.set()
        self.ready.clear()
        await self.broker.disconnect()
        self.health.state = ConnectionState.DISCONNECTED

    # ------------------------------------------------------------------ #

    async def watchdog(self, interval: float = 10.0) -> None:
        """Reconnect loop. Runs for the life of the process."""
        while not self._stop.is_set():
            try:
                await self._check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("watchdog_iteration_failed", error=str(exc))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval)

    async def _check_once(self) -> None:
        if self._reconnecting:
            return

        if not self.broker.is_connected:
            self.ready.clear()
            await self._reconnect()
            return

        # Connected. Now check the session is actually alive — this is what
        # catches the weekly token reset, which `is_connected` cannot see.
        authenticated = await self.broker.is_authenticated()
        if not authenticated:
            if self.health.state != ConnectionState.UNAUTHENTICATED:
                log.error(
                    "GATEWAY_UNAUTHENTICATED",
                    detail=(
                        "The socket is up but IBKR reports no usable session. This is "
                        "normally the weekly token reset (Sunday ~01:00 ET). Trading is "
                        "halted until you approve the IB Key push on your phone."
                    ),
                )
            self.health.state = ConnectionState.UNAUTHENTICATED
            self.health.needs_manual_2fa = True
            self.ready.clear()
            return

        if self.health.state != ConnectionState.CONNECTED:
            log.info("gateway_session_recovered")
            self.health.state = ConnectionState.CONNECTED
            self.health.needs_manual_2fa = False
            await self._after_connect()

    async def _reconnect(self) -> None:
        self._reconnecting = True
        try:
            attempt = self.health.reconnect_attempts
            delay = min(self.base_delay * (2 ** min(attempt, 6)), self.max_delay)
            delay *= 0.5 + random.random()  # jitter, so retries don't synchronize
            self.health.reconnect_attempts += 1

            log.info(
                "gateway_reconnect_attempt",
                attempt=self.health.reconnect_attempts,
                delay=round(delay, 1),
            )
            await asyncio.sleep(delay)

            try:
                await self.connect()
            except Exception as exc:  # noqa: BLE001
                self.health.last_error = str(exc)
                log.warning(
                    "gateway_reconnect_failed",
                    attempt=self.health.reconnect_attempts,
                    error=str(exc)[:300],
                )
                return
            await self._after_connect()
        finally:
            self._reconnecting = False

    async def _after_connect(self) -> None:
        """Reconcile before allowing trading to resume.

        The decision loop waits on `ready`, so a reconnect can never leave us
        trading against a stale picture of our own positions.
        """
        if self.on_reconnect is not None:
            try:
                result = self.on_reconnect()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.exception("post_reconnect_hook_failed", error=str(exc))
                return
        self.ready.set()

    @property
    def is_tradeable(self) -> bool:
        return self.health.is_tradeable and self.ready.is_set()

    def status(self) -> dict[str, Any]:
        return {
            "state": self.health.state.value,
            "account_id": self.account_id,
            "needs_manual_2fa": self.health.needs_manual_2fa,
            "reconnect_attempts": self.health.reconnect_attempts,
            "last_error": self.health.last_error,
            "last_connected_at": (
                self.health.last_connected_at.isoformat()
                if self.health.last_connected_at
                else None
            ),
            "ready": self.ready.is_set(),
            "tradeable": self.is_tradeable,
        }
