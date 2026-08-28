"""The kill switch.

Three independent sources, because the failure modes are independent:

1. An in-process event — instant, but dies with the process.
2. A row in SQLite — survives a crash of the web layer, settable from the CLI.
3. A sentinel file — works even if Python's HTTP stack is wedged. `touch
   data/KILL` from any shell stops the system.

Worth being honest about the limit: if the trader process is dead or the
Gateway is unauthenticated, no kill switch can help. The only protection then is
the resting stop already at IBKR — which is the argument for native brackets.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..domain.enums import KillSwitchAction
from ..logging_setup import get_logger

log = get_logger(__name__)

STATE_KEY = "kill_switch"


@dataclass
class KillSwitch:
    sentinel: Path
    store: Any = None
    action: KillSwitchAction = KillSwitchAction.HALT_NEW_ENTRIES

    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _reason: str = ""
    _tripped_at: datetime | None = None

    def __post_init__(self) -> None:
        # A kill switch set before a restart must survive that restart.
        if self.sentinel.exists():
            self._event.set()
            self._reason = self._read_sentinel() or "sentinel file present at startup"
            self._tripped_at = datetime.now(UTC)
            log.warning("kill_switch_active_at_startup", reason=self._reason)
        elif self.store is not None:
            saved = self.store.get_state(STATE_KEY)
            if isinstance(saved, dict) and saved.get("tripped"):
                self._event.set()
                self._reason = saved.get("reason", "persisted kill switch")
                log.warning("kill_switch_restored_from_store", reason=self._reason)

    def _read_sentinel(self) -> str:
        try:
            return self.sentinel.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    # ------------------------------------------------------------------ #

    @property
    def is_tripped(self) -> bool:
        """Check all three sources.

        The file is stat'd on every call rather than cached, so an external
        `touch` takes effect within one loop tick.
        """
        if self._event.is_set():
            return True
        if self.sentinel.exists():
            self._event.set()
            self._reason = self._read_sentinel() or "sentinel file created"
            self._tripped_at = datetime.now(UTC)
            log.warning("kill_switch_tripped_by_sentinel", reason=self._reason)
            return True
        return False

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def tripped_at(self) -> datetime | None:
        return self._tripped_at

    def trip(self, reason: str, action: KillSwitchAction | None = None) -> None:
        if action is not None:
            self.action = action
        self._event.set()
        self._reason = reason
        self._tripped_at = datetime.now(UTC)
        try:
            self.sentinel.parent.mkdir(parents=True, exist_ok=True)
            self.sentinel.write_text(reason, encoding="utf-8")
        except OSError as exc:
            log.error("kill_switch_sentinel_write_failed", error=str(exc))
        if self.store is not None:
            try:
                self.store.set_state(
                    STATE_KEY,
                    {
                        "tripped": True,
                        "reason": reason,
                        "action": self.action.value,
                        "ts": self._tripped_at.isoformat(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log.error("kill_switch_persist_failed", error=str(exc))
        log.error("KILL_SWITCH_TRIPPED", reason=reason, action=self.action.value)

    def reset(self) -> None:
        """Clear the switch. Deliberately manual — never automatic."""
        self._event.clear()
        self._reason = ""
        self._tripped_at = None
        try:
            self.sentinel.unlink(missing_ok=True)
        except OSError as exc:
            log.error("kill_switch_sentinel_unlink_failed", error=str(exc))
        if self.store is not None:
            self.store.set_state(STATE_KEY, {"tripped": False})
        log.warning("kill_switch_reset")

    @property
    def should_flatten(self) -> bool:
        return self.is_tripped and self.action == KillSwitchAction.FLATTEN_ALL

    def status(self) -> dict[str, Any]:
        return {
            "tripped": self.is_tripped,
            "reason": self._reason,
            "action": self.action.value,
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
        }
