"""Write-ahead audit log for every model call.

The record is flushed to disk *before* any order derived from it is placed, so a
crash mid-order still leaves the reasoning behind that order on disk. Without
that ordering an audit could not answer the only question that matters after an
incident: why did it do that?
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger

log = get_logger(__name__)


class AuditLog:
    """Append-only JSONL, one file per trading day."""

    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        day = datetime.now(UTC).date().isoformat()
        return self.directory / f"llm-{day}.jsonl"

    def record(
        self,
        *,
        cycle_id: str,
        role: str,
        provider: str,
        host: str,
        model: str,
        strategy: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        raw_response: str,
        thinking: str = "",
        parsed: Any = None,
        validation_error: str = "",
        repair_attempted: bool = False,
        latency_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        from_cache: bool = False,
    ) -> None:
        if not self.enabled:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "cycle_id": cycle_id,
            "role": role,
            "provider": provider,
            "host": host,
            "model": model,
            "strategy": strategy,
            "messages": messages,
            "options": options,
            "raw_response": raw_response[:20000],
            "thinking": (thinking or "")[:8000],
            "parsed": parsed,
            "validation_error": validation_error,
            "repair_attempted": repair_attempted,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "from_cache": from_cache,
        }
        try:
            path = self._path()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
                # Force it to disk: the whole point is surviving a crash that
                # happens between this call and the order it justifies.
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as exc:  # noqa: BLE001 - auditing must never break trading
            log.warning("audit_write_failed", error=str(exc))

    def read_today(self, limit: int = 100) -> list[dict[str, Any]]:
        path = self._path()
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as exc:
            log.warning("audit_read_failed", error=str(exc))
            return []
        return out[-limit:]
