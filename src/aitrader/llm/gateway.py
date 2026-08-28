"""The single entry point for every model call.

Responsibilities: serialize calls to the account's concurrency tier, retry with
backoff, negotiate the structured-output strategy once and cache it, enforce a
hard deadline, record an audit trail, and degrade safely when the provider is
unreachable or out of quota.

Degradation is the important part. Ollama Cloud tiers allow 1 (Free), 3 (Pro) or
10 (Max) concurrent models with weekly GPU-time quotas, and no endpoint exposes
remaining quota. So when calls start failing we stop opening new positions and
keep managing existing ones — which is safe precisely because stops and targets
are real broker-side orders, not something this process is holding in memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from pydantic import BaseModel

from ..logging_setup import get_logger
from .audit import AuditLog
from .base import (
    LLMError,
    LLMQuotaExhausted,
    LLMRequest,
    LLMResponse,
    StructuredStrategy,
)
from .providers import OllamaProvider
from .structured import StructuredCaller, negotiate_strategy, schema_instruction

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

CAPABILITY_TTL = timedelta(hours=24)


class _Probe(BaseModel):
    """Tiny schema used only to negotiate the structured-output strategy."""

    ok: bool = True


@dataclass
class LLMHealth:
    available: bool = True
    consecutive_failures: int = 0
    consecutive_schema_failures: int = 0
    quota_exhausted: bool = False
    last_error: str = ""
    last_success: datetime | None = None
    total_calls: int = 0
    total_tokens: int = 0
    #: Set once the model has failed schema validation too many times in a row;
    #: the system then runs on deterministic rules for the rest of the day.
    deterministic_only: bool = False
    strategy: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "consecutive_failures": self.consecutive_failures,
            "quota_exhausted": self.quota_exhausted,
            "deterministic_only": self.deterministic_only,
            "last_error": self.last_error[:300],
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "strategy": self.strategy,
        }


@dataclass
class LLMGateway:
    provider: OllamaProvider
    audit: AuditLog
    store: Any = None
    max_concurrent: int = 1
    max_retries: int = 3
    retry_base_delay: float = 2.0
    cache_enabled: bool = True
    #: True only in replay/backtest. Serving a cached decision to a live cycle
    #: would act on a market that has since moved, so this stays False in
    #: paper and live trading and the cache is write-only there.
    cache_read_enabled: bool = False
    max_schema_failures: int = 3

    health: LLMHealth = field(default_factory=LLMHealth)
    _sem: asyncio.Semaphore = field(init=False)
    _strategies: dict[str, StructuredStrategy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrent)

    # ------------------------------------------------------------------ #
    # strategy negotiation
    # ------------------------------------------------------------------ #

    async def ensure_strategy(self, model: str) -> StructuredStrategy:
        """Resolve (and cache) how to get structured output from this model."""
        if model in self._strategies:
            return self._strategies[model]

        if self.store is not None:
            row = self.store.load_capability(self.provider.host, model)
            if row:
                try:
                    checked = datetime.fromisoformat(row["checked_at"])
                    if datetime.now(UTC) - checked < CAPABILITY_TTL:
                        strategy = StructuredStrategy(row["strategy"])
                        self._strategies[model] = strategy
                        self.health.strategy = strategy.value
                        log.info(
                            "structured_strategy_from_cache",
                            model=model, strategy=strategy.value,
                        )
                        return strategy
                except (ValueError, KeyError):
                    pass

        strategy = await negotiate_strategy(self.provider, model, _Probe)
        self._strategies[model] = strategy
        self.health.strategy = strategy.value
        if self.store is not None:
            caps = await self.provider.show(model)
            self.store.save_capability(
                self.provider.host, model, strategy.value, caps.capabilities
            )
        return strategy

    # ------------------------------------------------------------------ #
    # the call path
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(req: LLMRequest) -> str:
        payload = json.dumps(
            {
                "model": req.model,
                "messages": req.messages,
                "options": req.options(),
                "strategy": req.strategy.value,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(
        self,
        *,
        role: str,
        cycle_id: str,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
        context: dict[str, Any] | None = None,
        temperature: float = 0.0,
        num_ctx: int = 32768,
        num_predict: int = 2048,
        seed: int | None = 42,
        keep_alive: str = "30m",
        timeout: float = 180.0,
        think: bool | str | None = None,
    ) -> T | None:
        """Ask the model for a validated object, or None if it could not deliver.

        None is a normal outcome, not an error: callers substitute a fail-safe
        default (usually "do nothing"). Raising here would risk a caller
        catching it and interpreting it as "no opinion".
        """
        if self.health.deterministic_only:
            return None

        strategy = await self.ensure_strategy(model)

        # The schema goes into the prompt regardless of strategy. On the cloud
        # path `format` does nothing, so this text is the only thing steering
        # the output shape.
        system_prompt = f"{system.strip()}\n\n{schema_instruction(response_model)}"

        req = LLMRequest(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            schema=response_model.model_json_schema(),
            strategy=strategy,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            seed=seed,
            keep_alive=keep_alive,
            timeout=timeout,
            think=think,
        )

        cache_key = self._cache_key(req)
        if self.cache_read_enabled and self.store is not None:
            cached = self.store.cache_get(cache_key)
            if cached is not None:
                try:
                    return response_model.model_validate(cached, context=context or {})
                except Exception:  # noqa: BLE001
                    pass

        caller = StructuredCaller(self.provider)
        parsed: T | None = None
        response: LLMResponse | None = None
        error = ""

        async with self._sem:
            for attempt in range(self.max_retries + 1):
                try:
                    async with asyncio.timeout(timeout):
                        parsed, response, error = await caller.call(
                            req, response_model, context=context
                        )
                    self._on_success(response)
                    break
                except LLMQuotaExhausted as exc:
                    self.health.quota_exhausted = True
                    self.health.available = False
                    self.health.last_error = str(exc)
                    log.error("llm_quota_exhausted", error=str(exc))
                    return None
                except TimeoutError:
                    error = f"timed out after {timeout}s"
                    self._on_failure(error)
                    if attempt >= self.max_retries:
                        log.error("llm_timeout_giving_up", role=role, model=model)
                        return None
                except LLMError as exc:
                    error = str(exc)
                    self._on_failure(error)
                    if attempt >= self.max_retries:
                        log.error("llm_failed_giving_up", role=role, error=error)
                        return None
                # Exponential backoff with jitter.
                delay = self.retry_base_delay * (2**attempt) * (0.5 + random.random())
                log.warning(
                    "llm_retry", role=role, attempt=attempt + 1,
                    delay=round(delay, 1), error=error[:200],
                )
                await asyncio.sleep(delay)

        # Audit before anything downstream can act on this.
        self.audit.record(
            cycle_id=cycle_id,
            role=role,
            provider=self.provider.name,
            host=self.provider.host,
            model=model,
            strategy=strategy.value,
            messages=req.messages,
            options=req.options(),
            raw_response=(response.content if response else ""),
            thinking=(response.thinking if response else ""),
            parsed=parsed.model_dump(mode="json") if parsed else None,
            validation_error=error,
            repair_attempted=bool(error) or (response is not None and parsed is None),
            latency_ms=(response.latency_ms if response else 0),
            prompt_tokens=(response.prompt_tokens if response else 0),
            completion_tokens=(response.completion_tokens if response else 0),
        )

        if parsed is None:
            self.health.consecutive_schema_failures += 1
            if self.health.consecutive_schema_failures >= self.max_schema_failures:
                self.health.deterministic_only = True
                log.error(
                    "llm_disabled_for_session",
                    reason="repeated schema failures",
                    failures=self.health.consecutive_schema_failures,
                )
            return None

        self.health.consecutive_schema_failures = 0
        if self.cache_enabled and self.store is not None:
            try:
                self.store.cache_put(
                    cache_key, model, {"messages": req.messages}, parsed.model_dump(mode="json")
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("llm_cache_write_failed", error=str(exc))

        return parsed

    # ------------------------------------------------------------------ #

    def _on_success(self, response: LLMResponse | None) -> None:
        self.health.available = True
        self.health.consecutive_failures = 0
        self.health.quota_exhausted = False
        self.health.last_success = datetime.now(UTC)
        self.health.total_calls += 1
        if response:
            self.health.total_tokens += response.prompt_tokens + response.completion_tokens

    def _on_failure(self, error: str) -> None:
        self.health.consecutive_failures += 1
        self.health.last_error = error
        if self.health.consecutive_failures >= 3:
            self.health.available = False

    async def check_health(self) -> bool:
        ok = await self.provider.healthy()
        self.health.available = ok and not self.health.quota_exhausted
        return self.health.available
