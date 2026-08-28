"""Ollama providers: cloud and local.

Both speak the native `/api/chat` endpoint rather than the OpenAI-compatible
`/v1/` layer. That is deliberate — `/v1/` does not support `tool_choice` (so a
tool call cannot be forced), and there are open reports of `json_schema` being
silently ignored and of tool calls being dropped when streaming through it.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from ollama import AsyncClient

from ..logging_setup import get_logger
from .base import (
    LLMError,
    LLMQuotaExhausted,
    LLMRequest,
    LLMResponse,
    ModelCapabilities,
    StructuredStrategy,
    schema_to_tool,
)

log = get_logger(__name__)


class OllamaProvider:
    """Shared implementation. Cloud and local differ only in host and auth."""

    name = "ollama"

    def __init__(self, host: str, api_key: str = "", timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = AsyncClient(host=self.host, headers=headers, timeout=timeout)
        self._headers = headers
        self._timeout = timeout

    # ------------------------------------------------------------------ #

    def _normalize_model(self, model: str) -> str:
        """Reconcile the `-cloud` suffix asymmetry.

        Cloud models carry a `-cloud` suffix when routed through a local ollama
        daemon, but calling ollama.com directly requires the bare tag. Callers
        should not have to care which host they are pointed at.
        """
        if "ollama.com" in self.host and model.endswith("-cloud"):
            return model[: -len("-cloud")]
        return model

    async def chat(self, req: LLMRequest) -> LLMResponse:
        model = self._normalize_model(req.model)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": req.messages,
            "stream": False,
            "options": req.options(),
            "keep_alive": req.keep_alive,
        }
        if req.think is not None:
            kwargs["think"] = req.think

        if req.strategy == StructuredStrategy.NATIVE_SCHEMA and req.schema:
            kwargs["format"] = req.schema
        elif req.strategy == StructuredStrategy.JSON_MODE:
            kwargs["format"] = "json"
        elif req.strategy == StructuredStrategy.SINGLE_TOOL and req.schema:
            kwargs["tools"] = [
                schema_to_tool(
                    req.schema,
                    req.tool_name,
                    "Submit your response. You must call this function exactly once.",
                )
            ]

        started = time.monotonic()
        try:
            raw = await self._client.chat(**kwargs)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise LLMQuotaExhausted(
                    f"Ollama returned 429 (rate limit or quota exhausted) from {self.host}"
                ) from exc
            if status in (401, 403):
                raise LLMError(
                    f"Ollama rejected credentials ({status}). Check OLLAMA_API_KEY."
                ) from exc
            raise LLMError(f"Ollama HTTP {status}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama request to {self.host} timed out") from exc
        except Exception as exc:
            msg = str(exc)
            # The client wraps some errors; sniff the ones that matter.
            if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
                raise LLMQuotaExhausted(f"Ollama quota/rate limit: {msg}") from exc
            if "401" in msg or "403" in msg or "unauthor" in msg.lower():
                raise LLMError(f"Ollama auth failure: {msg}") from exc
            raise LLMError(f"Ollama request failed: {msg}") from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        return self._to_response(raw, model, latency_ms)

    @staticmethod
    def _to_response(raw: Any, model: str, latency_ms: int) -> LLMResponse:
        data = raw if isinstance(raw, dict) else getattr(raw, "model_dump", lambda: {})()
        message = data.get("message") or {}

        tool_calls: list[dict[str, Any]] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            # Ollama returns arguments already parsed into an object, unlike
            # OpenAI which returns a JSON string.
            tool_calls.append({"name": fn.get("name", ""), "arguments": fn.get("arguments", {})})

        return LLMResponse(
            content=message.get("content") or "",
            thinking=message.get("thinking") or "",
            tool_calls=tool_calls,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            raw=data,
        )

    async def show(self, model: str) -> ModelCapabilities:
        """Read the model's advertised capabilities.

        This is how we find out whether a tag supports tool calling. It matters
        for Hermes in particular: some `hermes3` Modelfile templates predate
        Ollama's native tools plumbing, so the tool strategy has to be skipped
        and the `<tool_call>` XML convention driven by hand instead.
        """
        name = self._normalize_model(model)
        try:
            raw = await self._client.show(name)
            data: dict[str, Any] = (
                raw if isinstance(raw, dict) else getattr(raw, "model_dump", lambda: {})()
            )
            caps = data.get("capabilities") or []
            return ModelCapabilities(model=name, capabilities=[str(c) for c in caps])
        except Exception as exc:  # noqa: BLE001
            log.warning("model_show_failed", model=name, host=self.host, error=str(exc))
            return ModelCapabilities(model=name, capabilities=[])

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=self._headers) as c:
                r = await c.get(f"{self.host}/api/tags")
                return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=self._headers) as c:
                r = await c.get(f"{self.host}/api/tags")
                r.raise_for_status()
                return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001
            log.warning("list_models_failed", host=self.host, error=str(exc))
            return []


class OllamaCloudProvider(OllamaProvider):
    """https://ollama.com with a Bearer API key.

    Note that Ollama Cloud does not support schema-constrained structured
    outputs, so `StructuredStrategy.NATIVE_SCHEMA` will not hold here and
    capability negotiation is expected to settle on JSON mode.
    """

    name = "ollama_cloud"

    def __init__(self, api_key: str, host: str = "https://ollama.com", timeout: float = 300.0):
        if not api_key:
            raise LLMError(
                "Ollama Cloud requires an API key. Set OLLAMA_API_KEY in your .env "
                "(create one at https://ollama.com/settings/keys)."
            )
        super().__init__(host=host, api_key=api_key, timeout=timeout)


class OllamaLocalProvider(OllamaProvider):
    """A local ollama daemon. Supports schema-constrained decoding."""

    name = "ollama_local"

    def __init__(self, host: str = "http://localhost:11434", timeout: float = 600.0):
        super().__init__(host=host, api_key="", timeout=timeout)


def build_provider(
    provider: str, api_key: str = "", cloud_host: str = "https://ollama.com",
    local_host: str = "http://localhost:11434",
) -> OllamaProvider:
    if provider == "ollama_cloud":
        return OllamaCloudProvider(api_key=api_key, host=cloud_host)
    if provider == "ollama_local":
        return OllamaLocalProvider(host=local_host)
    raise ValueError(f"unknown LLM provider {provider!r}")
