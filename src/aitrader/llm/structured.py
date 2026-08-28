"""Getting reliably-typed objects out of a model that may not cooperate.

Ollama Cloud does not support schema-constrained decoding, so on the cloud path
nothing *mechanically* prevents the model from emitting prose, truncated JSON,
or a chatty preamble around the object we asked for. This module is what stands
between that and the risk gate.

Extraction is deliberately layered, and the terminal state is always a fail-safe
default rather than an exception that a caller might swallow: a malformed
generation must never become an order.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..logging_setup import get_logger
from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    StructuredStrategy,
    describe_schema,
)

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
#: Hermes models express structure with this convention rather than native tools.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove reasoning traces.

    Reasoning models sometimes leak `<think>` blocks into `content` instead of
    the separate `thinking` field. An unclosed block means the response hit the
    token cap mid-thought, so everything after it is discarded.
    """
    text = _THINK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def find_balanced_json(text: str) -> str | None:
    """Scan for the first balanced JSON object, ignoring braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json(response: LLMResponse) -> dict[str, Any] | None:
    """Recover a JSON object from a response, whatever shape it arrived in."""
    # 1. A native tool call carries already-parsed arguments.
    for call in response.tool_calls:
        args = call.get("arguments")
        if isinstance(args, dict) and args:
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    text = strip_thinking(response.content or "")
    if not text:
        return None

    # 2. The whole body is JSON (what JSON mode should produce).
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3. Hermes-style tool call tags.
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                # Hermes wraps the payload under "arguments".
                inner = parsed.get("arguments")
                return inner if isinstance(inner, dict) else parsed
        except json.JSONDecodeError:
            pass

    # 4. A fenced code block.
    for block in _FENCE_RE.findall(text):
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            candidate = find_balanced_json(block)
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

    # 5. Last resort: the first balanced object anywhere in the text.
    candidate = find_balanced_json(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def compact_errors(exc: ValidationError, limit: int = 8) -> str:
    """Render validation errors small enough to feed back for repair."""
    lines: list[str] = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        lines.append(f"- {loc}: {err.get('msg', 'invalid')}")
    return "\n".join(lines)


def schema_instruction(model_cls: type[BaseModel]) -> str:
    """The system-prompt fragment that carries the schema.

    On the cloud path this text is the *only* thing constraining output shape,
    so it is explicit to the point of being blunt.
    """
    return (
        "You must reply with a single JSON object and nothing else. "
        "No prose before or after it, no markdown fences, no explanation.\n\n"
        "The object must conform to this JSON Schema:\n\n"
        f"{describe_schema(model_cls.model_json_schema())}\n"
    )


class StructuredCaller:
    """Calls a provider and returns a validated pydantic model, or a safe default."""

    def __init__(self, provider: LLMProvider, audit: Any = None) -> None:
        self.provider = provider
        self.audit = audit

    async def call(
        self,
        req: LLMRequest,
        model_cls: type[T],
        context: dict[str, Any] | None = None,
        allow_repair: bool = True,
    ) -> tuple[T | None, LLMResponse | None, str]:
        """Return (parsed, response, error).

        `parsed` is None when the model could not produce valid output even
        after one repair attempt. The caller is expected to substitute a
        fail-safe default; this method never raises for a schema problem,
        because an exception here could be caught somewhere that treats it as
        "no opinion" rather than "do nothing".
        """
        req.schema = model_cls.model_json_schema()
        response = await self.provider.chat(req)

        parsed, err = self._validate(response, model_cls, context)
        if parsed is not None:
            return parsed, response, ""

        if not allow_repair:
            return None, response, err

        log.warning(
            "llm_schema_invalid_attempting_repair",
            model=req.model,
            strategy=req.strategy.value,
            error=err[:400],
        )

        repair_req = LLMRequest(
            model=req.model,
            messages=[
                *req.messages,
                {"role": "assistant", "content": (response.content or "")[:4000]},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply did not match the required schema.\n\n"
                        f"Problems:\n{err}\n\n"
                        "Reply again with ONLY the corrected JSON object. "
                        "No prose, no markdown fences."
                    ),
                },
            ],
            schema=req.schema,
            strategy=req.strategy,
            temperature=0.0,
            num_ctx=req.num_ctx,
            num_predict=req.num_predict,
            seed=req.seed,
            keep_alive=req.keep_alive,
            timeout=req.timeout,
            tool_name=req.tool_name,
        )
        repair_response = await self.provider.chat(repair_req)
        parsed, err2 = self._validate(repair_response, model_cls, context)
        if parsed is not None:
            log.info("llm_schema_repair_succeeded", model=req.model)
            return parsed, repair_response, ""

        log.error(
            "llm_schema_repair_failed",
            model=req.model,
            first_error=err[:200],
            second_error=err2[:200],
        )
        return None, repair_response, f"{err} | after repair: {err2}"

    @staticmethod
    def _validate(
        response: LLMResponse, model_cls: type[T], context: dict[str, Any] | None
    ) -> tuple[T | None, str]:
        payload = extract_json(response)
        if payload is None:
            preview = (response.content or "")[:200].replace("\n", " ")
            return None, f"no JSON object found in response (starts with: {preview!r})"
        try:
            return model_cls.model_validate(payload, context=context or {}), ""
        except ValidationError as exc:
            return None, compact_errors(exc)
        except Exception as exc:  # noqa: BLE001
            return None, f"unexpected validation failure: {exc}"


async def negotiate_strategy(
    provider: LLMProvider,
    model: str,
    probe_cls: type[BaseModel],
    candidates: list[StructuredStrategy] | None = None,
) -> StructuredStrategy:
    """Find the cheapest strategy that actually works for this model and host.

    Negotiated once at startup and cached, rather than re-tried on every call:
    walking the whole ladder each cycle would multiply both latency and cloud
    quota usage for no benefit.
    """
    from .base import STRATEGY_ORDER

    caps = await provider.show(model)
    order = candidates or STRATEGY_ORDER

    for strategy in order:
        if strategy == StructuredStrategy.SINGLE_TOOL and not caps.supports_tools:
            log.info("strategy_skipped_no_tool_capability", model=model)
            continue

        req = LLMRequest(
            model=model,
            messages=[
                {"role": "system", "content": schema_instruction(probe_cls)},
                {
                    "role": "user",
                    "content": "Reply with a minimal valid object for the schema above.",
                },
            ],
            schema=probe_cls.model_json_schema(),
            strategy=strategy,
            temperature=0.0,
            num_predict=512,
            num_ctx=4096,
            timeout=90.0,
        )
        try:
            response = await provider.chat(req)
        except Exception as exc:  # noqa: BLE001
            log.warning("strategy_probe_error", strategy=strategy.value, error=str(exc))
            continue

        if extract_json(response) is not None:
            log.info(
                "structured_strategy_negotiated",
                model=model,
                host=getattr(provider, "host", ""),
                strategy=strategy.value,
                capabilities=caps.capabilities,
            )
            return strategy
        log.info("strategy_probe_no_json", strategy=strategy.value, model=model)

    log.warning("no_structured_strategy_worked_falling_back", model=model)
    return StructuredStrategy.TEXT_EXTRACT
